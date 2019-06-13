from functools import lru_cache
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data import all_video_paths, count_images, plot
from utils import constants, personal_constants


class X300VWDataset(Dataset):
    def __init__(
        self,
        window_size_gaussian: int = 7,
        n_images_per_sample: int = 3,
        mu: float = 0.0,
        sigma: float = 1 / 3,
        transform: Optional = None,
    ) -> None:
        self._all_videos = all_video_paths(personal_constants.DATASET_300VW_OUTPUT_PATH)
        self._n_images_per_video = count_images(
            self._all_videos,
            constants.DATASET_300VW_ANNOTATIONS_OUTPUT_FOLDER,
            constants.DATASET_300VW_ANNOTATIONS_OUTPUT_EXTENSION,
        )
        self._n_images = sum(self._n_images_per_video)
        self._cumulative_n_images = self._cumulative_sum()
        self._n_images_per_sample = n_images_per_sample
        print(f'n images in dataset: {self._n_images}')

        self._window_size_gaussian = window_size_gaussian
        assert self._window_size_gaussian > 0 and self._window_size_gaussian % 2 == 1
        self._window_radius = self._window_size_gaussian // 2
        self._gaussian = self._precompute_gaussian(mu, sigma)

        self._transform = transform

    def _cumulative_sum(self) -> List[int]:
        cumulative_sum = 0
        cumulative_n_images = [cumulative_sum]
        for n_images_in_video in self._n_images_per_video:
            cumulative_sum += n_images_in_video
            cumulative_n_images.append(cumulative_sum)

        return cumulative_n_images

    def _precompute_gaussian(self, mu: float, sigma: float) -> np.ndarray:
        x, y = np.meshgrid(
            np.linspace(-1, 1, self._window_size_gaussian),
            np.linspace(-1, 1, self._window_size_gaussian),
        )
        d = np.sqrt(x * x + y * y)
        return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    def __len__(self):
        return self._n_images

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:

        for video_index, (lower_bound, upper_bound) in enumerate(
            zip(self._cumulative_n_images, self._cumulative_n_images[1:])
        ):
            if lower_bound <= index < upper_bound:
                break

        assert upper_bound - lower_bound == self._n_images_per_video[video_index]
        # +1 because frames are numerated starting 1
        frame_index = index - lower_bound + 1
        frame_indices = self._random_sample_indices(video_index, frame_index)

        sample = [
            {
                'image': self._load_image(video_index, fi),
                'landmarks': self._load_landmarks(video_index, fi),
            }
            for fi in frame_indices
        ]

        if self._transform:
            sample = self._transform(sample)

        return sample

    def _random_sample_indices(self, video_index: int, frame_index: int) -> List[int]:
        frame_indices = torch.randint(
            0,
            self._n_images_per_video[video_index],
            size=(self._n_images_per_sample,),
            dtype=torch.int64,
        ).numpy()
        frame_indices += 1
        frame_indices = [fi for fi in frame_indices if fi != frame_index]
        frame_indices = [frame_index] + frame_indices[: self._n_images_per_sample - 1]
        assert len(frame_indices) == self._n_images_per_sample
        return frame_indices

    def _load_image(self, video_index: int, frame_index: int) -> np.ndarray:
        frame_input_path = (
            self._all_videos[video_index]
            / constants.DATASET_300VW_IMAGES_OUTPUT_FOLDER
            / (
                f'{frame_index:{constants.DATASET_300VW_NUMBER_FORMAT}}'
                + f'.{constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION}'
            )
        )
        if not frame_input_path.exists():
            raise Exception(f'Image does not exist: {frame_input_path}')
        image = cv2.imread(str(frame_input_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(float)
        image = (image / 255) * 2 - 1
        assert -1 <= image.min() <= image.max() <= 1
        image = np.moveaxis(image, -1, 0)
        assert image.shape == (
            constants.INPUT_CHANNELS,
            constants.IMSIZE,
            constants.IMSIZE,
        ), f"wrong shape {image.shape}"
        return image

    def _load_landmarks(self, video_index: int, frame_index: int) -> np.ndarray:
        annotation_input_path = (
            self._all_videos[video_index]
            / constants.DATASET_300VW_ANNOTATIONS_OUTPUT_FOLDER
            / (
                f'{frame_index:{constants.DATASET_300VW_NUMBER_FORMAT}}'
                + f'.{constants.DATASET_300VW_ANNOTATIONS_OUTPUT_EXTENSION}'
            )
        )
        if not annotation_input_path.exists():
            raise Exception(f'Landmarks file does not exist: {annotation_input_path}')
        single_dim_landmarks = np.loadtxt(annotation_input_path)

        landmarks = np.empty(
            (constants.DATASET_300VW_N_LANDMARKS, constants.IMSIZE, constants.IMSIZE)
        )
        assert single_dim_landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
        for landmark_index in range(single_dim_landmarks.shape[0]):
            start_indices = single_dim_landmarks[landmark_index, :]
            landmarks[landmark_index, :, :] = self._landmark_to_channel(
                start_indices[0], start_indices[1]
            )

        return landmarks

    @lru_cache()
    def _landmark_to_channel(self, x_1: int, y_1: int) -> np.ndarray:
        landmark_channel = np.zeros((constants.IMSIZE, constants.IMSIZE))
        start_indices_landmarks = np.asarray([x_1, y_1], dtype=int)
        start_indices_landmarks -= self._window_radius

        end_indices_landmarks = start_indices_landmarks + self._window_size_gaussian
        if any(start_indices_landmarks > constants.IMSIZE) or any(
            end_indices_landmarks < 0
        ):
            return landmark_channel

        start_indices_gaussian = np.where(
            start_indices_landmarks < 0, abs(start_indices_landmarks), 0
        )
        start_indices_landmarks = np.where(
            start_indices_landmarks < 0, 0, start_indices_landmarks
        )
        end_indices_gaussian = self._window_size_gaussian - np.where(
            end_indices_landmarks > constants.IMSIZE,
            end_indices_landmarks - constants.IMSIZE,
            0,
        )
        end_indices_landmarks = np.where(
            end_indices_landmarks > constants.IMSIZE,
            constants.IMSIZE,
            end_indices_landmarks,
        )

        assert all(
            (end_indices_landmarks - start_indices_landmarks)
            == (end_indices_gaussian - start_indices_gaussian)
        )

        landmark_channel[
            start_indices_landmarks[1] : end_indices_landmarks[1],
            start_indices_landmarks[0] : end_indices_landmarks[0],
        ] = self._gaussian[
            start_indices_gaussian[1] : end_indices_gaussian[1],
            start_indices_gaussian[0] : end_indices_gaussian[0],
        ]

        return landmark_channel


def _test_return() -> None:
    dataset = X300VWDataset()
    n_images = len(dataset)
    dataset_indices = np.random.randint(0, n_images, size=3)
    for batch_index, dataset_index in enumerate(dataset_indices):
        batch = dataset[dataset_index]
        for sample_index, sample in enumerate(batch):
            image, landmarks = sample['image'], sample['landmarks']
            assert image.shape == (
                constants.INPUT_CHANNELS,
                constants.IMSIZE,
                constants.IMSIZE,
            )
            assert landmarks.shape == (
                constants.DATASET_300VW_N_LANDMARKS,
                constants.IMSIZE,
                constants.IMSIZE,
            )
            print((batch_index, sample_index), image.shape, landmarks.shape)
            image = np.moveaxis(image, 0, -1)
            image = ((image + 1) / 2) * 255

            overlay_alpha = 1.0
            color_index = 'rgb'.index('r')

            mask = np.zeros(image.shape, dtype=float)
            mask[..., color_index] = 255
            for index in range(landmarks.shape[0]):
                image += overlay_alpha * mask * landmarks[index, :, :, np.newaxis]

            image[image > 255] = 255
            image = image.astype('uint8')
            plot(image)

    input('Press [enter] to exit.')


if __name__ == '__main__':
    _test_return()
