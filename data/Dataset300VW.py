from typing import Dict, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from data import all_video_paths, count_images, plot
from utils import constants, personal_constants


class X300VWDataset(Dataset):
    def __init__(self, transform: Optional = None) -> None:
        self._all_videos = all_video_paths(personal_constants.DATASET_300VW_OUTPUT_PATH)
        n_images_per_video = count_images(
            self._all_videos,
            constants.DATASET_300VW_ANNOTATIONS_OUTPUT_FOLDER,
            constants.DATASET_300VW_ANNOTATIONS_OUTPUT_EXTENSION,
        )
        self._n_images = sum(n_images_per_video)
        cumulative_sum = 0
        self._cumulative_n_images = [cumulative_sum]
        for n_images_in_video in n_images_per_video:
            cumulative_sum += n_images_in_video
            self._cumulative_n_images.append(cumulative_sum)
        self._cumulative_n_images.append(float('inf'))
        self._n_images_per_sample = 3

        self._window_size_gaussian = 7
        assert self._window_size_gaussian > 0 and self._window_size_gaussian % 2 == 1
        self._window_radius = self._window_size_gaussian // 2
        x, y = np.meshgrid(
            np.linspace(-1, 1, self._window_size_gaussian),
            np.linspace(-1, 1, self._window_size_gaussian),
        )
        d = np.sqrt(x * x + y * y)
        mu, sigma = 0.0, 1 / 3
        self._gaussian = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

        self._transform = transform

    def __len__(self):
        return self._n_images

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:

        for video_index, (lower_bound, upper_bound) in enumerate(
            zip(self._cumulative_n_images, self._cumulative_n_images[1:])
        ):
            if lower_bound <= index < upper_bound:
                break

        if np.isinf(upper_bound):
            upper_bound = self._n_images

        frame_indices = np.random.randint(
            0, upper_bound - lower_bound, self._n_images_per_sample
        )
        # +1 because frames are numerated starting 1
        frame_index = index - lower_bound + 1
        frame_indices += 1
        frame_indices = [fi for fi in frame_indices if fi != frame_index]
        frame_indices = [frame_index] + frame_indices[: self._n_images_per_sample - 1]
        assert len(frame_indices) == self._n_images_per_sample

        samples = [
            (self._load_image(video_index, fi), self._load_landmarks(video_index, fi))
            for fi in frame_indices
        ]
        keys = ('image', 'landmarks')
        sample = [{k: v for k, v in zip(keys, values)} for values in samples]

        if self._transform:
            sample = self._transform(sample)

        return sample

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

        landmarks = np.zeros(
            (
                constants.DATASET_300VW_N_LANDMARKS,
                constants.IMSIZE + self._window_size_gaussian - 1,
                constants.IMSIZE + self._window_size_gaussian - 1,
            )
        )
        assert single_dim_landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
        for landmark_index in range(single_dim_landmarks.shape[0]):
            # because landmarks is zero padded, the start indices are the actual landmark centers
            start_indices = single_dim_landmarks[landmark_index, :]
            start_indices = np.round(start_indices).astype(int)
            end_indices = start_indices + self._window_size_gaussian
            if np.any(start_indices < 0) or np.any(
                end_indices >= constants.IMSIZE + self._window_size_gaussian - 1
            ):
                continue
            # landmarks[0] is x, landmarks[1] is y
            landmarks[
                landmark_index,
                start_indices[1] : end_indices[1],
                start_indices[0] : end_indices[0],
            ] = np.copy(self._gaussian)
        landmarks = landmarks[
            :,
            self._window_radius : constants.IMSIZE + self._window_radius,
            self._window_radius : constants.IMSIZE + self._window_radius,
        ]

        return landmarks


def _test():
    dataset = X300VWDataset()
    n_images = len(dataset)
    print(f'n videos: {n_images}')
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

            overlay_color = 'r'
            if overlay_color == 'r':
                color_index = 2
            elif overlay_color == 'g':
                color_index = 1
            elif overlay_color == 'b':
                color_index = 0

            mask = np.zeros(image.shape, dtype=float)
            mask[..., color_index] = 255
            output = image
            for index in range(landmarks.shape[0]):
                output += overlay_alpha * mask * landmarks[index, :, :, np.newaxis]

            output[output > 255] = 255
            output = output.astype('uint8')

            image = image.astype('uint8')
            plot(image, None)
            plot(output, None)
    input('Press [enter] to exit.')


if __name__ == '__main__':
    _test()
