from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from data import all_video_paths, count_images, plot
from utils import constants, personal_constants


class X300VWDataset(Dataset):
    def __init__(
        self,
        mode: constants.Dataset300VWMode,
        window_size_gaussian: int = 7,
        n_images_per_sample: int = 3,
        mu: float = 0.0,
        sigma: float = 1 / 3,
        transform: Optional = None,
    ) -> None:
        self._all_videos = all_video_paths(personal_constants.DATASET_300VW_OUTPUT_PATH)
        self._all_videos = self._filter(mode)

        self._n_images_per_video = count_images(
            self._all_videos,
            constants.DATASET_300VW_IMAGES_OUTPUT_FOLDER,
            constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION,
        )
        self._n_images = sum(self._n_images_per_video)
        self._cumulative_n_images = self._cumulative_sum()
        self._n_images_per_sample = n_images_per_sample
        print(f'n images in dataset: {self._n_images}')

        self._all_landmarks = self._load_all_landmarks()

        self._window_size_gaussian = window_size_gaussian
        assert self._window_size_gaussian > 0 and self._window_size_gaussian % 2 == 1
        self._window_radius = self._window_size_gaussian // 2
        self._gaussian = self._precompute_gaussian(mu, sigma)

        self._transform = transform

    def _filter(self, mode: constants.Dataset300VWMode) -> List[Path]:
        filtered_list = [
            video_path
            for video_path in self._all_videos
            if video_path.stem in mode.value
        ]
        if len(mode.value) != len(filtered_list):
            raise Exception(
                f'Videos are missing from dataset. Should have the following: {mode.value}'
            )
        return filtered_list

    def _load_all_landmarks(self) -> List[np.ndarray]:
        print('Loading landmarks positions into memory...')
        return [
            np.load(video_path / 'annotations.npy')
            for video_path in tqdm(self._all_videos)
            if (video_path / 'annotations.npy').exists()
        ]

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
        frames_indices = self._random_sample_indices(video_index, frame_index)

        sample = [
            {
                'image': self._load_image(video_index, fi),
                'landmarks': self._load_landmarks(video_index, fi),
            }
            for fi in frames_indices
        ]

        if self._transform:
            sample = self._transform(sample)

        return sample

    def _random_sample_indices(self, video_index: int, frame_index: int) -> List[int]:
        frames_indices = torch.randperm(self._n_images_per_video[video_index], dtype=torch.int64).numpy()
        frames_indices = frames_indices[: self._n_images_per_sample]
        frames_indices += 1
        frames_indices = [fi for fi in frames_indices if fi != frame_index]
        frames_indices = [frame_index] + frames_indices[: self._n_images_per_sample - 1]
        assert len(frames_indices) == self._n_images_per_sample
        return frames_indices


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
        return image

    def _load_landmarks(self, video_index: int, frame_index: int) -> np.ndarray:
        single_dim_landmarks = self._all_landmarks[video_index][frame_index - 1, :, :]
        landmarks = np.empty(
            (constants.IMSIZE, constants.IMSIZE, constants.DATASET_300VW_N_LANDMARKS)
        )
        assert single_dim_landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
        for landmark_index in range(single_dim_landmarks.shape[0]):
            start_indices = single_dim_landmarks[landmark_index, :]
            landmarks[:, :, landmark_index] = self._landmark_to_channel(
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
    dataset = X300VWDataset(constants.Dataset300VWMode.ALL)
    n_images = len(dataset)
    dataset_indices = np.random.randint(0, n_images, size=3)
    for batch_index, dataset_index in enumerate(dataset_indices):
        batch = dataset[dataset_index]
        for sample_index, sample in enumerate(batch):
            image, landmarks = sample['image'], sample['landmarks']
            assert image.shape == (
                constants.IMSIZE,
                constants.IMSIZE,
                constants.INPUT_CHANNELS,
            )
            assert landmarks.shape == (
                constants.IMSIZE,
                constants.IMSIZE,
                constants.DATASET_300VW_N_LANDMARKS,
            )
            print((batch_index, sample_index), image.shape, landmarks.shape)
            plot(image, landmarks_in_channel=landmarks)

    input('Press [enter] to exit.')


def _test_random_sampling() -> None:
    from torch.utils.data import DataLoader

    dataset = X300VWDataset(constants.Dataset300VWMode.ALL)
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=constants.DEBUG_BATCH_SIZE
    )
    # change return of __get_item__ to
    # return frames_indices
    raise Exception()
    n_epochs = 3
    frames_indices = [
        _get_all_frames_indices(dataloader, dataset) for _ in range(n_epochs)
    ]
    for i in range(n_epochs):
        for j in range(i + 1, n_epochs):
            assert (frames_indices[i] != frames_indices[j]).any()
        print(frames_indices[i][:2, :2, :])


def _get_all_frames_indices(dataloader, dataset):
    # not empty because last batch might be smaller
    frames_indices = np.zeros(
        (len(dataloader), constants.DEBUG_BATCH_SIZE, dataset._n_images_per_sample),
        dtype=int,
    )
    for batch_index, batch in enumerate(dataloader):
        for sample_index, sample in enumerate(batch):
            frames_indices[batch_index, : len(sample), sample_index] = sample
    return frames_indices


if __name__ == '__main__':
    _test_return()
    # _test_random_sampling()
