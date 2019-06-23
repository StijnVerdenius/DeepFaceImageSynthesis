from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import cv2
from data import all_video_paths, count_images, plot
from utils import constants, data_utils, personal_constants


class X300VWDataset(Dataset):
    def __init__(
        self,
        mode: constants.Dataset300VWMode,
        window_size_gaussian: int = 7,
        n_images_per_sample: int = 3,
        mu: float = 0.0,
        sigma: float = 1 / 3,
        transform: Optional = None,
        n_videos_limit: Optional[int] = None,
    ) -> None:
        self._all_videos = all_video_paths(personal_constants.DATASET_300VW_OUTPUT_PATH)
        if n_videos_limit is None:
            self._all_videos = self._filter(mode)
            self._all_videos = self._sort(mode)
        else:
            print(f'Limited dataset to first {n_videos_limit} videos!')
            self._all_videos = self._all_videos[:n_videos_limit]

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

        self._transform = transform

    def _filter(self, mode: constants.Dataset300VWMode) -> List[Path]:
        filtered_list = [
            video_path
            for video_path in self._all_videos
            if video_path.stem in mode.value
        ]
        if len(mode.value) != len(filtered_list):
            # raise Exception(
            #     f'WARNING: Videos are missing from dataset. Should have the following:'
            #     + f'{mode.value} but actually have {filtered_list}'
            # )
            print(
                f'WARNING: Videos are missing from dataset. Should have the following:'
                + f'{mode.value} but actually have {filtered_list}'
            )
        return filtered_list

    def _sort(self, mode: constants.Dataset300VWMode) -> List[Path]:
        name_to_path = {p.name: p for p in self._all_videos}
        sorted_list = [name_to_path[video_name] for video_name in mode.value]
        return sorted_list

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
                'landmarks': data_utils.single_to_multi_dim_landmarks(
                    self._all_landmarks[video_index][fi - 1, :, :],
                    constants.DATASET_300VW_IMSIZE,
                ),
            }
            for fi in frames_indices
        ]

        if self._transform:
            sample = self._transform(sample)

        return sample

    def _random_sample_indices(self, video_index: int, frame_index: int) -> List[int]:
        frames_indices = torch.randperm(
            self._n_images_per_video[video_index], dtype=torch.int64
        ).numpy()
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


def _test_return() -> None:
    dataset = X300VWDataset(constants.Dataset300VWMode.ALL)
    n_images = len(dataset)
    dataset_indices = np.random.randint(0, n_images, size=3)[:3]
    for batch_index, dataset_index in enumerate(dataset_indices):
        batch = dataset[dataset_index]
        for sample_index, sample in enumerate(batch):
            image, landmarks = sample['image'], sample['landmarks']
            assert image.shape == (
                constants.DATASET_300VW_IMSIZE,
                constants.DATASET_300VW_IMSIZE,
                constants.INPUT_CHANNELS,
            )
            assert landmarks.shape == (
                constants.DATASET_300VW_IMSIZE,
                constants.DATASET_300VW_IMSIZE,
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
