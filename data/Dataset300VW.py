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

        self._transform = transform

    def __len__(self):
        return self._n_images

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:

        for video_index, (lower_bound, upper_bound) in enumerate(
                zip(self._cumulative_n_images, self._cumulative_n_images[1:])
        ):
            if lower_bound <= index < upper_bound:
                break

        # +1 because frames are numerated starting 1
        frame_index = index - lower_bound + 1
        frame_input_path = (
                self._all_videos[video_index]
                / constants.DATASET_300VW_IMAGES_OUTPUT_FOLDER
                / (
                        f'{frame_index:{constants.DATASET_300VW_NUMBER_FORMAT}}'
                        + f'.{constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION}'
                )
        )
        image = cv2.imread(str(frame_input_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(float)
        image = (image / 255) * 2 - 1
        image = np.moveaxis(image, 2, 0)
        assert image.shape == (constants.INPUT_CHANNELS, constants.IMSIZE, constants.IMSIZE), f"wrong shape {image.shape}"
        assert -1 <= image.min() <= image.max() <= 1

        annotation_input_path = (
                self._all_videos[video_index]
                / constants.DATASET_300VW_ANNOTATIONS_OUTPUT_FOLDER
                / (
                        f'{frame_index:{constants.DATASET_300VW_NUMBER_FORMAT}}'
                        + f'.{constants.DATASET_300VW_ANNOTATIONS_OUTPUT_EXTENSION}'
                )
        )
        landmarks = np.loadtxt(annotation_input_path)
        sample = {'image': image, 'landmarks': landmarks}

        if self._transform:
            sample = self._transform(sample)

        return sample


def _test():
    dataset = X300VWDataset()
    n_images = len(dataset)
    print(f'n videos: {n_images}')
    dataset_indices = np.random.randint(0, n_images, size=4)
    for index, sample_index in enumerate(dataset_indices):
        sample = dataset[sample_index]
        image, landmarks = sample['image'], sample['landmarks']
        assert image.shape == (constants.IMSIZE, constants.IMSIZE, 3)
        assert landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
        print(index, image.shape, landmarks.shape)
        image = ((image + 1) / 2) * 255
        image = image.astype('uint8')
        plot(image, landmarks)
    input('Press [enter] to exit.')


if __name__ == '__main__':
    _test()
