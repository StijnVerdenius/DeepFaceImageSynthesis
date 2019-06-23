from functools import lru_cache
from typing import Tuple

import numpy as np

from utils import constants


def landmarks_to_box(
    landmarks: np.ndarray, image_size: Tuple[int, int, int]
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [
        landmarks[:, 0].min(),
        landmarks[:, 1].min(),
        landmarks[:, 0].max(),
        landmarks[:, 1].max(),
    ]

    x1, y1 = [t - constants.DATASET_300VW_PADDING for t in (x1, y1)]
    x2, y2 = [t + constants.DATASET_300VW_PADDING for t in (x2, y2)]

    box_height, box_width = y2 - y1 + 1, x2 - x1 + 1
    assert box_height > 1 and box_width > 1
    box_radius = box_height if box_height > box_width else box_width
    box_radius /= 2
    box_radius *= constants.DATASET_300VW_EXPAND_RATIO

    # landmarks can be out of image, but that's okay, we'll still export them.
    image_height, image_width, _ = image_size
    center_x = int(landmarks[constants.DATASET_300VW_CENTER_LANDMARK_INDEX, 0])
    center_y = int(landmarks[constants.DATASET_300VW_CENTER_LANDMARK_INDEX, 1])
    box_radius = min(
        box_radius, center_x, center_y, image_width - center_x, image_height - center_y
    )
    box_radius = int(box_radius)
    x1, y1 = [t - box_radius for t in (center_x, center_y)]
    x2, y2 = [t + box_radius for t in (center_x, center_y)]
    assert all(isinstance(t, int) for t in (x1, y1, x2, y2))

    assert abs((x2 - x1) - (y2 - y1)) == 0
    assert 0 <= x1 <= x2 <= image_width and 0 <= y1 <= y2 <= image_height

    return x1, y1, x2, y2


def extract(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    extraction = image[y1 : y2 + 1, x1 : x2 + 1, ...]
    return extraction


def offset_landmarks(
    landmarks: np.ndarray, box: Tuple[int, int, int, int]
) -> np.ndarray:
    x1, y1, x2, y2 = box
    landmarks = np.copy(landmarks)
    landmarks[:, 0] -= x1
    landmarks[:, 1] -= y1
    return landmarks


def rescale_landmarks(
    landmarks: np.ndarray, image_shape: Tuple[int, int, int], image_size: int
) -> np.ndarray:
    height, width, n_channels = image_shape
    landmarks = np.copy(landmarks)
    height_factor = 1 / height * image_size
    width_factor = 1 / width * image_size
    landmarks[:, 0] *= width_factor
    landmarks[:, 1] *= height_factor
    return landmarks


def single_to_multi_dim_landmarks(
    single_dim_landmarks: np.ndarray, image_size
) -> np.ndarray:
    assert single_dim_landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
    landmarks = np.empty((image_size, image_size, constants.DATASET_300VW_N_LANDMARKS))
    for landmark_index in range(single_dim_landmarks.shape[0]):
        start_indices = single_dim_landmarks[landmark_index, :]
        landmarks[:, :, landmark_index] = _landmark_to_channel(
            start_indices[0], start_indices[1], image_size
        )

    return landmarks


@lru_cache()
def _landmark_to_channel(x_1: int, y_1: int, image_size: int) -> np.ndarray:
    landmark_channel = np.zeros((image_size, image_size))
    start_indices_landmarks = np.asarray([int(x_1), int(y_1)])
    start_indices_landmarks -= constants.DATASET_300VW_WINDOW_RADIUS_GAUSSIAN

    end_indices_landmarks = (
        start_indices_landmarks + constants.DATASET_300VW_WINDOW_SIZE_GAUSSIAN
    )
    if any(start_indices_landmarks > image_size) or any(end_indices_landmarks < 0):
        return landmark_channel

    start_indices_gaussian = np.where(
        start_indices_landmarks < 0, abs(start_indices_landmarks), 0
    )
    start_indices_landmarks = np.where(
        start_indices_landmarks < 0, 0, start_indices_landmarks
    )
    end_indices_gaussian = constants.DATASET_300VW_WINDOW_SIZE_GAUSSIAN - np.where(
        end_indices_landmarks > image_size, end_indices_landmarks - image_size, 0
    )
    end_indices_landmarks = np.where(
        end_indices_landmarks > image_size, image_size, end_indices_landmarks
    )

    assert all(
        (end_indices_landmarks - start_indices_landmarks)
        == (end_indices_gaussian - start_indices_gaussian)
    )

    landmark_channel[
        start_indices_landmarks[1] : end_indices_landmarks[1],
        start_indices_landmarks[0] : end_indices_landmarks[0],
    ] = constants.DATASET_300VW_GAUSSIAN[
        start_indices_gaussian[1] : end_indices_gaussian[1],
        start_indices_gaussian[0] : end_indices_gaussian[0],
    ]

    return landmark_channel
