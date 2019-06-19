import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import dlib
import numpy as np
import torch
from tqdm import tqdm

import _pickle as pickle
import cv2
from data import transformations
from models.generators.ResnetGenerator import ResnetGenerator as Generator
from utils import constants, general_utils, personal_constants

ESCAPE_KEY_CODE = 27
RECTANGLE_COLOR = (0, 0, 255)
RECTANGLE_THICKNESS = 3
LANDMARK_COLOR = (0, 255, 0)
LANDMARK_RADIUS = 3
LANDMARK_THICKNESS = -1
# negative thickness means fill


def main(arguments: argparse.Namespace) -> None:
    network = get_network(
        arguments.network_path, arguments.use_network, arguments.device
    )
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(personal_constants.DLIB_PREDICTOR_PATH))

    transform_to_input = [
        transformations.Resize._f,
        transformations.RescaleValues._f,
        transformations.ChangeChannels._f,
        lambda image: torch.from_numpy(image[np.newaxis, ...])
        .float()
        .to(arguments.device),
    ]
    transform_from_input = [
        network,
        lambda batch: batch[0],
        general_utils.de_torch,
        general_utils.denormalize_picture,
    ]

    from_image_path = Path(arguments.from_image_path)
    from_image = cv2.imread(str(from_image_path))
    for t in transform_to_input:
        from_image = t(from_image)

    base_image_path = from_image_path.parent / (
        from_image_path.stem + '_base' + from_image_path.suffix
    )
    base_image = cv2.imread(str(base_image_path))

    cam = cv2.VideoCapture(arguments.webcam)
    bar = tqdm()

    while True:
        n_bounding_boxes = show_image(
            cam,
            arguments.use_mirror,
            detector,
            predictor,
            transform_to_input,
            from_image,
            arguments.device,
            transform_from_input,
            base_image,
        )
        bar.update(1)
        bar.set_postfix(n_bounding_boxes=n_bounding_boxes)
        if cv2.waitKey(1) == ESCAPE_KEY_CODE:
            break

    bar.close()
    cv2.destroyAllWindows()


def get_network(network_path: str, use_network: bool, device: str) -> Optional:
    if use_network:
        with (open(network_path, 'rb')) as openfile:
            weights = pickle.load(openfile)
        network = Generator(n_hidden=24, use_dropout=False)
        network.load_state_dict(weights['generator'])
        network = network.to(device)
    else:
        network = None
    return network


def show_image(
    cam: cv2.VideoCapture,
    use_mirror: bool,
    detector,
    predictor,
    transform_to_input: List[Callable],
    from_image: torch.Tensor,
    device: str,
    transform_from_input: List[Callable],
    base_image: np.ndarray,
) -> int:
    image_success, image = cam.read()
    if not image_success:
        return 0

    if use_mirror:
        image = cv2.flip(image, 1)

    bounding_boxes = detector(image, 1)
    n_rectangles = len(bounding_boxes)

    display_image = np.copy(image)
    dlib_landmarks = [predictor(display_image, rectangle).parts() for rectangle in bounding_boxes]
    for dl in dlib_landmarks:
        assert len(dl) == constants.DATASET_300VW_N_LANDMARKS
        image_landmarks = np.asarray([(lm.x, lm.y) for lm in dl], dtype=float)
        image_box = _landmarks_to_box(image_landmarks, image.shape)

        cv2.rectangle(
            display_image,
            (image_box[0], image_box[1]),
            (image_box[2], image_box[3]),
            RECTANGLE_COLOR,
            RECTANGLE_THICKNESS,
        )

    for dl in dlib_landmarks:
        for lm in dl:
            cv2.circle(
                display_image,
                (lm.x, lm.y),
                LANDMARK_RADIUS,
                LANDMARK_COLOR,
                LANDMARK_THICKNESS,
            )

    cv2.imshow('display_image', display_image)

    if n_rectangles != 1:
        return n_rectangles

    single_dim_landmarks = extract(image, bounding_boxes[0], predictor)

    multi_dim_landmarks = single_to_multi_dim_landmarks(single_dim_landmarks)
    for t in transform_to_input:
        multi_dim_landmarks = t(multi_dim_landmarks)

    output = torch.cat(
        (from_image, multi_dim_landmarks.to(device)), dim=constants.CHANNEL_DIM
    )
    for t in transform_from_input:
        output = t(output)

    original_image_box = (671, 95, 949, 373)
    target_width = original_image_box[2] - original_image_box[0]
    target_height = original_image_box[3] - original_image_box[1]
    output = cv2.resize(
        output,
        dsize=(target_width, target_height),
        interpolation=constants.INTERPOLATION,
    )
    base_image[
        original_image_box[1] : original_image_box[3],
        original_image_box[0] : original_image_box[2],
        ...,
    ] = output
    cv2.imshow('merged', base_image)

    return n_rectangles


def extract(image: np.ndarray, bounding_box, predictor) -> np.ndarray:
    landmarks = predictor(image, bounding_box).parts()
    assert len(landmarks) == constants.DATASET_300VW_N_LANDMARKS
    image_landmarks = np.asarray([(lm.x, lm.y) for lm in landmarks], dtype=float)
    image_box = _landmarks_to_box(image_landmarks, image.shape)

    extracted_image = _extract(image, image_box)
    extracted_landmarks = _offset_landmarks(image_landmarks, image_box)

    output_landmarks = _rescale_landmarks(
        extracted_landmarks, extracted_image.shape, constants.IMSIZE
    )

    return output_landmarks


def _landmarks_to_box(
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
    # // 2 returns numpy float but we need ints
    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
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

def _extract(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    extraction = image[y1 : y2 + 1, x1 : x2 + 1, ...]
    return extraction


def _offset_landmarks(
    landmarks: np.ndarray, box: Tuple[int, int, int, int]
) -> np.ndarray:
    x1, y1, x2, y2 = box
    landmarks = np.copy(landmarks)
    landmarks[:, 0] -= x1
    landmarks[:, 1] -= y1
    return landmarks


def _rescale_landmarks(
    landmarks: np.ndarray, image_shape: Tuple[int, int, int], image_size: int
) -> np.ndarray:
    height, width, n_channels = image_shape
    landmarks = np.copy(landmarks)
    height_factor = 1 / height * image_size
    width_factor = 1 / width * image_size
    landmarks[:, 0] *= width_factor
    landmarks[:, 1] *= height_factor
    return landmarks


def single_to_multi_dim_landmarks(single_dim_landmarks: np.ndarray) -> np.ndarray:
    assert single_dim_landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
    landmarks = np.empty(
        (constants.IMSIZE, constants.IMSIZE, constants.DATASET_300VW_N_LANDMARKS)
    )
    for landmark_index in range(single_dim_landmarks.shape[0]):
        start_indices = single_dim_landmarks[landmark_index, :]
        landmarks[:, :, landmark_index] = _landmark_to_channel(
            start_indices[0], start_indices[1], constants.IMSIZE
        )

    return landmarks


@lru_cache()
def _landmark_to_channel(x_1: int, y_1: int, image_size: int) -> np.ndarray:
    landmark_channel = np.zeros((image_size, image_size))
    start_indices_landmarks = np.asarray([x_1, y_1], dtype=int)
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


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--webcam', default=0, type=int)
    parser.add_argument('--mirror', type=bool, default=False)
    parser.add_argument('--no-use-network', dest='use_network', action='store_false')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--no-use-mirror', dest='use_mirror', action='store_false')
    parser.add_argument('--overlay', type=bool, default=True)
    parser.add_argument('--overlay-color', type=str, default='r')
    parser.add_argument('--overlay-alpha', type=float, default=1.0)
    parser.add_argument('--predictor-path', type=str, default=1.0)
    parser.add_argument(
        '--from-image-path', type=str, default='./data/local_data/0.jpg'
    )
    parser.add_argument(
        '--network-path', type=str, default='./data/local_data/weights.pickle'
    )

    return parser.parse_args()


if __name__ == '__main__':
    print(
        # 'cuda_version:',
        # torch.version.cuda,
        # 'pytorch version:',
        # torch.__version__,
        'python version:',
        sys.version,
    )
    print('Working directory: ', os.getcwd())
    args = parse()
    main(args)
