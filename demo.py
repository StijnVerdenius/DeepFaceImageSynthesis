import argparse
import os
import sys
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
from utils import constants, data_utils, general_utils, personal_constants
from utils.constants import (
    ESCAPE_KEY_CODE,
    LANDMARK_COLOR,
    LANDMARK_RADIUS_OTHER,
    LANDMARK_RADIUS_SELECTED,
    LANDMARK_THICKNESS,
    ORIGINAL_IMAGE_BOX,
    RECTANGLE_COLOR,
    RECTANGLE_THICKNESS_OTHER,
    RECTANGLE_THICKNESS_SELECTED,
)


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
        image_to_batch,
        lambda batch: batch.to(arguments.device),
    ]
    transform_from_input = [
        network,
        image_from_batch,
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
    if arguments.image_to_box_size:
        rescale_factor_x = constants.IMSIZE / (
            ORIGINAL_IMAGE_BOX[2] - ORIGINAL_IMAGE_BOX[0]
        )
        rescale_factor_y = constants.IMSIZE / (
            ORIGINAL_IMAGE_BOX[3] - ORIGINAL_IMAGE_BOX[1]
        )
        base_image_box = [
            ORIGINAL_IMAGE_BOX[0] * rescale_factor_x,
            ORIGINAL_IMAGE_BOX[1] * rescale_factor_y,
            ORIGINAL_IMAGE_BOX[2] * rescale_factor_x,
            ORIGINAL_IMAGE_BOX[3] * rescale_factor_y,
        ]
        base_image_box = [int(bib) for bib in base_image_box]
        target_height, target_width, _ = base_image.shape
        target_height, target_width = (
            int(target_height * rescale_factor_y),
            int(target_width * rescale_factor_x),
        )
        base_image = cv2.resize(
            base_image,
            dsize=(target_width, target_height),
            interpolation=constants.INTERPOLATION,
        )
    else:
        base_image_box = ORIGINAL_IMAGE_BOX

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
            base_image_box,
        )
        bar.update(1)
        bar.set_postfix(n_bounding_boxes=n_bounding_boxes)
        if cv2.waitKey(1) == ESCAPE_KEY_CODE:
            break

    bar.close()
    cv2.destroyAllWindows()


def image_to_batch(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image[np.newaxis, ...]).float()


def image_from_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch[0]


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
    base_image_box: Tuple[int, int, int, int],
) -> int:
    image_success, image = cam.read()
    if not image_success:
        return 0

    if use_mirror:
        image = cv2.flip(image, 1)

    bounding_boxes = detector(image, 1)
    n_rectangles = len(bounding_boxes)

    if n_rectangles == 0:
        selected_bounding_box_index = -1
    else:
        bounding_boxes_sizes = [
            (
                bounding_boxes[bb_index].br_corner().x
                - bounding_boxes[bb_index].tl_corner().x
            )
            * (
                bounding_boxes[bb_index].br_corner().y
                - bounding_boxes[bb_index].tl_corner().y
            )
            for bb_index in range(n_rectangles)
        ]
        selected_bounding_box_index = bounding_boxes_sizes.index(
            max(bounding_boxes_sizes)
        )

    display_webcam_image(image, predictor, bounding_boxes, selected_bounding_box_index)

    if n_rectangles > 0:
        landmarks = predictor(
            image, bounding_boxes[selected_bounding_box_index]
        ).parts()
        display_output_image(
            image,
            base_image,
            landmarks,
            device,
            from_image,
            transform_from_input,
            transform_to_input,
            base_image_box,
        )

    return n_rectangles


def display_output_image(
    image: np.ndarray,
    base_image: np.ndarray,
    landmarks,
    device: str,
    from_image: torch.Tensor,
    transform_from_input: List[Callable],
    transform_to_input: List[Callable],
    base_image_box: Tuple[int, int, int, int],
):
    single_dim_landmarks = extract(image, landmarks)
    multi_dim_landmarks = data_utils.single_to_multi_dim_landmarks(
        single_dim_landmarks, constants.IMSIZE
    )
    for t in transform_to_input:
        multi_dim_landmarks = t(multi_dim_landmarks)
    output = torch.cat(
        (from_image, multi_dim_landmarks.to(device)), dim=constants.CHANNEL_DIM
    )
    for t in transform_from_input:
        output = t(output)
    target_width = base_image_box[2] - base_image_box[0]
    target_height = base_image_box[3] - base_image_box[1]
    output = cv2.resize(
        output,
        dsize=(target_width, target_height),
        interpolation=constants.INTERPOLATION,
    )
    base_image[
        base_image_box[1] : base_image_box[3],
        base_image_box[0] : base_image_box[2],
        ...,
    ] = output
    cv2.imshow('merged', base_image)


def display_webcam_image(image, predictor, bounding_boxes, selected_bounding_box_index):
    display_image = np.copy(image)
    dlib_landmarks = [
        predictor(display_image, rectangle).parts() for rectangle in bounding_boxes
    ]
    for dl_index, dl in enumerate(dlib_landmarks):
        assert len(dl) == constants.DATASET_300VW_N_LANDMARKS
        image_landmarks = np.asarray([(lm.x, lm.y) for lm in dl], dtype=float)
        image_box = data_utils.landmarks_to_box(image_landmarks, image.shape)

        thickness = (
            RECTANGLE_THICKNESS_SELECTED
            if dl_index == selected_bounding_box_index
            else RECTANGLE_THICKNESS_OTHER
        )
        cv2.rectangle(
            display_image,
            (image_box[0], image_box[1]),
            (image_box[2], image_box[3]),
            RECTANGLE_COLOR,
            thickness,
        )

    for dl_index, dl in enumerate(dlib_landmarks):
        radius = (
            LANDMARK_RADIUS_SELECTED
            if dl_index == selected_bounding_box_index
            else LANDMARK_RADIUS_OTHER
        )
        for lm in dl:
            cv2.circle(
                display_image, (lm.x, lm.y), radius, LANDMARK_COLOR, LANDMARK_THICKNESS
            )
    cv2.imshow('display_image', display_image)


def extract(image: np.ndarray, landmarks) -> np.ndarray:
    assert len(landmarks) == constants.DATASET_300VW_N_LANDMARKS
    image_landmarks = np.asarray([(lm.x, lm.y) for lm in landmarks], dtype=float)
    image_box = data_utils.landmarks_to_box(image_landmarks, image.shape)

    extracted_image = data_utils.extract(image, image_box)
    extracted_landmarks = data_utils.offset_landmarks(image_landmarks, image_box)

    output_landmarks = data_utils.rescale_landmarks(
        extracted_landmarks, extracted_image.shape, constants.IMSIZE
    )

    return output_landmarks


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--webcam', default=0, type=int)
    parser.add_argument('--mirror', type=bool, default=False)
    parser.add_argument('--no-use-network', dest='use_network', action='store_false')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--no-use-mirror', dest='use_mirror', action='store_false')
    parser.add_argument('--predictor-path', type=str, default=1.0)
    parser.add_argument(
        '--image-to-box-size', dest='image_to_box_size', action='store_true'
    )
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
