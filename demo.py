import argparse
import os
import sys
from typing import Optional, Tuple

import dlib
import numpy as np
from tqdm import tqdm

import cv2
from utils import personal_constants

ESCAPE_KEY_CODE = 27
RECTANGLE_COLOR = (0, 0, 255)
RECTANGLE_THICKNESS = 3
LANDMARK_COLOR = (0, 255, 0)
LANDMARK_RADIUS = 3
LANDMARK_THICKNESS = -1
# negative thickness means fill


def main(arguments: argparse.Namespace) -> None:
    network = get_network(arguments.use_network, arguments.use_cuda)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(personal_constants.DLIB_PREDICTOR_PATH))

    cam = cv2.VideoCapture(arguments.webcam)
    bar = tqdm()

    while True:
        n_bounding_boxes = show_image(
            cam, arguments.use_mirror, network, detector, predictor
        )
        bar.update(1)
        bar.set_postfix(n_bounding_boxes=n_bounding_boxes)
        if cv2.waitKey(1) == ESCAPE_KEY_CODE:
            break

    bar.close()
    cv2.destroyAllWindows()


def get_network(use_network: bool, use_cuda: bool) -> Optional:
    if use_network:
        net = ''
        if use_cuda:
            net = net.cuda()
    else:
        net = None
    return net


def show_image(
    cam: cv2.VideoCapture, use_mirror: bool, network: Optional, detector, predictor
) -> int:
    image_success, image = cam.read()
    if not image_success:
        return 0

    if use_mirror:
        image = cv2.flip(image, 1)

    bounding_boxes = detector(image, 1)
    n_rectangles = len(bounding_boxes)

    display_image = np.copy(image)
    for rectangle in bounding_boxes:
        top_left = rectangle.tl_corner()
        bottom_right = rectangle.br_corner()
        cv2.rectangle(
            display_image,
            (top_left.x, top_left.y),
            (bottom_right.x, bottom_right.y),
            RECTANGLE_COLOR,
            RECTANGLE_THICKNESS,
        )

        landsmarks = predictor(display_image, rectangle).parts()
        assert len(landsmarks) == 68  # change to from constants
        for lm in landsmarks:
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

    output_landmarks = extract(image, bounding_boxes[0], predictor)

    return n_rectangles


def extract(
    image: np.ndarray, bounding_box, predictor
) -> Tuple[np.ndarray, np.ndarray]:
    top_left = bounding_box.tl_corner()
    bottom_right = bounding_box.br_corner()
    image_box = (top_left.x, top_left.y, bottom_right.x, bottom_right.y)
    landmarks = predictor(image, bounding_box).parts()
    assert len(landmarks) == 68  # change to from constants
    image_landmarks = np.asarray([(lm.x, lm.y) for lm in landmarks], dtype=float)

    extracted_image = _extract(image, image_box)
    extracted_landmarks = _offset_landmarks(image_landmarks, image_box)

    # output_image = _rescale_image(extracted_image)
    output_landmarks = _rescale_landmarks(extracted_landmarks, extracted_image.shape)

    # return output_image, output_landmarks
    return output_landmarks


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


def _rescale_image(image: np.ndarray) -> np.ndarray:
    output_width, output_height = 128, 128  # change to constants
    _, _, n_channels = image.shape
    image = cv2.resize(
        image,
        dsize=(output_width, output_height),
        interpolation=cv2.INTER_CUBIC,  # change to constants
    )
    # notice that the order is swapped
    assert image.shape == (output_height, output_width, n_channels)
    return image


def _rescale_landmarks(
    landmarks: np.ndarray, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    height, width, n_channels = image_shape
    landmarks = np.copy(landmarks)
    height_factor = 1 / height * 128  # change to constants
    width_factor = 1 / width * 128  # change to constants
    landmarks[:, 0] *= width_factor
    landmarks[:, 1] *= height_factor
    return landmarks


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--webcam', default=0, type=int)
    parser.add_argument('--mirror', type=bool, default=False)
    parser.add_argument('--use-network', dest='use_network', action='store_true')
    parser.add_argument('--no-use-cuda', dest='use_cuda', action='store_false')
    parser.add_argument('--no-use-mirror', dest='use_mirror', action='store_false')
    parser.add_argument('--overlay', type=bool, default=True)
    parser.add_argument('--overlay-color', type=str, default='r')
    parser.add_argument('--overlay-alpha', type=float, default=1.0)
    parser.add_argument('--predictor-path', type=str, default=1.0)

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
