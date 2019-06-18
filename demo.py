import argparse
import os
import sys
from typing import Optional

import dlib
from tqdm import tqdm

import cv2

ESCAPE_KEY_CODE = 27
RECTANGLE_COLOR = (0, 0, 255)
RECTANGLE_THICKNESS = 3


def main(arguments: argparse.Namespace) -> None:
    network = get_network(arguments.use_network, arguments.use_cuda)
    detector = dlib.get_frontal_face_detector()

    cam = cv2.VideoCapture(arguments.webcam)
    bar = tqdm()

    while True:
        n_bounding_boxes = show_image(cam, arguments.use_mirror, network, detector)
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
    cam: cv2.VideoCapture, use_mirror: bool, network: Optional, detector
) -> int:
    image_success, image = cam.read()
    if not image_success:
        return

    if use_mirror:
        image = cv2.flip(image, 1)

    dets = detector(image, 0)
    n_rectangles = len(dets)
    for rectangle in dets:
        top_left = rectangle.tl_corner()
        bottom_right = rectangle.br_corner()
        cv2.rectangle(
            image,
            (top_left.x, top_left.y),
            (bottom_right.x, bottom_right.y),
            RECTANGLE_COLOR,
            RECTANGLE_THICKNESS,
        )

    cv2.imshow('pix2pix', image)

    return n_rectangles


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
