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
from models.generators import ResnetGenerator, UNetGenerator
from utils import constants, data_utils, general_utils, personal_constants
from utils.constants import (
    ESCAPE_KEY_CODE,
    LANDMARK_COLOR,
    LANDMARK_RADIUS_OTHER,
    LANDMARK_RADIUS_SELECTED,
    LANDMARK_THICKNESS,
    RECTANGLE_COLOR,
    RECTANGLE_THICKNESS_OTHER,
    RECTANGLE_THICKNESS_SELECTED,
    VGG
)

if constants.IMSIZE == 64:
    model_name_to_instance_settings = {
        'model1': (ResnetGenerator.ResnetGenerator, {'n_hidden': 24, 'use_dropout': False}),
        'hinge1': (UNetGenerator.UNetGenerator, {'n_hidden': 24, 'use_dropout': True}),
    }
elif constants.IMSIZE == 128:
    model_name_to_instance_settings = {
        'stijn1': (UNetGenerator.UNetGenerator, {'n_hidden': 64, 'use_dropout': True}),
        'klaus': (UNetGenerator.UNetGenerator, {'n_hidden': 64, 'use_dropout': True}),
        'klaus_monday': (UNetGenerator.UNetGenerator, {'n_hidden': 64, 'use_dropout': True}),
        'klaus_wednesday': (UNetGenerator.UNetGenerator, {'n_hidden': 64, 'use_dropout': True}),
    }


def main(arguments: argparse.Namespace) -> None:
    network = get_model(
        arguments.use_model,
        arguments.model_base_path,
        arguments.model_name,
        arguments.device,
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

    if arguments.use_outer_image:
        outer_image = cv2.imread(arguments.outer_image_path)
    else:
        outer_image = None

    ORIGINAL_IMAGE_BOX = (arguments.x1, arguments.y1, arguments.x2, arguments.y2)
    if arguments.use_outer_image and arguments.image_to_box_size:
        rescale_factor_x = constants.IMSIZE / (arguments.x2 - arguments.x1)
        rescale_factor_y = constants.IMSIZE / (arguments.y2 - arguments.y1)
        base_image_box = [
            arguments.x1 * rescale_factor_x,
            arguments.y1 * rescale_factor_y,
            arguments.x2 * rescale_factor_x,
            arguments.y2 * rescale_factor_y,
        ]
        base_image_box = [int(bib) for bib in base_image_box]
        target_height, target_width, _ = outer_image.shape
        target_height, target_width = (
            int(target_height * rescale_factor_y),
            int(target_width * rescale_factor_x),
        )
        outer_image = cv2.resize(
            outer_image,
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
            outer_image,
            base_image_box,
        )
        bar.update(1)
        bar.set_postfix(n_bounding_boxes=n_bounding_boxes)
        if cv2.waitKey(1) == ESCAPE_KEY_CODE:
            break

    bar.close()
    cv2.destroyAllWindows()


def get_model(
        use_model: bool, model_base_path: str, model_name: str, device: str
) -> Optional[torch.nn.Module]:
    if use_model:
        model_path = Path(model_base_path) / f'{model_name}.pickle'
        with (open(str(model_path), 'rb')) as openfile:
            weights = pickle.load(openfile)

        model_class, model_kwargs = model_name_to_instance_settings[model_name]
        model = model_class(**model_kwargs)
        model.load_state_dict(weights['generator'])
        model = model.to(device)
        model = model.eval()
    else:
        model = None
    return model


def image_to_batch(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image[np.newaxis, ...]).float()


def image_from_batch(batch: torch.Tensor) -> torch.Tensor:
    return batch[0]


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

    ratio = 1920/(2*640)
    display_image = cv2.resize(
        display_image,
        dsize=tuple([int(x) for x in (640*ratio, 480*ratio)]),
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.imshow('display_image', display_image)


def display_output_image(
        image: np.ndarray,
        outer_image: Optional[np.ndarray],
        landmarks,
        device: str,
        from_image: torch.Tensor,
        transform_from_input: List[Callable],
        transform_to_input: List[Callable],
        base_image_box: Tuple[int, int, int, int],
):
    single_dim_landmarks = extract(image, landmarks)
    multi_dim_landmarks = data_utils.single_to_multi_dim_landmarks(
        single_dim_landmarks, constants.DATASET_300VW_IMSIZE
    )
    for t in transform_to_input:
        multi_dim_landmarks = t(multi_dim_landmarks)

    output = torch.cat(
        (from_image, multi_dim_landmarks.to(device)), dim=constants.CHANNEL_DIM
    )

    for t in transform_from_input:
        output = t(output)

    # plottable = general_utils.BGR2RGB_numpy(output)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.imshow(plottable)
    # plt.show()
    # exit()

    if outer_image is None:
        size = int(1920/2)
        output = cv2.resize(
            output,
            dsize=(size, size),
            interpolation=cv2.INTER_CUBIC,
        )
        outer_image = output
    else:
        target_width = base_image_box[2] - base_image_box[0]
        target_height = base_image_box[3] - base_image_box[1]
        output = cv2.resize(
            output,
            dsize=(target_width, target_height),
            interpolation=constants.INTERPOLATION,
        )

        outer_image[
        base_image_box[1]: base_image_box[3],
        base_image_box[0]: base_image_box[2],
        ...,
        ] = output

    from data import plot
    cv2.imshow('merged', outer_image)
    # plot(outer_image)
    #
    # exit()


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

    # camera
    parser.add_argument('--webcam', default=0, type=int)
    parser.add_argument('--no-use-mirror', dest='use_mirror', action='store_false')

    # image
    parser.add_argument(
        '--from-image-path', type=str, default='./data/local_data/300VW_Dataset_processed_dim128/516/images/000098.jpg'
    )
    parser.add_argument(
        '--use-outer-image', dest='use_outer_image', action='store_true'
    )
    parser.add_argument(
        '--outer-image-path', type=str, default='./data/local_data/0_base.jpg'
    )
    parser.add_argument(
        '--image-to-box-size', dest='image_to_box_size', action='store_true'
    )
    parser.add_argument('--x1', default=671, type=int)
    parser.add_argument('--y1', default=95, type=int)
    parser.add_argument('--x2', default=949, type=int)
    parser.add_argument('--y2', default=373, type=int)

    # model
    parser.add_argument('--no-use-model', dest='use_model', action='store_false')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument(
        '--model-base-path', type=str, default='./data/local_data/'
    )
    parser.add_argument('--model-name', type=str, default='klaus_wednesday')

    return parser.parse_args()


if __name__ == '__main__':
    print(
        'cuda_version:',
        torch.version.cuda,
        'pytorch version:',
        torch.__version__,
        'python version:',
        sys.version,
    )
    print('Working directory: ', os.getcwd())
    args = parse()
    main(args)
