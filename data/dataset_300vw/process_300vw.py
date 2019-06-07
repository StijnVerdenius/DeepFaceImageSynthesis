import copy
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from tqdm import tqdm

from utils import constants, personal_constants

OUTER_LOOP_DESCRIPTION = 'video'
INNER_LOOP_DESCRIPTION = 'frame'


def count_images(all_videos: List[Path]) -> List[int]:
    assert len(all_videos) == constants.DATASET_300VW_N_VIDEOS
    n_images_per_video = [
        len(
            list(
                (video_path / constants.DATASET_300VW_ANNOTATIONS_INPUT_FOLDER).glob(
                    f'*.{constants.DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION}'
                )
            )
        )
        for video_path in tqdm(all_videos, desc=OUTER_LOOP_DESCRIPTION)
    ]
    return n_images_per_video


def extract_frames(all_videos: List[Path], n_images_per_video: List[int]) -> None:
    for video_input_path, n_images_in_video in tqdm(
        list(zip(all_videos, n_images_per_video)), desc=OUTER_LOOP_DESCRIPTION
    ):
        frames_output_dir = (
            personal_constants.DATASET_300VW_TEMP_PATH
            / video_input_path.stem
            / constants.DATASET_300VW_IMAGES_TEMP_FOLDER
        )
        if (
            frames_output_dir.exists()
            and len(list(frames_output_dir.iterdir())) == n_images_in_video
        ):
            continue

        frames_output_dir.mkdir(exist_ok=True, parents=True)
        avi_path = video_input_path / constants.DATASET_300VW_VIDEO_FILE_NAME
        video = cv2.VideoCapture(str(avi_path))

        counter = 1
        success, image = video.read()
        while success == 1:
            frame_output_path = (
                frames_output_dir
                / f'{counter:06d}.{constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION}'
            )
            cv2.imwrite(
                str(frame_output_path),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), constants.DATASET_300VW_IMAGE_QUALITY],
            )
            counter += 1
            success, image = video.read()


def visualize(video_id: str, frame_id: str) -> None:
    annotation_input_path = (
        personal_constants.DATASET_300VW_RAW_PATH
        / video_id
        / constants.DATASET_300VW_ANNOTATIONS_INPUT_FOLDER
        / f'{frame_id}.{constants.DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION}'
    )
    frame_input_path = (
        personal_constants.DATASET_300VW_TEMP_PATH
        / video_id
        / constants.DATASET_300VW_IMAGES_TEMP_FOLDER
        / f'{frame_id}.{constants.DATASET_300VW_IMAGES_TEMP_EXTENSION}'
    )
    image = cv2.imread(str(frame_input_path))
    image_points = _load_pts_file(annotation_input_path)
    image_box = _points_to_box(image_points, image.shape)

    extraction = _extract(image, image_box)
    extraction_points = _offset_points(image_points, image_box)

    output = _rescale_image(extraction)
    output_points = _rescale_points(extraction_points, extraction.shape)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _plot(image)
    _plot(image, image_points)
    _plot(image, image_points, image_box)
    extraction = cv2.cvtColor(extraction, cv2.COLOR_BGR2RGB)
    _plot(extraction)
    _plot(extraction, extraction_points)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    _plot(output, output_points)


def _plot(
    image, points: np.ndarray = None, box: Tuple[int, int, int, int] = None
) -> None:
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.ion()
    plt.show()

    if points is not None:
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(points))]
        for index, c in zip(range(len(points)), colors):
            plt.plot([points[index, 0]], [points[index, 1]], 'x', color=c)

    if box is not None:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax = plt.gca()
        ax.add_patch(rect)

    plt.draw()
    plt.pause(0.001)


def _load_pts_file(file_path: Path) -> np.ndarray:
    with open(str(file_path), 'r') as file:
        lines = file.readlines()

    assert lines[0].strip().startswith('version: 1'), str(file_path)
    assert lines[1] == f'n_points: {constants.DATASET_300VW_N_POINTS}\n', str(file_path)

    lines = [l.strip() for l in lines]
    # remove
    # version: 1
    # n_points: 68
    # {
    lines = lines[3:]
    # remove
    # }
    lines = lines[:-1]
    points = [[float(x) for x in p.split()] for p in lines]
    points = np.asarray(points)
    assert points.shape == (constants.DATASET_300VW_N_POINTS, 2)
    return points


def _points_to_box(
    points: np.ndarray, image_size: Tuple[int, int, int]
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [
        points[:, 0].min(),
        points[:, 1].min(),
        points[:, 0].max(),
        points[:, 1].max(),
    ]
    x1, y1 = [t - constants.DATASET_300VW_PADDING for t in (x1, y1)]
    x2, y2 = [t + constants.DATASET_300VW_PADDING for t in (x2, y2)]
    box_height, box_width = y2 - y1 + 1, x2 - x1 + 1
    assert box_height > 1 and box_width > 1

    if constants.DATASET_300VW_EXPAND_RATIO is not None:
        box_height *= constants.DATASET_300VW_EXPAND_RATIO
        box_width *= constants.DATASET_300VW_EXPAND_RATIO
        x1, y1 = [math.floor(t - s) for t, s in zip((x1, y1), (box_width, box_height))]
        x2, y2 = [math.ceil(t + s) for t, s in zip((x2, y2), (box_width, box_height))]

    image_height, image_width, _ = image_size
    # landmarks can be out of image, but that's okay, we'll still export them.
    x1, y1 = [t if t >= 0 else 0 for t in (x1, y1)]
    x2, y2 = [t if t < m else m for t, m in zip((x2, y2), (image_width, image_height))]
    assert x1 <= x2 and y1 <= y2

    return x1, y1, x2, y2


def _extract(image: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    extraction = image[y1 : y2 + 1, x1 : x2 + 1, ...]
    return extraction


def _offset_points(
    points: np.ndarray, box: Tuple[float, float, float, float]
) -> np.ndarray:
    x1, y1, x2, y2 = box
    points = copy.copy(points)
    points[:, 0] -= x1
    points[:, 1] -= y1
    return points


def _rescale_image(image: np.ndarray) -> np.ndarray:
    output_width, output_height = constants.IMSIZE, constants.IMSIZE
    _, _, n_channels = image.shape
    image = cv2.resize(
        image, dsize=(output_width, output_height), interpolation=cv2.INTER_CUBIC
    )
    # notice that the order is swapped
    assert image.shape == (output_height, output_width, n_channels)
    return image


def _rescale_points(
    points: np.ndarray, image_shape: Tuple[int, int, int]
) -> np.ndarray:
    height, width, n_channels = image_shape
    points = copy.copy(points)
    height_factor = 1 / height * constants.IMSIZE
    width_factor = 1 / width * constants.IMSIZE
    points[:, 0] *= width_factor
    points[:, 1] *= height_factor
    return points


def process_temp_folder(all_videos: List[Path]) -> None:
    for video_input_path in tqdm(all_videos, desc=OUTER_LOOP_DESCRIPTION):
        video_output_path = (
            personal_constants.DATASET_300VW_OUTPUT_PATH / video_input_path.stem
        )
        annotations_output_dir = (
            video_output_path / constants.DATASET_300VW_ANNOTATIONS_OUTPUT_FOLDER
        )
        annotations_output_dir.mkdir(exist_ok=True, parents=True)
        frames_output_dir = (
            video_output_path / constants.DATASET_300VW_IMAGES_OUTPUT_FOLDER
        )
        frames_output_dir.mkdir(exist_ok=True, parents=True)

        for annotation_input_path in tqdm(
            sorted(
                list(
                    (
                        video_input_path
                        / constants.DATASET_300VW_ANNOTATIONS_INPUT_FOLDER
                    ).glob(f'*.{constants.DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION}')
                )
            ),
            desc=INNER_LOOP_DESCRIPTION,
            leave=False,
        ):
            frame_output_path = (
                frames_output_dir
                / f'{annotation_input_path.stem}.{constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION}'
            )
            annotation_output_path = (
                annotations_output_dir
                / f'{annotation_input_path.stem}.{constants.DATASET_300VW_ANNOTATIONS_OUTPUT_EXTENSION}'
            )
            if frame_output_path.exists() and annotation_output_path.exists():
                continue

            frame_input_path = (
                personal_constants.DATASET_300VW_TEMP_PATH
                / video_input_path.stem
                / constants.DATASET_300VW_IMAGES_TEMP_FOLDER
                / f'{annotation_input_path.stem}.{constants.DATASET_300VW_IMAGES_TEMP_EXTENSION}'
            )
            image = cv2.imread(str(frame_input_path))
            image_points = _load_pts_file(annotation_input_path)
            image_box = _points_to_box(image_points, image.shape)

            extraction = _extract(image, image_box)
            extraction_points = _offset_points(image_points, image_box)

            if not frame_output_path.exists():
                output = _rescale_image(extraction)
                cv2.imwrite(
                    str(frame_output_path),
                    output,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        constants.DATASET_300VW_IMAGE_QUALITY,
                    ],
                )

            if not annotation_output_path.exists():
                output_points = _rescale_points(extraction_points, extraction.shape)
                np.savetxt(str(annotation_output_path), output_points)


def main() -> None:
    all_videos = sorted(
        [p for p in personal_constants.DATASET_300VW_RAW_PATH.iterdir() if p.is_dir()]
    )
    print(f'n videos: {len(all_videos)}')
    all_videos = all_videos[: constants.DATASET_300VW_LIMIT_N_VIDEOS]
    print(f'Taking first n videos: {len(all_videos)}')

    print('Counting images...')
    n_images_per_video = count_images(all_videos)
    n_images = sum(n_images_per_video)
    print(f'n images: {n_images}')

    print('Visualizing extraction process...')
    visualize('001', '000001')

    print('Extracting frames from videos...')
    extract_frames(all_videos, n_images_per_video)

    print('Extracting faces')
    process_temp_folder(all_videos)

    print('Done.')
    # use input because the plots will disappear once the program exits
    input('Press [enter] to exit.')


if __name__ == '__main__':
    main()
