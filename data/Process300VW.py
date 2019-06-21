import copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import cv2
from data import all_video_paths, count_images, plot
from utils import constants, data_utils, personal_constants


def extract_frames(all_videos: List[Path], n_images_per_video: List[int]) -> None:
    for video_input_path, n_images_in_video in tqdm(
        list(zip(all_videos, n_images_per_video)),
        desc=constants.DATASET_300VW_OUTER_LOOP_DESCRIPTION,
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
            frame_output_path = frames_output_dir / (
                f'{counter:{constants.DATASET_300VW_NUMBER_FORMAT}}'
                + f'.{constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION}'
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_landmarks = _load_pts_file(annotation_input_path)
    image_box = data_utils.landmarks_to_box(image_landmarks, image.shape)

    extracted_image = data_utils.extract(image, image_box)
    extracted_landmarks = data_utils.offset_landmarks(image_landmarks, image_box)

    output = _rescale_image(extracted_image)
    output_landmarks = data_utils.rescale_landmarks(
        extracted_landmarks, extracted_image.shape, constants.DATASET_300VW_IMSIZE
    )

    # plot(image)
    # plot(image, image_landmarks)
    # plot(image, image_landmarks, image_box)
    # plot(extracted_image)
    # plot(extracted_image, extraction_landmarks)
    plot(output, output_landmarks)


def _load_pts_file(file_path: Path) -> np.ndarray:
    with open(str(file_path), 'r') as file:
        lines = file.readlines()

    assert lines[0].strip().startswith('version: 1'), str(file_path)
    assert lines[1] == f'n_points: {constants.DATASET_300VW_N_LANDMARKS}\n', str(
        file_path
    )

    lines = [l.strip() for l in lines]
    # remove
    # version: 1
    # n_landmarks: 68
    # {
    lines = lines[3:]
    # remove
    # }
    lines = lines[:-1]
    landmarks = [[float(x) for x in p.split()] for p in lines]
    landmarks = np.asarray(landmarks)
    assert landmarks.shape == (constants.DATASET_300VW_N_LANDMARKS, 2)
    return landmarks


def _rescale_image(image: np.ndarray) -> np.ndarray:
    output_width, output_height = constants.DATASET_300VW_IMSIZE, constants.DATASET_300VW_IMSIZE
    _, _, n_channels = image.shape
    image = cv2.resize(
        image,
        dsize=(output_width, output_height),
        interpolation=constants.INTERPOLATION,
    )
    # notice that the order is swapped
    assert image.shape == (output_height, output_width, n_channels)
    return image


def process_temp_folder(all_videos: List[Path]) -> None:
    for video_input_path in tqdm(
        all_videos, desc=constants.DATASET_300VW_OUTER_LOOP_DESCRIPTION
    ):
        video_output_path = (
            personal_constants.DATASET_300VW_OUTPUT_PATH / video_input_path.stem
        )
        annotation_file_output_path = video_output_path / 'annotations.npy'
        frames_output_dir = (
            video_output_path / constants.DATASET_300VW_IMAGES_OUTPUT_FOLDER
        )
        frames_output_dir.mkdir(exist_ok=True, parents=True)

        annotations_paths = sorted(
            list(
                (
                    video_input_path / constants.DATASET_300VW_ANNOTATIONS_INPUT_FOLDER
                ).glob(f'*.{constants.DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION}')
            )
        )
        landmarks = np.empty(
            (len(annotations_paths), constants.DATASET_300VW_N_LANDMARKS, 2)
        )
        for frame_index, annotation_input_path in enumerate(
            tqdm(
                annotations_paths,
                desc=constants.DATASET_300VW_INNER_LOOP_DESCRIPTION,
                leave=False,
            )
        ):
            frame_output_path = (
                frames_output_dir
                / f'{annotation_input_path.stem}.{constants.DATASET_300VW_IMAGES_OUTPUT_EXTENSION}'
            )
            if frame_output_path.exists() and annotation_file_output_path.exists():
                continue

            frame_input_path = (
                personal_constants.DATASET_300VW_TEMP_PATH
                / video_input_path.stem
                / constants.DATASET_300VW_IMAGES_TEMP_FOLDER
                / f'{annotation_input_path.stem}.{constants.DATASET_300VW_IMAGES_TEMP_EXTENSION}'
            )
            image = cv2.imread(str(frame_input_path))
            image_landmarks = _load_pts_file(annotation_input_path)
            image_box = data_utils.landmarks_to_box(image_landmarks, image.shape)

            extracted_image = data_utils.extract(image, image_box)
            extracted_landmarks = data_utils.offset_landmarks(
                image_landmarks, image_box
            )

            if not frame_output_path.exists():
                output = _rescale_image(extracted_image)
                cv2.imwrite(
                    str(frame_output_path),
                    output,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        constants.DATASET_300VW_IMAGE_QUALITY,
                    ],
                )

            output_landmarks = data_utils.rescale_landmarks(
                extracted_landmarks,
                extracted_image.shape,
                constants.DATASET_300VW_IMSIZE,
            )

            landmarks[frame_index, :, :] = output_landmarks

        if not annotation_file_output_path.exists():
            np.save(str(annotation_file_output_path), landmarks)


def main() -> None:
    n_videos_limit = None
    all_videos = all_video_paths(personal_constants.DATASET_300VW_RAW_PATH)
    assert len(all_videos) == constants.DATASET_300VW_N_VIDEOS
    print(f'n videos: {len(all_videos)}')
    all_videos = all_videos[:n_videos_limit]
    print(f'Taking first {len(all_videos)} videos!')

    print('Counting images...')
    n_images_per_video = count_images(
        all_videos,
        constants.DATASET_300VW_ANNOTATIONS_INPUT_FOLDER,
        constants.DATASET_300VW_ANNOTATIONS_INPUT_EXTENSION,
    )
    n_images = sum(n_images_per_video)
    print(f'n images: {n_images}')

    print('Visualizing extraction process...')
    visualize('001', '000001')
    visualize('007', '000020')

    print('Extracting frames from videos...')
    extract_frames(all_videos, n_images_per_video)

    print('Extracting faces')
    process_temp_folder(all_videos)

    print('Done.')
    # use input because the plots will disappear once the program exits
    input('Press [enter] to exit.')


if __name__ == '__main__':
    main()
