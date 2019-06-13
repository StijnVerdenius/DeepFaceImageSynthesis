from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from tqdm import tqdm

from utils import constants, general_utils


def all_video_paths(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.is_dir()])


def count_images(all_videos: List[Path], folder: str, extension: str) -> List[int]:
    n_images_per_video = [
        len(list((video_path / folder).glob(f'*.{extension}')))
        for video_path in tqdm(
            all_videos, desc=constants.DATASET_300VW_OUTER_LOOP_DESCRIPTION
        )
    ]
    return n_images_per_video


def plot(
    image: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
    landmarks_in_channel: Optional[np.ndarray] = None,
    overlay_alpha: float = 1.0,
    color_index: str = 'r',
    title: Optional[str] = None,
) -> None:
    plt.figure()
    plt.axis('off')

    if landmarks_in_channel is not None:
        color_index = 'bgr'.index(color_index)
        image = image.astype(float)
        mask = np.zeros(image.shape, dtype=float)
        mask[..., color_index] = 255
        for index in range(landmarks_in_channel.shape[-1]):
            image += (
                overlay_alpha * mask * landmarks_in_channel[:, :, index, np.newaxis]
            )

        image[image > 255] = 255
        image = image.astype('uint8')

    image = general_utils.BGR2RGB(image)
    plt.imshow(image)

    if landmarks is not None:
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, len(landmarks))]
        for index, c in zip(range(len(landmarks)), colors):
            plt.plot(
                [landmarks[index, 0]], [landmarks[index, 1]], 'o', color=c, markersize=1
            )

    if box is not None:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'
        )
        ax = plt.gca()
        ax.add_patch(rect)

    plt.title(title)
    plt.draw()
    plt.pause(0.001)
    plt.show()
