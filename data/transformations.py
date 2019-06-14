from typing import Callable, List, Optional

from torchvision import transforms

from data import plot
from utils import constants, general_utils
from utils.training_helpers import *


def _loop_all(
    sample: List[Dict[str, np.ndarray]], sample_setup: Callable, f: Callable
) -> List[Dict[str, np.ndarray]]:
    for s in sample:
        process_sample, *args = sample_setup(s)
        if not process_sample:
            continue

        for key, value in s.items():
            s[key] = f(value, *args)

    return sample


def _process_all(s: Dict[str, np.ndarray]) -> Tuple[bool, None]:
    return True, None


class RandomHorizontalFlip:
    """RandomHorizontalFlip should be applied to all n images together, not just one

    """

    def __init__(self, probability: float = 0.5):
        self._probability = probability

    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        if random.random() < self._probability:
            return _loop_all(sample, _process_all, self._f)
        else:
            return sample

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        return cv2.flip(value, flipCode=1)


class RandomRescale:
    def __init__(
        self, probability: float = 1 / 3, scales: Optional[List[float]] = None
    ):
        self._probability = probability
        if scales is None:
            scales = [1.1, 1.2]
        self._scales = scales

    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        return _loop_all(sample, self._process_s, self._f)

    def _process_s(self, s: Dict[str, np.ndarray]) -> Tuple[bool, float]:
        return random.random() < self._probability, random.choice(self._scales)

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        scale = args[0]
        return cv2.resize(
            value, None, fx=scale, fy=scale, interpolation=constants.INTERPOLATION
        )


class RandomCrop:
    def __init__(
        self, probability: float = 1 / 3, scales: Optional[List[float]] = None
    ):
        self._probability = probability
        if scales is None:
            scales = [0.8, 0.9]
        self._scales = scales

    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        return _loop_all(sample, self._process_s, self._f)

    def _process_s(self, s: Dict[str, np.ndarray]) -> Tuple[bool, float, float]:
        scale = random.choice(self._scales)
        input_height, input_width, _ = s['image'].shape
        target_height, target_width = (
            int(input_height * scale),
            int(input_width * scale),
        )
        top = np.random.randint(0, input_height - target_height)
        left = np.random.randint(0, input_width - target_width)
        return (
            random.random() < self._probability,
            target_height,
            target_width,
            top,
            left,
        )

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        target_height, target_width, top, left = args
        return value[top : top + target_height, left : left + target_width]


class Resize:
    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        return _loop_all(sample, _process_all, self._f)

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        return cv2.resize(
            value,
            (constants.IMSIZE, constants.IMSIZE),
            interpolation=constants.INTERPOLATION,
        )


class RescaleValues:
    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        return _loop_all(sample, _process_all, self._f)

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        # don't rescale landmarks
        if value.shape[-1] == constants.DATASET_300VW_N_LANDMARKS:
            return value

        value = value.astype(float)
        value = (value / 255) * 2 - 1
        # assert -1 <= value.min() <= value.max() <= 1
        return value


class ChangeChannels:
    def __call__(
        self, sample: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        return _loop_all(sample, _process_all, self._f)

    @staticmethod
    def _f(value: np.ndarray, *args) -> np.ndarray:
        # numpy image: H x W x C
        # torch image: C X H X W
        value = np.moveaxis(value, -1, 0)
        # assert value.shape == (
        #     constants.INPUT_CHANNELS,
        #     constants.IMSIZE,
        #     constants.IMSIZE,
        # ), f"wrong shape {image.shape}"
        return value


def _test():
    from data.Dataset300VW import X300VWDataset

    dataset = X300VWDataset(constants.Dataset300VWMode.ALL)
    sample = dataset[0]

    image, landmarks = sample[0]['image'], sample[0]['landmarks']
    plot(image, landmarks_in_channel=landmarks, title='original')

    for t in (RandomHorizontalFlip, RandomRescale, RandomCrop):
        sample = t(probability=1)(sample)
        image, landmarks = sample[0]['image'], sample[0]['landmarks']
        plot(image, landmarks_in_channel=landmarks, title=t.__name__)

    transform = transforms.Compose(
        [
            RandomHorizontalFlip(probability=1),
            RandomRescale(probability=1),
            RandomCrop(probability=1),
            Resize(),
            RescaleValues(),
            ChangeChannels(),
        ]
    )
    sample = transform(sample)
    image, landmarks = sample[0]['image'], sample[0]['landmarks']
    image = general_utils.move_color_channel(image)
    image = general_utils.denormalize_picture(image)
    landmarks = general_utils.move_color_channel(landmarks)
    plot(image, landmarks_in_channel=landmarks, title='all')


if __name__ == '__main__':
    _test()
