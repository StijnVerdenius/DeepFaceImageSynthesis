import random
import torch

import copy
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

from data import all_video_paths, count_images, plot
from utils import constants, personal_constants

from data.Dataset300VW import X300VWDataset  ### THESE ARE FOR TESTING PURPOSES!!!!
from torch.utils.data import DataLoader
from utils.training_helpers import *


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """

    def __init__(self, rotations=(-30, 30), scalings=(.75, 1.25)):
        assert (isinstance(rotations, type(scalings)))
        self.rotations = rotations
        self.scalings = scalings

    def __call__(self, batch):

        if type(self.rotations) == tuple:

            # Continuous range of scales and rotations
            rotation = (self.rotations[1] - self.rotations[0]) * random.random() - \
                  (self.rotations[1] - self.rotations[0]) / 2

            scaling = (self.scalings[1] - self.scalings[0]) * random.random() - \
                 (self.scalings[1] - self.scalings[0]) / 2 + 1

        elif type(self.rotations) == list:
            # Fixed range of scales and rotations
            rotation = self.rotations[random.randint(0, len(self.rotations) - 1)]
            scaling = self.scalings[random.randint(0, len(self.scalings) - 1)]

        for e in batch.keys():

            sample = batch[e]
            print(sample.shape)

            # get img dimensions
            h, w = sample.shape[-2:]

            # get centre of rotation in the source sample
            center = (w / 2, h / 2)

            # get the affine matrix of a 2D rotation
            R = cv2.getRotationMatrix2D(center, rotation, scaling)

            # interpolate over 4x4 pixel neighborhood
            flagval = cv2.INTER_CUBIC


            # apply affine transformation to source sample
            t_sample = cv2.warpAffine(sample, R, dsize=(w, h), flags=flagval)


            # if sample.min() < 0.0:
            #     sample = sample - sample.min()
            #
            # if sample.max() > 1.0:
            #     sample = sample / sample.max()

            # normalize transformed sample
            t_sample = cv2.normalize(t_sample, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            batch[e] = t_sample

            cv2.imshow('original',sample)
            cv2.imshow("result", t_sample)

        return batch


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for elem in sample.keys():
            if elem in ['fname', 'seq_name']:
                continue
            else:
                tmp = sample[elem]

                if tmp.ndim == 2:
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_CUBIC

                tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

                sample[elem] = tmp

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['fname', 'seq_name']:
                    continue
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if elem in ['fname', 'seq_name']:
                continue
            else:
                tmp = sample[elem]

                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]

                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W

                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.from_numpy(tmp)

        return sample


# if __name__ == '__main__':
#     mode = "test"
#     batch_size = 1
#
#     # # initialize data
#     # data = DataLoader(X300VWDataset(), shuffle=(False or mode == "test"), batch_size=batch_size, drop_last=True)
#
#     # # get one batch to test
#     # (batch_1, batch_2, batch_3) = next(iter(data))
#
#     # Each index is a lsit of 3 dicts, and each dict is the image and landmarks
#     data = X300VWDataset()
#
#
#     # transform image
#     transform = ScaleNRotate()
#     sample = transform(data[0][0])
#
#
#     # sample = transform(batch_1)

class ChangeChannels:
    def __call__(self, sample: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        print('before')
        for s in sample:
            for key, value in s.items():
                print(key, value.shape)

        for s in sample:
            for key, value in s.items():
                # numpy image: H x W x C
                # torch image: C X H X W
                s[key] = value.transpose((2, 0, 1))

        print('after')
        for s in sample:
            for key, value in s.items():
                print(key, value.shape)

        return sample


if __name__ == '__main__':
    data = X300VWDataset()
    transform = ChangeChannels()
    transform(data[0])


    # transform image
    transform = ScaleNRotate()
    sample = transform(data[0][0])
