from typing import Tuple, Dict

import torch
from torch import __init__
from torchvision.utils import save_image
from utils.constants import *
import torchvision.utils as vutils
import numpy as np
import random
import numpy as np


def plot_some_pictures(images, batches_done, suffix, filesort):
    """
    save some plots in PIC_DIR

    """

    save_image(images, f'./{PREFIX_OUTPUT}/{DATA_MANAGER.stamp}/{PIC_DIR}/{batches_done}_{suffix}.{filesort}',
               nrow=4, normalize=True)


def calculate_accuracy(predictions, targets):
    """
    Gets the accuracy for discriminator

    """

    actual_predictions = predictions > 0.5
    true_positives = (actual_predictions == (targets > 0.5)).type(torch.DoubleTensor)
    accuracy = (torch.mean(true_positives))

    actual_predictions.detach()
    true_positives.detach()

    return accuracy.item()


def unpack_batch(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return batch["image"], batch["landmarks"]


def combine_real_and_fake(indices, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines a set of real and fake images along the batch dimension
    Also generates targets.

    """

    # random indices
    # random.shuffle(indices)
    # shuffle_indices_local = torch.LongTensor(indices).to(DEVICE)

    # combine fake and real images
    composite = torch.cat((fake, real), dim=0) #.index_select(0, shuffle_indices_local)

    # combine real and fake targets
    ground_truth = labels #.index_select(0, shuffle_indices_local).to(DEVICE)

    # shuffle_indices_local.detach()

    return composite, ground_truth


def L2_distance(tensor1, tensor2, batch_dim=0):

    # get number of dimensions
    n_dims = len(tensor1.shape)

    # get dims to sum over
    dims = tuple([n for n in range(n_dims) if n != batch_dim])

    # get normalization term
    n_term = np.prod([dim for i,dim in enumerate(tensor1.shape) if i != batch_dim])

    distance = torch.sqrt(torch.sum((tensor1 - tensor2).pow(2), dim=dims))/n_term

    return distance


def L1_distance(tensor1, tensor2, batch_dim=0):
    # get number of dimensions
    n_dims = len(tensor1.shape)

    # get dims to sum over
    dims = tuple([n for n in range(n_dims) if n != batch_dim])

    distance = torch.mean(torch.abs(tensor1 - tensor2), dim=dims) # changed from sum to mean

    return distance
