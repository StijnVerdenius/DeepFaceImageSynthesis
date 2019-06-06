from typing import Tuple

import torch
from torchvision.utils import save_image
from utils.constants import *
import random


def plot_some_pictures(feedback, images, batches_done):
    """
    save some plots in PIC_DIR

    """

    save_image(images[:25].view(-1, 3, IMSIZE, IMSIZE),
               f'./{PREFIX_OUTPUT}/{DATA_MANAGER.stamp}/{PIC_DIR}/{batches_done}.png',
               nrow=5, normalize=True)

    if (feedback):
        # TODO: if feedback is on, run the following script from the result-image directory in terminal while it is running:
        # watch xdg-open latests_plot.png
        save_image(images[:25].view(-1, 3, IMSIZE, IMSIZE),
                   f'./{PREFIX_OUTPUT}/{DATA_MANAGER.stamp}/{PIC_DIR}/latests_plot.png',
                   nrow=5, normalize=True)


def calculate_accuracy(predictions, targets):
    """
    Gets the accuracy for discriminator

    """

    actual_predictions = predictions > 0.5
    true_positives = (actual_predictions == (targets > 0.5)).type(torch.DoubleTensor)
    accuracy = (torch.mean(true_positives))

    return accuracy.item()


def combine_real_and_fake(indices, real: torch.Tensor, fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines a set of real and fake images along the batch dimension
    Also generates targets.

    """

    # random indices
    random.shuffle(indices)
    shuffle_indices_local = torch.LongTensor(indices).to(DEVICE)

    # combine fake and real images
    composite = torch.cat((fake, real), dim=0).index_select(0, shuffle_indices_local)

    # combine real and fake targets
    labels = (torch.zeros(fake.shape[0]).to(DEVICE), torch.ones(real.shape[0]).to(DEVICE))
    ground_truth = torch.cat(labels, dim=0).index_select(0, shuffle_indices_local).to(DEVICE)

    return composite, ground_truth
