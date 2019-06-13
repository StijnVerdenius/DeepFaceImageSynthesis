from utils.training_helpers import unpack_batch
from torch.utils.data import DataLoader

from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
from models.embedders import GeneralEmbedder
from models.generators.GeneralGenerator import GeneralGenerator
from utils.constants import CHANNEL_DIM, DEVICE
from utils.general_utils import ensure_current_directory, torch, denormalize_picture, de_torch
import numpy as np
import matplotlib.pyplot as plt
import random


def compare(dataloader: DataLoader,
            embedder: GeneralEmbedder,
            generator: GeneralGenerator,
            arguments,
            number_of_pictures: int = 4,
            number_of_batches: int = 1
            ):
    """ visually compares generated images with target """


    for i, (batch_1, batch_2, batch_3) in enumerate(dataloader):

        if (i >= number_of_batches):
            break

        image_1, landmarks_1 = unpack_batch(batch_1)
        image_2, landmarks_2 = unpack_batch(batch_2)

        generator.eval()

        combined = torch.cat((image_1, landmarks_2), dim=CHANNEL_DIM).to(DEVICE).float()

        generated_images = generator.forward(combined[:number_of_pictures, :, :, :])

        landmarks_1 = torch.sum(landmarks_1, dim=CHANNEL_DIM)
        landmarks_2 = torch.sum(landmarks_2, dim=CHANNEL_DIM)

        plots = 5

        for image_index in range(number_of_pictures):
            plottable_generated = denormalize_picture(de_torch(generated_images[image_index, :, :, :]))
            plottable_landmarks_1 = denormalize_picture(de_torch(-1 * landmarks_1[image_index, :, :]), binarised=True)
            plottable_landmarks_2 = denormalize_picture(de_torch(-1 * landmarks_2[image_index, :, :]), binarised=True)
            plottable_source = denormalize_picture(de_torch(image_1[image_index, :, :, :]))
            plottable_target = denormalize_picture(de_torch(image_2[image_index, :, :, :]))


            plt.subplot(number_of_pictures, plots, image_index * plots + 1)
            plt.imshow(np.stack((plottable_landmarks_1.T, plottable_landmarks_1.T, plottable_landmarks_1.T), axis=2))
            plt.title("source")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(number_of_pictures, plots, image_index * plots + 2)
            plt.imshow(plottable_source)
            plt.title("source")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(number_of_pictures, plots, image_index * plots + 3)
            plt.imshow(plottable_generated)
            plt.title("generated")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(number_of_pictures, plots, image_index * plots + 4)
            plt.imshow(plottable_target)
            plt.title("target")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(number_of_pictures, plots, image_index * plots + 5)
            plt.imshow(np.stack((plottable_landmarks_2.T, plottable_landmarks_2.T, plottable_landmarks_2.T), axis=2))
            plt.title("target")
            plt.xticks([])
            plt.yticks([])

        plt.show()


def local_test():
    """ for testing something in this file specifically """
    pass


if __name__ == '__main__':
    ensure_current_directory()
    local_test()
