from models.losses.GeneralLoss import GeneralLoss
from models.generators.GeneralGenerator import GeneralGenerator
import torch.nn as nn
import torch
from models.generators.ResnetGenerator import ResnetGenerator as G
from utils.constants import CHANNEL_DIM, DEVICE
from utils.training_helpers import L1_distance


class ConsistencyLoss(GeneralLoss):

    def __init__(self, weight: float = 1, **kwargs):
        super(ConsistencyLoss, self).__init__(weight)

    def custom_forward(self,
                       batch: torch.Tensor,
                       gen_img: torch.Tensor,
                       image_ls: torch.Tensor,
                       generator: GeneralGenerator) \
            -> torch.Tensor:

        image_ls = image_ls.to(DEVICE).float()

        # Concatanate generated img with input landmark channels
        input2 = torch.cat((gen_img, image_ls), CHANNEL_DIM)

        # Generate approximation of input batch
        gen_img_2 = generator.forward(input2)

        # Get L1**2 distance between generated approx. and original input img
        loss = L1_distance(gen_img_2, batch).pow(2).mean()

        return loss


if __name__ == '__main__':
    G = G()
    get_loss = ConsistencyLoss()
