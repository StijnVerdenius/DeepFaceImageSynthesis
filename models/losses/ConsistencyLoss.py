from models.losses.GeneralLoss import GeneralLoss
from models.generators.GeneralGenerator import GeneralGenerator
import torch.nn as nn
import torch
from models.generators.ResnetGenerator import ResnetGenerator as G
from utils.constants import CHANNEL_DIM
from utils.training_helpers import L1_distance


class ConsistencyLoss(GeneralLoss):

    def __init__(self, weight: float=1, **kwargs):
        super(ConsistencyLoss, self).__init__(weight)

    def custom_forward(self, image: torch.Tensor, image_ls: torch.Tensor, target_ls: torch.Tensor, generator: GeneralGenerator)\
            -> torch.Tensor:

        # Concatanate input image with target landmark channels
        input = torch.cat((image, target_ls), 1)

        # Generate conditioned img
        gen_img = generator.forward(input)

        # Concatanate generated img with input landmark channels
        input2 = torch.cat((gen_img, image_ls), CHANNEL_DIM)

        # Generate approximation of input image
        gen_img_2 = generator.forward(input2)

        # Get L1**2 distance between generated approx. and original input img
        loss = L1_distance(gen_img_2, image).pow(2).mean()  # CHECK AGAIN!!!!!!!!!

        return loss


if __name__ == '__main__':
    # Test if working
    dummy_batch = torch.rand((20, 3, 28, 28))

    dummy_ls1 = torch.rand((20, 68, 28, 28))
    dummy_ls2 = torch.rand((20, 68, 28, 28))

    G = G()
    get_loss = ConsistencyLoss()

    loss = get_loss.forward(dummy_batch, dummy_ls1, dummy_ls2, G)[0]
    loss.backward()

    print(loss.item())
