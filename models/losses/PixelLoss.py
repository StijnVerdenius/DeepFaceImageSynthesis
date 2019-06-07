from models.generators import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from models.generators.pix2pixGenerator import pix2pixGenerator as G


class PixelLoss(GeneralLoss):

    def __init__(self, **kwargs):
        super(PixelLoss).__init__()

    def forward(self, image: torch.Tensor, target_ls: torch.Tensor, generator: GeneralGenerator):

        # Concatanate input image with target landmark channels
        input = torch.cat((image, target_ls), 1)

        # Generate conditioned img
        gen_img = generator.forward(input)

        # Get L2 **2 distance between generated approx. and original input img
        loss = torch.sum((gen_img-image).pow(2), dim=(1,2,3)).mean() # CHECK AGAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return loss

if __name__ == '__main__':


    # Test if working
    dummy_batch = torch.rand((20,3,28,28))

    dummy_ls1 = torch.rand((20,68,28,28))
    dummy_ls2 = torch.rand((20,68,28,28))

    G = G()
    get_loss = PixelLoss()


    loss = get_loss.forward(dummy_batch, dummy_ls1, G)
    loss.backward()



    print(loss.item())
