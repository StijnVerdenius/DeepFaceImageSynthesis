from models.generators import GeneralGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from models.generators.ResnetGenerator import ResnetGenerator as G
from utils.training_helpers import L2_distance, CHANNEL_DIM


class PixelLoss(GeneralLoss):

    def __init__(self, weight=1, **kwargs):
        super(PixelLoss,self).__init__(weight)

    def custom_forward(self, image: torch.Tensor, target_ls: torch.Tensor, generator: GeneralGenerator)\
            -> torch.Tensor:
        # Concatanate input image with target landmark channels
        input = torch.cat((image, target_ls), CHANNEL_DIM)

        # Generate conditioned img
        gen_img = generator.forward(input)

        # Get L2 **2 distance between generated approx. and original input img
        loss = L2_distance(gen_img, image).pow(2).mean() # todo: squaring and square rooting?

        return loss



if __name__ == '__main__':


    # Test if working
    dummy_batch = torch.rand((20,3,28,28))

    dummy_ls1 = torch.rand((20,68,28,28))
    dummy_ls2 = torch.rand((20,68,28,28))

    G = G()
    get_loss = PixelLoss()


    loss = get_loss.forward(dummy_batch, dummy_ls1, G)[0]
    loss.backward()



    print(loss.item())
