from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from models.generators.pix2pixGenerator import pix2pixGenerator as G


class ConsistencyLoss(GeneralLoss):

    def __init__(self):
        super(ConsistencyLoss).__init__()

    # todo: add methods here that are shared for all generators, inheret your costum version from this object

    def forward(self, image, image_ls, target_ls, generator):

        # Concatanate input image with target landmark channels
        input = torch.cat((image, target_ls), 1)

        # Generate conditioned img
        gen_img = generator.forward(input)

        # Concatanate generated img with input landmark channels
        input2 = torch.cat((gen_img,image_ls),1)

        # Generate approximation of input image
        gen_img_2 = generator.forward(input2)

        # Get L1**2 distance between generated approx. and original input img
        loss = torch.sum(torch.abs(gen_img_2-image), dim=(1,2,3)).pow(2).mean() # CHECK AGAIN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return loss



if __name__ == '__main__':


    # Test if working
    dummy_batch = torch.rand((20,3,28,28))

    dummy_ls1 = torch.rand((20,68,28,28))
    dummy_ls2 = torch.rand((20,68,28,28))

    G = G()
    get_loss = ConsistencyLoss()


    loss = get_loss.forward(dummy_batch, dummy_ls1, dummy_ls2, G)
    loss.backward()



    print(loss.item())
