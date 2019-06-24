from models.generators.GeneralGenerator import GeneralGenerator
from models.generators.ResnetGenerator import ResnetGenerator
from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch
from typing import Tuple
from utils.constants import *
# from utils.model_utils import
from utils.training_helpers import L1_distance
from typing import Tuple, Dict

class PerceptualLoss(GeneralLoss):

    def __init__(self, weight: float, **kwargs):
        super(PerceptualLoss, self).__init__(weight=weight)

    def custom_forward(self, real_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                       fake_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])\
            -> torch.Tensor:
        """ forward pass """

        real_feats = real_feats[1:]
        fake_feats = fake_feats[1:]

        # reconstruction loss
        l1_part = 0
        for real, fake in zip(real_feats, fake_feats):
            l1_part += L1_distance(real, fake).mean()

        # style loss
        frobenius = self.frobenius_norm(self.gram_matrix(real_feats[2]), self.gram_matrix(fake_feats[2]))

        return frobenius + l1_part

    def frobenius_norm(self, batch_1: torch.Tensor, batch_2: torch.Tensor)\
            -> torch.Tensor:
        """ forbenius norm (just normal norm?)"""

        return torch.norm((batch_1 - batch_2))  # todo: revisit

    def gram_matrix(self, batch: torch.Tensor)\
            -> torch.Tensor:
        """ calculates gram matrix """

        # dims
        a, b, c, d = batch.size()

        # make 2d (batch * ch) x (h * w)
        features = batch.view((a * b, c * d))

        # get gram-matrix
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps. todo: is this necessary?
        return G.div(a * b * c * d)



if __name__ == '__main__':
    z = PerceptualLoss(1)

    testinput = torch.rand((20, 3, 28, 28))
    testinput_2 = torch.rand((20, 3, 28, 28))

    bana = z(testinput, testinput_2)[0]

    print(bana.shape, bana)
