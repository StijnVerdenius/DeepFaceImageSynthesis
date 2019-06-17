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

class IdLoss(GeneralLoss):

    def __init__(self, weight: float, **kwargs):
        super(IdLoss, self).__init__(weight=weight)

    def custom_forward(self, real_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                       fake_feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor])\
            -> torch.Tensor:
        """ forward pass """

        real_feats = real_feats[0]
        fake_feats = fake_feats[0]

        # reconstruction loss
        l1 = 0
        for real, fake in zip(real_feats, fake_feats):
            l1 += L1_distance(real, fake).mean()

        return l1




if __name__ == '__main__':
    z = IdLoss(1)

    testinput = torch.rand((20, 3, 28, 28))
    testinput_2 = torch.rand((20, 3, 28, 28))

    bana = z.forward(testinput, testinput_2)[0]

    print(bana.shape,bana)
