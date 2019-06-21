from models.losses.GeneralLoss import GeneralLoss
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch

from utils.constants import DEVICE


class HingeAdverserialDLoss(GeneralLoss):

    def __init__(self, weight: float = 1, **kwargs):
        super(HingeAdverserialDLoss, self).__init__(weight)
        self.zero = torch.zeros((1)).to(DEVICE)

    def custom_forward(self, real_scores: torch.Tensor, fake_scores: torch.Tensor, *args) \
            -> torch.Tensor:


        loss = torch.max(self.zero,1 + fake_scores).mean() + \
               torch.max(self.zero, 1 - real_scores).mean()

        return loss

