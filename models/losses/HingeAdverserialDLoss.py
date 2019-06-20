from models.losses.GeneralLoss import GeneralLoss
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch

from utils.constants import DEVICE


class HingeAdverserialDLoss(GeneralLoss):

    def __init__(self, weight: float = 1, **kwargs):
        super(HingeAdverserialDLoss, self).__init__(weight)
        self.zero = torch.zeros((1)).to(DEVICE)

    def custom_forward(self, combined: torch.Tensor, *args) \
            -> torch.Tensor:
        split_index = combined.shape[0] // 2

        fake_scores = combined[:split_index]
        real_scores = combined[split_index:]

        loss = - torch.min(self.zero, -1 + fake_scores).mean() - \
               torch.min(self.zero, -real_scores - 1).mean()

        return loss

