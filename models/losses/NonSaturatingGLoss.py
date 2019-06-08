from models.losses.GeneralLoss import GeneralLoss
from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch


class NonSaturatingGLoss(GeneralLoss):

    def __init__(self, weight:float, **kwargs):
        super(NonSaturatingGLoss, self).__init__(weight=weight)

    def custom_forward(self, fake: torch.Tensor, discriminator: GeneralDiscriminator):
        loss = -1 * torch.mean(torch.log(discriminator.forward(fake)))  # CHECKK

        return loss
