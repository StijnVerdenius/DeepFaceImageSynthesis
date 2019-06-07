from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch


class NonSaturatingGLoss(GeneralLoss):

    def __init__(self, weight, **kwargs):
        super(NonSaturatingGLoss, self).__init__(weight=weight)

    def custom_forward(self, fake, discriminator):
        loss = -1 * torch.mean(torch.log(discriminator.forward(fake)))  # CHECKK

        return loss
