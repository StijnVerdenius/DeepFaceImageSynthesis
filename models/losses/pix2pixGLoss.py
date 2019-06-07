from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch


class pix2pixGLoss(GeneralLoss):

    def __init__(self, weight, **kwargs):
        super(pix2pixGLoss, self).__init__(weight=weight)

    # todo: add methods here that are shared for all generators, inheret your costum version from this object

    def custom_forward(self, fake, discriminator):
        loss = -1 * torch.mean(torch.log(discriminator.forward(fake)))  # CHECKK

        return loss
