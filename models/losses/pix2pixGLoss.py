from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch


class pix2pixGLoss(GeneralLoss):

    def __init__(self):
        super(pix2pixGLoss).__init__()

    def forward(self, fake, discriminator):
        loss = -1 * torch.mean(torch.log(discriminator.forward(fake)))  # CHECKK

        return loss
