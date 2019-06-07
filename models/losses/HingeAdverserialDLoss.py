from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch


class HingeAdverserialDLoss(GeneralLoss):

    def __init__(self, weight=1, **kwargs):
        super(HingeAdverserialDLoss).__init__(weight)

    def custom_forward(self, imgs, gen_imgs, discriminator):
        loss = torch.min(0, -1 + discriminator.forward(gen_imgs)).mean() - \
               torch.min(0, -discriminator.forward(imgs) - 1).mean()

        return loss
