from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch

class pix2pixDLoss(GeneralLoss):

    def __init__(self):
        super(pix2pixDLoss).__init__()


    def forward(self,predictions, labels):

        loss = nn.BCELoss() ##CHECK

        return loss(predictions,labels)
