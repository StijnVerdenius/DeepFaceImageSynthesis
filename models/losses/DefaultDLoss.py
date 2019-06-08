from models.losses.GeneralLoss import GeneralLoss
import torch.nn as nn
import torch

class DefaultDLoss(GeneralLoss):

    def __init__(self, **kwargs):
        super(DefaultDLoss, self).__init__()


    def custom_forward(self,predictions, labels):

        loss = nn.BCELoss() ##CHECK

        return loss(predictions,labels)
