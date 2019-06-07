import torch.nn as nn


class GeneralLoss(nn.Module):

    def __init__(self, weight=1):
        super(GeneralLoss, self).__init__()
        self.weight = weight  # hyperparameter

    def forward(self, *input):
        """
        wrapper forward for other child class forward-methods so that weight can be applied

        :param input: any number of params
        :return:
        """

        loss = self.custom_forward(*input)

        return self.weight * loss

    def custom_forward(self, *input):
        """
        Method place-holder to be overridden in child-class


        :param input: any number of params
        :return:
        """
        raise Exception("PLEASE IMPLEMENT costum_forward METHOD IN CHILD-CLASS")
