import torch.nn as nn
from models.GeneralModel import GeneralModel


class GeneralDiscriminator(GeneralModel):

    def __init__(self, n_input, device="cpu"):
        super(GeneralDiscriminator,self).__init__(n_input, device)

    # todo: add methods here that are shared for all discriminators, inheret your costum version from this object
