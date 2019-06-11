import torch.nn as nn
from models.GeneralModel import GeneralModel


class GeneralDiscriminator(GeneralModel):

    def __init__(self, n_channels_in: int, device:str="cpu", **kwargs):
        super(GeneralDiscriminator,self).__init__(n_channels_in, device, **kwargs)

    # todo: add methods here that are shared for all discriminators, inheret your costum version from this object
