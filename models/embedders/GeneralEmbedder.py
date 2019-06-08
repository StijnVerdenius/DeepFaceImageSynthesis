import torch.nn as nn

from models.GeneralModel import GeneralModel


class GeneralEmbedder(GeneralModel):

    def __init__(self, n_channels_in: int = 1, n_channels_out: int=1, device: str="cpu"): # CHECK DEFAULT PARAMETERS!!!!!!
        super().__init__(n_channels_in, device)
        self.n_channels_out = n_channels_out

    # todo: add methods here that are shared for all embedders, inheret your costum version from this object
