import torch.nn as nn

from models.GeneralModel import GeneralModel


class GeneralGenerator(GeneralModel):

    def __init__(self, n_channels_in=(1), n_channels_out=(1), device="cpu", **kwargs):  # CHECK DEFAULT VALUES!!!!!!!!
        super(GeneralGenerator, self).__init__(n_channels_in, device, **kwargs)

        self.n_channels_out = n_channels_out

    # todo: add methods here that are shared for all generators, inheret your custom version from this object
