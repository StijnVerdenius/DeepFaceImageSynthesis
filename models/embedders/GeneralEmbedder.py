import torch.nn as nn

from models.GeneralModel import GeneralModel


class GeneralEmbedder(GeneralModel):

    def __init__(self, n_input = 1, n_output=1, device="cpu"): # CHECK DEFAULT PARAMETERS!!!!!!
        super().__init__(n_input, device)
        self.n_output = n_output

    # todo: add methods here that are shared for all embedders, inheret your costum version from this object
