import torch.nn as nn

from models.GeneralModel import GeneralModel


class GeneralGenerator(GeneralModel):

    def __init__(self, n_input=(1, 1), n_output=(1, 1), device="cpu"):  # CHECK DEFAULT VALUES!
        super(GeneralGenerator, self).__init__(n_input, device)

        self.n_output = n_output

    # todo: add methods here that are shared for all generators, inheret your costum version from this object
