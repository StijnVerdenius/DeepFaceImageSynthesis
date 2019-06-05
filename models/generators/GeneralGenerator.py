import torch.nn as nn

from models.GeneralModel import GeneralModel


class GeneralGenerator(GeneralModel):

    def __init__(self, n_input, n_output, device="cpu"):
        super().__init__(input_size, device)

    # todo: add methods here that are shared for all generators, inheret your costum version from this object



