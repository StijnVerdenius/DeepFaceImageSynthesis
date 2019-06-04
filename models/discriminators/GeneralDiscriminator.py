import torch.nn as nn
from models.GeneralModel import GeneralModel


class GeneralDiscriminator(GeneralModel):

    def __init__(self, input_size = (1,1), device="cpu"):
        super().__init__(input_size, device)

    # todo: add methods here that are shared for all discriminators, inheret your costum version from this object
