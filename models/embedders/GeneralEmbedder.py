import torch.nn as nn

from models.GeneralModel import GeneralModel


class GeneralEmbedder(GeneralModel):

    def __init__(self, embedding_size=(1,), input_size = (1,1), device="cpu"):
        super().__init__(input_size, device)
        self.embedding_size = embedding_size

    # todo: add methods here that are shared for all embedders, inheret your costum version from this object
