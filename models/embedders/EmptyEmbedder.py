import torch

from models.embedders.GeneralEmbedder import GeneralEmbedder
import torch.nn as nn


class EmptyEmbedder(GeneralEmbedder):
    """ for running without embedder """

    def __init__(self, **kwargs):
        super().__init__()
        print("Note: Running without embedder")

    @staticmethod
    def forward(_):
        return None

    @staticmethod
    def parameters():
        return [torch.LongTensor([])]

    @staticmethod
    def state_dict():
        return None
