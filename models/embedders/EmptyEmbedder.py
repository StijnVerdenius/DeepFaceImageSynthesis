import torch

from models.embedders.GeneralEmbedder import GeneralEmbedder
import torch.nn as nn


class EmptyEmbedder(GeneralEmbedder):
    """ for running without embedder """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Note: Running without embedder")

    @staticmethod
    def forward(_):
        return None

    @staticmethod
    def parameters(*args):
        return [torch.LongTensor([])]

    @staticmethod
    def state_dict(**kwargs):
        return None

    @staticmethod
    def load_state_dict(self, *args, **kwargs):
        pass
