import torch

from models.embedders.GeneralEmbedder import GeneralEmbedder
import torch.nn as nn


class EmptyEmbedder():
    """ for running without embedder """

    def __init__(self, **kwargs):
        print("Note: Running without embedder")
        pass

    @staticmethod
    def forward(_):
        return None

    @staticmethod
    def parameters():
        return None

    @staticmethod
    def state_dict():
        return None
