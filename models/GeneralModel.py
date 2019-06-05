import torch.nn as nn


class GeneralModel(nn.Module):

    def __init__(self, n_input, device):
        self.n_input = n_input
        self.device = device
        super().__init__()
