import torch.nn as nn


class GeneralModel(nn.Module):

    def __init__(self, input_size, device):
        self.input_size = input_size
        self.device = device
        super().__init__()
