from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn


class InitialDiscriminator(GeneralDiscriminator):

    def __init__(self, input_size=(1, 1), device="cpu"):
        super().__init__(input_size=input_size, device=device)

        self.dummy_variable = nn.Linear(self.input_size, 1).to(self.device)

    def forward(self, inp):
        return self.dummy_variable(inp)
