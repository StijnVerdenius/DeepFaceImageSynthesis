from models.embedders.GeneralEmbedder import GeneralEmbedder
import torch.nn as nn


class InitialEmbedder(GeneralEmbedder):

    def __init__(self, embedding_size=3, input_size=(1, 1), device="cpu"):
        super().__init__(input_size=input_size, device=device, embedding_size=embedding_size)

        self.dummy_variable = nn.Linear(self.input_size, self.embedding_size).to(self.device)

    def forward(self, inp):
        return self.dummy_variable(inp)
