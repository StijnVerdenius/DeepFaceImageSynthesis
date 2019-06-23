from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch
from models.losses.DefaultDLoss import DefaultDLoss as DLoss
from utils.constants import IMSIZE


class PixelDiscriminator(GeneralDiscriminator):
    """ Defines a PixelGAN discriminator"""

    def __init__(self, n_channels_in: int=3, n_hidden: int=64, norm_layer: nn.Module=nn.BatchNorm2d, use_dropout: bool=True, device: str="cpu",
                 **kwargs):
        """
        n_input (int)      - no. of channels in input images
        n_hidden (int)     - no. of filters in the last hidden layer
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        """
        super(PixelDiscriminator, self).__init__(n_channels_in, device, **kwargs)

        # If normalizing layer is batch normalization, don't add bias because nn.BatchNorm2d has affine params
        use_bias = norm_layer != nn.BatchNorm2d

        # Initialize Discriminator
        layers = [
            nn.Conv2d(n_channels_in, n_hidden, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5 * int(use_dropout)),  # todo: revisit dropout rate
            nn.Conv2d(n_hidden, n_hidden * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(n_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5 * int(use_dropout)),  # todo: revisit dropout rate
            nn.Conv2d(n_hidden * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid()
        ]

        # Save model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor)\
            -> torch.Tensor:
        return self.model.forward(x).view(-1, IMSIZE * IMSIZE).mean(dim=1)


if __name__ == '__main__':
    # Test if working

    dummy_batch = torch.rand((10, 3, 128, 128))

    G = PixelDiscriminator()

    score = G.forward(dummy_batch)

    print(score.shape)

    get_loss = DLoss()

    target = torch.rand((10))

    loss = get_loss.forward(score, target)[0]

    loss.backward()

    print(loss)

    # print(score)
