from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch
from models.losses.pix2pixDLoss import pix2pixDLoss as DLoss


class PixelDiscriminator(GeneralDiscriminator):
    """ Defines a PixelGAN discriminator"""

    def __init__(self, n_channels_in=3, n_hidden=64, norm_layer=nn.BatchNorm2d, use_dropout=False, device="cpu"): #havent used dropout!!!!!!!!
        """
        n_input (int)      - no. of channels in input images
        n_hidden (int)     - no. of filters in the last hidden layer
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        """
        super(PixelDiscriminator,self).__init__(n_channels_in, device)

        # If normalizing layer is batch normalization, don't add bias because nn.BatchNorm2d has affine params
        use_bias = norm_layer != nn.BatchNorm2d

        # Initialize Discriminator
        layers = [
            nn.Conv2d(n_channels_in, n_hidden, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_hidden, n_hidden * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(n_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_hidden * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid()
        ]


        # Save model
        self.model = nn.Sequential(*layers)


    def forward(self, x):

        return self.model(x)


if __name__ == '__main__':

    # Test if working

    dummy_batch = torch.rand((10,3,28,28))

    G = PixelDiscriminator()

    score = G.forward(dummy_batch)

    get_loss = DLoss()

    target = torch.rand((10,1,28,28))

    loss = get_loss.forward(score, target)

    loss.backward()

    print(loss)


    # print(score)

