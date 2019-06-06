from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn


class PixelDiscriminator(GeneralDiscriminator):
    """ Defines a PixelGAN discriminator"""

    def __init__(self, n_input, n_hidden=64, norm_layer=nn.BatchNorm2d, use_dropout=False, device="cpu"):
        """
        n_input (int)      - no. of channels in input images
        n_hidden (int)     - no. of filters in the last hidden layer
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        """
        super(PixelDiscriminator,self).__init__(n_input, device)

        # If normalizing layer is batch normalization, don't add bias because nn.BatchNorm2d has affine params
        use_bias = norm_layer != nn.BatchNorm2d

        # Initialize Discriminator

        layers = [
            nn.Conv2d(n_input, n_hidden, kernel_size=1, stride=1, padding=0),
            nn.Dropout(int(use_dropout) * 0.33),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_hidden, n_hidden * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            nn.Dropout(int(use_dropout) * 0.33),
            norm_layer(n_hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_hidden * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)
        ]

        # Save model
        self.model = nn.Sequential(*layers)

        # todo: scalar output?


    def forward(self, x):

        return self.model(x)
