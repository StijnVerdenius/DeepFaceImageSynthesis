from models.discriminators.GeneralDiscriminator import GeneralDiscriminator
import torch.nn as nn
import torch


class PatchDiscriminator(GeneralDiscriminator):
    """ Defines a PatchGAN discriminator"""

    def __init__(self, n_channels_in:int=3, n_hidden:int=64, n_layers:int=3, norm_layer: nn.Module=nn.BatchNorm2d, use_dropout: bool=True,
                 device: str="cpu", **kwargs):
        """
        n_channels_in (int)      - no. of channels in input images
        n_hidden (int)     - no. of filters in the last hidden layer
        n_layers (int)     - no. of layers of the Discriminator
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        """
        super(PatchDiscriminator, self).__init__(n_channels_in, device, **kwargs)

        # If normalizing layer is batch normalization, don't add bias because nn.BatchNorm2d has affine params
        use_bias = norm_layer != nn.BatchNorm2d

        # Initialize Discriminator input block
        layers = [nn.Conv2d(n_channels_in, n_hidden, kernel_size=4, stride=2, padding=1)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        # Set factor of change for input and output channels for hidden layers
        mult_in = 1
        mult_out = 1

        # Add hidden layers
        for i in range(1, n_layers + 1):

            mult_in = mult_out
            mult_out = min(2 ** i, 8)

            if i == n_layers:
                layers += [nn.Conv2d(n_hidden * mult_in, n_hidden * mult_out, kernel_size=4, stride=1, padding=1,
                                     bias=use_bias)]  # stride = 1
            else:
                layers += [nn.Conv2d(n_hidden * mult_in, n_hidden * mult_out, kernel_size=4, stride=2, padding=1,
                                     bias=use_bias)]  # stride = 2

            layers += [norm_layer(n_hidden * mult_out)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]
            layers += [nn.Dropout(0.33*int(use_dropout))] # todo: revisit dropout rate

        # Add output layer (1 channel prediction map)
        layers += [nn.Conv2d(n_hidden * mult_out, 1, kernel_size=4, stride=1, padding=1)]
        layers += [nn.Sigmoid()]

        # Save model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor)\
            -> torch.Tensor:

        batch_size = x.shape[0]

        return self.model(x).squeeze().view(batch_size, -1).mean(dim=1).squeeze()


if __name__ == '__main__':
    # Test if working

    dummy_batch = torch.rand((32, 71, 128, 128))

    D = PatchDiscriminator(n_channels_in=71)

    score = D.forward(dummy_batch)

    print(D.model[0].weight.grad)
    print(score.shape)
    print(score)

    # Test if working

    dummy_batch = torch.rand((10, 3, 28, 28))

    D = PatchDiscriminator(n_channels_in=3)

    score = D.forward(dummy_batch)

    print(D.model[0].weight.grad)
    print(score.shape)
    print(score)

