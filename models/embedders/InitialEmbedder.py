from models.embedders.GeneralEmbedder import GeneralEmbedder
import torch.nn as nn
import torch


class InitialEmbedder(GeneralEmbedder):
    """ Takes a video frame and associated landmark image and maps these to an N-dimensional vector
    that is invariant to the pose and mimics in a particular frame"""

    def __init__(self, n_channels_in: int=784, n_hidden: int=64, n_channels_out: int=1, n_layers: int=3, device: str="cpu", **kwargs): # CHECK DEFAULT PARAMETERS!!!!! + Arch. is the same as PatchDiscriminator except output block
        """
        n_channels_in (int)      - no. of channels in input images
        n_hidden (int)     - no. of filters in the last hidden layer
        n_channels_out (int)     - no. number of channels in output images
        n_layers           - no. of convolutional layers for the Encoder
        use_dropout (bool) - use dropout layers or not
        """

        # super().__init__(input_size=input_size, device=device, embedding_size=embedding_size)

        super(InitialEmbedder,self).__init__(n_channels_in, n_channels_out, device)

        # Initialize Embedder input block
        layers = [nn.Conv2d(n_channels_in, n_hidden, kernel_size=4, stride=2, padding=1)]
        layers += [nn.BatchNorm2d(n_hidden)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        # Set factor of change for input and output channels for hidden layers
        mult_in = 1
        mult_out = 1

        for i in range(1,n_layers+1):

            mult_in = mult_out
            mult_out = min(2 ** i, 8)

            if i == n_layers:
                layers += [nn.Conv2d(n_hidden * mult_in, n_hidden * mult_out, kernel_size=4, stride=1, padding=1)] # stride = 1
            else:
                layers += [nn.Conv2d(n_hidden * mult_in, n_hidden * mult_out, kernel_size=4, stride=2, padding=1)] # stride = 2

            layers += [nn.BatchNorm2d(n_hidden * mult_out)]
            layers += [nn.LeakyReLU(0.2, inplace=True)]


        # Add output block
        layers += [nn.Linear(n_hidden * mult_out, n_channels_out)]
        layers += [nn.BatchNorm2d(n_hidden)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Linear(n_channels_out, n_channels_out)]

        # Save model
        self.model = nn.Sequential(*layers)



    def forward(self, x: torch.Tensor)\
            -> torch.Tensor:

        return self.model(x)
