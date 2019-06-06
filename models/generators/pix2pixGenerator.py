from models.generators.GeneralGenerator import GeneralGenerator
import torch.nn as nn
import torch
from models.losses.pix2pixGLoss import pix2pixGLoss as GLoss
from models.discriminators.PatchDiscriminator import PatchDiscriminator

class pix2pixGenerator(GeneralGenerator):
    """ Defines the pix2pix (CycleGAN) Generator"""


    # CHECK DEFAULT VALUES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, n_channels_in=71, n_channels_out=3, n_hidden=64, norm_layer=nn.InstanceNorm2d, use_dropout = False, n_downsampling=2, n_blocks=6, padding_type ='reflect', device="cpu"):
        """
        n_channels_in (int)      - no. of channels in input images
        n_channels_out (int)     - no. number of channels in output images
        n_hidden (int)     - no. of filters in the last hidden layer
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        n_blocks (int)     - no of ResNet blocks
        padding_type (str) - type of padding: zero, replicate, or reflect
        """
        super(pix2pixGenerator,self).__init__(n_channels_in, n_channels_out, device)

        # If normalizing layer is instance normalization, add bias
        use_bias = norm_layer == nn.InstanceNorm2d

        # Initialize model input block
        layers = []

        # Initialize padding
        pad = 0

        # Set padding
        if padding_type == 'zero':
            pad = 1
        elif padding_type == 'replicate':
            layers += [nn.ReflectionPad2d(3)]
        elif padding_type == 'reflect':
            layers += [nn.ReflectionPad2d(3)]
        else:
            raise NotImplementedError('Padding is not implemented! (padding type not zero, replicate or reflect)')

        # Add input block layers
        layers += [nn.Conv2d(n_channels_in, n_hidden, kernel_size=7, padding=pad, bias=use_bias)]
        layers += [nn.InstanceNorm2d(n_hidden)]
        layers += [nn.LeakyReLU(0.2,inplace=True)]


        # Add downsampling blocks
        for i in range(n_downsampling):
            mult_ch = 2 ** i # set factor to update current no. of channels
            layers += [nn.Conv2d(n_hidden * mult_ch, n_hidden * mult_ch * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
            layers += [nn.InstanceNorm2d(n_hidden * mult_ch * 2)]
            layers += [nn.LeakyReLU(0.2,inplace=True)]

        # Add ResNet blocks
        mult_ch = 2 ** n_downsampling # set factor to update current no. of channels
        for i in range(n_blocks):
            layers += [ResidualBlock(n_hidden * mult_ch, padding_type=padding_type, norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout, use_bias=use_bias)]

        # Add upsampling blocks
        for i in range(n_downsampling):
            mult_ch = 2 ** (n_downsampling - i) # set factor to update current no. of channel
            layers += [nn.ConvTranspose2d(n_hidden * mult_ch, int(n_hidden * mult_ch / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)]
            layers += [nn.InstanceNorm2d(int(n_hidden * mult_ch / 2))]
            layers += [nn.LeakyReLU(0.2,inplace=True)]


        # Add output block layers
        if padding_type == 'replicate':
            layers += [nn.ReflectionPad2d(3)]
        elif padding_type == 'reflect':
            layers += [nn.ReflectionPad2d(3)]
        layers += [nn.Conv2d(n_hidden, n_channels_out, kernel_size=7, padding=pad)]
        layers += [nn.Tanh()]


        # Save model
        self.model = nn.Sequential(*layers)


    def forward(self, x):

        return self.model(x)


class ResidualBlock(nn.Module):
    """ Defines a Residual block for the bottleneck of the pix2pix Generator"""

    def __init__(self, n_channels, padding_type, norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        """
        n_channels         - no. of input and output channels
        padding_type (str) - type of padding: reflect, replicate, or zero
        norm_layer         - normalization layer
        use_dropout (bool) - use dropout layers or not
        use_bias (bool)    - use bias or not
        """
        super(ResidualBlock, self).__init__()

        # Initialize residual block
        residual = []

        # Initialize padding
        pad = 0

        # Set padding
        if padding_type == 'zero':
            pad = 1
        elif padding_type == 'replicate':
            residual += [nn.ReflectionPad2d(1)]
        elif padding_type == 'reflect':
            residual += [nn.ReflectionPad2d(1)]

        # Add convolutional block
        residual += [nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=pad, bias=use_bias)]
        residual += [norm_layer(n_channels)]
        residual += [nn.LeakyReLU(0.2,inplace=True)]

        # Add dropout if required
        if use_dropout:
            residual += [nn.Dropout(0.5)]

        # Add padding
        if padding_type == 'replicate':
            residual += [nn.ReflectionPad2d(1)]
        elif padding_type == 'reflect':
            residual += [nn.ReflectionPad2d(1)]

        # Add convolutional block
        residual += [nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=pad, bias=use_bias)]
        residual += [norm_layer(n_channels)]


        self.resBlock = nn.Sequential(*residual)


    def forward(self, x):
        residual = self.resBlock(x)
        return x + residual # add skip connections



if __name__ == '__main__':

    # Test if working

    dummy_batch = torch.rand((10,71,28,28))

    G = pix2pixGenerator()
    D = PatchDiscriminator()

    gen_imgs = G.forward(dummy_batch)


    get_loss = GLoss()

    loss = get_loss.forward(gen_imgs,D)

    loss.backward()



    print(loss.item())
