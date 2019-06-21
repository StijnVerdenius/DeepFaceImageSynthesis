from models.generators.GeneralGenerator import GeneralGenerator
import torch.nn as nn
import torch
from models.generators.ResnetGenerator import ResidualBlock


class UNetGenerator(GeneralGenerator):

    def __init__(self, n_channels_in: int = 71, n_channels_out: int = 3, n_hidden: int = 64,
                 norm_layer: nn.Module = nn.InstanceNorm2d, use_dropout: bool = True,
                 n_downsampling: int = 4, padding_type: str = 'zero', device: str = "cpu",
                 **kwargs):
        super(UNetGenerator, self).__init__(n_channels_in, n_channels_out, device, **kwargs, **kwargs)

        self.n_downsampling = n_downsampling

        # If normalizing layer is instance normalization, add bias
        use_bias = norm_layer == nn.InstanceNorm2d

        # Initialize model input block
        self.layers = []

        # Initialize padding
        pad = 0

        # Set padding
        if padding_type == 'zero':
            pad = 1
        elif padding_type == 'replicate':
            self.layers += [nn.ReflectionPad2d(3)]
        elif padding_type == 'reflect':
            self.layers += [nn.ReflectionPad2d(3)]
        else:
            raise NotImplementedError('Padding is not implemented! (padding type not zero, replicate or reflect)')

        self.layers = []

        # Add input block layers
        self.layers += [nn.Conv2d(n_channels_in, n_hidden, kernel_size=5, padding=pad, bias=use_bias)]
        self.layers += [nn.InstanceNorm2d(n_hidden)]
        self.layers += [nn.LeakyReLU(0.2, inplace=True)]

        # Add downsampling blocks
        for i in range(n_downsampling):
            block_down = []
            mult_ch = 2 ** i  # set factor to update current no. of channels
            block_down += [nn.Conv2d(n_hidden * mult_ch, n_hidden * mult_ch * 2, kernel_size=3, stride=2, padding=1,
                                     bias=use_bias)]
            block_down += [nn.InstanceNorm2d(n_hidden * mult_ch * 2)]
            block_down += [nn.LeakyReLU(0.2, inplace=True)]
            block_down += [nn.Dropout2d(0.3)]
            self.layers += [nn.Sequential(*block_down)]

        mult_ch = 2 ** n_downsampling
        self.residual_block = ResidualBlock(n_hidden * mult_ch, padding_type="zero", norm_layer=nn.InstanceNorm2d,
                                            use_dropout=use_dropout, use_bias=use_bias)
        self.layers += [self.residual_block]

        # Add upsampling blocks
        for i in range(n_downsampling):
            block_up = []
            mult_ch = 2 ** (n_downsampling - i)  # set factor to update current no. of channel
            block_up += [
                nn.ConvTranspose2d(n_hidden * mult_ch, int(n_hidden * mult_ch / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1 if (i == 0 or i == 1 or i == 3) else 0, bias=use_bias)]
            block_up += [nn.InstanceNorm2d(int(n_hidden * mult_ch / 2))]
            block_up += [nn.LeakyReLU(0.2, inplace=True)]
            block_up += [nn.Dropout2d(0.3)]
            self.layers += [nn.Sequential(*block_up)]

        # Add output block layers
        if padding_type == 'replicate':
            self.layers += [nn.ReflectionPad2d(3)]
        elif padding_type == 'reflect':
            self.layers += [nn.ReflectionPad2d(3)]
        self.layers += [nn.Conv2d(n_hidden, n_channels_out, kernel_size=7, padding=4)]
        self.layers += [nn.Tanh()]

        # Save model
        self.model = nn.Sequential(*self.layers)

    def forward(self, x) \
            -> torch.Tensor:

        memory = []

        started_downsampling = False
        started_upsampling = False
        temp = None

        layer: nn.Module
        for i, layer in enumerate(self.model):

            temp = layer.forward(x)

            if (isinstance(layer, nn.Sequential)):
                if (not started_downsampling and not started_upsampling):
                    started_downsampling = True
                if (started_downsampling):
                    memory.append(x)
                elif (started_upsampling):
                    temp = temp + memory.pop()
            else:
                if (started_downsampling and not started_upsampling):
                    started_upsampling = True
                    started_downsampling = False

            x = temp

        del memory
        del temp

        return x


if __name__ == '__main__':
    gen = UNetGenerator(device="cuda")

    dummy_batch = torch.rand((10, 71, 64, 64))

    banana = gen.forward(dummy_batch)

    print(banana.shape)
