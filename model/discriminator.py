import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .blocks import ResBlockEncoder, get_norm_layer, get_nonlinearity_layer
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

class ResDiscriminator(nn.Module):
    """
    ResNet Discriminator Network
    :param input_nc: number of channels in input
    :param ndf: base filter channel
    :param layers: down and up sample layers
    :param img_f: the largest feature channels
    :param norm: normalization function 'instance, batch, group'
    :param activation: activation function 'ReLU, SELU, LeakyReLU, PReLU'
    """
    def __init__(self, input_nc=3, ndf=64, img_f=1024, layers=6, norm='none', activation='LeakyReLU', use_spect=True,
                 use_coord=False):
        super(ResDiscriminator, self).__init__()

        self.layers = layers

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.nonlinearity = nonlinearity

        # first encoder in Discriminator # [3,256,256] -> [64,128,128] 
        self.block0 = ResBlockEncoder(input_nc, ndf, ndf, norm_layer, nonlinearity, use_spect, use_coord)

        mult = 1
        for i in range(layers - 1):
            mult_prev = mult
            mult = min(2 ** (i + 1), img_f//ndf) #layers=6, mult 2, 4, 8, 16, 16#  [128,64,64] [256,32,32] [512,16,16] [1024,8,8] [1024,4,4]
            block = ResBlockEncoder(ndf*mult_prev, ndf*mult, ndf*mult_prev, norm_layer, nonlinearity, use_spect, use_coord)
            setattr(self, 'encoder' + str(i), block)
        self.conv = SpectralNorm(nn.Conv2d(ndf*mult, 1, 1)) # [1024,4,4] -> [1, 4, 4]

    def forward(self, x):
        out = self.block0(x)
        for i in range(self.layers - 1):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
        out = self.conv(self.nonlinearity(out))
        return out