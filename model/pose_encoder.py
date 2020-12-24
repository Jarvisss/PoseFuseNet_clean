from .blocks import *
import torch
import torch.nn as nn 

class AutoEncoder(nn.Module):

    def __init__(self, n_layers=3, input_nc=18, nf=32, max_nc=256, norm_type='in',activation='LeakyReLU',use_spectral_norm=False):
        super(AutoEncoder, self).__init__()

        self.n_layers = n_layers
        self.nf = nf
        self.max_nc = max_nc
        self.input_nc = input_nc
        self.use_spect = use_spectral_norm
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self._make_layers(norm_layer, nonlinearity)

    def _make_layers(self, norm_layer, nonlinearity):
        mult = 1
        in_layer = EncoderBlock(self.input_nc, self.nf, norm_layer, nonlinearity,self.use_spect )
        setattr(self, 'in_layer', in_layer)
        for i in range(0, self.n_layers-1): # 18 32 64 128 64 32 18
            mult_prev = mult
            mult = mult * 2
            encoder = EncoderBlock(mult_prev*self.nf, mult*self.nf, norm_layer, nonlinearity, use_spect=self.use_spect)
            setattr(self, 'encoder'+str(i), encoder)

        
        for i in range(self.n_layers-1):
            mult_prev = mult
            mult = mult // 2
            decoder = ResBlockDecoder(mult_prev*self.nf, mult*self.nf, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=self.use_spect)
            setattr(self, 'decoder'+str(i), decoder)

        out_layer = ResBlockDecoder(mult*self.nf, self.input_nc,norm_layer=norm_layer, nonlinearity=nonlinearity,  use_spect=self.use_spect)
        setattr(self, 'out_layer', out_layer)

    def forward(self, pose):
        in_layer = getattr(self,'in_layer')
        out = in_layer(pose)
        for i in range(self.n_layers-1):
            encoder = getattr(self, 'encoder'+str(i))
            out = encoder(out)
        mid = out
        for i in range(self.n_layers-1):
            decoder = getattr(self, 'decoder'+str(i))
            out = decoder(out)

        out_layer = getattr(self,'out_layer')
        out = out_layer(out)

        return out, mid


