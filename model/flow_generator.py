from .blocks import *
import torch
import torch.nn as nn 

class FlowGenerator(nn.Module):
    def __init__(self, inc=43,n_layers=3, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True, mask_use_sigmoid=True):
        super(FlowGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        

        self.resDown1 = ResBlockEncoder(inc, 64, 64,  norm_layer, nonlinearity, use_spectral_norm) # (64, 128, 128)
        self.resDown2 = ResBlockEncoder(64, 128,128,  norm_layer, nonlinearity, use_spectral_norm) # (128, 64, 64)
        self.resDown3 = ResBlockEncoder(128, 256,256, norm_layer, nonlinearity, use_spectral_norm) # (256, 32, 32)
        self.resDown4 = ResBlockEncoder(256, 256,256, norm_layer, nonlinearity, use_spectral_norm) # (256, 16, 16)

        self.resUp1 = ResBlockUpNorm(256, 256, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (256, 32, 32)
        self.resUp2 = ResBlockUpNorm(256, 128, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (128, 64, 64)
        self.resUp3 = ResBlockUpNorm(128, 64, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (64, 128, 128)
        self.resUp4 = ResBlockUpNorm(64, 32, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (32, 256, 256)

        
        self.flow_module = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 2, 3, padding = 0), # (2, 256, 256)
        )
        
        self.mask_module = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 1, 3, padding = 0) # (1, 256, 256)
        )

        self.mask_use_sigmoid = mask_use_sigmoid
        
    def forward(self, image1, pose1, pose2, sim_map=None):
        if sim_map is None:
            x_in = torch.cat((image1, pose1, pose2), dim=1) # (43, 256, 256)
        else:
            x_in = torch.cat((image1, pose1, pose2, sim_map), dim=1) # (43+13, 256, 256)

        x_resdown1 = self.resDown1(x_in) # (64, 128, 128)
        x_resdown2 = self.resDown2(x_resdown1) # (128, 64, 64)
        x_resdown3 = self.resDown3(x_resdown2) # (256, 32, 32)
        x_resdown4 = self.resDown4(x_resdown3) # (256, 16, 16)

        x_resup1 = self.resUp1(x_resdown4) # (256, 32, 32)
        x_resup2 = self.resUp2(x_resup1) # (128, 64, 64)
        x_resup3 = self.resUp3(x_resup2) # (64, 128, 128)
        x_resup4 = self.resUp4(x_resup3) # (32, 256, 256)

        flow = self.flow_module(x_resup2) # (2, 256, 256)
        # mask = torch.sigmoid(self.mask_module(x_resup4))
        mask = self.mask_module(x_resup2)
        if self.mask_use_sigmoid:
            mask = torch.sigmoid(mask)
        # x_hat = warp_flow(image1, flow)
        return flow, mask

class AppearanceEncoder(nn.Module):
    '''
    appearance encoder
    '''
    def __init__(self, n_layers=3, inc=3,norm_type='bn',activation='LeakyReLU',  use_spectral_norm=True):
        super(AppearanceEncoder, self).__init__()
        self.n_layers = n_layers
        self.inc = inc
        self.max_nc = 256
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.down = self._make_layers(self.max_nc, norm_layer, nonlinearity, use_spectral_norm)
        

    def _make_layers(self, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        '''
        Encoder:
        layer1 : (3,256,256) -> (64,128,128)
        layer2 : (64,128,128) -> (128,64,64)
        layer3 : (128,64,64) -> (256,32,32)
        
        Decoder
        layer1 : (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (32,256,256)

        layer4 : BN, ReLU
        layer5 :        
        
        '''
        onc = 64
        layers = [ResBlockEncoder(self.inc, onc, onc,  norm_layer, nonlinearity, use_spectral_norm)]
        for i in range(1, int(self.n_layers)):
            if onc * 2 > max_nc:
                layers.append(ResBlockEncoder(onc, onc, onc, norm_layer, nonlinearity, use_spectral_norm))
            else:
                layers.append(ResBlockEncoder(onc, onc * 2, onc,  norm_layer, nonlinearity, use_spectral_norm))
            onc = onc * 2

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.down(x)

        
class AppearanceDecoder(nn.Module):
    '''
    Decode part of the generator
    '''
    def __init__(self, n_bottleneck_layers=4, n_decode_layers=3, norm_type='bn', use_spectral_norm=True):
        super(AppearanceDecoder, self).__init__()
        self.n_bottleneck_layers = n_bottleneck_layers
        self.n_decode_layers = n_decode_layers
        self.bottleneck_nc = 64 * 2**(n_decode_layers-1)


        self.decoder_onc = 32
        self.norm_layer = make_norm_layer(norm_type, self.decoder_onc)
        self.relu = nn.LeakyReLU(inplace = False)
        self.conv2d = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.decoder_onc, 3, 3, padding = 0)
        )
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        
        self.bottleneck = self._make_bottle_neck_layers(norm_type,use_spectral_norm)
        self.decoder = self._make_decoder_layers(norm_type,use_spectral_norm)

    def _make_bottle_neck_layers(self, norm_type, use_spectral_norm):
        layers = []
        for i in range(self.n_bottleneck_layers):
            layers.append(ResBlock2d(self.bottleneck_nc, 3, 1, norm_type, use_spectral_norm))
        
        return nn.Sequential(*layers)

    def _make_decoder_layers(self, norm_type,use_spectral_norm):
        layers = []
        for i in range(self.n_decode_layers ):
            # layers.append(ConvUp(self.bottleneck_nc//(2**i), self.bottleneck_nc//(2**i*2), norm_type=norm_type, norm_layer, nonlinearity, use_spectral_norm))
            layers.append(ResBlockUpNorm(self.bottleneck_nc//(2**i), self.bottleneck_nc//(2**i*2), norm_type=norm_type, use_spectral_norm=use_spectral_norm))
            
            # if i == self.n_decode_layers -1:
            #     layers.append(ResBlockUpNorm(self.bottleneck_nc//(2**i), 3, norm_type=norm_type, norm_layer, nonlinearity, use_spectral_norm))
            # else:
            #     layers.append(ResBlockUpNorm(self.bottleneck_nc//(2**i), self.bottleneck_nc//(2**i*2), norm_type=norm_type, norm_layer, nonlinearity, use_spectral_norm))
        
        layers.append(self.norm_layer) 
        layers.append(self.relu) 
        layers.append(self.conv2d)        
        layers.append(self.tanh)
        # layers.append(self.sigmoid)

        return nn.Sequential(*layers)
        
    def forward(self, feature):
        return self.decoder(self.bottleneck(feature))
        # return self.decoder(feature)








