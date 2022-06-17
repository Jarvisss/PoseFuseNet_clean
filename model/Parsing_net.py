from .blocks import *
import torch
import torch.nn as nn 

class ParsingGenerator(nn.Module):
    def __init__(self, inc=43, norm_type='bn', use_spectral_norm=True, use_attn=True,mask_use_sigmoid=True):
        super(ParsingGenerator, self).__init__()

        self.resDown1 = ResBlockDown(inc, 64, use_spectral_norm=use_spectral_norm) # (64, 128, 128)
        self.in1 = make_norm_layer(norm_type, 64)
        self.resDown2 = ResBlockDown(64, 128, use_spectral_norm=use_spectral_norm) # (128, 64, 64)
        self.in2 = make_norm_layer(norm_type, 128)

        self.resDown3 = ResBlockDown(128, 256, use_spectral_norm=use_spectral_norm) # (256, 32, 32)
        self.in3 = make_norm_layer(norm_type, 256)
        
        # self.self_att_down = SelfAttention(256) #
        self.resDown4 = ResBlockDown(256, 256, use_spectral_norm=use_spectral_norm) # (256, 16, 16)
        self.in4 = make_norm_layer(norm_type, 256)

        # self.res1 = ResBlock2d(256, 3, 1, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
        # self.res2 = ResBlock2d(256, 3, 1, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
        # self.res3 = ResBlock2d(256, 3, 1, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
        # self.res4 = ResBlock2d(256, 3, 1, norm_type=norm_type, use_spectral_norm=use_spectral_norm)

        self.resUp1 = ResBlockUpNorm(256, 256, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (256, 32, 32)
        self.resUp2 = ResBlockUpNorm(256, 128, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (128, 64, 64)
        # self.self_att_up = SelfAttention(128) #
        self.resUp3 = ResBlockUpNorm(128, 64, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (64, 128, 128)
        self.resUp4 = ResBlockUpNorm(64, 32, norm_type=norm_type, use_spectral_norm=use_spectral_norm) # (32, 256, 256)

        
        self.outlayer = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 20, 3, padding = 0), # (20, 256, 256)
        )
        
        # self.flow_module = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(32, 2, 3, padding = 0) # (2, 256, 256)
        # )
        self.attention_module = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 1, 3, padding = 0) # (1, 256, 256)
        )
        self.use_attn = use_attn
    def forward(self, image1, parsing1, pose2):
        _,_,H,W = image1.shape
        x_in = torch.cat((image1, parsing1, pose2), dim=1) # (43, 256, 256)
        x_resdown1 = self.resDown1(x_in) # (64, 128, 128)
        x_resdown2 = self.resDown2(x_resdown1) # (128, 64, 64)

        x_resdown3 = self.resDown3(x_resdown2) # (256, 32, 32)
        x_resdown4 = self.resDown4(x_resdown3) # (256, 16, 16)

        # x_att_down = self.self_att_down(x_resdown4) #
        
        # x_res1 = self.res1(x_resdown4)
        # x_res2 = self.res2(x_res1)
        # x_res3 = self.res3(x_res2)
        # x_res4 = self.res4(x_res3)


        x_resup1 = self.resUp1(x_resdown4) # (256, 32, 32)
        x_resup2 = self.resUp2(x_resup1) # (128, 64, 64)

        # x_att_up = self.self_att_up(x_resup2)
        x_resup3 = self.resUp3(x_resup2) # (64, 128, 128)
        x_resup4 = self.resUp4(x_resup3) # (32, 256, 256)

        output_logits = self.outlayer(x_resup4) # (20, 256, 256)
        if self.use_attn:
            attn = self.attention_module(x_resup4) # (1, 256, 256)
            # mask = torch.sigmoid(self.mask_module(x_resup4))
            return output_logits, attn
        else:
            return output_logits


class MaskGenerator(nn.Module):
    '''doc
    '''
    def __init__(self, inc=3+1+21,onc=1, ngf=64, n_layers=6,max_nc=512, norm_type='bn', activation='LeakyReLU', use_spectral_norm=False):
        super(MaskGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.inc = inc
        self.onc = onc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = n_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.inc
        onc = self.onc
        self.encoder_block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.encoder_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'encoder' + str(i), block)
        
        mult_prev = mult
        mult = min(2 ** (self.decoder_layers-2), max_nc//ngf)
        self.decoder_block0 = ResBlockUpNorm(ngf*mult_prev, ngf*mult,norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)

        for i in range(1, self.decoder_layers-1):
            mult_prev = mult * 2
            mult = min(2 ** (self.decoder_layers-i-2), max_nc//ngf)
            block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            setattr(self, 'decoder' + str(i), block)
        
        mult_prev = mult * 2
        self.decoder_out = ResBlockUpNorm(ngf*mult_prev, onc, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
        self.attn_out = ResBlockUpNorm(ngf*mult_prev, onc, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
        

    def forward(self, image1, mask1, pose2, pose_1=None):
        _,_,H,W = image1.shape
        if pose_1 is None:
            x_in = torch.cat((image1, mask1, pose2), dim=1) # (43, 256, 256)
        else:
            x_in = torch.cat((image1, mask1,pose_1, pose2), dim=1) # (43, 256, 256)
        out = self.encoder_block0(x_in)
        result = [out]
        for i in range(1, self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        
        out = self.decoder_block0(out)
        for i in range(1, self.decoder_layers-1):
            skip = result[self.encoder_layers-i-1]
            model = getattr(self, 'decoder'+str(i))
            out = model(torch.cat((out,skip),dim=1))
        
        skip = result[0]
        output_logits = self.decoder_out(torch.cat((out,skip),dim=1))
        output_attn = self.attn_out(torch.cat((out,skip),dim=1))
        return output_logits, output_attn

class MaskGenerator(nn.Module):
    '''doc
    '''
    def __init__(self, inc=3+1+21,onc=1, ngf=64, n_layers=6,max_nc=512, norm_type='bn', activation='LeakyReLU', use_spectral_norm=False):
        super(MaskGenerator, self).__init__()

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.inc = inc
        self.onc = onc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = n_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.inc
        onc = self.onc
        self.encoder_block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.encoder_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'encoder' + str(i), block)
        
        mult_prev = mult
        mult = min(2 ** (self.decoder_layers-2), max_nc//ngf)
        self.decoder_block0 = ResBlockUpNorm(ngf*mult_prev, ngf*mult,norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)

        for i in range(1, self.decoder_layers-1):
            mult_prev = mult * 2
            mult = min(2 ** (self.decoder_layers-i-2), max_nc//ngf)
            block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            setattr(self, 'decoder' + str(i), block)
        
        mult_prev = mult * 2
        self.decoder_out = ResBlockUpNorm(ngf*mult_prev, onc, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
        self.attn_out = ResBlockUpNorm(ngf*mult_prev, onc, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
        

    def forward(self, image1, mask1, pose2, pose_1=None):
        _,_,H,W = image1.shape
        if pose_1 is None:
            x_in = torch.cat((image1, mask1, pose2), dim=1) # (43, 256, 256)
        else:
            x_in = torch.cat((image1, mask1,pose_1, pose2), dim=1) # (43, 256, 256)
        out = self.encoder_block0(x_in)
        result = [out]
        for i in range(1, self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        
        out = self.decoder_block0(out)
        for i in range(1, self.decoder_layers-1):
            skip = result[self.encoder_layers-i-1]
            model = getattr(self, 'decoder'+str(i))
            out = model(torch.cat((out,skip),dim=1))
        
        skip = result[0]
        output_logits = self.decoder_out(torch.cat((out,skip),dim=1))
        output_attn = self.attn_out(torch.cat((out,skip),dim=1))
        return output_logits, output_attn




