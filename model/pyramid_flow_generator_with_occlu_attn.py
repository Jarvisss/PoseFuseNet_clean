from .blocks import *
from model.resample2d_package .resample2d import Resample2d
import torch
import torch.nn as nn 

'''
Input source feature and target feature at level L, L assume to be 4
'''
class FlowGenerator(nn.Module):
    """Flow Generator
    """
    def __init__(self, inc=3+21+22, n_layers=5, flow_layers=[2,3], ngf=32, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True):
        super(FlowGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.inc = inc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = 1 + len(flow_layers)
        self.flow_layers = flow_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)


    def _make_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.inc
        self.block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.encoder_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'encoder' + str(i), block)
        
        '''32, 64, 128, 256, 256
        128, 64, 32, 16, 8
        '''

        for i in range(self.decoder_layers):
            mult_prev = mult
            mult = min(2 ** (self.encoder_layers-i-2), max_nc//ngf) if i != self.encoder_layers-1 else 1
            block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            setattr(self, 'decoder' + str(i), block)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spectral_norm, False)
            setattr(self, 'jump'+str(i), jumpconv)

            if self.encoder_layers-i-1 in self.flow_layers:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'flow' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)

                flow_attn = nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'attn'+str(i), flow_attn)
        
        '''256, 128, 64
        16, 32, 64
        '''
        pass
    
    '''Get flow and mask at certain layer, [32,32,128] [64,64,64]
    '''
    def forward(self, image1, pose1, pose2, sim_map=None):
        if sim_map is None:
            x_in = torch.cat((image1, pose1, pose2), dim=1) # (43, 256, 256)
        else:
            x_in = torch.cat((image1, pose1, pose2, sim_map), dim=1) # (43+13, 256, 256)

        flow_fields=[]
        masks=[]
        attns=[]

        out = self.block0(x_in)
        result = [out]
        
        for i in range(1, self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layers-i-2])
            out = out+jump

            if self.encoder_layers-i-1 in self.flow_layers:
                model = getattr(self, 'flow'+str(i))
                flow_field = model(out)
                model = getattr(self, 'mask'+str(i))
                mask = model(out)
                model = getattr(self, 'attn'+str(i))
                attn = model(out)

                flow_fields.append(flow_field)
                masks.append(mask)
                attns.append(attn)

        return flow_fields, masks, attns



class ShapeNetFlowGenerator(nn.Module):
    """Flow Generator
    """
    def __init__(self, inc, struct_nc, n_layers=5, flow_layers=[2,3], ngf=32, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True):
        super(ShapeNetFlowGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.inc = inc
        self.structure_nc = struct_nc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = 1 + len(flow_layers)
        self.flow_layers = flow_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)


    def _make_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.inc
        self.block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.encoder_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'encoder' + str(i), block)
        
        '''32, 64, 128, 256, 256
        128, 64, 32, 16, 8
        '''
        self.cat = ResBlock2d(ngf*mult+self.structure_nc,ngf*mult,None, 3,1,self.norm_type, use_spectral_norm)
        for i in range(self.decoder_layers):
            mult_prev = mult
            mult = min(2 ** (self.encoder_layers-i-2), max_nc//ngf) if i != self.encoder_layers-1 else 1
            block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            setattr(self, 'decoder' + str(i), block)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spectral_norm, False)
            setattr(self, 'jump'+str(i), jumpconv)

            if self.encoder_layers-i-1 in self.flow_layers:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'flow' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)

                flow_attn = nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'attn'+str(i), flow_attn)
        
        '''256, 128, 64
        16, 32, 64
        '''
        pass
    
    '''Get flow and mask at certain layer, [32,32,128] [64,64,64]
    '''
    def forward(self, image1, pose1, pose2):
        x_in = image1

        flow_fields=[]
        masks=[]
        attns=[]

        out = self.block0(x_in)
        result = [out]
        
        for i in range(1, self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        
        # add this line to inject pose info
        out = self.encode_pose(pose1, pose2, out)    
        for i in range(self.decoder_layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layers-i-2])
            out = out+jump

            if self.encoder_layers-i-1 in self.flow_layers:
                model = getattr(self, 'flow'+str(i))
                flow_field = model(out)
                model = getattr(self, 'mask'+str(i))
                mask = model(out)
                model = getattr(self, 'attn'+str(i))
                attn = model(out)

                flow_fields.append(flow_field)
                masks.append(mask)
                attns.append(attn)

        return flow_fields, masks, attns

    '''
    encode pose info at feature level
    '''
    def encode_pose(self, pose1, pose2, out):
        B=pose1-pose2
        # print(B)
        _,_,w,h = out.size()
        B=B.repeat(1, 1, w, h)
        out = torch.cat((out,B), 1) 
        out = self.cat(out)  
        return out


class FlowGeneratorWithMask(nn.Module):
    '''
    generate flow with mask
    '''
    def __init__(self, image_nc=3, structure_nc=21, n_layers=5, flow_layers=[2,3], ngf=32, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True):
        super(FlowGeneratorWithMask, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.image_nc = image_nc
        self.structure_nc = structure_nc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = 1 + len(flow_layers)
        self.flow_layers = flow_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)


    def _make_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.image_nc + self.structure_nc * 2
        self.block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.encoder_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'encoder' + str(i), block)
        
        '''32, 64, 128, 256, 256
        128, 64, 32, 16, 8
        '''

        for i in range(self.decoder_layers):
            mult_prev = mult
            mult = min(2 ** (self.encoder_layers-i-2), max_nc//ngf) if i != self.encoder_layers-1 else 1
            block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            setattr(self, 'decoder' + str(i), block)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spectral_norm, False)
            setattr(self, 'jump'+str(i), jumpconv)

            if self.encoder_layers-i-1 in self.flow_layers:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'flow' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)

                flow_attn = nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'attn'+str(i), flow_attn)
        
        '''256, 128, 64
        16, 32, 64
        '''
        pass
    
    '''Get flow and mask at certain layer, [32,32,128] [64,64,64]
    '''
    def forward(self, image1, pose1, pose2, sim_map=None):
        if sim_map is None:
            x_in = torch.cat((image1, pose1, pose2), dim=1) # (43, 256, 256)
        else:
            x_in = torch.cat((image1, pose1, pose2, sim_map), dim=1) # (43+13, 256, 256)

        flow_fields=[]
        masks=[]
        attns=[]

        out = self.block0(x_in)
        result = [out]
        
        for i in range(1, self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layers-i-2])
            out = out+jump

            if self.encoder_layers-i-1 in self.flow_layers:
                model = getattr(self, 'flow'+str(i))
                flow_field = model(out)
                model = getattr(self, 'mask'+str(i))
                mask = model(out)
                model = getattr(self, 'attn'+str(i))
                attn = model(out)

                flow_fields.append(flow_field)
                masks.append(mask)
                attns.append(attn)

        return flow_fields, masks, attns

#### TODO
class PoseAttnFCNGenerator(nn.Module):
    '''
    pose attention generator
    '''
    def __init__(self, inc=21, n_layers=5, attn_layers=[2,3], ngf=32, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True):
        super(PoseAttnFCNGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.inc = inc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = 1 + len(attn_layers)
        self.attn_layers = attn_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self,ngf,max_nc,norm_layer,nonlinearity, use_spectral_norm):
        inc =self.inc
        # self.block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(inc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.encoder_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'encoder' + str(i), block)
        
        '''32, 64, 128, 256, 256
        128, 64, 32, 16, 8
        '''

        for i in range(self.decoder_layers):
            mult_prev = mult
            mult = min(2 ** (self.encoder_layers-i-2), max_nc//ngf) if i != self.encoder_layers-1 else 1
            block = ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
            # block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            setattr(self, 'decoder' + str(i), block)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spectral_norm, False)
            setattr(self, 'jump'+str(i), jumpconv)

            if self.encoder_layers-i-1 in self.attn_layers:
                flow_attn = nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'attn'+str(i), flow_attn)
        
        '''256, 128, 64
        16, 32, 64
        '''
        pass

    def forward(self, x_in):

        attns=[]

        out = self.block0(x_in)
        result = [out]
        
        for i in range(1, self.encoder_layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out)
            result.append(out) 
        for i in range(self.decoder_layers):
            model = getattr(self, 'decoder' + str(i))
            out = model(out)

            model = getattr(self, 'jump' + str(i))
            jump = model(result[self.encoder_layers-i-2])
            out = out+jump

            if self.encoder_layers-i-1 in self.attn_layers:
                model = getattr(self, 'attn'+str(i))
                attn = model(out)
                attns.append(attn)

        return attns


class AppearanceEncoder(nn.Module):
    '''
    appearance encoder
    '''
    def __init__(self, n_layers=3, inc=3, ngf=64, max_nc=256,norm_type='bn',activation='LeakyReLU',  use_spectral_norm=True):
        super(AppearanceEncoder, self).__init__()
        self.n_layers = n_layers
        self.inc = inc
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self._make_layers(ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm)
        
    def _make_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        '''
        Encoder:
        layer1 : (3,256,256) -> (64,128,128)
        layer2 : (64,128,128) -> (128,64,64)
        layer3 : (128,64,64) -> (256,32,32)
        
        '''
        self.block0 = ResBlockEncoder(self.inc, ngf, ngf,  norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, int(self.n_layers)):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult,  norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder'+str(i), block)

    def forward(self, x):
        out = self.block0(x)
        feature_list = [out]
        for i in range(1, self.n_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
            feature_list.append(out)

        feature_list = list(reversed(feature_list))
        return feature_list

class AppearanceDecoder(nn.Module):
    '''
    Decode part of the generator
    '''
    def __init__(self, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True, align_corners=True, use_resample=False):
        super(AppearanceDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        '''
        Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            if i>0 and self.n_decode_layers-i in self.flow_layers: # [2,3]
                mult_prev = mult_prev*2
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None,None, 3,1,self.norm_type, use_spectral_norm),
                ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, source_features, flows, masks):
        counter = 0
        K = len(source_features)
        # assert(K==2)
        for i in range(self.n_decode_layers):
            model = getattr(self, 'decoder' + str(i))

            if self.n_decode_layers-i in self.flow_layers:
                merge = None
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    # print(out_k.shape)
                    # print(masks[k][counter].shape)
                    out_k = out_k * masks[k][counter]
                    
                    if merge is None:
                        merge = out_k
                    else:
                        merge += out_k
                counter += 1
                # print('merge:',merge.shape)
                if counter == 1: # the first input to decoder is not concat
                    out = merge
                else: # the following input to decoder is concat to the network generated feature
                    out = torch.cat((out, merge),dim=1)
                    # print('out:',out.shape)
            
            out = model(out)
        out_image = self.outconv(out)
        return out_image

class PoseSOAdaINDecoder(nn.Module):
    '''
    Pose Self occlusion decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,n_bottle_neck_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(PoseSOAdaINDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.n_bottle_neck_layers = n_bottle_neck_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc
        self.max_nc = max_nc
        self.P_LEN = 2*max_nc*2*self.n_bottle_neck_layers 
        self.p = nn.Parameter(torch.rand(self.P_LEN,self.max_nc).normal_(0.0,0.02))

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)   

        bottle_neck_nc = ngf*mult
        for i in range(self.n_bottle_neck_layers):
            block = ResBlock(bottle_neck_nc)
            setattr(self, 'btneck' + str(i), block)   

        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers - i in self.flow_layers:
                direct_attn = nn.Sequential(
                    nn.Conv2d(ngf*mult_prev, 1, kernel_size=3,stride=1,padding=1,bias=True),
                    nn.Sigmoid()
                )
                setattr(self, 'so'+str(i), direct_attn )
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None, 3,1,self.norm_type, use_spectral_norm),
                # ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            
                
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    # P: B, P_LEN, max_nc
    # e: B, max_nc, 1
    def forward(self, g_y, source_features, flows, attns, e):
        p = self.p.unsqueeze(0)
        p = p.expand(e.shape[0],self.P_LEN,self.max_nc) #B, p_len, max_nc
        e_psi = torch.bmm(p, e) #B, p_len, 1

        out = self.block0(g_y)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
        
        # finally got [256,32,32] output
        for i in range(self.n_bottle_neck_layers):
            model = getattr(self, 'btneck'+str(i))
            out = model(out, e_psi[:,i*self.max_nc*4:(i+1)*self.max_nc*4,:])

        counter = 0
        K = len(source_features)
        normed_attns = []
        pose_out = []
        for i in range(self.n_decode_layers):
            model = getattr(self, 'decoder' + str(i))

            if self.n_decode_layers-i in self.flow_layers:
                direct_attn = getattr(self, 'so'+str(i))
                attn_p = direct_attn(out) # attention of direct generated part
                all_attns = attns[0][counter]
                for k in range(1,K):
                    all_attns = torch.cat((all_attns,attns[k][counter]),dim=1)
                all_attns = nn.Softmax(dim=1)(all_attns)

                fw = None
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    if fw is None:
                        fw = out_k * all_attns[:,k:k+1,...]
                    else:
                        fw += out_k * all_attns[:,k:k+1,...]

                counter += 1
                pose_out += [attn_p * out]
                out = attn_p * out + (1-attn_p)*fw # warp + direct feature
                normed_attns += [torch.cat((attn_p,all_attns),dim=1)]
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        out_image = self.outconv(out)
        return out_image, normed_attns, pose_out


class PoseSODecoder(nn.Module):
    '''
    Pose Self occlusion decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(PoseSODecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)         
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers - i in self.flow_layers:
                direct_attn = nn.Conv2d(ngf*mult_prev, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'so'+str(i), direct_attn )
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None, 3,1,self.norm_type, use_spectral_norm),
                # ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            
                
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows, attns):
        out = self.block0(g_y)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
        
        # finally got [256,32,32] output

        counter = 0
        K = len(source_features)
        normed_attns = []
        pose_out = []
        for i in range(self.n_decode_layers):
            model = getattr(self, 'decoder' + str(i))

            if self.n_decode_layers-i in self.flow_layers:
                direct_attn = getattr(self, 'so'+str(i))
                attn_p = direct_attn(out) # attention of direct generated part
                all_attns = attn_p
                for k in range(K):
                    all_attns = torch.cat((all_attns,attns[k][counter]),dim=1)
                all_attns = nn.Softmax(dim=1)(all_attns)

                merge = all_attns[:,0:1,...] * out
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    merge += out_k * all_attns[:,k+1:k+2,...]
                counter += 1
                out = merge # warp + direct feature
                normed_attns += [all_attns]
                pose_out += [all_attns[:,0:1,...] * out]
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        out_image = self.outconv(out)
        return out_image, normed_attns, pose_out


class PoseAwareAppearanceDecoder(nn.Module):
    '''Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(PoseAwareAppearanceDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)         
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows, masks, attns):
        out = self.block0(g_y)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
        
        # finally got [256,32,32] output

        counter = 0
        K = len(source_features)
        for i in range(self.n_decode_layers):
            model = getattr(self, 'decoder' + str(i))

            if self.n_decode_layers-i in self.flow_layers:
                
                merge = None
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    
                    # print(out_k.shape)
                    # print(masks[k][counter].shape)
                    out_k = attns[k][counter] * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                    # out_k = 1/K * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                    
                    if merge is None:
                        merge = out_k
                    else:
                        merge += out_k
                counter += 1
                out = merge # warp + direct feature
                
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        
        out_image = self.outconv(out)
        return out_image

    # def forward_hook_function(self, g_y, source_features, flows, masks, attns):
    #     out = self.block0(g_y)
    #     for i in range(1, self.n_decode_layers):
    #         model = getattr(self, 'encoder'+str(i))
    #         out = model(out)
        
    #     hook_target = []
    #     for i in range(self.n_decode_layers):
    #         model = getattr(self, 'decoder'+str(i))
    #         if self.n_decode_layers-i in self.flow_layers:
                
    #             merge = None
    #             for k in range(K):
    #                 if self.use_resample:
    #                     out_k = self.resample(source_features[k][i], flows[k][counter])
    #                 else:
    #                     out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    
    #                 # print(out_k.shape)
    #                 # print(masks[k][counter].shape)
    #                 out_k = attns[k][counter] * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                    
    #                 if merge is None:
    #                     merge = out_k
    #                 else:
    #                     merge += out_k
    #             counter += 1
    #             out = merge
            
    #         out = model(out)


class PoseAwareAppearanceDecoderShapeNet(nn.Module):
    '''Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(PoseAwareAppearanceDecoderShapeNet, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = ResBlockDecoder(self.structure_nc, ngf, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        self.block1 = ResBlockDecoder(ngf, mult*ngf, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)         
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows, masks, attns):
        g_y = g_y.repeat(1, 1, 8, 8)
        out = self.block0(g_y)
        out = self.block1(out) # 32*32
        K = len(attns)
        L = len(attns[0])
        
        # finally got [256,32,32] output

        counter = 0
        K = len(source_features)
        for i in range(self.n_decode_layers):

            if self.n_decode_layers-i in self.flow_layers:
                
                merge = None
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    
                    # print(out_k.shape)
                    # print(masks[k][counter].shape)
                    out_k = attns[k][counter] * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                    # out_k = 1/K * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                    
                    if merge is None:
                        merge = out_k
                    else:
                        merge += out_k
                counter += 1
                out = merge # warp + direct feature
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        out_image = self.outconv(out)
        return out_image


class PoseAwareSelfAttentionDecoder(nn.Module):
    '''Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(PoseAwareSelfAttentionDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)         
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            if self.n_decode_layers-i in self.flow_layers: # [2,3]
                mult_prev = mult_prev*2
                attn = nn.Conv2d(ngf*mult_prev, 1, 1,1,0) # [1×1 conv to make dimention down]
                setattr(self, 'attn'+str(i), attn)

            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
                
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

                    

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows):
        out = self.block0(g_y)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
        
        # finally got [256,32,32] output

        counter = 0
        all_attns = []
        K = len(source_features)
        for i in range(self.n_decode_layers):

            if self.n_decode_layers-i in self.flow_layers:
                
                merge = None
                attns = None
                out_ks = []
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    model = getattr(self, 'attn'+str(i))
                    attn_k = model(torch.cat((out, out_k),dim=1))
                    attns = attn_k if attns is None else torch.cat((attns, attn_k),dim=1)
                    out_ks += [out_k]
                attns_normed = nn.Softmax(dim=1)(attns)
                all_attns += [attns_normed]
                for k in range(K):
                    if merge is not None:
                        merge += out_ks[k] * attns_normed[:,k:k+1,...]
                    else:
                        merge = out_ks[k] * attns_normed[:,k:k+1,...]
                counter += 1
                out = torch.cat((merge,out),dim=1) # warp + direct feature
            model = getattr(self, 'decoder' + str(i))
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        out_image = self.outconv(out)
        return out_image, all_attns

class PoseAwareResidualFlowDecoder(nn.Module):
    '''Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_res_blocks=1,n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False,use_res_attn=False):
        super(PoseAwareResidualFlowDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.n_res_blocks = n_res_blocks
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_res_attn = use_res_attn
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)         
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers-i in self.flow_layers: # [2,3]
                if self.n_res_blocks==1:
                    res_flow = nn.Conv2d(ngf*mult_prev*2, 2, 3,1,1) # [3×3 conv to make dimention down]
                    res_attn = nn.Conv2d(ngf*mult_prev*2, 1, 3,1,1)
                    setattr(self, 'res_flow'+str(i), res_flow)
                    if self.use_res_attn:
                        setattr(self, 'res_attn'+str(i), res_attn)
                else:
                    for n in range(self.n_res_blocks):
                        res_flow = nn.Conv2d(ngf*mult_prev*2, 2, 3,1,1) # [3×3 conv to make dimention down]
                        res_attn = nn.Conv2d(ngf*mult_prev*2, 1, 3,1,1)
                        setattr(self, 'res_flow'+str(i)+'_'+str(n), res_flow)
                        if self.use_res_attn:
                            setattr(self, 'res_attn'+str(i)+'_'+str(n), res_attn)
                # res_attn = nn.Conv2d(ngf*mult_prev*2, 1, 3,1,1) # [1×1 conv to make dimention down]
                # setattr(self, 'res_attn'+str(i), res_attn)

            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

            
        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows, masks, attns, debug=False):
        out = self.block0(g_y)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
        
        K = len(attns)
        L = len(attns[0])
        # finally got [256,32,32] output
        if debug:
            for k in range(0, len(attns)):
                for i in range(0, len(attns[0])):
                    print('before gd', i, k)
                    print(torch.max(attns[k][i]))
        counter = 0
        out_flows = [[0 for _ in range(L)] for _ in range(K)]
        out_attns = [[0 for _ in range(L)] for _ in range(K)]
        decode_feats = [0 for _ in range(L)]
        merged_feats = [0 for _ in range(L)]
        K = len(source_features)
        for i in range(self.n_decode_layers):
            if self.n_decode_layers-i in self.flow_layers:
                

                out_ks = []
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    out_ks.append(out_k)
                
                if self.use_res_attn:
                    res_attns = []
                    for k in range(K):
                        model = getattr(self, 'res_attn'+str(i))
                        residual_attn = model(torch.cat((out, out_ks[k]),dim=1))
                        res_attns.append(residual_attn)
                    residual_attns = torch.cat(res_attns, dim=1)
                    residual_attns = nn.Softmax(dim=1)(residual_attns) - 1/K
                    # softmax - 1/K, the output will be in range [-1/K, (K-1)/K]
                    # 理想情况下，每个残差注意力都接近0，在0轴附近浮动，可视化结果显示
                
                for n in range(self.n_res_blocks):
                    merge = None
                    for k in range(K):
                        if self.n_res_blocks==1:
                            model = getattr(self, 'res_flow'+str(i))
                        else:
                            model = getattr(self, 'res_flow'+str(i)+'_'+str(n))

                        residual_flow = model(torch.cat((out, out_ks[k]),dim=1))
                        
                        if n==0:
                            out_flows[k][counter] = flows[k][counter] + residual_flow
                        else:
                            out_flows[k][counter] += residual_flow

                        # out_k = warp_flow(out_ks[k], residual_flow, align_corners=self.align_corners)
                        # this is a residual flow actually 
                        out_k = warp_flow(source_features[k][i], out_flows[k][counter], align_corners=self.align_corners)
                        out_ks[k] = out_k
                        # print(out_k.shape)
                        # print(masks[k][counter].shape)
                        if self.use_res_attn:
                            fused_out_k = (attns[k][counter]+ residual_attns[:,k:k+1,...]) * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                            out_attns[k][counter] = attns[k][counter]+ residual_attns[:,k:k+1,...]
                        else:
                            fused_out_k = attns[k][counter] * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                            out_attns[k][counter] = attns[k][counter]
                        # out_k = 1/K * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                        
                        if merge is None:
                            merge = fused_out_k
                        else:
                            merge += fused_out_k
                    
                    out = merge # warp + direct feature
                counter += 1
            if self.n_decode_layers-i in self.flow_layers:
                merged_feats[counter-1] = out
            model = getattr(self, 'decoder' + str(i))    
            out = model(out)
            if self.n_decode_layers-i in self.flow_layers:
                decode_feats[counter-1] = out
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        # print(out.shape)
        out_image = self.outconv(out)
        # print(out_image.shape)

        if debug:
            for k in range(0, len(attns)):
                for i in range(0, len(attns[0])):
                    print('after gd', i, k)
                    print(torch.max(attns[k][i]))
        return out_image, out_flows, out_attns, decode_feats, merged_feats
        # return out_image, out_flows, out_attns, decode_feats

class PoseAwareResidualFlowDecoderShapenet(nn.Module):
    '''Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False,use_res_attn=False):
        super(PoseAwareResidualFlowDecoderShapenet, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_res_attn = use_res_attn
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        self.block0 = ResBlockDecoder(self.structure_nc, ngf, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        self.block1 = ResBlockDecoder(ngf, mult*ngf, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
        
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers-i in self.flow_layers: # [2,3]
                res_flow = nn.Conv2d(ngf*mult_prev*2, 2, 3,1,1) # [3×3 conv to make dimention down]
                res_attn = nn.Conv2d(ngf*mult_prev*2, 1, 3,1,1)
                setattr(self, 'res_flow'+str(i), res_flow)
                if self.use_res_attn:
                    setattr(self, 'res_attn'+str(i), res_attn)
                # res_attn = nn.Conv2d(ngf*mult_prev*2, 1, 3,1,1) # [1×1 conv to make dimention down]
                # setattr(self, 'res_attn'+str(i), res_attn)

            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

            
        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows, masks, attns, debug=False):
        g_y = g_y.repeat(1, 1, 8, 8)
        out = self.block0(g_y)
        out = self.block1(out) # 32*32
        
        K = len(attns)
        L = len(attns[0])
        # finally got [256,32,32] output
        if debug:
            for k in range(0, len(attns)):
                for i in range(0, len(attns[0])):
                    print('before gd', i, k)
                    print(torch.max(attns[k][i]))
        counter = 0
        out_flows = [[0 for _ in range(L)] for _ in range(K)]
        out_attns = [[0 for _ in range(L)] for _ in range(K)]
        K = len(source_features)
        for i in range(self.n_decode_layers):
            if self.n_decode_layers-i in self.flow_layers:
                merge = None

                out_ks = []
                for k in range(K):
                    if self.use_resample:
                        out_k = self.resample(source_features[k][i], flows[k][counter])
                    else:
                        out_k = warp_flow(source_features[k][i], flows[k][counter], align_corners=self.align_corners)
                    out_ks.append(out_k)
                
                if self.use_res_attn:
                    res_attns = []
                    for k in range(K):
                        model = getattr(self, 'res_attn'+str(i))
                        residual_attn = model(torch.cat((out, out_ks[k]),dim=1))
                        res_attns.append(residual_attn)
                    residual_attns = torch.cat(res_attns, dim=1)
                    residual_attns = nn.Softmax(dim=1)(residual_attns) - 1/K
                    # softmax - 1/K, the output will be in range [-1/K, (K-1)/K]
                    # 理想情况下，每个残差注意力都接近0，在0轴附近浮动，可视化结果显示
                    
                for k in range(K):
                    model = getattr(self, 'res_flow'+str(i))
                    residual_flow = model(torch.cat((out, out_ks[k]),dim=1))
                    
                    out_flows[k][counter] = flows[k][counter] + residual_flow
                    out_k = warp_flow(out_ks[k], residual_flow, align_corners=self.align_corners)
                    # print(out_k.shape)
                    # print(masks[k][counter].shape)
                    if self.use_res_attn:
                        out_k = (attns[k][counter]+ residual_attns[:,k:k+1,...]) * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                        out_attns[k][counter] = attns[k][counter]+ residual_attns[:,k:k+1,...]
                    else:
                        out_k = attns[k][counter] * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                        out_attns[k][counter] = attns[k][counter]
                    # out_k = 1/K * (out_k * masks[k][counter] + out * (1-masks[k][counter])) # a_k(w_k*m_k + p*(1-m_k))
                    
                    if merge is None:
                        merge = out_k
                    else:
                        merge += out_k
                counter += 1
                out = merge # warp + direct feature
            model = getattr(self, 'decoder' + str(i))    
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        out_image = self.outconv(out)

        if debug:
            for k in range(0, len(attns)):
                for i in range(0, len(attns[0])):
                    print('after gd', i, k)
                    print(torch.max(attns[k][i]))
        return out_image, out_flows, out_attns


class PoseSaSoDecoder(nn.Module):
    '''
    Pose self-attn self-occl decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(PoseSaSoDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        self._make_layers(ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Encoder 
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'encoder' + str(i), block)         
        '''Decoder:
        layer1 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        layer2 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        layer3 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers - i in self.flow_layers:
                self_occlusion = nn.Conv2d(ngf*mult_prev, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'so'+str(i), self_occlusion)
                multi_attention = nn.Conv2d(ngf*mult_prev*2, 1, kernel_size=3, stride=1, padding=1, bias=True)
                setattr(self, 'ma'+str(i), multi_attention)
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                # ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            
                
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, g_y, source_features, flows):
        out = self.block0(g_y)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'encoder'+str(i))
            out = model(out)
        
        # finally got [256,32,32] output

        counter = 0
        K = len(source_features)
        normed_attns = []
        pose_out = []
        for i in range(self.n_decode_layers):
            model = getattr(self, 'decoder' + str(i))

            if self.n_decode_layers-i in self.flow_layers:
                self_occ_layer = getattr(self, 'so'+str(i))
                attn_p = self_occ_layer(out) # target自己生成一个attention
                multi_attns = [attn_p]
                multi_attn_layer = getattr(self, 'ma'+str(i))
                for k in range(K):
                    out_k = warp_flow(source_features[k][i], flows[k][i], align_corners=self.align_corners)
                    attn_k = multi_attn_layer(torch.cat((out, out_k), dim=1)) # 每个warped source和target之间都有一个attention
                    multi_attns += [attn_k]
                
                all_attns = nn.Softmax(dim=1)( torch.cat((multi_attns),dim=1) )

                merge = all_attns[:,0:1,...] * out # target 自己的部分

                # 加上各个source的部分
                for k in range(K):
                    out_k = warp_flow(source_features[k][i], flows[k][i], align_corners=self.align_corners)
                    merge += out_k * all_attns[:,k+1:k+2,...]
                
                counter += 1
                out = merge # warp + direct feature
                normed_attns += [all_attns]
                pose_out += [all_attns[:,0:1,...] * out]
            out = model(out)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    model = getattr(self, 'sa'+str(i),)
                    out = model(out)
        out_image = self.outconv(out)
        return out_image, normed_attns, pose_out

class UnifyGenerator(nn.Module):
    '''Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_encode_layers=5,n_decode_layers=3,output_nc=3, flow_layers=[2,3], flow_ngf=32, target_ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(UnifyGenerator, self).__init__()
        self.n_encode_layers = n_encode_layers
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        
        self._make_flow_layers(flow_ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm)
        self._make_target_layers(target_ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)
        # self.flow_generator = FlowGenerator(inc=3+structure_nc*2, n_layers=5, flow_layers=[2,3], ngf=32, max_nc=256, norm_type,activation, use_spectral_norm)

    def _make_flow_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.structure_nc*2 + 3
        ''' Flow Encoder:
            32 *128*128,
            64 *64*64,
            128 *32*32,
            256 *16*16,
            256 *8*8
        '''
        self.flow_block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_encode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'flow_encoder' + str(i), block)
        
        ''' Flow Decoder
            256 *16*16,
            128 *32*32, -> flow (2,32,32) -> mask (1,32,32)
            64 *64*64,  -> flow (2,64,64) -> mask (1,64,64)
        '''        
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_encode_layers-i-2), max_nc//ngf) if i != self.n_encode_layers-1 else 1
            if self.n_encode_layers-i in self.flow_layers:
                block = ResBlockUpNorm(ngf*mult_prev+4, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            else:
                block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)

            setattr(self, 'flow_decoder' + str(i), block)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spectral_norm, False)
            setattr(self, 'flow_jump'+str(i), jumpconv)

            if self.n_encode_layers-i-1 in self.flow_layers:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'flow_flow' + str(i), flow_out)

                flow_mask = nn.Sequential(nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'flow_mask' + str(i), flow_mask)

                flow_attn = nn.Conv2d(ngf*mult, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'flow_attn'+str(i), flow_attn)
        

        pass

    def _make_target_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Target Encoder
            64 *128*128,
            128 *64*64,
            256 *32*32,
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.target_block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'target_encoder' + str(i), block)         
        '''Decoder:
        target_decoder0 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        target_decoder1 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        target_decoder2 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers-i in self.flow_layers: # [2,3]
                res_flow = nn.Conv2d(ngf*mult_prev*2, 2, 3,1,1) # [1×1 conv to make dimention down]
                setattr(self, 'res_flow'+str(i), res_flow)
            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'target_decoder'+str(i), up)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, source_imgs, source_poses, source_features,  target_pose, sim_maps=None):
        K = len(source_features)
        assert(len(source_imgs)==K)
        assert(len(source_poses)==K)

        target_out = self.target_block0(target_pose)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'target_encoder'+str(i))
            target_out = model(target_out)

        flow_outs = [0]*K
        flow_encode_features = [0]*K
        for k in range(K):
            if sim_maps is None:
                x_in = torch.cat((source_imgs[k], source_poses[k], target_pose), dim=1) # (43, 256, 256)
            else:
                x_in = torch.cat((source_imgs[k], source_poses[k], target_pose, sim_maps[k]), dim=1) # (43+13, 256, 256)

            out = self.flow_block0(x_in)
            result = [out]
            for i in range(1, self.n_encode_layers):
                model = getattr(self, 'flow_encoder' + str(i))
                out = model(out)
                result.append(out)

            # get flow decoder 0 result
            model = getattr(self, 'flow_decoder' + str(0))
            out = model(out)
            model = getattr(self, 'flow_jump' + str(0))
            jump = model(result[self.n_encode_layers-2])
            out = out+jump
            flow_outs[k] = out
            flow_encode_features[k] = result
        
        # now that we have k outputs of flow generator at decoder0 with resolution 16*16
        # and k outputs of flow encoders each has 5 resolutions []

        # pre-define the intermediate outputs
        layers = len(self.flow_layers)
        flows=[[0]*layers for k in range(K)]
        residual_flows=[[0]*layers for k in range(K)]
        masks=[[0]*layers for k in range(K)]
        attns=[[0]*layers for k in range(K)]
        outs=[[0]*layers for k in range(K)]
        for i in range(self.n_decode_layers-1): # 0,1
            flow_decoder = getattr(self, 'flow_decoder' + str(i+1))
            target_merge = None
            for k in range(K):
                # if the last layer has flow output, then it should have residual flow from target stream
                if self.n_encode_layers-i-1 in self.flow_layers: # 2,3
                    out = flow_decoder(torch.cat((outs[k][i-1], (flows[k][i-1]+residual_flows[k][i-1])*2, masks[k][i-1], attns[k][i-1]),dim=1)) # flow ×2 upsampling 
                else:
                    out = flow_decoder(flow_outs[k])
                    
                model = getattr(self, 'flow_jump' + str(i+1))
                jump = model(flow_encode_features[k][self.n_encode_layers-i-1-2])
                out = out+jump
                outs[k][i] = out
                if self.n_decode_layers-i in self.flow_layers:
                    model = getattr(self, 'flow_flow'+str(i+1))
                    flow = model(outs[k][i])
                    model = getattr(self, 'flow_mask'+str(i+1))
                    mask = model(outs[k][i])
                    model = getattr(self, 'flow_attn'+str(i+1))
                    attn = model(outs[k][i])
                    flows[k][i] = flow
                    masks[k][i] = mask
                    attns[k][i] = attn

            # print(torch.mean(attns[0][i]))
            # print(torch.mean(attns[1][i]))
            # softmax normalize the attentions
            a = torch.cat([attns[k][i] for k in range(K)], dim=1)
            a = nn.Softmax(dim=1)(a)
            for k in range(K):
                attns[k][i] = a[:,k:k+1,...]
            # print(torch.mean(attns[0][i]))
            # print(torch.mean(attns[1][i]))
            for k in range(K):
                # if this layer has flow output
                if self.n_decode_layers-i in self.flow_layers:
                    # merge the flowed sources with the target
                    out_k = warp_flow(source_features[k][i], flows[k][i], align_corners=self.align_corners) # 1. warp the source by the initial flow
                    model = getattr(self, 'res_flow'+str(i)) 
                    residual_flow = model(torch.cat((target_out, out_k),dim=1))   # 2. get the residual flow by '''flow_res = f(source_w, target)'''
                    # residual_flow = residual_flow * 0
                    out_k = warp_flow(source_features[k][i], residual_flow + flows[k][i], align_corners=self.align_corners) # 3. warp the source again by the residual flow
                    out_k = attns[k][i] * (out_k * masks[k][i] + target_out * (1-masks[k][i])) # 4. recalculate the masked source by the new one 
                    # a_k(w_k*m_k + p*(1-m_k))
                    if target_merge is None:
                        target_merge = out_k
                    else:
                        target_merge += out_k
                    
                    residual_flows[k][i]=residual_flow

                
                    
            target_out = target_merge # set the target merge as the output of the current layer
            model = getattr(self, 'target_decoder'+str(i))
            target_out = model(target_out)
        
        # finish the last layer of decoder
        for i in range(self.n_decode_layers-1, self.n_decode_layers):
            model = getattr(self, 'target_decoder'+str(i))
            target_out = model(target_out)

        out_image = self.outconv(target_out)
        return out_image, flows,residual_flows, attns, masks

class PoseGenerator(nn.Module):
    def __init__(self,  image_nc=3, structure_nc=21, output_nc=3, ngf=64, max_nc=256, layers=3, num_blocks=2, 
                norm='batch', activation='ReLU', attn_layer=[2,3], extractor_kz={'1':5,'2':5}, use_spect=True, align_corners=True):  
        super(PoseGenerator, self).__init__()
        self.source = AppearanceEncoder(layers, image_nc, ngf, max_nc,norm, activation, use_spect)
        self.target = AppearanceDecoder(layers,output_nc,attn_layer, ngf, max_nc, norm, activation, use_spect, align_corners)
        self.flow_net = FlowGenerator(image_nc, structure_nc,n_layers=5,flow_layers=attn_layer, ngf=32, max_nc=256, norm_type=norm, activation=activation, use_spectral_norm=use_spect)       

    def forward(self, source, source_B, target_B):
        feature_list = self.source(source)
        flow_fields, masks = self.flow_net(source, source_B, target_B)
        image_gen = self.target(feature_list, flow_fields, masks)

        return image_gen, flow_fields, masks  

    # def forward_hook_function(self, source_feature, flows, masks):
    #     hook_target=[]
    #     hook_source=[]      
    #     hook_mask=[]      
    #     out = self.block0(target_B)
    #     for i in range(self.layers-1):
    #         model = getattr(self, 'encoder' + str(i))
    #         out = model(out) 

    #     counter=0
    #     for i in range(self.layers):
    #         if self.layers-i in self.attn_layer:
    #             model = getattr(self, 'attn' + str(i))

    #             attn_param, out_attn = model.hook_attn_param(source_feature[i], out, flow_fields[counter])        
    #             out = out*(1-masks[counter]) + out_attn*masks[counter]

    #             hook_target.append(out)
    #             hook_source.append(source_feature[i])
    #             hook_attn.append(attn_param)
    #             hook_mask.append(masks[counter])
    #             counter += 1

    #         model = getattr(self, 'decoder' + str(i))
    #         out = model(out)

    #     out_image = self.outconv(out)
    #     return hook_target, hook_source, hook_attn, hook_mask 

class UnifyGenerator2(nn.Module):
    ''' generate attentions and occlusions in target stream
    '''
    def __init__(self, structure_nc=21, n_encode_layers=5,n_decode_layers=3,output_nc=3, flow_layers=[2,3], flow_ngf=32, target_ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True,use_self_attention=False, align_corners=True, use_resample=False):
        super(UnifyGenerator, self).__init__()
        self.n_encode_layers = n_encode_layers
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
        self.use_self_attention = use_self_attention
        self.structure_nc = structure_nc

        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        
        self._make_flow_layers(flow_ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm)
        self._make_target_layers(target_ngf, max_nc, output_nc, norm_layer, nonlinearity, use_spectral_norm)
        # self.flow_generator = FlowGenerator(inc=3+structure_nc*2, n_layers=5, flow_layers=[2,3], ngf=32, max_nc=256, norm_type,activation, use_spectral_norm)

    def _make_flow_layers(self, ngf, max_nc, norm_layer, nonlinearity, use_spectral_norm):
        inc = self.structure_nc*2 + 3
        ''' Flow Encoder:
            32 *128*128,
            64 *64*64,
            128 *32*32,
            256 *16*16,
            256 *8*8
        '''
        self.flow_block0 = ResBlockEncoder(inc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_encode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            block = ResBlockEncoder(ngf*mult_prev, ngf*mult, ngf*mult, norm_layer, nonlinearity, use_spectral_norm)    
            setattr(self, 'flow_encoder' + str(i), block)
        
        ''' Flow Decoder
            256 *16*16,
            128 *32*32, -> flow (2,32,32) -> mask (1,32,32)
            64 *64*64,  -> flow (2,64,64) -> mask (1,64,64)
        '''        
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_encode_layers-i-2), max_nc//ngf) if i != self.n_encode_layers-1 else 1
            if self.n_encode_layers-i in self.flow_layers:
                block = ResBlockUpNorm(ngf*mult_prev+4, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            else:
                block = ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)

            setattr(self, 'flow_decoder' + str(i), block)

            jumpconv = Jump(ngf*mult, ngf*mult, 3, None, nonlinearity, use_spectral_norm, False)
            setattr(self, 'flow_jump'+str(i), jumpconv)

            if self.n_encode_layers-i-1 in self.flow_layers:
                flow_out = nn.Conv2d(ngf*mult, 2, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'flow_flow' + str(i), flow_out)

        pass

    def _make_target_layers(self, ngf, max_nc,output_nc, norm_layer, nonlinearity, use_spectral_norm):
        ''' Target Encoder
            64 *128*128,
            128 *64*64,
            256 *32*32,
        '''
        # self.block0 = ResBlockEncoder(self.structure_nc, ngf, ngf, norm_layer, nonlinearity, use_spectral_norm)
        self.target_block0 = EncoderBlock(self.structure_nc, ngf, norm_layer, nonlinearity, use_spectral_norm)
        mult = 1
        for i in range(1, self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** i, max_nc//ngf)
            # block = ResBlockEncoder(ngf*mult_prev, ngf*mult,ngf*mult, norm_layer,
            #                      nonlinearity, use_spectral_norm)
            block = EncoderBlock(ngf*mult_prev, ngf*mult, norm_layer,
                                 nonlinearity, use_spectral_norm)
            setattr(self, 'target_encoder' + str(i), block)         
        '''Decoder:
        target_decoder0 : (256,32,32) w (2,32,32) -> (256,32,32) + (256,32,32) -> (128,64,64)
        target_decoder1 : (128,64,64) w (2,64,64) -> (128,64,64) + (128,64,64) -> (64,128,128)
        target_decoder2 : (64,128,128) -> (64,256,256)
        conv  : (64,256,256) -> (3,256,256)
        '''
        mult = min(2 ** (self.n_decode_layers-1), max_nc//ngf)
        for i in range(self.n_decode_layers):
            mult_prev = mult
            mult = min(2 ** (self.n_decode_layers-i-2), max_nc//ngf) if i != self.n_decode_layers-1 else 1
            # if self.n_decode_layers-i in self.flow_layers: # [2,3]
            #     mult_prev = mult_prev*2
            if self.n_decode_layers-i in self.flow_layers: # [2,3]
                res_flow = nn.Conv2d(ngf*mult_prev*2, 2, 3,1,1) # [1×1 conv to make dimention down]
                setattr(self, 'res_flow'+str(i), res_flow)
                
                attn = nn.Conv2d(ngf*mult_prev*2, 1, kernel_size=3,stride=1,padding=1,bias=True)
                setattr(self, 'attn'+str(i), attn)

                mask = nn.Sequential(nn.Conv2d(ngf*mult_prev, 1, kernel_size=3,stride=1,padding=1,bias=True),
                                          nn.Sigmoid())
                setattr(self, 'mask' + str(i), flow_mask)

            up = nn.Sequential(
                ResBlock2d(ngf*mult_prev,None, None,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'target_decoder'+str(i), up)
            if self.use_self_attention:
                if self.n_decode_layers-i == self.flow_layers[1]:
                    selfattn = SelfAttention(ngf*mult)
                    setattr(self, 'sa'+str(i), selfattn)

        
        self.outconv = Output(ngf, output_nc, 3, None, nonlinearity, use_spectral_norm, False)
        if self.use_resample:
            self.resample = Resample2d(4, 1, sigma=2)


    # get 3 channel image
    # source_features is reversed 
    # flows: K * [2,32,32][2,64,64]
    # masks: K * [1,32,32][1,64,64]
    # attns: K * [1,32,32][1,64,64]
    # source_features: K * [256,32,32][128,64,64]
    def forward(self, source_imgs, source_poses, source_features,  target_pose, sim_maps=None):
        K = len(source_features)
        assert(len(source_imgs)==K)
        assert(len(source_poses)==K)

        target_out = self.target_block0(target_pose)
        for i in range(1, self.n_decode_layers):
            model = getattr(self, 'target_encoder'+str(i))
            target_out = model(target_out)

        flow_outs = [0]*K
        flow_encode_features = [0]*K
        for k in range(K):
            if sim_maps is None:
                x_in = torch.cat((source_imgs[k], source_poses[k], target_pose), dim=1) # (43, 256, 256)
            else:
                x_in = torch.cat((source_imgs[k], source_poses[k], target_pose, sim_maps[k]), dim=1) # (43+13, 256, 256)

            out = self.flow_block0(x_in)
            result = [out]
            for i in range(1, self.n_encode_layers):
                model = getattr(self, 'flow_encoder' + str(i))
                out = model(out)
                result.append(out)

            # get flow decoder 0 result
            model = getattr(self, 'flow_decoder' + str(0))
            out = model(out)
            model = getattr(self, 'flow_jump' + str(0))
            jump = model(result[self.n_encode_layers-2])
            out = out+jump
            flow_outs[k] = out
            flow_encode_features[k] = result
        
        # now that we have k outputs of flow generator at decoder0 with resolution 16*16
        # and k outputs of flow encoders each has 5 resolutions []

        # pre-define the intermediate outputs
        layers = len(self.flow_layers)
        flows=[[0]*layers for k in range(K)]
        
        outs=[[0]*layers for k in range(K)]
        for i in range(self.n_decode_layers-1): # 0,1
            flow_decoder = getattr(self, 'flow_decoder' + str(i+1))
            target_merge = None
            for k in range(K):
                # if the last layer has flow output, then it should have residual flow from target stream
                if self.n_encode_layers-i-1 in self.flow_layers: # 2,3
                    out = flow_decoder(torch.cat((outs[k][i-1], (flows[k][i-1]+residual_flows[k][i-1])*2, masks[k][i-1], attns[k][i-1]),dim=1)) # flow ×2 upsampling 
                else:
                    out = flow_decoder(flow_outs[k])
                    
                model = getattr(self, 'flow_jump' + str(i+1))
                jump = model(flow_encode_features[k][self.n_encode_layers-i-1-2])
                out = out+jump
                outs[k][i] = out
                if self.n_decode_layers-i in self.flow_layers:
                    model = getattr(self, 'flow_flow'+str(i+1))
                    flow = model(outs[k][i])
                    flows[k][i] = flow

            # print(torch.mean(attns[0][i]))
            # print(torch.mean(attns[1][i]))
            # softmax normalize the attentions
            residual_flows=[[0]*layers for k in range(K)]
            masks=[[0]*layers]
            attns=[[0]*layers for k in range(K)]
            for k in range(K):
                # if this layer has flow output
                if self.n_decode_layers-i in self.flow_layers:
                    out_k = warp_flow(source_features[k][i], flows[k][i], align_corners=self.align_corners)
                    model = getattr(self, 'attn'+str(i))
                    attn = model(torch.cat((target_out, out_k),dim=1))

            a = torch.cat([attns[k][i] for k in range(K)], dim=1)
            a = nn.Softmax(dim=1)(a)

            for k in range(K):
                attns[k][i] = a[:,k:k+1,...]
            # print(torch.mean(attns[0][i]))
            # print(torch.mean(attns[1][i]))
            for k in range(K):
                # if this layer has flow output
                if self.n_decode_layers-i in self.flow_layers:
                    # merge the flowed sources with the target
                    out_k = warp_flow(source_features[k][i], flows[k][i], align_corners=self.align_corners) # 1. warp the source by the initial flow
                    model = getattr(self, 'res_flow'+str(i)) 
                    residual_flow = model(torch.cat((target_out, out_k),dim=1))   # 2. get the residual flow by '''flow_res = f(source_w, target)'''
                    # residual_flow = residual_flow * 0
                    out_k = warp_flow(source_features[k][i], residual_flow + flows[k][i], align_corners=self.align_corners) # 3. warp the source again by the residual flow
                    out_k = attns[k][i] * (out_k * masks[k][i] + target_out * (1-masks[k][i])) # 4. recalculate the masked source by the new one 
                    # a_k(w_k*m_k + p*(1-m_k))
                    if target_merge is None:
                        target_merge = out_k
                    else:
                        target_merge += out_k
                    
                    residual_flows[k][i]=residual_flow

                
                    
            target_out = target_merge # set the target merge as the output of the current layer
            model = getattr(self, 'target_decoder'+str(i))
            target_out = model(target_out)
        
        # finish the last layer of decoder
        for i in range(self.n_decode_layers-1, self.n_decode_layers):
            model = getattr(self, 'target_decoder'+str(i))
            target_out = model(target_out)

        out_image = self.outconv(target_out)
        return out_image, flows,residual_flows, attns, masks





