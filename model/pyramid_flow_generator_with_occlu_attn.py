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
    def __init__(self, image_nc=3, structure_nc=21, n_layers=5, flow_layers=[2,3], ngf=32, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True):
        super(FlowGenerator, self).__init__()
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
    def __init__(self, structure_nc=21, n_layers=5, attn_layers=[2,3], ngf=32, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True):
        super(PoseAttnFCNGenerator, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm_type)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        self.structure_nc = structure_nc
        self.ngf = ngf
        self.norm_type = norm_type

        self.max_nc = max_nc
        self.encoder_layers = n_layers
        self.decoder_layers = 1 + len(attn_layers)
        self.attn_layers = attn_layers

        self._make_layers(self.ngf, self.max_nc, norm_layer, nonlinearity, use_spectral_norm)

    def _make_layers(self,ngf,max_nc,norm_layer,nonlinearity, use_spectral_norm):
        inc =self.structure_nc * 2
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

    def forward(self, pose_A, pose_B):

        x_in = torch.cat((pose_A, pose_B), dim=1) # (42, 256, 256)
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
                ResBlock2d(ngf*mult_prev,3,1,self.norm_type, use_spectral_norm),
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


class PoseAwareAppearanceDecoder(nn.Module):
    '''
    Pose Stream Decoder
    '''
    def __init__(self, structure_nc=21, n_decode_layers=3,output_nc=3, flow_layers=[2,3], ngf=64, max_nc=256, norm_type='bn',activation='LeakyReLU', use_spectral_norm=True, align_corners=True, use_resample=False):
        super(PoseAwareAppearanceDecoder, self).__init__()
        self.n_decode_layers = n_decode_layers
        self.flow_layers = flow_layers
        self.norm_type = norm_type
        self.align_corners = align_corners
        self.use_resample = use_resample
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
                ResBlock2d(ngf*mult_prev,3,1,self.norm_type, use_spectral_norm),
                ResBlockDecoder(ngf*mult_prev, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, use_spect=use_spectral_norm)
                # ResBlockUpNorm(ngf*mult_prev, ngf*mult, norm_type=self.norm_type, use_spectral_norm=use_spectral_norm)
            )
            setattr(self, 'decoder'+str(i), up)
            if self.n_decode_layers-i in self.flow_layers:
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
                out = merge
                sa_layer = getattr(self, 'sa'+str(i))
                out = sa_layer(out)
            
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




