import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import functools


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    # elif norm_type == 'adain':
    #     norm_layer = functools.partial(ADAIN)
    # elif norm_type == 'spade':
    #     norm_layer = functools.partial(SPADE, config_text='spadeinstance3x3')        
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    if norm_type != 'none':
        norm_layer.__name__ = norm_type

    return norm_layer

def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer

def make_norm_layer(norm, channels):
    if norm=='bn':
        return nn.BatchNorm2d(channels, affine=True)
    elif norm=='in':
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        return None


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module

# grid:(b, 2, H, W) [-1,1]
def gen_uniform_grid(source):
    [b, _, h, w] = source.shape
    # mesh grid
    x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
    y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
    grid = torch.stack([x,y], dim=0)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    grid = 2*grid - 1
    return grid

def warp_flow(source, flow, align_corners=True, mode='bilinear', mask=None, mask_value=-1):
    '''
    Warp a image x according to the given flow
    Input:
        x: (b, c, H, W)
        flow: (b, 2, H, W) # range [-w/2, w/2] [-h/2, h/2]
        mask: (b, 1, H, W)
    Ouput:
        y: (b, c, H, W)
    '''
    [b, c, h, w] = source.shape
    # mesh grid
    x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
    y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
    grid = torch.stack([x,y], dim=0)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    grid = 2*grid - 1

    flow = 2* flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
    
    grid = (grid+flow).permute(0, 2, 3, 1)
    
    '''grid = grid + flow # in this way flow is -1 to 1
    '''
    # to (b, h, w, c) for F.grid_sample
    output = F.grid_sample(source, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)

    if mask is not None:
        output = torch.where(mask>0.5, output, output.new_ones(1).mul_(mask_value))
    return output
    pass

# def warp_flow(source, flow, mode='bilinear'):
#     [b, c, h, w] = source.shape
#     x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
#     y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
#     grid = torch.stack([x,y], dim=0)
#     grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
#     grid = 2*grid - 1
#     flow = 2*flow / torch.tensor([w-1, h-1]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
#     grid = (grid+flow).permute(0, 2, 3, 1)
#     input_sample = F.grid_sample(source, grid)
#     return input_sample

class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta

        return out

class Output(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out

class Jump(nn.Module):
    """
    Define the output layer
    """
    def __init__(self, input_nc, output_nc, kernel_size = 3, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(Jump, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding':0, 'bias': True}

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size/2)), self.conv1)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1)

    def forward(self, x):
        out = self.model(x)
        return out

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, output_nc=None, hidden_nc=None, kernel_size=3, padding=1, norm_type='bn', use_spectral_norm=False, learnable_shortcut=False):
        super(ResBlock2d, self).__init__()
        hidden_nc = in_features if hidden_nc is None else hidden_nc
        output_nc = in_features if output_nc is None else output_nc

        self.learnable_shortcut = True if in_features != output_nc else learnable_shortcut

        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_features, out_channels=hidden_nc, kernel_size=kernel_size,
                                padding=padding))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=hidden_nc, out_channels=output_nc, kernel_size=kernel_size,
                                padding=padding))
        else:
            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_nc, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(in_channels=hidden_nc, out_channels=output_nc, kernel_size=kernel_size, padding=padding)

        if self.learnable_shortcut:
            bypass = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_features, out_channels=output_nc, kernel_size=kernel_size,
                                padding=padding))
            self.shortcut = nn.Sequential(bypass,)
        self.norm1 = make_norm_layer(norm_type, in_features)
        self.norm2 = make_norm_layer(norm_type, in_features)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.learnable_shortcut:
            out += self.shortcut(x)
        else:
            out += x
        return out


class ResBlockUpNorm(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, conv_size=3, padding_size = 1, is_bilinear = True, norm_type='bn', use_spectral_norm=False):
        super(ResBlockUpNorm, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if is_bilinear:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale, mode='bilinear',align_corners=True)
        else:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale,align_corners=True)
        self.relu = nn.LeakyReLU(inplace = False)
        
        self.norml1 = make_norm_layer(norm_type, in_channel)
        
        self.normr1 = make_norm_layer(norm_type, in_channel)
        self.normr2 = make_norm_layer(norm_type, out_channel)
        
        if use_spectral_norm:
            #left
            self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))

            #right
            self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
            self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))
        else:
            #left
            self.conv_l1 = nn.Conv2d(in_channel, out_channel, 1)
            #right
            self.conv_r1 = nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size)
            self.conv_r2 = nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size)
    
    def forward(self,x):
        res = x
        
        #left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)
        
        #right
        out = self.normr1(x)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.normr2(out)
        out = self.relu(out)
        out = self.conv_r2(out)

        out = out + out_res
        
        return out

class ConvUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, \
        conv_size=3, padding_size = 1, is_bilinear = True, norm_type='bn', use_spectral_norm=False):
        super(ConvUp, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        
        self.relu = nn.LeakyReLU(inplace = False)
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)
        self.norm = make_norm_layer(norm_type, out_channel)
    
    def forward(self,x):
        out = nn.Upsample(x, scale_factor=2)
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        
        return out

 

class ResBlockEncoder(nn.Module):
    """
    Define a encoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockEncoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.Conv2d(hidden_nc, output_nc, kernel_size=4, stride=2, padding=1), use_spect)
        bypass = spectral_norm(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, padding=0), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
                                       norm_layer(hidden_nc), nonlinearity, conv2,)
        
        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

class EncoderBlock(nn.Module):
    '''d'''
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False):
        super(EncoderBlock, self).__init__()


        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        
        conv1 = spectral_norm(nn.Conv2d(input_nc,  output_nc, **kwargs_down), use_spect)
        conv2 = spectral_norm(nn.Conv2d(output_nc, output_nc, **kwargs_fine), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc),  nonlinearity, conv1, 
                                       norm_layer(output_nc), nonlinearity, conv2,)

    def forward(self, x):
        out = self.model(x)
        return out


class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1, use_spectral_norm=False):
        super(ResBlockDown, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = False)
        self.avg_pool2d = nn.AvgPool2d(2)

        if use_spectral_norm:
            #left
            self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1,))
            
            #right
            self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
            self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))
        else:
            #left
            self.conv_l1 = nn.Conv2d(in_channel, out_channel, 1,)
            
            #right
            self.conv_r1 = nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size)
            self.conv_r2 = nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size)

    def forward(self, x):
        res = x
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out

class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """
    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
                 use_spect=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = input_nc if hidden_nc is None else hidden_nc

        conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        conv2 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        bypass = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, norm_layer(hidden_nc), nonlinearity, conv2,)

        self.shortcut = nn.Sequential(bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)
        return out

# class ResBlock(nn.Module):
#     """
#     Define an Residual block for different types
#     """
#     def __init__(self, input_nc, output_nc=None, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity= nn.LeakyReLU(),
#                 learnable_shortcut=False, use_spect=False, use_coord=False):
#         super(ResBlock, self).__init__()

#         hidden_nc = input_nc if hidden_nc is None else hidden_nc
#         output_nc = input_nc if output_nc is None else output_nc
#         self.learnable_shortcut = True if input_nc != output_nc else learnable_shortcut

#         kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
#         kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

#         conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, **kwargs)
#         conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, **kwargs)

#         if type(norm_layer) == type(None):
#             self.model = nn.Sequential(nonlinearity, conv1, nonlinearity, conv2,)
#         else:
#             self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, conv1, 
#                                        norm_layer(hidden_nc), nonlinearity, conv2,)

#         if self.learnable_shortcut:
#             bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, **kwargs_short)
#             self.shortcut = nn.Sequential(bypass,)


#     def forward(self, x):
#         if self.learnable_shortcut:
#             out = self.model(x) + self.shortcut(x)
#         else:
#             out = self.model(x) + x
#         return out

class ResBlockDownFirst(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDownFirst, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = False)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1,))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

    def forward(self, x):
        res = x
        #left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)
        
        #right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)
        
        #merge
        out = out_res + out
        
        return out  


class ConvDownResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ConvDownResBlock, self).__init__()
        
        self.convdown = nn.Conv2d(in_channel, out_channel, conv_size,stride=2, padding=padding_size)
        self.relu = nn.LeakyReLU()
        self.relu_inplace = nn.LeakyReLU(inplace = False)

        # res block
        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))

    def forward(self, x):
        x_down = self.convdown(x)
        #left
        identity = x_down
        
        #right
        out = self.relu(x_down)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        
        #merge
        out = identity + out
        
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()
        
        #conv f
        self.query_conv  = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.value_conv  = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query_conv(x) #BxC'xHxW, C'=C//8
        proj_key = self.key_conv(x) #BxC'xHxW
        proj_value = self.value_conv(x) #BxCxHxW
        
        proj_query = proj_query.view(B,-1,H*W).permute(0,2,1) #BxNxC', N=H*W
        proj_key = proj_key.view(B,-1,H*W) #BxC'xN
        proj_value = proj_value.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(proj_query, proj_key) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(proj_value, attention_map) #BxCxN
        out = out.view(B,C,H,W)
        
        out = self.gamma*out + x
        return out
        


        
def adaIN(feature, mean_style, std_style, eps = 1e-5):
    B,C,H,W = feature.shape
    
    
    feature = feature.view(B,C,-1)
            
    std_feat = (torch.std(feature, dim = 2) + eps).view(B,C,1)
    mean_feat = torch.mean(feature, dim = 2).view(B,C,1)
    
    adain = std_style * (feature - mean_feat)/std_feat + mean_style
    
    adain = adain.view(B,C,H,W)
    return adain

class Embedder(nn.Module):
    def __init__(self, in_height, structure_nc=21, max_nc=256):
        super(Embedder, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace=False)
        
        #in 6*224*224
        self.resDown1 = ResBlockDown(structure_nc+3, 32) #out 32*128*128
        self.resDown2 = ResBlockDown(32, 64) #out 64*64*64
        self.resDown3 = ResBlockDown(64, 128) #out 128*32*32
        self.self_att = SelfAttention(128) #out 128*32*32
        self.resDown4 = ResBlockDown(128, 256) #out 256*16*16
        self.resDown5 = ResBlockDown(256, 256) #out 256*8*8
        self.sum_pooling = nn.AdaptiveMaxPool2d((1,1)) #out 256*1*1

    def forward(self, x, y):
        out = torch.cat((x,y),dim = 1) #out 24*224*224
        out = self.resDown1(out) #out 32*128*128
        out = self.resDown2(out) #out 64*64*64
        out = self.resDown3(out) #out 256*32*32
        
        out = self.self_att(out) #out 256*32*32
        
        out = self.resDown4(out) #out 256*16*16
        out = self.resDown5(out) #out 256*8*8
        
        out = self.sum_pooling(out) #out 256*1*1
        out = self.relu(out) #out 256*1*1
        out = out.view(-1,max_nc,1) #out B*256*1
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        
        #using no ReLU method
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, padding = 1)
        
    def forward(self, x, psi_slice):
        C = psi_slice.shape[1]
        
        res = x
        out = self.conv1(x)
        out = adaIN(out, psi_slice[:, 0:C//4, :], psi_slice[:, C//4:C//2, :])
        out = self.relu(out)
        out = self.conv2(out)
        out = adaIN(out, psi_slice[:, C//2:3*C//4, :], psi_slice[:, 3*C//4:C, :])
        
        out = out + res
        
        return out
        
class ResBlockD(nn.Module):
    def __init__(self, in_channel):
        super(ResBlockD, self).__init__()
        
        #using no ReLU method
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = out + res
        
        return out


class ResBlockUp(nn.Module):
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, conv_size=3, padding_size = 1, is_bilinear = True):
        super(ResBlockUp, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if is_bilinear:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale, mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale,align_corners=False)
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1))
        
        #right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding = padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding = padding_size))
    
    def forward(self,x, psi_slice):
        mean1 = psi_slice[:, 0:self.in_channel, :]
        std1 = psi_slice[:, self.in_channel:2*self.in_channel, :]
        mean2 = psi_slice[:, 2*self.in_channel:2*self.in_channel + self.out_channel, :]
        std2 = psi_slice[:, 2*self.in_channel + self.out_channel: 2*(self.in_channel+self.out_channel), :]
        
        res = x
        
        #left
        out_res = self.upsample(res)
        out_res = self.conv_l1(out_res)
        
        #right
        out = adaIN(x, mean1, std1)
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = adaIN(out, mean2, std2)
        out = self.relu(out)
        out = self.conv_r2(out)
        
        out = out + out_res
        
        return out

class Padding(nn.Module):
    def __init__(self, in_shape):
        super(Padding, self).__init__()
        
        self.zero_pad = nn.ZeroPad2d(self.findPadSize(in_shape))
    
    def forward(self,x):
        out = self.zero_pad(x)
        return out
    
    def findPadSize(self,in_shape):
        if in_shape < 256:
            pad_size = (256 - in_shape)//2
        else:
            pad_size = 0
        return pad_size

