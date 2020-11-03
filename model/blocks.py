import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm_layer(norm, channels):
    if norm=='bn':
        return nn.BatchNorm2d(channels, affine=True)
    elif norm=='in':
        return nn.InstanceNorm2d(channels, affine=True)
    else:
        return None


# grid:(b, 2, H, W) [-1,1]
def gen_uniform_grid(x):
    [b, c, h, w] = x.shape
    # mesh grid
    xx = x.new_tensor(range(w)).view(1,-1).repeat(h,1)
    yy = x.new_tensor(range(h)).view(-1,1).repeat(1,w)
    xx = xx.view(1,1,h,w).repeat(b,1,1,1)
    yy = yy.view(1,1,h,w).repeat(b,1,1,1)
    grid = torch.cat((xx,yy), dim=1).float()

    grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0
    return grid

def warp_flow(x, flow, mode='bilinear', mask=None, mask_value=-1):
    '''
    Warp a image x according to the given flow
    Input:
        x: (b, c, H, W)
        flow: (b, 2, H, W)
        mask: (b, 1, H, W)
    Ouput:
        y: (b, c, H, W)
    '''
    [b, c, h, w] = x.shape
    # mesh grid
    xx = x.new_tensor(range(w)).view(1,-1).repeat(h,1)
    yy = x.new_tensor(range(h)).view(-1,1).repeat(1,w)
    xx = xx.view(1,1,h,w).repeat(b,1,1,1)
    yy = yy.view(1,1,h,w).repeat(b,1,1,1)
    grid = torch.cat((xx,yy), dim=1).float()
     
    grid = grid + flow

    # scale to [-1, 1]
    grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0


    # to (b, h, w, c) for F.grid_sample
    grid = grid.permute(0,2,3,1)
    output = F.grid_sample(x, grid, mode=mode, padding_mode='zeros')

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

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding, norm_type='bn', use_spectral_norm=False):
        super(ResBlock2d, self).__init__()

        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                padding=padding))
            self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                                padding=padding))
        else:
            self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)

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
        out += x
        return out

class ResBlockUpNorm(nn.Module):
    
    def __init__(self, in_channel, out_channel, out_size=None, scale = 2, \
        conv_size=3, padding_size = 1, is_bilinear = True, norm_type='bn', use_spectral_norm=False):
        super(ResBlockUpNorm, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        if is_bilinear:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale, mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size = out_size, scale_factor=scale,align_corners=False)
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
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel//8, 1))
        #conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))
        
        self.softmax = nn.Softmax(-2) #sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x) #BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x) #BxC'xHxW
        h_projection = self.conv_h(x) #BxCxHxW
        
        f_projection = f_projection.view(B,-1,H*W).permute(0,2,1) #BxNxC', N=H*W
        g_projection = g_projection.view(B,-1,H*W) #BxC'xN
        h_projection = h_projection.view(B,-1,H*W) #BxCxN
        
        attention_map = torch.bmm(f_projection, g_projection) #BxNxN
        attention_map = self.softmax(attention_map) #sum_i_N (A i,j) = 1
        
        #sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map) #BxCxN
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


class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        
        #using no ReLU method
        
        #general
        self.relu = nn.LeakyReLU(inplace = False)
        
        #left
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 3, padding = 1))
        
    def forward(self, x, psi_slice):
        C = psi_slice.shape[1]
        
        res = x
        
        out = adaIN(x, psi_slice[:, 0:C//4, :], psi_slice[:, C//4:C//2, :])
        out = self.relu(out)
        out = self.conv1(out)
        out = adaIN(out, psi_slice[:, C//2:3*C//4, :], psi_slice[:, 3*C//4:C, :])
        out = self.relu(out)
        out = self.conv2(out)
        
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

