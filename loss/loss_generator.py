from .externel_functions import VGGLoss,VGG19
from model.blocks import warp_flow
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossG(nn.Module):
    """
    Loss for generator meta training
    Inputs: x, x_hat, r_hat, D_res_list, D_hat_res_list, e, W, i
    output: lossG
    """
    def __init__(self, device):
        super(LossG, self).__init__()
        
        self.vggLoss = VGGLoss().to(device)
        self.L1Loss = nn.L1Loss().to(device)
        
    def forward(self, x, x_hat):
        loss_content, loss_style  = self.vggLoss(x, x_hat)
        loss_rec  = self.L1Loss(x, x_hat)
        return loss_content, loss_style, loss_rec

class ParsingLoss(nn.Module):
    """
    Loss Parsing
    Inputs: HW*C logits output by the network, HW target class labels
    
    """
    def __init__(self, device):
        super(ParsingLoss, self).__init__()
        self.ceLoss = nn.CrossEntropyLoss().to(device)

    def forward(self, logits, target):
        loss_ce = self.ceLoss(logits, target)
        return loss_ce

class PerceptualCorrectness(nn.Module):
    r"""
    """

    def __init__(self, layer=['rel1_1','relu2_1','relu3_1','relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer  
        self.eps=1e-8 
        # self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=True):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)

        return loss

    def calculate_loss(self, flow, layer, mask=None, use_bilinear_sampling=True):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape

        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        flow = F.interpolate(flow, [h,w])

        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps)
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        try:
            correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        except:
            print("An exception occurred")
            print(source_norm.shape)
            print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        # if use_bilinear_sampling:
        # input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        input_sample = self.warp_flow(source_vgg, flow).view(b, c, -1)
        # else:
        #     input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
        if mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            mask=F.interpolate(mask, size=(target_vgg.size(2), target_vgg.size(3)))
            mask=mask.view(-1, target_vgg.size(2)*target_vgg.size(3)) #[b 1 N2]
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.sum(mask * loss_map)/(torch.sum(mask)+self.eps)
            
        return loss

    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w-1, h-1]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid).view(b, c, -1)
        return input_sample

    def warp_flow(self,x,flow):
        [b, c, h, w] = x.shape
        # mesh grid
        xx = x.new_tensor(range(w)).view(1,-1).repeat(h,1)
        yy = x.new_tensor(range(h)).view(-1,1).repeat(1,w)
        xx = xx.view(1,1,h,w).repeat(b,1,1,1)
        yy = yy.view(1,1,h,w).repeat(b,1,1,1)
        grid = torch.cat((xx,yy), dim=1).float()
        # grid: (b, 2, H, W)
        grid = grid + flow

        # scale to [-1, 1]
        grid[:,0,:,:] = 2.0*grid[:,0,:,:]/max(w-1,1) - 1.0
        grid[:,1,:,:] = 2.0*grid[:,1,:,:]/max(h-1,1) - 1.0

        # to (b, h, w, c) for F.grid_sample
        grid = grid.permute(0,2,3,1)
        output = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros')

        return output
        

class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) +
                        torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self):
        super(GicLoss, self).__init__()
        self.dT = DT()

    def forward(self, grid):
        B,H,W,_ = grid.size()
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]
        Gxcenter = Gx[:, 1:H - 1, 1:W - 1]
        Gxleft = Gx[:, 1:H - 1, 0:W - 2]
        Gxright = Gx[:, 1:H - 1, 2:W]

        Gycenter = Gy[:, 1:H - 1, 1:W - 1]
        Gyup = Gy[:, 0:H - 2, 1:W - 1]
        Gydown = Gy[:, 2:H, 1:W - 1]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown)) / (B*H*W)