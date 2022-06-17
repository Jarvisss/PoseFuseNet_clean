from .externel_functions import VGGLoss,VGG19
from model.blocks import warp_flow
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resample2d_package.resample2d import Resample2d
from model.block_extractor.block_extractor   import BlockExtractor
from model.local_attn_reshape.local_attn_reshape   import LocalAttnReshape
import numpy as np
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

# class MaskLoss(nn.Module):
#     def __init__(self, device):
#         super(MaskLoss, self).__init__()
#         self.bceloss = nn.BCELoss().to(device)
#     def forward(self, source, target):
#         loss_bce = self.bceloss(source, target)
#         return loss_bce

class MultiAffineRegularizationLoss(nn.Module):
    def __init__(self, kz_dic):
        super(MultiAffineRegularizationLoss, self).__init__()
        self.kz_dic=kz_dic
        self.method_dic={}
        for key in kz_dic:
            instance = AffineRegularizationLoss(kz_dic[key])
            self.method_dic[key] = instance
        self.layers = sorted(kz_dic, reverse=True) 
 
    def __call__(self, flow_fields):
        loss=0
        for i in range(len(flow_fields)):
            method = self.method_dic[self.layers[i]]
            loss += method(flow_fields[i])
        return loss



class AffineRegularizationLoss(nn.Module):
    """docstring for AffineRegularizationLoss"""
    # kernel_size: kz
    def __init__(self, kz):
        super(AffineRegularizationLoss, self).__init__()
        self.kz = kz
        self.criterion = torch.nn.L1Loss()
        self.extractor = BlockExtractor(kernel_size=kz)
        self.reshape = LocalAttnReshape()

        temp = np.arange(kz)
        A = np.ones([kz*kz, 3])
        A[:, 0] = temp.repeat(kz)
        A[:, 1] = temp.repeat(kz).reshape((kz,kz)).transpose().reshape(kz**2)
        AH = A.transpose()
        k = np.dot(A, np.dot(np.linalg.inv(np.dot(AH, A)), AH)) - np.identity(kz**2) #K = (A((AH A)^-1)AH - I)
        self.kernel = np.dot(k.transpose(), k)
        self.kernel = torch.from_numpy(self.kernel).unsqueeze(1).view(kz**2, kz, kz).unsqueeze(1)

    def __call__(self, flow_field):
        grid = self.flow2grid(flow_field)

        grid_x = grid[:,0,:,:].unsqueeze(1)
        grid_y = grid[:,1,:,:].unsqueeze(1)
        weights = self.kernel.type_as(flow_field)
        loss_x = self.calculate_loss(grid_x, weights)
        loss_y = self.calculate_loss(grid_y, weights)
        return loss_x+loss_y
    
    def calculate_loss(self, grid, weights):
        results = nn.functional.conv2d(grid, weights)   # KH K B [b, kz*kz, w, h]
        b, c, h, w = results.size()
        kernels_new = self.reshape(results, self.kz)
        f = torch.zeros(b, 2, h, w).type_as(kernels_new) + float(int(self.kz/2))
        grid_H = self.extractor(grid, f)
        result = torch.nn.functional.avg_pool2d(grid_H*kernels_new, self.kz, self.kz)
        loss = torch.mean(result)*self.kz**2
        return loss

    def flow2grid(self, flow_field):
        b,c,h,w = flow_field.size()
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(flow_field).float() 
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(flow_field).float()
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        return flow_field+grid



class FusingCompactnessLoss(nn.Module):
    r"""
    """

    def __init__(self, K, layer=['relu1_1','relu2_1','relu3_1','relu4_1']):
        super(FusingCompactnessLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps = 1e-8
        self.resample = Resample2d(4,1,sigma=2)
        self.criterion = torch.nn.L1Loss()
        self.K = K
        
        pass

    def __call__(self, target, source_list, flow_list, attention_list,used_layers):
        used_layers=sorted(used_layers, reverse=True)
        assert(len(source_list) == self.K)
        assert(len(flow_list) == self.K)
        # assert(len(mask_list) == self.K) # mask wont be useful here
        assert(len(attention_list) == self.K)

        assert(len(flow_list[0])==len(used_layers))
        # assert(len(mask_list[0])==len(used_layers))
        assert(len(attention_list[0])==len(used_layers))
        loss = 0

        self.target_vgg = self.vgg(target)
        self.sources_vggs = []
        for source in source_list:
            self.sources_vggs.append(self.vgg(source))


        for i in range(len(flow_list[0])): # i 表示是第几层网络
            vgg_layer = self.layer[used_layers[i]]
            layer_fused_sources_vgg = self.calculate_fused_vgg_at_layer(self.sources_vggs, attention_list, flow_list, i, vgg_layer)
            loss += self.criterion(layer_fused_sources_vgg, self.target_vgg[vgg_layer])

        return loss
        pass

    def calculate_fused_vgg_at_layer(self,source_vggs, attentions, flows, layer, vgg_layer):
        for k in range(len(flows)):
            source_k_vgg_layer = source_vggs[k][vgg_layer]
            flow = flows[k][layer]
            attention = attentions[k][layer]
            warped_k = self.bilinear_warp(source_k_vgg_layer, flow)
            if k == 0:
                fused = warped_k * attention
            else:
                fused += warped_k * attention
        
        return fused
                


    
    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid, align_corners=True)
        return input_sample

class PerceptualCorrectness(nn.Module):
    r"""
    """

    def __init__(self, layer=['relu1_1','relu2_1','relu3_1','relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer  
        self.eps=1e-8 
        self.resample = Resample2d(4, 1, sigma=2)

    def __call__(self, target, source, flow_list, used_layers, mask=None, use_bilinear_sampling=True, target_flow_list=None):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        if target_flow_list is not None:
            assert len(target_flow_list)==len(flow_list), 'target flow length not equal to source flow'
            for i in range(len(flow_list)):
                loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling, target_flow_list[i])
        else:
            for i in range(len(flow_list)):
                loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling, None)
        return loss

    def calculate_loss(self, flow , layer, mask=None, use_bilinear_sampling=True, target_flow=None):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        [bf,cf,hf,wf] = flow.shape

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
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        if target_flow is not None:
            if use_bilinear_sampling:
                target_sample = self.bilinear_warp(target_vgg, target_flow).view(b, c, -1)
            else:
                target_sample = self.resample(target_vgg, target_flow).view(b, c, -1)
            target_all = target_sample
            
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
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid, align_corners=True).view(b, c, -1)
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


class FlowAttnLoss(nn.Module):
    r"""
    """

    def __init__(self, layer=['relu1_1','relu2_1','relu3_1','relu4_1']):
        super(FlowAttnLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer  
        self.eps=1e-8 
        self.resample = Resample2d(4, 1, sigma=2)
        self.criterion = nn.L1Loss()
        

    def __call__(self, target, source, flow_list, attention_list, used_layers,mask=None, use_bilinear_sampling=True):
        used_layers=sorted(used_layers, reverse=True)
        # self.target=target
        # self.source=source
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], attention_list[i], self.layer[used_layers[i]], mask, use_bilinear_sampling)

        return loss

    # def calculate_loss(self, flow, attention, layer, mask=None, use_bilinear_sampling=True):
    #     target_vgg = self.target_vgg[layer]
    #     source_vgg = self.source_vgg[layer]
        
    #     if use_bilinear_sampling:
    #         input_sample = self.bilinear_warp(source_vgg, flow)
    #     else:
    #         input_sample = self.resample(source_vgg, flow)
        
    #     # weight = torch.exp(-1 * attention)
    #     weight = attention # 对于采样不正确的点，我们希望 attention weight小，L = a1*10 + a2*100, s.t. a1+a2=1, a1 a2 > 0
    #     loss = self.criterion(weight * target_vgg, weight * input_sample)
    #     return loss
    def calculate_loss(self, flow, attention, layer, mask=None, use_bilinear_sampling=True):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow)
        else:
            input_sample = self.resample(source_vgg, flow)
        
        [b, c, h, w] = target_vgg.shape
        [bf,cf,hf,wf] = flow.shape
        # maps = F.interpolate(maps, [h,w]).view(b,-1)
        flow = F.interpolate(flow, [h,w])
        target_all = target_vgg.view(b, c, -1)                      #[b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1,2)       #[b N2 C]


        source_norm = source_all/(source_all.norm(dim=2, keepdim=True)+self.eps) #[b C N2] norm 1
        target_norm = target_all/(target_all.norm(dim=1, keepdim=True)+self.eps)
        try:
            correction = torch.bmm(source_norm, target_norm)                       #[b N2 N2]
        except:
            print("An exception occurred")
            print(source_norm.shape)
            print(target_norm.shape)
        (correction_max,max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)    #[b 1 N2]
        loss_map = torch.exp(-correction_sample/(correction_max+self.eps))
        if attention is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            attention=F.interpolate(attention, size=(h,w))
            attention=attention.view(-1, h*w) #[b 1 N2]
            loss_map = loss_map - torch.exp(torch.tensor(-1).type_as(loss_map))
            loss = torch.mean(attention * loss_map)
        return loss


    def bilinear_warp(self, source, flow):
        [b, c, h, w] = source.shape
        x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
        y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
        grid = torch.stack([x,y], dim=0)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        grid = 2*grid - 1
        flow = 2*flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
        grid = (grid+flow).permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid, align_corners=True).view(b, c, -1)
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

    # expect input to be BHWC
    def forward(self, grid, mask=None):
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

        # calculate the mask after dt
        if mask is not None:
            assert mask.shape[0]==B
            assert mask.shape[2]==H
            assert mask.shape[3]==W

            # calculate mask dt
            mask_center = mask[:,:, 1:H - 1, 1: W - 1]
            mask_left = mask[:,:, 1:H - 1, 0:W - 2]
            mask_right = mask[:,:, 1:H - 1, 2:W]

            mask_up = mask[:,:, 0:H - 2, 1:W - 1]
            mask_down = mask[:,:, 2:H, 1:W - 1]

            dtm_left = self.dT(mask_left, mask_center)
            dtm_right = self.dT(mask_right, mask_center)
            dtm_up = self.dT(mask_up, mask_center)
            dtm_down = self.dT(mask_down, mask_center)

            # we need to calculate inside part of a mask
            dtm_left_inside = (dtm_left==0) * mask_center
            dtm_right_inside = (dtm_right==0) * mask_center
            dtm_up_inside = (dtm_up==0) * mask_center
            dtm_down_inside = (dtm_down==0) * mask_center

            return torch.sum(torch.abs(dtleft * dtm_left_inside - dtright*dtm_right_inside) + torch.abs(dtup*dtm_up_inside - dtdown*dtm_down_inside)) / (B*H*W)
        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown)) / (B*H*W)

