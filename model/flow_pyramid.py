from .blocks import *
import torch
import torch.nn as nn 

class FlowPyramid(nn.Module):
    '''
    [flow list] = FP(x1, y1, y2)
    '''
    def __init__(self,source_nc=23,target_nc=20, mode='bilinear'):
        
        super(FlowPyramid, self).__init__()
        self.sourceEncoder = SourcePyramidEncoder(inc=source_nc)
        self.targetEncoder = TargetPyramidEncoder(inc=target_nc)
        self.flowGenerator = FlowGenerator(mode=mode)
    
    def forward(self, x1, y1, y2):
        '''
        doc string
        '''
        source_features = self.sourceEncoder(x1, y1)
        target_features = self.targetEncoder(y2)

        flows = self.flowGenerator(source_features, target_features)
        x_hat = warp_flow(x1, flows[0])
        return flows, x_hat

class SourcePyramidEncoder(nn.Module):
    def __init__(self, inc):

        super(SourcePyramidEncoder, self).__init__()

        self.resDown1 = ConvDownResBlock(inc, 64) # (64, 128, 128)
        self.resDown2 = ConvDownResBlock(64, 128) # (128, 64, 64)
        self.resDown3 = ConvDownResBlock(128, 256) # (256, 32, 32)
        self.resDown4 = ConvDownResBlock(256, 256) # (256, 16, 16)
        self.resDown5 = ConvDownResBlock(256, 256) # (256, 8, 8)

        self.latres5 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latres4 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latres3 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latres2 = nn.Conv2d(128, 256, 1, 1, 0)
        self.latres1 = nn.Conv2d(64, 256, 1, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, image1, pose1):

        x_in = torch.cat((image1, pose1), dim=1) # (23, 256, 256)
        x_resdown1 = self.resDown1(x_in) # (64, 128, 128)
        x_resdown2 = self.resDown2(x_resdown1) # (128, 64, 64)
        x_resdown3 = self.resDown3(x_resdown2) # (256, 32, 32)
        x_resdown4 = self.resDown4(x_resdown3) # (256, 16, 16)
        x_resdown5 = self.resDown5(x_resdown4) # (256, 8, 8)

        x_lat5 = self.latres5(x_resdown5) # (256, 8, 8)
        x_lat4 = self.latres4(x_resdown4) # (256, 16, 16)
        x_lat3 = self.latres3(x_resdown3) # (256, 32, 32)
        x_lat2 = self.latres2(x_resdown2) # (256, 64, 64)
        x_lat1 = self.latres1(x_resdown1) # (256, 128, 128)

        # first coarse 
        x5_out = x_lat5 # (256, 8, 8)
        # coarse + residual 
        x4_out = self.upsample(x_lat5) + x_lat4 # (256, 16, 16)
        x3_out = self.upsample(x_lat4) + x_lat3 # (256, 32, 32)
        x2_out = self.upsample(x_lat3) + x_lat2 # (256, 64, 64)
        x1_out = self.upsample(x_lat2) + x_lat1 # (256, 128, 128)

        return [x1_out, x2_out, x3_out, x4_out, x5_out]

class TargetPyramidEncoder(nn.Module):
    def __init__(self, inc):

        super(TargetPyramidEncoder, self).__init__()

        self.resDown1 = ConvDownResBlock(inc, 64) # (64, 128, 128)
        self.resDown2 = ConvDownResBlock(64, 128) # (128, 64, 64)
        self.resDown3 = ConvDownResBlock(128, 256) # (256, 32, 32)
        self.resDown4 = ConvDownResBlock(256, 256) # (256, 16, 16)
        self.resDown5 = ConvDownResBlock(256, 256) # (256, 8, 8)

        self.latres5 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latres4 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latres3 = nn.Conv2d(256, 256, 1, 1, 0)
        self.latres2 = nn.Conv2d(128, 256, 1, 1, 0)
        self.latres1 = nn.Conv2d(64, 256, 1, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, pose):

        x_in = pose # (20, 256, 256)
        x_resdown1 = self.resDown1(x_in) # (64, 128, 128)
        x_resdown2 = self.resDown2(x_resdown1) # (128, 64, 64)
        x_resdown3 = self.resDown3(x_resdown2) # (256, 32, 32)
        x_resdown4 = self.resDown4(x_resdown3) # (256, 16, 16)
        x_resdown5 = self.resDown5(x_resdown4) # (256, 8, 8)

        x_lat5 = self.latres5(x_resdown5) # (256, 8, 8)
        x_lat4 = self.latres4(x_resdown4) # (256, 16, 16)
        x_lat3 = self.latres3(x_resdown3) # (256, 32, 32)
        x_lat2 = self.latres2(x_resdown2) # (256, 64, 64)
        x_lat1 = self.latres1(x_resdown1) # (256, 128, 128)

        # first coarse 
        x5_out = x_lat5 # (256, 8, 8)
        # coarse + residual 
        x4_out = self.upsample(x_lat5) + x_lat4 # (256, 16, 16)
        x3_out = self.upsample(x_lat4) + x_lat3 # (256, 32, 32)
        x2_out = self.upsample(x_lat3) + x_lat2 # (256, 64, 64)
        x1_out = self.upsample(x_lat2) + x_lat1 # (256, 128, 128)

        return [x1_out, x2_out, x3_out, x4_out, x5_out]

class FlowGenerator(nn.Module):
    '''
    F_list = G(S_list, T_list)
    '''
    def __init__(self, mode='bilinear'):
        
        super(FlowGenerator, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode=mode)

        self.E5 = nn.Conv2d(512,2,3,1,1)
        self.E4 = nn.Conv2d(512,2,3,1,1)
        self.E3 = nn.Conv2d(512,2,3,1,1)
        self.E2 = nn.Conv2d(512,2,3,1,1)
        self.E1 = nn.Conv2d(512,2,3,1,1)

    
    def forward(self, S_list, T_list):
        '''
        S_list: source features [S1,S2,S3,...,Sn], with shape (256, n, n)
        T_list: target features [T1,T2,T3,...,Tn], with shape (256, n, n)
        '''
        assert(len(S_list) == len(T_list))
        
        F5 = self.E5(torch.cat((S_list[4], T_list[4]), dim=1)) # (512,8,8) -> (2,8,8)
        UF5 = self.upsample(F5) # (2,16,16)
        F4 = UF5 + self.E4(torch.cat((warp_flow(S_list[3], UF5), T_list[3]), dim=1)) # (512,16,16) -> (2,16,16)
        UF4 = self.upsample(F4) # (2,32,32)
        F3 = UF4 + self.E3(torch.cat((warp_flow(S_list[2], UF4), T_list[2]), dim=1)) # (512,32,32) -> (2,32,32)
        UF3 = self.upsample(F3) # (2,64,64)
        F2 = UF3 + self.E2(torch.cat((warp_flow(S_list[1], UF3), T_list[1]), dim=1)) # (512,64,64) -> (2,64,64)
        UF2 = self.upsample(F2) # (2,128,128)
        F1 = UF2 + self.E1(torch.cat((warp_flow(S_list[0], UF2), T_list[0]), dim=1)) # (512,128,128) -> (2,128,128)
        UF1 = self.upsample(F1) # (2,256,256)

        return [UF1, UF2, UF3, UF4, UF5]

class AttentionGenerator(nn.Module):
    def __init__(self, inc=40):
        super(FlowGeAttentionGeneratornerator, self).__init__()

        self.resDown1 = ResBlockDown(inc, 64) # (64, 128, 128)
        self.resDown2 = ResBlockDown(64, 128) # (128, 64, 64)
        self.resDown3 = ResBlockDown(128, 256) # (256, 32, 32)
        self.self_att_down = SelfAttention(256) #
        self.resDown4 = ResBlockDown(256, 256) # (256, 16, 16)

        # self.res1 = ResBlock(256)
        # self.res2 = ResBlock(256)

        self.resUp1 = ResBlockUp(256, 256) # (256, 32, 32)
        self.resUp2 = ResBlockUp(256, 128) # (128, 64, 64)
        self.self_att_up = SelfAttention(128) #
        self.resUp3 = ResBlockUp(128, 64) # (64, 128, 128)
        self.resUp4 = ResBlockUp(64, 32) # (32, 256, 256)

        self.attention_module = nn.utils.spectral_norm(nn.Conv2d(32, 1, 3, padding = 1)) # (1, 256, 256)
  
    
    def forward(self, image1, pose1, pose2):
        
        x_in = torch.cat((image1, pose1, pose2), dim=1) # (43, 256, 256)
        x_resdown1 = self.resDown1(x_in) # (64, 128, 128)
        x_resdown2 = self.resDown2(x_resdown1) # (128, 64, 64)
        x_resdown3 = self.resDown3(x_resdown2) # (256, 32, 32)
        x_resdown4 = self.resDown4(x_resdown3) # (256, 16, 16)

        x_att_down = self.self_att_down(x_resdown4) #
        
        # x_res1 = self.res1(x_att_down)
        # x_res2 = self.res2(x_res1)

        x_resup1 = self.resUp1(x_att_down) # (256, 32, 32)
        x_resup2 = self.resUp2(x_resup1) # (128, 64, 64)

        x_att_up = self.self_att_up(x_resup2)
        x_resup3 = self.resUp3(x_att_up) # (64, 128, 128)
        x_resup4 = self.resUp4(x_resup3) # (32, 256, 256)

        attn = nn.Sigmoid(self.attention_module(x_resup4)) # (1, 256, 256)

        return attn






