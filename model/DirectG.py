import torch.nn as nn
import torch
from .blocks import ResBlockDown, SelfAttention, ResBlock, ResBlockUp, adaIN
import math

class DirectGenerator(nn.Module):
    P_LEN = 2*(512*2*5 + 512+256 + 256+128 + 128+64 + 64+32 + 32)
    slice_idx = [0,
                512*4, #res1
                512*4, #res2
                512*4, #res3
                512*4, #res4
                512*4, #res5 ## not used
                512*2 + 256*2, #resUp1
                256*2 + 128*2, #resUp2
                128*2 + 64*2, #resUp3
                64*2 + 32*2, #resUp4
                32*2] #last adain
    for i in range(1, len(slice_idx)):
        slice_idx[i] = slice_idx[i-1] + slice_idx[i]
    
    # def __init__(self, in_height, finetuning=False, e_finetuning=None):
    def __init__(self, inc=20, out_size=256):
        super(DirectGenerator, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(inplace = False)
        
        #in 3*224*224 for voxceleb2
        # self.pad = Padding(in_height) #out 3*256*256
        
        #Down
        self.resDown1 = ResBlockDown(inc, 64, conv_size=9, padding_size=4) #out 64*128*128
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.in2 = nn.InstanceNorm2d(128, affine=True)
        
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.in3 = nn.InstanceNorm2d(256, affine=True)
        
        self.self_att_Down = SelfAttention(256) #out 256*32*32
        
        self.resDown4 = ResBlockDown(256, 512) #out 512*16*16
        self.in4 = nn.InstanceNorm2d(512, affine=True)
        
        #Res
        #in 512*16*16
        self.res1 = ResBlock(512)
        self.res2 = ResBlock(512)
        self.res3 = ResBlock(512)
        self.res4 = ResBlock(512)
        #out 512*16*16
        
        #Up
        #in 512*16*16
        self.resUp1 = ResBlockUp(512, 256) #out 256*32*32
        self.resUp2 = ResBlockUp(256, 128) #out 128*64*64
        
        self.self_att_Up = SelfAttention(128) #out 128*64*64

        self.resUp3 = ResBlockUp(128, 64) #out 64*128*128
        self.resUp4 = ResBlockUp(64, 32, out_size=(out_size, out_size), scale=None, conv_size=3, padding_size=1) #out 3*224*224
        self.conv2d = nn.Conv2d(32, 3, 3, padding = 1)
        
        self.p = nn.Parameter(torch.rand(self.P_LEN,512).normal_(0.0,0.02))
        

        # self.finetuning = finetuning
        # self.psi = nn.Parameter(torch.rand(self.P_LEN,1))
        # self.e_finetuning = e_finetuning
        
    # def finetuning_init(self):
    #     if self.finetuning:
    #         self.psi = nn.Parameter(torch.mm(self.p, self.e_finetuning.mean(dim=0)))
            
    def forward(self, y, e):
        if math.isnan(self.p[0,0]):
            sys.exit()
        
        # if self.finetuning:
        #     e_psi = self.psi.unsqueeze(0)
        #     e_psi = e_psi.expand(e.shape[0],self.P_LEN,1)
        # else:
        p = self.p.unsqueeze(0)
        p = p.expand(e.shape[0],self.P_LEN,512)
        e_psi = torch.bmm(p, e) #B, p_len, 1
        
        #in 3*224*224 for voxceleb2
        # out = self.pad(y)
        
        #Encoding
        out = self.resDown1(y)
        out = self.in1(out)
        
        out = self.resDown2(out)
        out = self.in2(out)
        
        out = self.resDown3(out)
        out = self.in3(out)
        
        out = self.self_att_Down(out)
        
        out = self.resDown4(out)
        out = self.in4(out)
        
        
        #Residual
        out = self.res1(out, e_psi[:, self.slice_idx[0]:self.slice_idx[1], :])
        out = self.res2(out, e_psi[:, self.slice_idx[1]:self.slice_idx[2], :])
        out = self.res3(out, e_psi[:, self.slice_idx[2]:self.slice_idx[3], :])
        out = self.res4(out, e_psi[:, self.slice_idx[3]:self.slice_idx[4], :])
        # out = self.res5(out, e_psi[:, self.slice_idx[4]:self.slice_idx[5], :])
        
        
        #Decoding
        out = self.resUp1(out, e_psi[:, self.slice_idx[5]:self.slice_idx[6], :])
        
        out = self.resUp2(out, e_psi[:, self.slice_idx[6]:self.slice_idx[7], :])
        
        out = self.self_att_Up(out)

        out = self.resUp3(out, e_psi[:, self.slice_idx[7]:self.slice_idx[8], :])
        
        out = self.resUp4(out, e_psi[:, self.slice_idx[8]:self.slice_idx[9], :])
        
        out = adaIN(out,
                    e_psi[:,
                          self.slice_idx[9]:(self.slice_idx[10]+self.slice_idx[9])//2,
                          :],
                    e_psi[:,
                          (self.slice_idx[10]+self.slice_idx[9])//2:self.slice_idx[10],
                          :]
                   )
        
        out = self.relu(out)
        
        out = self.conv2d(out)
        
        out = self.sigmoid(out)
        
        #out = out*255
        self.img_gen = out
        #out 3*224*224
        return out

    
