import torch.nn as nn
import torch
from .blocks import ResBlockDown, SelfAttention

    
class DirectEmbedder(nn.Module):
    def __init__(self, inc=23):
        super(DirectEmbedder, self).__init__()
        
        self.relu = nn.LeakyReLU(inplace=False)
        
        #in 23*256*256
        # self.pad = Padding(in_height) #out 6*256*256
        self.resDown1 = ResBlockDown(inc, 64) #out 64*128*128
        self.resDown2 = ResBlockDown(64, 128) #out 128*64*64
        self.resDown3 = ResBlockDown(128, 256) #out 256*32*32
        self.self_att = SelfAttention(256) #out 256*32*32
        self.resDown4 = ResBlockDown(256, 512) #out 515*16*16
        self.resDown5 = ResBlockDown(512, 512) #out 512*8*8
        self.resDown6 = ResBlockDown(512, 512) #out 512*4*4
        # self.sum_pooling = nn.AdaptiveAvgPool2d((1,1)) #out 512*1*1
        self.avg_pooling = nn.AvgPool2d(kernel_size=(4,4)) #out 512*1*1

    def forward(self, x, y):
        # x 3*256*256
        # y (17+3)*256*256,
        out = torch.cat((x,y),dim = -3) #out 23*256*256
        # out = self.pad(out) #out 6*256*256
        out = self.resDown1(out) #out 64*128*128
        out = self.resDown2(out) #out 128*64*64
        out = self.resDown3(out) #out 256*32*32
        
        out = self.self_att(out) #out 256*32*32
        
        out = self.resDown4(out) #out 512*16*16
        out = self.resDown5(out) #out 512*8*8
        out = self.resDown6(out) #out 512*4*4
        
        out = self.avg_pooling(out) #out 512*1*1
        out = out * 4 * 4 # perform average pooling first and multiply by kernel size to perform sum pooling

        out = self.relu(out) #out 512*1*1
        out = out.view(-1,512,1) #out B*512*1
        return out
