import torch
import torch.nn as nn
class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
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

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))

def make_grid(x):
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

x = torch.ones((2,3,4,4))

grid = make_grid(x)
print(grid)
gic = GicLoss()
print(gic(grid))
