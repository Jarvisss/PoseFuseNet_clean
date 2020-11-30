from model.blocks import warp_flow

# def warp_flow(source, flow, align_corners=True, mode='bilinear', mask=None, mask_value=-1):
#     '''
#     Warp a image x according to the given flow
#     Input:
#         x: (b, c, H, W)
#         flow: (b, 2, H, W) # range [w-1, h-1]
#         mask: (b, 1, H, W)
#     Ouput:
#         y: (b, c, H, W)
#     '''
#     [b, c, h, w] = source.shape
#     # mesh grid
#     x = torch.arange(w).view(1, -1).expand(h, -1).type_as(source).float() / (w-1)
#     y = torch.arange(h).view(-1, 1).expand(-1, w).type_as(source).float() / (h-1)
#     grid = torch.stack([x,y], dim=0)
#     grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

#     grid = 2*grid - 1
#     print(grid)
#     flow = 2* flow/torch.tensor([w, h]).view(1, 2, 1, 1).expand(b, -1, h, w).type_as(flow)
#     print(flow)
#     grid = (grid+flow).permute(0, 2, 3, 1)
    
#     '''grid = grid + flow # in this way flow is -1 to 1
#     '''
#     # to (b, h, w, c) for F.grid_sample
#     output = F.grid_sample(source, grid, mode=mode, padding_mode='zeros', align_corners=align_corners)

#     if mask is not None:
#         output = torch.where(mask>0.5, output, output.new_ones(1).mul_(mask_value))
#     return output
#     pass

img_path = '/dataset/ljw/deepfashion/img/MEN/Shirts_Polos/id_00003706/01_1_front.jpg'

from PIL import Image
import torchvision.transforms.functional as F
import torch
image = Image.open(img_path)
image.save('test_in.jpg')
image = F.to_tensor(image)

image = image.unsqueeze(0)
flow = torch.zeros(image.shape)[:,0:2,:,:]

flow[:,0:1,100:120,100:120] = 20

warp_image = warp_flow(image, flow)

out_img = warp_image[0].permute(1,2,0) * 255
out_np = out_img.type(torch.uint8).numpy()
Image.fromarray(out_np).save('test_out.jpg')