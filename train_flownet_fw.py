import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt

# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
from model.flow_generator import FlowGenerator, AppearanceEncoder, AppearanceDecoder
from model.blocks import warp_flow
from util.flow_utils import flow2img

from dataset.fashionvideo_dataset import FashionVideoDataset
from loss.loss_generator import PerceptualCorrectness, LossG

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
from skimage.transform import resize
import torch.nn.functional as F
import random



"""
set random seed
"""
torch.manual_seed(7) # cpu
torch.cuda.manual_seed(7) #gpu
np.random.seed(7) #numpy
random.seed(7) #random and transforms
torch.backends.cudnn.deterministic=True

"""Training Parameters"""
# cuda visible devices, related to batchsize, defalut the batchsize should be 2 times cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
# default for cuda:0 to use gpu
device = torch.device("cuda:0") 
cpu = torch.device("cpu")

# experiment_name = 'FuseG_v1_wosim-0921'
# experiment_name = 'FuseG_v3_2shot_wosim_feature_warp-0927'
# experiment_name = 'FuseG_v3_2shot_wosim_feature_warp_no_spec_instance_norm-0927'
experiment_name = 'FuseG_v3_2shot_wosim_feature_warp_use_spec_instance_norm-0927'

visualize_result_dir = '/home/ljw/playground/poseFuseNet/visualize_result/{0}/'.format(experiment_name)
path_to_ckpt_dir = '/home/ljw/playground/poseFuseNet/checkpoints/{0}/'.format(experiment_name)
path_to_log_dir = '/home/ljw/playground/poseFuseNet/logs/{0}'.format(experiment_name)

path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
path_to_backup = path_to_ckpt_dir + 'backup_model_weights.tar'
batch_size = 2

K = 3
lambda_style = 500.0
lambda_content = 0.5
lambda_rec = 5.0
pose_normalize = True
scale_pose = 255 if pose_normalize else 1

norm_type = 'in' # 'bn' | 'in'

path_to_train_A = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/danceFashion/train_256/train_A/'

is_clean_pose = False
path_to_train_kps = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/danceFashion/train_256/train_alphapose/'
if is_clean_pose:
    path_to_train_kps = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/danceFashion/train_256/train_video2d/'



if not os.path.isdir(path_to_ckpt_dir):
    os.makedirs(path_to_ckpt_dir)
if not os.path.isdir(visualize_result_dir):
    os.makedirs(visualize_result_dir)
if not os.path.isdir(path_to_log_dir):
    os.makedirs(path_to_log_dir)

"""Create dataset and net"""

dataset = FashionVideoDataset(path_to_train_A=path_to_train_A, path_to_train_kps=path_to_train_kps, K=K, is_clean_pose=is_clean_pose, pose_scale=scale_pose)
print(dataset.__len__())

fashionVideoDataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

GF = nn.DataParallel(FlowGenerator(inc=43, norm_type=norm_type).to(device)) # dx + dx + dy = 3 + 20 + 20
GE = nn.DataParallel(AppearanceEncoder(n_layers=3, inc=3).to(device)) # dx = 3
GD = nn.DataParallel(AppearanceDecoder(n_bottleneck_layers=4, n_decode_layers=3, norm_type=norm_type).to(device)) # df = 256
# GA = nn.DataParallel(AttentionGenerator(inc=40).to(device))

GF.train()
GE.train()
GD.train()
# GA.train()

optimizerG = optim.Adam(params = list(GF.parameters()) + list(GE.parameters()) + list(GD.parameters()) ,
                        lr=1e-4,
                        amsgrad=False)

criterionG = LossG(device=device)
# criterionGF = PerceptualCorrectness()

matplotlib.use('agg') 


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
i_batch_current = 0
i_batch_total = 0
num_epochs = 15000

"""initiate checkpoint if inexistant"""
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    GF.apply(init_weights)
    GE.apply(init_weights)
    GD.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            'GE_state_dict': GE.module.state_dict(),
            'GF_state_dict': GF.module.state_dict(),
            'GD_state_dict': GD.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)
    print('...Done')


"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
GE.module.load_state_dict(checkpoint['GE_state_dict'], strict=False)
GD.module.load_state_dict(checkpoint['GD_state_dict'], strict=False)
epochCurrent = checkpoint['epoch']
lossesG = checkpoint['lossesG']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] + 1
optimizerG.load_state_dict(checkpoint['optimizerG'])

i_batch_total = epochCurrent * fashionVideoDataLoader.__len__() + i_batch_current

"""
create tensorboard writter
"""
TIMESTAMP = "/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer = SummaryWriter(path_to_log_dir+TIMESTAMP)

pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)

""" Training start """
for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
    
    pbar.set_postfix(epoch=epoch)
    for i_batch, (ref_xs, ref_ys, g_x, g_y, idx) in enumerate(pbar, start=0):


        ref_xs = ref_xs.squeeze().to(device) # [B, 2, 3, 256, 256]
        ref_ys = ref_ys.squeeze().to(device) # [B, 2, 20, 256, 256]
        g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]

        optimizerG.zero_grad()

        # print(g_y.shape, e_hat.shape)
        flow_1, mask_1 = GF(ref_xs[:,0,...], ref_ys[:,0,...], g_y)
        flow_2, mask_2 = GF(ref_xs[:,1,...], ref_ys[:,1,...], g_y)

        assert flow_1.shape == flow_2.shape #(B,2,256,256)
        assert mask_1.shape == mask_2.shape #(B,1,256,256)
        
        # (B,2,H,W)
        mask = F.softmax(torch.cat((mask_1, mask_2), dim=1), dim=1) # pixel wise sum to 1

        xf1 = GE(ref_xs[:,0,...]) #(B,256,32,32)
        xf2 = GE(ref_xs[:,1,...])

        assert xf1.shape == xf2.shape

        if mask is not None:
            if not xf1.shape[2:] == mask.shape[2:]:
                # (B,1,32,32)
                mask1 = F.interpolate(mask[:,0:1,...], size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
                mask2 = F.interpolate(mask[:,1:2,...], size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
        if flow_1 is not None:
            if not xf1.shape[2:] == flow_1.shape[2:]:
                flow1 = F.interpolate(flow_1, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)
                flow2 = F.interpolate(flow_2, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)

        xf1_warped = warp_flow(xf1, flow1) #(B,256,32,32)
        xf2_warped = warp_flow(xf2, flow2) #(B,256,32,32)
        # (B,256,32,32)
        xf = mask1.repeat(1,xf1_warped.shape[1],1,1) * xf1_warped + mask2.repeat(1,xf2_warped.shape[1],1,1) * xf2_warped
        x_hat = GD(xf)

        lossG_content, lossG_style, lossG_L1 = criterionG(g_x, x_hat)
        lossG_content = lossG_content * lambda_content
        lossG_style = lossG_style * lambda_style
        lossG_L1 = lossG_L1 * lambda_rec
        lossG = lossG_content + lossG_style + lossG_L1

        lossG.backward(retain_graph=False)
        optimizerG.step()

        writer.add_scalar('loss/lossG', lossG.item(), global_step=i_batch_total, walltime=None)
        # writer.add_scalar('loss/lossG_vggface', loss_face.item(), global_step=i_batch_total, walltime=None)
        writer.add_scalar('loss/lossG_content', lossG_content.item(), global_step=i_batch_total, walltime=None)
        writer.add_scalar('loss/lossG_style', lossG_style.item(), global_step=i_batch_total, walltime=None)
        writer.add_scalar('loss/lossG_L1', lossG_L1.item(), global_step=i_batch_total, walltime=None)
        i_batch_total += 1
        pbar.set_postfix(epoch=epoch, G_loss=lossG.item())

    lossesG.append(lossG.item())
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GE_state_dict': GE.module.state_dict(),
        'GF_state_dict': GF.module.state_dict(),
        'GD_state_dict': GD.module.state_dict(),
        'num_vid': dataset.__len__(),
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
        }, path_to_chkpt)
    # def get_group_ref_imgs(ref_xs, batch_dim):
    #     ref0 = (ref_xs[batch_dim][0]*255).transpose(0,2)
    #     ref1 = (ref_xs[batch_dim][1]*255).transpose(0,2)
    #     ref2 = (ref_xs[batch_dim][2]*255).transpose(0,2)
    #     ref3 = (ref_xs[batch_dim][3]*255).transpose(0,2)

    #     refup = torch.cat((ref0, ref1), dim=0)
    #     refdown = torch.cat((ref2, ref3), dim=0)

    #     ref_group = torch.cat((refup, refdown), dim=1)
    #     ref_group_numpy = ref_group.to(cpu).numpy()
    #     ref_group_numpy = resize(ref_group_numpy, (256,256))
    #     ref_group = torch.from_numpy(ref_group_numpy).to(device)
    #     return ref_group

    # ref_group = get_group_ref_imgs(ref_xs, batch_dim=0)
    DISPLAY_BATCH = 0

    '''display items'''
    ref_x1 = ref_xs[DISPLAY_BATCH][0].permute(1,2,0) * 255.0
    ref_x2 = ref_xs[DISPLAY_BATCH][1].permute(1,2,0) * 255.0
    g_x = g_x[DISPLAY_BATCH].permute(1,2,0) * 255.0

    ref_y1 = ref_ys[DISPLAY_BATCH][0][17:20,...].permute(1,2,0)*scale_pose
    ref_y2 = ref_ys[DISPLAY_BATCH][1][17:20,...].permute(1,2,0)*scale_pose
    g_y = g_y[DISPLAY_BATCH][17:20,...].permute(1,2,0)*scale_pose

    flow_1 = flow_1[DISPLAY_BATCH].permute(1,2,0)
    flow_2 = flow_2[DISPLAY_BATCH].permute(1,2,0)

    flowimg1 = torch.from_numpy(flow2img(flow_1.detach().to(cpu).numpy())).to(device).float()
    flowimg2 = torch.from_numpy(flow2img(flow_2.detach().to(cpu).numpy())).to(device).float()

    mask_1 = torch.cat((mask[:,0:1,...],)*3, dim=1)
    mask_2 = torch.cat((mask[:,1:2,...],)*3, dim=1)
    maskimg1 = mask_1[DISPLAY_BATCH].permute(1,2,0) * 255.0
    maskimg2 = mask_2[DISPLAY_BATCH].permute(1,2,0) * 255.0

    out = x_hat[DISPLAY_BATCH].permute(1,2,0) * 255.0 # (256,256,3)
    outf1 = F.interpolate(xf1_warped, size=out.shape[0:2], mode='bilinear',align_corners=False)[DISPLAY_BATCH].permute(1,2,0) * 255.0
    outf2 = F.interpolate(xf2_warped, size=out.shape[0:2], mode='bilinear',align_corners=False)[DISPLAY_BATCH].permute(1,2,0) * 255.0
    white = out.new_tensor(torch.ones(out.size()) * 255.0)


    '''rearrange to display'''
    ref1 = torch.cat((ref_x1, ref_y1, white), dim =0)
    ref2 = torch.cat((ref_x2, ref_y2, white), dim =0)
    gt = torch.cat((g_x, g_y, white), dim=0)
    out1_col = torch.cat((outf1[...,0:3], flowimg1, maskimg1),dim=0)
    out2_col = torch.cat((outf2[...,0:3], flowimg2, maskimg2),dim=0)
    out_col = torch.cat((out, white, white),dim=0)

    final_img = torch.cat((ref1, ref2, out1_col, out2_col, out_col, gt), dim=1)

    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    # out = np.transpose(out, (1, 0, 2))
    plt.imsave(visualize_result_dir+"epoch_{}_batch_{}.png".format(epoch, i_batch), final_img)

writer.close()


