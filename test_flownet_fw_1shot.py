import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
from PIL import Image
import json
# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
from model.flow_generator import FlowGenerator, AppearanceEncoder, AppearanceDecoder
from model.blocks import warp_flow
from util.flow_utils import flow2img
from util import openpose_utils

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
# default for cuda:0 to use gpu
device = torch.device("cuda:0") 
cpu = torch.device("cpu")

# experiment_name = 'FuseG_v1_wosim-0921'
norm_type = 'in' # 'bn' | 'in'
use_spectral = False
use_mask = False
soft_mask = False

n_enc_dec_layers = 2
n_bottleneck_layers= 4

experiment_name = 'FuseG_v5_oneshot_{0}_spec_{1}_1e4_enc_{2}_btn_{3}-1004'.format(norm_type, use_spectral, n_enc_dec_layers, n_bottleneck_layers)
# experiment_name = 'FuseG_v6_oneshot_mask_{0}_soft_{1}_enc_{2}_btn_{3}_lr1e4-1007'.format(use_mask, soft_mask,n_enc_dec_layers, n_bottleneck_layers)

# test_datset = 'iPER'
test_datset = 'danceFashion'
test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/'.format(experiment_name,test_datset)

if not os.path.isdir(test_result_dir):
    os.makedirs(test_result_dir)

path_to_ckpt_dir = '/home/ljw/playground/poseFuseNet/checkpoints/{0}/'.format(experiment_name)
path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
path_to_backup = path_to_ckpt_dir + 'backup_model_weights.tar'

batch_size = 1
K = 2
pose_normalize = True
scale_pose = 255 if pose_normalize else 1

is_clean_pose = False

if test_datset == 'danceFashion':
    path_to_test_A = '/dataset/ljw/danceFashion/test_256/train_A/'
    path_to_test_kps = '/dataset/ljw/danceFashion/test_256/train_alphapose/'
    if is_clean_pose:
        path_to_test_kps = '/dataset/ljw/danceFashion/test_256/train_video2d/'
else: # iPER
    path_to_test_A = '/dataset/ljw/iper/test_256/train_A/'
    path_to_test_kps = '/dataset/ljw/iper/test_256/train_alphapose/'
    if is_clean_pose:
        path_to_test_kps = '/dataset/ljw/iper/test_256/train_video2d/'

video_names = os.listdir(path_to_test_A)

# video_name = '91+xeI+ijRS' # train

GF = nn.DataParallel(FlowGenerator(inc=43, norm_type=norm_type, use_spectral_norm=use_spectral).to(device)) # dx + dx + dy = 3 + 20 + 20
GE = nn.DataParallel(AppearanceEncoder(n_layers=n_enc_dec_layers, inc=3, use_spectral_norm=use_spectral).to(device)) # dx = 3
GD = nn.DataParallel(AppearanceDecoder(n_bottleneck_layers=n_bottleneck_layers, n_decode_layers=n_enc_dec_layers, norm_type=norm_type, use_spectral_norm=use_spectral).to(device)) # df = 256

GF.eval()
GE.eval()
GD.eval()


matplotlib.use('agg') 

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
GE.module.load_state_dict(checkpoint['GE_state_dict'], strict=False)
GD.module.load_state_dict(checkpoint['GD_state_dict'], strict=False)
epochCurrent = checkpoint['epoch']
print('current experiment name:', experiment_name)
print('current epoch:', epochCurrent)


def transform_image(image, resize_param, method=Image.BICUBIC, affine=None, normalize=False, toTensor=True, fillWhiteColor=False):
    import torchvision.transforms.functional as F
    image = F.resize(image, resize_param, interpolation=method)
    if affine is not None:
        angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
        fillcolor = None if not fillWhiteColor else (255,255,255)
        image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
    if toTensor:
        image = F.to_tensor(image)
    if normalize:
        image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    return image  

def load_image(A_path, name='fashion', load_size=(256,256)):
    A_img = Image.open(A_path).convert('RGB')

    # padding white color after affine transformation  
    fillWhiteColor = True if name =='fashion' else False
    Ai = transform_image(A_img, load_size, fillWhiteColor=fillWhiteColor)
    return Ai

def load_skeleton(B_path, load_size=(256,256), is_clean_pose=False, no_bone_RGB=False, pose_scale=255):
    B_coor = json.load(open(B_path))["people"]
    B_coor = B_coor[0]
    pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=load_size, org_size=load_size)
    pose_body = pose_dict['body']
    if not is_clean_pose:
        pose_body = openpose_utils.openpose18_to_coco17(pose_body)

    pose_numpy = openpose_utils.obtain_map(pose_body, load_size) 
    pose = np.transpose(pose_numpy,(2, 0, 1))
    pose = torch.Tensor(pose)
    Bi = pose
    if not no_bone_RGB:
        color = np.zeros(shape=load_size + (3, ), dtype=np.uint8)
        LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
        color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
        color = np.transpose(color,(2,0,1))
        color = color / pose_scale # normalize to 0-1
        color = torch.Tensor(color)
        Bi = torch.cat((Bi, color), dim=0)

    return Bi

""" Test start """
with torch.no_grad():
    for video_name in video_names:
        
        print(video_name)
        path_to_test_imgs = os.path.join(path_to_test_A, video_name)
        path_to_test_img_kps = os.path.join(path_to_test_kps, video_name)
        
        ref1_name = '00000'
        ref2_name = '00203'
        gt_name = '00282'
        
        ref_x1 = load_image(os.path.join(path_to_test_imgs, ref1_name+'.png')).unsqueeze(0).to(device)
        g_x = load_image(os.path.join(path_to_test_imgs, gt_name+'.png')).unsqueeze(0).to(device)
        
        ref_y1 = load_skeleton(os.path.join(path_to_test_img_kps, ref1_name+'.json')).unsqueeze(0).to(device)
        g_y = load_skeleton(os.path.join(path_to_test_img_kps, gt_name+'.json')).unsqueeze(0).to(device)


        # print(g_y.shape, e_hat.shape)
        flow_1, mask_1 = GF(ref_x1, ref_y1, g_y)
        xf1 = GE(ref_x1) 
        if flow_1 is not None:
            if not xf1.shape[2:] == flow_1.shape[2:]:
                flow1 = F.interpolate(flow_1, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)
        
        if mask_1 is not None:
            if not xf1.shape[2:] == mask_1.shape[2:]:
                mask1 = F.interpolate(mask_1, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
        
        xf1_warped = warp_flow(xf1, flow1) #(B,256,32,32)
        if use_mask:
            if soft_mask:
                xf1_warped_masked = xf1_warped * mask1 #(B,256,32,32)
            else:
                xf1_warped_masked = warp_flow(xf1, flow1, mask=mask1) #(B,256,32,32)
        else:
            xf1_warped_masked = xf1_warped

        x_hat = GD(xf1_warped_masked)
        # x_hat1 = warp_flow(ref_x1, flow_1)

        ref_group1 = (ref_x1[0]*255).permute(1,2,0)
        ref_pose1 = (ref_y1[0][17:20,...]*scale_pose).permute(1,2,0)

        flowimg1 = flow2img(flow_1[0].detach().permute(1,2,0).to(cpu).numpy())
        flowimg1 = torch.from_numpy(flowimg1).to(device).float()
        
        mask_1 = mask_1[0].permute(1,2,0)
        maskimg_1 = torch.cat((mask_1,)*3,dim=2) * 255.0
        # for img_no in range(1,batch_size):
            # ref_group = torch.cat((ref_group, get_group_ref_imgs(ref_xs, img_no)), dim = 0)

        
        out = (x_hat[0]*255).permute(1,2,0)
        outf1 = F.interpolate(xf1, size=out.shape[0:2], mode='bilinear',align_corners=False)[0].permute(1,2,0) * 255.0
        outf1_warp = F.interpolate(xf1_warped, size=out.shape[0:2], mode='bilinear',align_corners=False)[0].permute(1,2,0) * 255.0
        outf1_warp_masked = F.interpolate(xf1_warped_masked, size=out.shape[0:2], mode='bilinear',align_corners=False)[0].permute(1,2,0) * 255.0
        

        visualize_f1_warp = outf1_warp[...,0:3]
        visualize_f1_warp = (visualize_f1_warp - torch.min(visualize_f1_warp))* 255.0/(torch.max(visualize_f1_warp) - torch.min(visualize_f1_warp))


        visualize_f1_warp_masked = outf1_warp_masked[...,0:3]
        visualize_f1_warp_masked = (visualize_f1_warp_masked - torch.min(visualize_f1_warp_masked))* 255.0/(torch.max(visualize_f1_warp_masked) - torch.min(visualize_f1_warp_masked))


        visualize_f1 = outf1[...,0:3]
        visualize_f1 = (visualize_f1 - torch.min(visualize_f1))* 255.0/(torch.max(visualize_f1) - torch.min(visualize_f1))

        # for img_no in range(1,batch_size):
        #     out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 0)
        white = flowimg1.new_tensor(torch.ones(flowimg1.size()) * 255)
        pose = (g_y[0][17:20,...]*scale_pose).permute(1,2,0)
        # for img_no in range(1,batch_size):
        #     pose = torch.cat((pose, (g_y[img_no][17:20,...]*scale_pose).transpose(0,2)), dim = 0)

        gtruth = (g_x[0]*255).permute(1,2,0)
        # for img_no in range(1,batch_size):
        #     gtruth = torch.cat((gtruth, (g_x[img_no]*255).transpose(0,2)), dim = 0)

        ref1 = torch.cat((ref_group1, ref_pose1), dim =0)
        out1_col = torch.cat((flowimg1,visualize_f1_warp),dim=0)
        out1_mask_col = torch.cat((maskimg_1,visualize_f1_warp_masked),dim=0)
        out_col = torch.cat((out, visualize_f1),dim=0)
        gt = torch.cat((gtruth, pose), dim=0)

        if use_mask:
            full_out = torch.cat((ref1, out1_col,out1_mask_col, out_col, gt), dim=1)
        else:
            full_out = torch.cat((ref1, out1_col, out_col, gt), dim=1)

        full_out = full_out.type(torch.uint8).to(cpu).numpy()
        # out = np.transpose(out, (1, 0, 2))
        out = out.type(torch.uint8).to(cpu).numpy()
        gtruth = gtruth.type(torch.uint8).to(cpu).numpy()
        ref = ref_group1.type(torch.uint8).to(cpu).numpy()
        test_result_vid_dir = os.path.join(test_result_dir,video_name)
        # if not os.path.isdir(test_result_vid_dir):
        #     os.makedirs(test_result_vid_dir)
        plt.imsave(test_result_vid_dir+"_result.png", full_out)
        plt.imsave(test_result_vid_dir+"_out.png", out)
        plt.imsave(test_result_vid_dir+"_gt.png", gtruth)
        plt.imsave(test_result_vid_dir+"_ref.png", ref)

