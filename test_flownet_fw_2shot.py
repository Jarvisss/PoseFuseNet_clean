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
from util.vis_util import visualize_feature
from util import openpose_utils
from util.io import load_image, load_skeleton, transform_image

from dataset.fashionvideo_dataset import FashionVideoDataset
from loss.loss_generator import PerceptualCorrectness, LossG

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
from skimage.transform import resize
import torch.nn.functional as F

import random
from PIL import Image
import json
import argparse

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


norm_type = 'in' # 'bn' | 'in'
use_spectral = False
use_mask = True
soft_mask = True

n_enc_dec_layers = 2
n_bottleneck_layers= 2

# experiment_name = 'FuseG_v5_oneshot_{0}_spec_{1}_1e4_enc_{2}_btn_{3}-1004'.format(norm_type, use_spectral, n_enc_dec_layers, n_bottleneck_layers)
experiment_name = 'FuseG_v7_2shot_mask_{0}_soft_{1}_enc_{2}_btn_{3}_lr1e4-1007'.format(use_mask, soft_mask,n_enc_dec_layers, n_bottleneck_layers)
# experiment_name = 'FuseG_v7_iper_2shot_mask_{0}_soft_{1}_enc_{2}_btn_{3}_lr1e4-1007'.format(use_mask, soft_mask,n_enc_dec_layers, n_bottleneck_layers)


test_datset = 'danceFashion'
# test_target = 'A1-cVlkGwjS'
# test_target = 'A1wrrhGRZmS'
# test_target = 'A1dLq8J8cjS'
# test_target = 'A14oLiUg7CS'
test_target = 'A15Ei5ve9BS'

# test_datset = 'iPER'
# test_target = '017_1_2'

test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/'.format(experiment_name,test_datset, test_target)
test_video_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/'.format(experiment_name,test_datset)

if not os.path.isdir(test_result_dir):
    os.makedirs(test_result_dir)

path_to_ckpt_dir = '/home/ljw/playground/poseFuseNet/checkpoints/{0}/'.format(experiment_name)

path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
path_to_backup = path_to_ckpt_dir + 'backup_model_weights.tar'
K = 3

pose_normalize = True
scale_pose = 255 if pose_normalize else 1

is_clean_pose = False

path_to_test_A = '/dataset/ljw/danceFashion/test_256/train_A/'

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

# test videos
# video_names = os.listdir(path_to_test_A)
video_names = [test_target]

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
lossesG = checkpoint['lossesG']
num_vid = checkpoint['num_vid']

print('current experiment name:', experiment_name)
print('current epoch:', epochCurrent)

parser = argparse.ArgumentParser()
parser.add_argument('--ref1', default=0, type=int, help='ref1 image id')
parser.add_argument('--ref2',default=158, type=int, help='ref2 image id')
args = parser.parse_args()


""" Test start """
with torch.no_grad():
    
    for video_name in video_names:

        print(video_name)
        path_to_test_imgs = os.path.join(path_to_test_A, video_name)
        path_to_test_img_kps = os.path.join(path_to_test_kps, video_name)
        

        ref1_id = args.ref1
        ref1_name = '{:05d}'.format(ref1_id)
        # gt_id = 282
        ref2_id = args.ref2
        ref2_name = '{:05d}'.format(ref2_id)
        
        test_result_vid_dir = test_result_dir + '{0}_{1}/'.format(ref1_name, ref2_name)
        if not os.path.isdir(test_result_vid_dir):
            os.makedirs(test_result_vid_dir)

        total_ids = len(os.listdir(path_to_test_imgs))
        assert(ref1_id <= total_ids)
        assert(ref2_id <= total_ids)
        for gt_id in tqdm(range(5, total_ids, 5)):
            gt_name = '{:05d}'.format(gt_id)
            ref_x2 = load_image(os.path.join(path_to_test_imgs, ref2_name+'.png')).unsqueeze(0).to(device)
            ref_y2 = load_skeleton(os.path.join(path_to_test_img_kps, ref2_name+'.json')).unsqueeze(0).to(device)

            ref_x1 = load_image(os.path.join(path_to_test_imgs, ref1_name+'.png')).unsqueeze(0).to(device)
            g_x = load_image(os.path.join(path_to_test_imgs, gt_name+'.png')).unsqueeze(0).to(device)
            ref_y1 = load_skeleton(os.path.join(path_to_test_img_kps, ref1_name+'.json')).unsqueeze(0).to(device)
            g_y = load_skeleton(os.path.join(path_to_test_img_kps, gt_name+'.json')).unsqueeze(0).to(device)
            # print(g_y.shape, e_hat.shape)
            flow_1, mask_1 = GF(ref_x1, ref_y1, g_y)
            flow_2, mask_2 = GF(ref_x2, ref_y2, g_y)

            mask = F.softmax(torch.cat((mask_1, mask_2), dim=1), dim=1) # pixel wise sum to 1
            xf1 = GE(ref_x1) #(B,256,32,32)
            xf2 = GE(ref_x2) #(B,256,32,32)
            
            if flow_1 is not None:
                if not xf1.shape[2:] == flow_1.shape[2:]:
                    flow1 = F.interpolate(flow_1, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)
                    flow2 = F.interpolate(flow_2, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)
                    # flow2 = F.interpolate(flow_2, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)

            if mask_1 is not None:
                if not xf1.shape[2:] == mask_1.shape[2:]:
                    mask1 = F.interpolate(mask[:,0:1,...], size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
                    mask2 = F.interpolate(mask[:,1:2,...], size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
            xf1_warped = warp_flow(xf1, flow1) #(B,256,32,32)
            xf2_warped = warp_flow(xf2, flow2) #(B,256,32,32)
            
            if use_mask:
                if soft_mask:
                    xf1_warped_masked = xf1_warped * mask1 #(B,256,32,32)
                    xf2_warped_masked = xf2_warped * mask2 #(B,256,32,32)
                else:
                    xf1_warped_masked = warp_flow(xf1, flow1, mask=mask1) #(B,256,32,32)
                    xf2_warped_masked = warp_flow(xf2, flow2, mask=mask2) #(B,256,32,32)

                xf_merge = xf1_warped_masked + xf2_warped_masked
            else:
                xf1_warped_masked = xf1_warped
                xf2_warped_masked = xf2_warped
                xf_merge = (xf1_warped_masked + xf2_warped_masked)/2

            x_hat = GD(xf_merge)
            
            
            DISPLAY_BATCH = 0
                

            ref_x1 = ref_x1[DISPLAY_BATCH].permute(1,2,0) * 255.0
            ref_x2 = ref_x2[DISPLAY_BATCH].permute(1,2,0) * 255.0
            g_x = g_x[DISPLAY_BATCH].permute(1,2,0) * 255.0

            ref_y1 = ref_y1[DISPLAY_BATCH][17:20,...].permute(1,2,0)*scale_pose
            ref_y2 = ref_y2[DISPLAY_BATCH][17:20,...].permute(1,2,0)*scale_pose
            g_y = g_y[DISPLAY_BATCH][17:20,...].permute(1,2,0)*scale_pose

            flow_1 = flow_1[DISPLAY_BATCH].permute(1,2,0)
            flowimg1 = torch.from_numpy(flow2img(flow_1.detach().to(cpu).numpy())).to(device).float()

            flow_2 = flow_2[DISPLAY_BATCH].permute(1,2,0)
            flowimg2 = torch.from_numpy(flow2img(flow_2.detach().to(cpu).numpy())).to(device).float()

            mask_1 = mask[DISPLAY_BATCH,0:1,...].permute(1,2,0)
            maskimg_1 = torch.cat((mask_1,)*3,dim=2) * 255.0

            mask_2 = mask[DISPLAY_BATCH,1:2,...].permute(1,2,0)
            maskimg_2 = torch.cat((mask_2,)*3,dim=2) * 255.0
            
            out = x_hat[DISPLAY_BATCH].permute(1,2,0) * 255.0 # (256,256,3)
            white = out.new_tensor(torch.ones(out.size()) * 255.0)
            img_shape = out.shape[0:2]
            

            visualize_f1 = visualize_feature(xf1, DISPLAY_BATCH, out_shape=img_shape)
            visualize_f1_warp = visualize_feature(xf1_warped, DISPLAY_BATCH, out_shape=img_shape)
            visualize_f1_warp_masked = visualize_feature(xf1_warped_masked, DISPLAY_BATCH, out_shape=img_shape)
            
            visualize_f2 = visualize_feature(xf2, DISPLAY_BATCH, out_shape=img_shape)
            visualize_f2_warp = visualize_feature(xf2_warped, DISPLAY_BATCH, out_shape=img_shape)
            visualize_f2_warp_masked = visualize_feature(xf2_warped_masked, DISPLAY_BATCH, out_shape=img_shape)

            merged_f = visualize_feature(xf_merge, DISPLAY_BATCH, out_shape=img_shape)

            '''rearrange to display'''
            ref1 = torch.cat((ref_x1, ref_y1, visualize_f1), dim =0)
            ref2 = torch.cat((ref_x2, ref_y2, visualize_f2), dim =0)

            feat1 = torch.cat((visualize_f1_warp, maskimg_1, visualize_f1_warp_masked), dim=0)
            feat2 = torch.cat((visualize_f2_warp, maskimg_2, visualize_f2_warp_masked), dim=0)

            out_col = torch.cat((out, g_y, merged_f),dim=0)
            gt = torch.cat((g_x, g_y, white), dim=0)

            final_img = torch.cat((ref1, ref2, feat1, feat2, out_col,gt), dim=1)

            final_img = final_img.type(torch.uint8).to(cpu).numpy()
            # out = np.transpose(out, (1, 0, 2))
            out = out.type(torch.uint8).to(cpu).numpy()
            gtruth = g_x.type(torch.uint8).to(cpu).numpy()
            ref1 = ref_x1.type(torch.uint8).to(cpu).numpy()
            ref2 = ref_x2.type(torch.uint8).to(cpu).numpy()

            
            # test_result_vid_dir = os.path.join(test_result_dir,video_name)

            plt.imsave(test_result_vid_dir+"{2}_result.png".format(ref1_name, ref2_name, gt_name), final_img)
            # plt.imsave(test_result_vid_dir+"{0}_{1}_{2}_out.png".format(ref2_name, ref1_name, gt_name), out)
            # plt.imsave(test_result_vid_dir+"{0}_{1}_{2}_gt.png".format(ref2_name, ref1_name, gt_name), gtruth)
            # plt.imsave(test_result_vid_dir+"{0}_{1}_{2}_ref1.png".format(ref2_name, ref1_name, gt_name), ref1)
            # plt.imsave(test_result_vid_dir+"{0}_{1}_{2}_ref2.png".format(ref2_name, ref1_name, gt_name), ref2)

            out2 = np.concatenate((ref1, ref2, out, gtruth), axis=1)

            plt.imsave(test_result_vid_dir+"{2}_result_simp.png".format(ref1_name, ref2_name, gt_name), out2)

        save_video_name_simp = '{0}_{1}_simple_result.mp4'.format(ref1_name, ref2_name)
        save_video_name = '{0}_{1}_result.mp4'.format(ref1_name, ref2_name)
        img_dir = test_result_vid_dir
        save_video_dir = test_result_dir

        imgs = os.listdir(img_dir)
        import cv2
        video_out_simp = cv2.VideoWriter(save_video_dir+save_video_name_simp, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (256*4, 256))
        video_out = cv2.VideoWriter(save_video_dir+save_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (256*6, 256*3))
        for img in tqdm(sorted(imgs)):
            if img.split('.')[0].split('_')[-1] == 'simp':
                frame = cv2.imread(os.path.join(img_dir, img))
                video_out_simp.write(frame)
            elif img.split('.')[0].split('_')[-1] == 'result':
                frame = cv2.imread(os.path.join(img_dir, img))
                video_out.write(frame)

        video_out_simp.release()
        video_out.release()





