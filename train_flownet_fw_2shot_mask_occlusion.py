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
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
# default for cuda:0 to use gpu
device = torch.device("cuda:0") 
cpu = torch.device("cpu")
batch_size = 2


K = 3

n_enc_dec_layers = 2
n_bottleneck_layers = 2

norm_type = 'in' # 'bn' | 'in'
use_spectral = False
use_mask = True
soft_mask = True
use_attention = True

# experiment_name = 'FuseG_v1_wosim-0921'
# experiment_name = 'FuseG_v3_2shot_wosim_feature_warp-0927'
# experiment_name = 'FuseG_v3_2shot_wosim_feature_warp_no_spec_instance_norm_oneshot_outsigmoid-0927'
# experiment_name = 'FuseG_v3_1shot_wosim_feature_warp_no_spec_instance_norm_sigmoid-0930'
# experiment_name = 'FuseG_v3_1shot_wosim_feature_warp_spec_instance_norm_sigmoid-0930'
# experiment_name = 'FuseG_v3_1shot_featurewarp_no_spec_in_sigmoid_lr1e4-0930'

# experiment_name = 'FuseG_v4_oneshot_in_no_spec_1e4_enc_2_btn_3-1003'
# experiment_name = 'FuseG_v4_oneshot_in_no_spec_1e4_enc_2_btn_3-1003'
# experiment_name = 'FuseG_v5_oneshot_{0}_spec_{1}_1e4_enc_{2}_btn_{3}-1004'.format(norm_type, use_spectral, n_enc_dec_layers, n_bottleneck_layers)

# start from v6, we default use spectral=False, and norm_type=in
# experiment_name = 'FuseG_v6_oneshot_mask_{0}_soft_{1}_enc_{2}_btn_{3}_lr1e4-1007'.format(use_mask, soft_mask,n_enc_dec_layers, n_bottleneck_layers)
# experiment_name = 'FuseG_v7_2shot_mask_{0}_soft_{1}_enc_{2}_btn_{3}_lr1e4-1007'.format(use_mask, soft_mask,n_enc_dec_layers, n_bottleneck_layers)
experiment_name = 'FuseG_v8_2shot_mask_mask_{0}_soft_{1}_enc_{2}_btn_{3}_lr1e4-1007'.format(use_mask, soft_mask,n_enc_dec_layers, n_bottleneck_layers)


visualize_result_dir = '/home/ljw/playground/poseFuseNet/visualize_result/{0}/'.format(experiment_name)
path_to_ckpt_dir = '/home/ljw/playground/poseFuseNet/checkpoints/{0}/'.format(experiment_name)
path_to_log_dir = '/home/ljw/playground/poseFuseNet/logs/{0}'.format(experiment_name)

path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
path_to_backup = path_to_ckpt_dir + 'backup_model_weights.tar'


lr = 1e-4
lambda_style = 500.0
lambda_content = 0.5
lambda_rec = 5.0
pose_normalize = True
scale_pose = 255 if pose_normalize else 1
forget_threshold = 0.1


path_to_train_A = '/dataset/ljw/danceFashion/train_256/train_A/'

is_clean_pose = False
path_to_train_kps = '/dataset/ljw/danceFashion/train_256/train_alphapose/'
if is_clean_pose:
    path_to_train_kps = '/dataset/ljw/danceFashion/train_256/train_video2d/'



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

GF = nn.DataParallel(FlowGenerator(inc=43, norm_type=norm_type, use_spectral_norm=use_spectral).to(device)) # dx + dx + dy = 3 + 20 + 20
GE = nn.DataParallel(AppearanceEncoder(n_layers=n_enc_dec_layers, inc=3, use_spectral_norm=use_spectral).to(device)) # dx = 3
GD = nn.DataParallel(AppearanceDecoder(n_bottleneck_layers=n_bottleneck_layers, n_decode_layers=n_enc_dec_layers, norm_type=norm_type, use_spectral_norm=use_spectral).to(device)) # df = 256
# GA = nn.DataParallel(AttentionGenerator(inc=40).to(device))


def writer_create(path_to_log_dir):
    TIMESTAMP = "/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(path_to_log_dir+TIMESTAMP)
    return writer

def init_weights(m, init_type='xavier'):
    if type(m) == nn.Conv2d:
        if init_type=='xavier':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='normal':
            torch.nn.init.normal_(m.weight)
        elif init_type=='kaiming':
            torch.nn.init.kaiming_normal_(m.weight)


def init_model(path_to_chkpt, GE, GF, GD, optimizerG):

    GF.apply(init_weights)
    GE.apply(init_weights)
    GD.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': 0,
            'lossesG': [],
            'GE_state_dict': GE.module.state_dict(),
            'GF_state_dict': GF.module.state_dict(),
            'GD_state_dict': GD.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': 0,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)
    print('...Done')

def train():
    GF.train()
    GE.train()
    GD.train()

    optimizerG = optim.Adam(params = list(GF.parameters()) + list(GE.parameters()) + list(GD.parameters()) ,
                            lr=lr,
                            amsgrad=False)

    if not os.path.isfile(path_to_chkpt):
        """initiate checkpoint if inexistant"""
        init_model(path_to_chkpt, GE, GF, GD, optimizerG)

    
    criterionG = LossG(device=device)
    # criterionGF = PerceptualCorrectness()

    matplotlib.use('agg') 


    """Training init"""
    num_epochs = 15000
    save_freq = 5 # every 5 epochs, save one result 
    


    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    GE.module.load_state_dict(checkpoint['GE_state_dict'], strict=False)
    GD.module.load_state_dict(checkpoint['GD_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    num_vid = checkpoint['num_vid']
    i_batch_current = checkpoint['i_batch']
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    i_batch_total = epochCurrent * fashionVideoDataLoader.__len__() // batch_size + i_batch_current

    """
    create tensorboard writter
    """
    
    writer = writer_create(path_to_log_dir)

    pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
    save_freq = 1
    """ Training start """
    for epoch in range(epochCurrent, num_epochs):
        if epoch >= 100:
            save_freq=5
        elif epoch >= 500:
            save_freq=10
        if epoch > epochCurrent:
            i_batch_current = 0
            pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
        
        pbar.set_postfix(epoch=epoch)
        for i_batch, (ref_xs, ref_ys, g_x, g_y, idx) in enumerate(pbar, start=0):


            ref_xs = ref_xs.to(device) # [B, 2, 3, 256, 256]
            ref_ys = ref_ys.to(device) # [B, 2, 20, 256, 256]
            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]

            optimizerG.zero_grad()

            # print(g_y.shape, e_hat.shape)
            flow_1, mask_1 = GF(ref_xs[:,0,...], ref_ys[:,0,...], g_y)
            flow_2, mask_2 = GF(ref_xs[:,1,...], ref_ys[:,1,...], g_y)
            

            mask = torch.cat((mask_1, mask_2), dim=1) # [B, 2, 256, 256]

            ####### drop away the part of mask where all channels smaller than threshold
            m = mask >= forget_threshold # 2 channel bool
            '''No gradient'''
            mm = mask_1.new_tensor( m[:,0:1,...] | m[:,1:2,...] ) # or operation makes all zeros zero, otherwise 1 
            softmax_mask = torch.softmax(mask, dim=1)
            softmax_mask_thresd = softmax_mask * mm


            xf1 = GE(ref_xs[:,0,...]) #(B,256,32,32)
            xf2 = GE(ref_xs[:,1,...]) #(B,256,32,32)
            if flow_1 is not None:
                if not xf1.shape[2:] == flow_1.shape[2:]:
                    flow1 = F.interpolate(flow_1, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)
                    flow2 = F.interpolate(flow_2, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)
                    # flow2 = F.interpolate(flow_2, size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,2,32,32)

            if mask_1 is not None:
                if not xf1.shape[2:] == mask_1.shape[2:]:
                    mask1 = F.interpolate(softmax_mask_thresd[:,0:1,...], size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
                    mask2 = F.interpolate(softmax_mask_thresd[:,1:2,...], size=xf1.shape[2:], mode='bilinear',align_corners=False) #(B,1,32,32)
            
            
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
            'epoch': epoch+1,
            'lossesG': lossesG,
            'GE_state_dict': GE.module.state_dict(),
            'GF_state_dict': GF.module.state_dict(),
            'GD_state_dict': GD.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)

        if epoch % save_freq == 0:
            '''display items'''
            DISPLAY_BATCH = 0
            

            ref_x1 = ref_xs[DISPLAY_BATCH][0].permute(1,2,0) * 255.0
            ref_x2 = ref_xs[DISPLAY_BATCH][1].permute(1,2,0) * 255.0
            g_x = g_x[DISPLAY_BATCH].permute(1,2,0) * 255.0

            ref_y1 = ref_ys[DISPLAY_BATCH][0][17:20,...].permute(1,2,0)*scale_pose
            ref_y2 = ref_ys[DISPLAY_BATCH][1][17:20,...].permute(1,2,0)*scale_pose
            g_y = g_y[DISPLAY_BATCH][17:20,...].permute(1,2,0)*scale_pose

            flow_1 = flow_1[DISPLAY_BATCH].permute(1,2,0)
            flowimg1 = torch.from_numpy(flow2img(flow_1.detach().to(cpu).numpy())).to(device).float()

            flow_2 = flow_2[DISPLAY_BATCH].permute(1,2,0)
            flowimg2 = torch.from_numpy(flow2img(flow_2.detach().to(cpu).numpy())).to(device).float()

            mask_1 = softmax_mask_thresd[:,0:1,...][DISPLAY_BATCH].permute(1,2,0)
            maskimg_1 = torch.cat((mask_1,)*3,dim=2) * 255.0

            mask_2 = softmax_mask_thresd[:,1:2,...][DISPLAY_BATCH].permute(1,2,0)
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
            plt.imsave(visualize_result_dir+"epoch_{}_batch_{}.png".format(epoch, i_batch), final_img)

    writer.close()


if __name__ == "__main__":
    train()

