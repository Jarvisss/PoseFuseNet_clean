"""multiview generation without direct generation branch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
matplotlib.use('agg')  # for image save not render

# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
from model.flow_generator import FlowGenerator, AppearanceEncoder, AppearanceDecoder
from model.blocks import warp_flow
from util.flow_utils import flow2img
from util.vis_util import visualize_feature, visualize_feature_group
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashionvideo_dataset import FashionVideoDataset
from loss.loss_generator import PerceptualCorrectness, LossG

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
from skimage.transform import resize
import torch.nn.functional as F
import random
import argparse
import json

device = torch.device("cuda:0") 
cpu = torch.device("cpu")


def save_parser(opt,fn):
    with open(fn, 'w') as f:
        json_obj = vars(opt)
        # json.dump(json_obj, f, separators=(',\n', ':\n'))
        json.dump(json_obj, f, indent = 0, separators = (',', ': '))

def set_random_seed(seed):
    """
    set random seed
    """
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed) #random and transforms
    torch.backends.cudnn.deterministic=True

def get_parser():
    parser = argparse.ArgumentParser()

    '''Common options'''
    parser.add_argument('--id', type=str, default='default', help = 'experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--seed', type=int, default=7, help = 'random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--K', type=int, default=2, help='source image views')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--root_dir',type=str, default='/home/ljw/playground/poseFuseNet/')
    parser.add_argument('--dataset',type=str, default='danceFashion', help='"danceFashion" or "iper"')

    '''Train options'''
    parser.add_argument('--epochs', type=int, default=2000, help='num epochs')
    parser.add_argument('--use_scheduler', action='store_true', help='open this to use learning rate scheduler')


    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')

    '''Model options'''
    parser.add_argument('--n_enc', type=int, default=2, help='encoder(decoder) layers ')
    parser.add_argument('--n_btn', type=int, default=2, help='bottle neck layers in generator')
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn"')
    parser.add_argument('--use_spectral', action='store_true', help='open this if use spectral normalization')


    '''Experiment options'''
    parser.add_argument('--no_mask', action='store_true', help='open this if do not use mask')
    parser.add_argument('--use_hard_mask', action='store_true', help='open this if want to use hard mask')
    parser.add_argument('--mask_norm_type', type=str, default='softmax', help='Normalize the masks of different views to sum 1, "divsum" or "softmax"')
    parser.add_argument('--use_mask_reg', action='store_true', help='open this if want to regularize the masks of different views to be as different as possible')
    parser.add_argument('--use_sample_correctness', action='store_true', help='open this if want to make flow learnt from each view to be correct')

    '''Loss options'''
    parser.add_argument('--lambda_style', type=float, default=500.0, help='learning rate')
    parser.add_argument('--lambda_content', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lambda_rec', type=float, default=5.0, help='learning rate')
    parser.add_argument('--lambda_correctness', type=float, default=5.0, help='learning rate')
    parser.add_argument('--lambda_regattn', type=float, default=1.0, help='learning rate')


    opt = parser.parse_args()
    return opt   


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
            'i_batch': 0,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)
    print('...Done')

def save_visualize_result(opt, ref_xs, ref_ys, gx, gy, xf_merge, x_hat, flows, masks_normed, features, features_warped, features_warped_masked, visualize_result_dir,epoch ):
    DISPLAY_BATCH = 0

    K = ref_xs.shape[1]
    assert(ref_ys.shape[1]==K)
    assert(masks_normed.shape[1]==K)
    assert(len(flows)==K)
    assert(len(features)==K)
    assert(len(features_warped)==K)
    assert(len(features_warped_masked)==K)

    g_x = gx[DISPLAY_BATCH].permute(1,2,0) * 255.0
    g_y = gy[DISPLAY_BATCH][17:20,...].permute(1,2,0)* 255.0
    out = x_hat[DISPLAY_BATCH].permute(1,2,0) * 255.0 # (256,256,3)
    white = torch.ones(out.size()).to(device) * 255.0

    masks = flows.copy()
    visualize_feat = flows.copy()
    visualize_feat_warp = flows.copy()
    ref = flows.copy()
    feat = flows.copy()
    visual_ref_xs = flows.copy()
    visual_ref_ys = flows.copy()


    img_shape = out.shape[0:2]
    features_warped_masked.append(xf_merge)
    visualize_feat_warp_masked = visualize_feature_group(features_warped_masked, DISPLAY_BATCH, out_shape=img_shape)
    visualize_feat_merged = visualize_feat_warp_masked[-1]
    visualize_feat_warp_masked = visualize_feat_warp_masked[:-1]
    for i in range(K):
        visual_ref_xs[i] = ref_xs[DISPLAY_BATCH][i].permute(1,2,0) * 255.0
        visual_ref_ys[i] = ref_ys[DISPLAY_BATCH][i][17:20].permute(1,2,0) * 255.0
        flows[i] =  torch.from_numpy(flow2img(flows[i][DISPLAY_BATCH].permute(1,2,0).detach().to(cpu).numpy())).to(device).float()
        masks[i] = masks_normed[:,i:i+1,...][DISPLAY_BATCH].permute(1,2,0)
        masks[i] = torch.cat((masks[i],)*3, dim=2) * 255.0
        visualize_feat[i] = visualize_feature(features[i], DISPLAY_BATCH, out_shape=img_shape)
        visualize_feat_warp[i] = visualize_feature(features_warped[i], DISPLAY_BATCH, out_shape=img_shape)

    '''Each col of result image'''    
    for i in range(K):
        ref[i] = torch.cat((visual_ref_xs[i], visual_ref_ys[i], visualize_feat[i]), dim=0)
        feat[i] = torch.cat((visualize_feat_warp[i], masks[i], visualize_feat_warp_masked[i]), dim=0)

    refs = torch.cat(ref, dim=1)
    feats = torch.cat(feat, dim=1)

    out_col = torch.cat((out, g_y, visualize_feat_merged),dim=0)
    gt = torch.cat((g_x, g_y, white), dim=0)

    final_img = torch.cat((refs, feats, out_col,gt), dim=1)

    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    # out = np.transpose(out, (1, 0, 2))
    plt.imsave(visualize_result_dir+"epoch_{}.png".format(epoch), final_img)

def train(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    '''Set experiment name and logging,checkpoint,vis dir'''

    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(exp_name)
    visualize_result_dir = opt.root_dir+ 'visualize_result/{0}/'.format(exp_name)
    path_to_log_dir = opt.root_dir+ 'logs/{0}'.format(exp_name)
    
    if not os.path.isdir(path_to_ckpt_dir):
        os.makedirs(path_to_ckpt_dir)
    if not os.path.isdir(visualize_result_dir):
        os.makedirs(visualize_result_dir)
    if not os.path.isdir(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
    

    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    """Create dataset and dataloader"""
    path_to_train_A = '/dataset/ljw/{0}/train_256/train_A/'.format(opt.dataset)
    path_to_train_kps = '/dataset/ljw/{0}/train_256/train_alphapose/'.format(opt.dataset)
    if opt.use_clean_pose:
        path_to_train_kps = '/dataset/ljw/{0}/train_256/train_video2d/'.format(opt.dataset)

    dataset = FashionVideoDataset(path_to_train_A=path_to_train_A, path_to_train_kps=path_to_train_kps, opt=opt)
    print(dataset.__len__())

    fashionVideoDataLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

    '''Create Model'''
    GF = nn.DataParallel(FlowGenerator(inc=43, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral).to(device)) # dx + dx + dy = 3 + 20 + 20
    GE = nn.DataParallel(AppearanceEncoder(n_layers=opt.n_enc, inc=3, use_spectral_norm=opt.use_spectral).to(device)) # dx = 3
    GD = nn.DataParallel(AppearanceDecoder(n_bottleneck_layers=opt.n_btn, n_decode_layers=opt.n_enc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral).to(device)) # df = 256

    GF.train()
    GE.train()
    GD.train()

    optimizerG = optim.Adam(params = list(GF.parameters()) + list(GE.parameters()) + list(GD.parameters()) ,
                            lr=opt.lr,
                            amsgrad=False)
    if opt.use_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizerG, 'min', factor=np.sqrt(0.1), patience=5, min_lr=5e-7)
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        init_model(path_to_chkpt, GE, GF, GD, optimizerG)


    '''Losses'''
    criterionG = LossG(device=device)
    criterionCorrectness = PerceptualCorrectness().to(device)

    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    GE.module.load_state_dict(checkpoint['GE_state_dict'], strict=False)
    GD.module.load_state_dict(checkpoint['GD_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    i_batch_current = checkpoint['i_batch']
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    i_batch_total = epochCurrent * fashionVideoDataLoader.__len__() // opt.batch_size + i_batch_current

    """
    create tensorboard writter
    """
    writer = writer_create(path_to_log_dir)
    
    save_freq = 1


    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        if epoch >= 100:
            save_freq=5
        if epoch >= 500:
            save_freq=10
        if epoch > epochCurrent:
            i_batch_current = 0
        epoch_loss_G = 0
        pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
        pbar.set_description('epoch[{0}/{1}], lr-{2}'.format(epoch,opt.epochs,optimizerG.param_groups[0]['lr']))
        for i_batch, (ref_xs, ref_ys, g_x, g_y, vid_path) in enumerate(pbar, start=0):


            ref_xs = ref_xs.to(device) # [B, 2, 3, 256, 256]
            ref_ys = ref_ys.to(device) # [B, 2, 20, 256, 256]
            assert(ref_xs.shape[1]==ref_ys.shape[1])
            assert(ref_xs.shape[1]==opt.K)
            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]

            optimizerG.zero_grad()

            '''Direct branch'''
            # [B*4, 3, 256, 256]
            ref_xs_compact = ref_xs.view(-1, ref_xs.shape[-3],ref_xs.shape[-2], ref_xs.shape[-1])
            # print(ref_xs_compact.shape)
            # [B*4, 20, 256, 256]
            ref_ys_compact = ref_ys.view(-1, ref_ys.shape[-3],ref_ys.shape[-2], ref_ys.shape[-1])
            # print(ref_ys_compact.shape)
            
            e_vectors = direct_E(ref_xs_compact, ref_ys_compact) #BK * 512 * 1
            e_vectors = e_vectors.view(-1, ref_xs.shape[1], 512, 1) # B*K*512*1

            e_hat = e_vectors.mean(dim=1) # B * 512 * 1
            # print(g_y.shape, e_hat.shape)
            x_direct = direct_G(g_y, e_hat) # (B,3,256,256)
            xf_direct = GE(x_direct) #(B,256,32,32)
            ''''''

            '''Get flows and masks and features'''
            flows, masks, xfs = None, None, None
            flows_down, masks_down, xfs_warp = None, None, None
            for k in range(0, opt.K):
                flow_k, mask_k = GF(ref_xs[:,k,...], ref_ys[:,k,...], g_y)
                xf_k = GE(ref_xs[:,k,...])
                flow_k_down = F.interpolate(flow_k, size=xf_k.shape[2:], mode='bilinear',align_corners=False)
                mask_k_down = F.interpolate(mask_k, size=xf_k.shape[2:], mode='bilinear',align_corners=False)
                xf_k_warp = warp_flow(xf_k, flow_k_down)

                if flows is None:
                    flows, masks, xfs = [flow_k],[mask_k],[xf_k]
                    flows_down, xfs_warp = [flow_k_down],[xf_k_warp]
                else:
                    flows.append(flow_k)
                    masks.append(mask_k)
                    xfs.append(xf_k)
                    flows_down.append(flow_k_down)
                    xfs_warp.append(xf_k_warp)
            
            '''normalize masks to sum to 1'''
            mask_cat = torch.cat(masks, dim=1)
            if opt.mask_norm_type == 'softmax':
                mask_normed = F.softmax(mask_cat, dim=1) # pixel wise sum to 1
            else:
                eps = 1e-12
                mask_normed = mask_cat / (torch.sum(mask_cat, dim=1).unsqueeze(1)+eps) # pixel wise sum to 1
            
            '''merge k features and 1 direct feature'''
            xfs_warp_masked = None
            xf_merge = None
            for k in range(0, opt.K):
                mask_normed_k_down = F.interpolate(mask_normed[:,k:k+1,...], size=xfs[0].shape[2:], mode='bilinear',align_corners=False)

                if xfs_warp_masked is None:
                    xfs_warp_masked = [xfs_warp[k] * mask_normed_k_down]
                    xf_merge = xfs_warp[k] * mask_normed_k_down
                else:
                    xfs_warp_masked.append(xfs_warp[k] * mask_normed_k_down)
                    xf_merge += xfs_warp[k] * mask_normed_k_down

            x_hat = GD(xf_merge)

            lossG_content, lossG_style, lossG_L1 = criterionG(g_x, x_hat)
            lossG_content = lossG_content * opt.lambda_content
            lossG_style = lossG_style * opt.lambda_style
            lossG_L1 = lossG_L1 * opt.lambda_rec
            lossG = lossG_content + lossG_style + lossG_L1
            if opt.use_mask_reg:
                lossAttentionReg = torch.mean(mask[:,0:1,...] * mask[:,1:2,...] * mask[:,2:3,...] * mask[:,3:4,...]) * opt.K**opt.K * lambda_regattn
                lossG += lossAttentionReg
                writer.add_scalar('loss/lossG_Reg', lossAttentionReg.item(), global_step=i_batch_total, walltime=None)


            if opt.use_sample_correctness:
                loss_correctness = (criterionCorrectness(g_x, ref_xs[:,0,...], [flow_1], [2]) + \
                    criterionCorrectness(g_x, ref_xs[:,1,...], [flow_2], [2])+\
                        criterionCorrectness(g_x, ref_xs[:,2,...], [flow_3], [2])+\
                            criterionCorrectness(g_x, ref_xs[:,3,...], [flow_4], [2]))/opt.K * lambda_correctness

                lossG += loss_correctness 
                writer.add_scalar('loss/lossG_correctness', loss_correctness.item(), global_step=i_batch_total, walltime=None)

            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)

            lossG.backward(retain_graph=False)
            optimizerG.step()
            
            writer.add_scalar('loss/lossG', lossG.item(), global_step=i_batch_total, walltime=None)
            # writer.add_scalar('loss/lossG_vggface', loss_face.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('loss/lossG_content', lossG_content.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('loss/lossG_style', lossG_style.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('loss/lossG_L1', lossG_L1.item(), global_step=i_batch_total, walltime=None)
            i_batch_total += 1
                        
            post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
            if opt.use_mask_reg:
                post_fix_str += ',L_reg=%.3f'%lossAttentionReg
            if opt.use_sample_correctness:
                post_fix_str += ',L_correctness=%.3f'%loss_correctness
            pbar.set_postfix_str(post_fix_str)
        
        if opt.use_scheduler:
            lr_scheduler.step(epoch_loss_G_moving)
        lossesG.append(lossG.item())
        
        
        '''save image result'''
        if epoch % save_freq == 0:
            torch.save({
            'epoch': epoch+1,
            'lossesG': lossesG,
            'GE_state_dict': GE.module.state_dict(),
            'GF_state_dict': GF.module.state_dict(),
            'GD_state_dict': GD.module.state_dict(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)

            save_visualize_result(opt, ref_xs, ref_ys, g_x, g_y, xf_merge, x_hat,\
                 flows, mask_normed, xfs, xfs_warp, xfs_warp_masked,\
                      visualize_result_dir, epoch)

    writer.close()


if __name__ == "__main__":
    opt = get_parser()
    for k,v in sorted(vars(opt).items()):
        print(k,':',v)
    set_random_seed(opt.seed)
    today = datetime.today().strftime("%Y%m%d")
    experiment_name = 'v{0}_direct_{1}shot_mask_{2}_soft_{3}_maskNormtype_{4}_lr{5}-{6}'.format(opt.id, opt.K, not opt.no_mask, not opt.use_hard_mask, opt.mask_norm_type, opt.lr, today)
    print(experiment_name)
    train(opt, experiment_name)

