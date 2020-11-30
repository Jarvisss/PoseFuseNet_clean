"""multiview generation without direct generation branch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import flow_vis
from datetime import datetime
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
from model.flow_generator import FlowGenerator, AppearanceEncoder, AppearanceDecoder
from model.Parsing_net import ParsingGenerator
from model.GMM import GMM
from model.blocks import warp_flow, gen_uniform_grid
from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_visualize_result
from util.openpose_utils import get_distance, get_pose_similarity_maps
from util.flow_utils import flow2img, flow2arrow
from util.io import load_image

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.fashionvideo_dataset import FashionVideoDataset
from loss.loss_generator import PerceptualCorrectness, LossG, GicLoss

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
from skimage.transform import resize
import torch.nn.functional as F
import random
import argparse
import json
from PIL import Image

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
    parser.add_argument('--test',  action='store_true', help='open this to test')
    parser.add_argument('--id', type=str, default='default', help = 'experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--seed', type=int, default=7, help = 'random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--K', type=int, default=2, help='source image views')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--root_dir',type=str, default='/home/ljw/playground/poseFuseNet/')
    parser.add_argument('--dataset',type=str, default='danceFashion', help='danceFashion | iper | fashion')
    parser.add_argument('--align_corner', action='store_true', help='behaviour in pytorch grid_sample, before torch=1.2.0 is default True, after 1.2.0 is default False')

    '''Train options'''
    parser.add_argument('--epochs', type=int, default=1000, help='num epochs')
    parser.add_argument('--use_scheduler', action='store_true', help='open this to use learning rate scheduler')


    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_parsing', action='store_true', help='use parsing map')
    parser.add_argument('--use_simmap', action='store_true', help='use similarity map')
    parser.add_argument('--use_dot', action='store_true', help='use dot similarity or cosine similarity')
    parser.add_argument('--step', type=int, default=1, help='every "step" use one sample ')
    parser.add_argument('--align_input', action='store_true', help='open this to align input by pose root')
    '''Model options'''
    parser.add_argument('--n_enc', type=int, default=2, help='encoder(decoder) layers ')
    parser.add_argument('--n_btn', type=int, default=2, help='bottle neck layers in generator')
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn" or "none"')
    parser.add_argument('--use_spectral', action='store_true', help='open this if use spectral normalization')
    
    '''GMM options'''
    parser.add_argument('--rigid_ckpt', type=str, default='Geo_v2_loss_parse_lr0.0001-20201023')
    parser.add_argument('--tps_ckpt', type=str, default='Geo_v2_tps_after_rigidaffine_use_parsingl1_lr0.0001-20201027')
    parser.add_argument('--use_attnflow', action='store_true', help='open this if want to use gmm module to guide the attention map, where large flow leads to low attention')
    parser.add_argument('--use_self_flow',action='store_true', help='open this if want to use self flow to guide the attention map, where large flow leads to low attention')
    parser.add_argument('--use_tv', action='store_true', help='open this if tps use tv l1 loss ')
    parser.add_argument('--move_rigid', action='store_true', help='open this if gmm module use rigid first and then tps')
    parser.add_argument('--gmm_pixel_wise', action='store_true', help='regularize the attention map by pixel wise gmm weight')
    parser.add_argument('--only_pose', action='store_true', help='use pose dot only ')

    '''Test options'''
    # if --test is open
    parser.add_argument('--test_id', type=str, default='default', help = 'test experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--test_id_parse', type=str, default='default', help = 'test parse experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--ref_ids', type=str, default='0', help='test ref ids')
    parser.add_argument('--test_source_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_source', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref images')
    parser.add_argument('--test_target_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_target', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref motions')
    parser.add_argument('--parse_use_attn', action='store_true', help='use attention for multi-view parsing generation')
    parser.add_argument('--no_gt_parsing',action='store_true', help='specified to test on no gt parsing dataset')
    parser.add_argument('--test_freq', type=int, default=5, help='every t images perform one test')

    '''Experiment options'''
    parser.add_argument('--no_mask', action='store_true', help='open this if do not use mask')
    parser.add_argument('--use_hard_mask', action='store_true', help='open this if want to use hard mask')
    parser.add_argument('--mask_sigmoid', action='store_true', help='Use Sigmoid() as mask output layer or not')
    parser.add_argument('--mask_norm_type', type=str, default='softmax', help='Normalize the masks of different views to sum 1, "divsum" or "softmax"')
    parser.add_argument('--use_mask_reg', action='store_true', help='open this if want to regularize the masks of different views to be as different as possible')
    parser.add_argument('--use_sample_correctness', action='store_true', help='open this if want to make flow learnt from each view to be correct')

    '''Loss options'''
    parser.add_argument('--lambda_style', type=float, default=500.0, help='style loss')
    parser.add_argument('--lambda_content', type=float, default=0.5, help='content loss')
    parser.add_argument('--lambda_rec', type=float, default=5.0, help='L1 loss')
    parser.add_argument('--lambda_correctness', type=float, default=5.0, help='sample correctness loss on flow map')
    parser.add_argument('--lambda_regattn', type=float, default=1.0, help='attention map loss ')
    parser.add_argument('--lambda_attnflow', type=float, default=1.0, help='gmm flow mul attention loss')
    parser.add_argument('--lambda_selfflow', type=float, default=10.0, help='self flow mul attention loss')
    parser.add_argument('--lambda_tv', type=float, default=10.0, help='regularize the tv loss of flow')

    return parser   


def create_writer(path_to_log_dir):
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

def make_ckpt_log_vis_dirs(opt, exp_name):
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(exp_name)
    path_to_visualize_dir = opt.root_dir+ 'visualize_result/{0}/'.format(exp_name)
    path_to_log_dir = opt.root_dir+ 'logs/{0}'.format(exp_name)
    
    path_to_visualize_dir_train = os.path.join(path_to_visualize_dir, 'train')
    path_to_visualize_dir_val = os.path.join(path_to_visualize_dir, 'val')

    if not os.path.isdir(path_to_ckpt_dir):
        os.makedirs(path_to_ckpt_dir)
    if not os.path.isdir(path_to_visualize_dir):
        os.makedirs(path_to_visualize_dir)
    if not os.path.isdir(path_to_visualize_dir_train):
        os.makedirs(path_to_visualize_dir_train)
    if not os.path.isdir(path_to_visualize_dir_val):
        os.makedirs(path_to_visualize_dir_val)
    if not os.path.isdir(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    
    return path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir

def make_dataset(opt):
    """Create dataset and dataloader"""
    path_to_train_A = '/dataset/ljw/{0}/train_256/train_A/'.format(opt.dataset)
    path_to_train_kps = '/dataset/ljw/{0}/train_256/train_alphapose/'.format(opt.dataset)
    path_to_train_parsing = '/dataset/ljw/{0}/train_256/parsing_A/'.format(opt.dataset)
    if opt.use_clean_pose:
        path_to_train_kps = '/dataset/ljw/{0}/train_256/train_video2d/'.format(opt.dataset)

    dataset = FashionVideoDataset(path_to_train_A=path_to_train_A, path_to_train_kps=path_to_train_kps,path_to_train_parsing=path_to_train_parsing, opt=opt)
    print(dataset.__len__())
    return dataset

def load_gmm(opt, rigid):
    print('[Loading gmm checkpoint...]')
    path_to_rigid_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(opt.rigid_ckpt)
    path_to_tps_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(opt.tps_ckpt)
    path_to_rigid_chkpt = path_to_rigid_ckpt_dir + 'model_weights.tar' 
    path_to_tps_chkpt = path_to_tps_ckpt_dir + 'model_weights.tar'
    setattr(opt, 'input_nc', 1) 
    setattr(opt, 'grid_size', 5) 
    setattr(opt, 'radius', 5) 
    setattr(opt, 'n_layers', 3) 
    setattr(opt, 'fine_height', 256) 
    setattr(opt, 'fine_width', 256) 
    print('[Loading Tps...]')
    tps_gmm = nn.DataParallel(GMM(opt, rigid=False)).to(device)
    tps_checkpoint = torch.load(path_to_tps_chkpt, map_location=cpu)
    tps_gmm.module.load_state_dict(tps_checkpoint['gmm_state_dict'], strict=False)
    print('[Loading Done...]')
    
    if rigid is True:
        print('[Loading rigid...]')
        rigid_gmm = nn.DataParallel(GMM(opt, rigid=True)).to(device)
        rigid_checkpoint = torch.load(path_to_rigid_chkpt, map_location=cpu)
        rigid_gmm.module.load_state_dict(rigid_checkpoint['gmm_state_dict'], strict=False)
        print('[Loading Done...]')
        return tps_gmm, rigid_gmm
    else:
        return tps_gmm
        

def init_generator(opt, path_to_chkpt):
    GF_inc = 43
    if opt.use_parsing:
        GF_inc+= 40 
    if opt.use_simmap:
        GF_inc += 13
    GF = nn.DataParallel(FlowGenerator(inc=GF_inc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral, mask_use_sigmoid=opt.mask_sigmoid).to(device)) # dx + dx + dy = 3 + 20 + 20
    GE = nn.DataParallel(AppearanceEncoder(n_layers=opt.n_enc, inc=3, use_spectral_norm=opt.use_spectral).to(device)) # dx = 3
    GD = nn.DataParallel(AppearanceDecoder(n_bottleneck_layers=opt.n_btn, n_decode_layers=opt.n_enc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral).to(device)) # df = 256


    optimizerG = optim.Adam(params = list(GF.parameters()) + list(GE.parameters()) + list(GD.parameters()) ,
                            lr=opt.lr,
                            amsgrad=False)
    if opt.use_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizerG, 'min', factor=np.sqrt(0.1), patience=5, min_lr=5e-7)
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
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

    return GF, GE, GD, optimizerG



def train(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    '''Set experiment name and logging,checkpoint,vis dir'''

    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 

    '''save option'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    """Train/val data split"""
    validation_split = 0.1

    dataset = make_dataset(opt)
    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=opt.batch_size , num_workers=4, drop_last=True)
    validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=opt.batch_size , num_workers=4)
    data_loaders = {"train": train_loader, "val": validation_loader}
    data_lengths = {"train": len(train_idx), "val": val_len}


    '''Load freezed eval model'''
    if opt.use_attnflow or opt.use_self_flow:
        if opt.move_rigid:
            tps_gmm, rigid_gmm = load_gmm(opt, True)
            tps_gmm.eval()
            rigid_gmm.eval()
        else:
            tps_gmm = load_gmm(opt, False)
            tps_gmm.eval()

    '''Create Train Model'''
    GF, GE, GD, optimizerG = init_generator(opt, path_to_chkpt)

 
    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    GE.module.load_state_dict(checkpoint['GE_state_dict'], strict=False)
    GD.module.load_state_dict(checkpoint['GD_state_dict'], strict=False)
    
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    i_batch_current = checkpoint['i_batch']
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    i_batch_total = epochCurrent * data_lengths['train'] // opt.batch_size + i_batch_current

    """
    create tensorboard writter
    """
    writer = create_writer(path_to_log_dir)
    
    '''Losses
    '''
    criterionG = LossG(device=device)
    criterionCorrectness = PerceptualCorrectness().to(device)

    save_freq = 1

    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        if epoch >= 100:
            save_freq=5
        if epoch >= 500:
            save_freq=10

        for phase in ['train','val']:
            if phase == 'train':
                GF.train()
                GE.train()
                GD.train()
                pbar = tqdm(train_loader, leave=True, initial=0)
                pbar.set_description('[{0:^7}] {1:>4}/{2:>4}, lr-{3}'.format(phase,epoch,opt.epochs,optimizerG.param_groups[0]['lr']))
                torch.set_grad_enabled(True)
            else:
                GF.eval()
                GE.eval()
                GD.eval()
                pbar = tqdm(validation_loader, leave=True, initial=0)
                pbar.set_description('[{0:^7}] {1:>4}/{2:>4}'.format(phase,epoch,opt.epochs))
                torch.set_grad_enabled(False)

            epoch_loss_G = 0
            epoch_loss_gmm = 0
            epoch_loss_sff = 0
            for i_batch, (ref_xs, ref_ys,ref_ps, g_x, g_y,g_p,sims,sim_maps, vid_path) in enumerate(pbar, start=0):
                assert(len(ref_xs)==len(ref_ys))
                assert(len(ref_xs)==len(ref_ps))
                assert(len(ref_xs)==opt.K)
                assert(len(sims)==opt.K)
                assert(len(sim_maps)==opt.K)

                for i in range(opt.K):
                    ref_xs[i] = ref_xs[i].to(device)
                    ref_ys[i] = ref_ys[i].to(device)
                    ref_ps[i] = ref_ps[i].to(device)
                    sims[i] = sims[i].to(device)
                    sim_maps[i] = sim_maps[i].to(device)

                # pixels = visualize_parsing(ref_ps[0].to(cpu).numpy(), 0, (256,256))
                # plt.imsave(path_to_visualize_dir+"parsing.png", pixels)

                g_x = g_x.to(device) # [B, 3, 256, 256]
                g_y = g_y.to(device) # [B, 20, 256, 256]
                g_p = g_p.to(device) # [B, 20, 256, 256]
                g_c = g_p[:,5:6,...] + g_p[:,6:7,...] + g_p[:,7:8,...] + g_p[:,12:13,...] #[B,1, H,W]


                '''Get flows and masks and features'''
                flows, masks, xfs = [], [], []
                flows_down, xfs_warp = [], []
                for k in range(0, opt.K):
                    if opt.move_rigid:
                        ref_pk = ref_ps[k] # [B,20, H,W]
                        ref_ck = ref_pk[:,5:6,...] + ref_pk[:,6:7,...] +ref_pk[:,7:8,...] + ref_pk[:,12:13,...] #[B,1, H,W]
                        with torch.no_grad():
                            rigid_grid, _ = rigid_gmm(ref_ck, g_c) # [B,H,W,2] # this grid is [-1,1]
                            ref_xs[k] = F.grid_sample(ref_xs[k], rigid_grid, padding_mode='border',align_corners=opt.align_corner)
                            ref_ys[k] = F.grid_sample(ref_ys[k], rigid_grid, padding_mode='zeros',align_corners=opt.align_corner)
                            ref_ps[k] = F.grid_sample(ref_ps[k], rigid_grid, padding_mode='zeros',align_corners=opt.align_corner)
                            sim_maps[k] = F.grid_sample(sim_maps[k], rigid_grid, padding_mode='zeros',align_corners=opt.align_corner)

                    if opt.use_parsing and not opt.use_simmap:
                        flow_k, mask_k = GF(ref_xs[k], torch.cat((ref_ys[k], ref_ps[k]), dim=1), torch.cat((g_y, g_p), dim=1))
                    elif opt.use_simmap and opt.use_parsing:
                        flow_k, mask_k = GF(ref_xs[k], torch.cat((ref_ys[k], ref_ps[k]), dim=1), torch.cat((g_y, g_p), dim=1), sim_maps[k])
                    else:
                        flow_k, mask_k = GF(ref_xs[k], ref_ys[k], g_y)
                    xf_k = GE(ref_xs[k])
                    flow_k_down = F.interpolate(flow_k * xf_k.shape[2] / flow_k.shape[2], size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
                    mask_k_down = F.interpolate(mask_k, size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
                    # x_k_down = F.interpolate(ref_xs[:,k,...], size=xf_k.shape[2:], mode='bilinear',align_corners=False)
                    # xf_k_warp = warp_flow(x_k_down, flow_k_down)
                    xf_k_warp = warp_flow(xf_k, flow_k_down, align_corners=opt.align_corner)
                    
                    flows += [flow_k]
                    masks += [mask_k]
                    xfs += [xf_k]
                    flows_down += [flow_k_down]
                    xfs_warp += [xf_k_warp]
                
                
                '''normalize masks to sum to 1'''
                mask_cat = torch.cat(masks, dim=1)
                if opt.mask_norm_type == 'softmax':
                    mask_normed = F.softmax(mask_cat, dim=1) # pixel wise sum to 1
                else:
                    eps = 1e-12
                    mask_normed = mask_cat / (torch.sum(mask_cat, dim=1).unsqueeze(1)+eps) # pixel wise sum to 1
                
                '''merge k features'''
                xfs_warp_masked = None
                xf_merge = None
                for k in range(0, opt.K):
                    mask_normed_k_down = F.interpolate(mask_normed[:,k:k+1,...], size=xfs[0].shape[2:], mode='bilinear',align_corners=opt.align_corner)

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
                writer.add_scalar('{0}/lossG_content'.format(phase), lossG_content.item(), global_step=i_batch_total, walltime=None)
                writer.add_scalar('{0}/lossG_style'.format(phase), lossG_style.item(), global_step=i_batch_total, walltime=None)
                writer.add_scalar('{0}/lossG_L1'.format(phase), lossG_L1.item(), global_step=i_batch_total, walltime=None)
                lossG = lossG_content + lossG_style + lossG_L1
                if opt.use_mask_reg:
                    lossAttentionReg = torch.mean(mask[:,0:1,...] * mask[:,1:2,...] * mask[:,2:3,...] * mask[:,3:4,...]) * opt.K**opt.K * lambda_regattn
                    lossG += lossAttentionReg
                    writer.add_scalar('{0}/lossG_Reg'.format(phase), lossAttentionReg.item(), global_step=i_batch_total, walltime=None)

                if opt.use_tv:
                    loss_tv = 0
                    ref_pk = ref_ps[k] # [B,20, H,W]
                    ref_ck = ref_pk[:,5:6,...] + ref_pk[:,6:7,...] +ref_pk[:,7:8,...] + ref_pk[:,12:13,...] #[B,1, H,W]
                    uniform_grid = gen_uniform_grid(ref_ps[0]) # this grid is [-1,1] [B2HW]
                    # print(torch.max(flows[0]))
                    # print(torch.min(flows[0]))
                    tv_crit = GicLoss()
                    for i in range(opt.K):
                        # tv loss requires input in [BHW2] order
                        # only compute for cloth region
                        loss_tv += tv_crit( (ref_ck * (flows[i] * 2/255) + uniform_grid).permute(0,2,3,1) ) 

                    loss_tv = loss_tv / opt.K * opt.lambda_tv
                    lossG += loss_tv
                if opt.use_sample_correctness:
                    loss_correctness = 0
                    for i in range(opt.K):
                        loss_correctness += criterionCorrectness(g_x, ref_xs[k], [flows[k]], used_layers=2)
                    loss_correctness = loss_correctness/opt.K * opt.lambda_correctness

                    lossG += loss_correctness 
                    writer.add_scalar('{0}/lossG_correctness'.format(phase), loss_correctness.item(), global_step=i_batch_total, walltime=None)

                if opt.use_attnflow:
                    loss_gmm = 0
                    ref_cs = []
                    rigid_grids = []
                    tps_grids = []
                    uniform_grid = gen_uniform_grid(ref_ps[0]) # this grid is [-1,1] [B2HW]
                    
                    ''' 计算k个source，在target cloth区域每个像素处的最大flow
                    '''
                    offset_flows = []
                    flow_means = []
                    for k in range(opt.K):
                        ref_pk = ref_ps[k] # [B,20, H,W]
                        ref_ck = ref_pk[:,5:6,...] + ref_pk[:,6:7,...] +ref_pk[:,7:8,...] + ref_pk[:,12:13,...] #[B,1, H,W]
                        with torch.no_grad():
                            tps_grid,_ = tps_gmm(ref_ck, g_c) # [BHW2]
                        offset_flow_xy = tps_grid.permute(0,3,1,2) - uniform_grid # [B2HW]
                        offset_flow_len = torch.sqrt(torch.pow(offset_flow_xy[:,0:1,...],2) + torch.pow(offset_flow_xy[:,1:2,...],2)) # sqrt(x^2+y^2) [B1HW]
                        offset_flow = offset_flow_len * g_c # [B,1,H,W],get the absolute value of flow at the interest area
                        offset_flows += [offset_flow]
                        flow_means += [torch.mean(offset_flows[k], dim=(1,2,3)).unsqueeze(1)] #[B] -> K*[B,1]
                        ref_cs += [ref_ck]
                        tps_grids += [tps_grid]

                    offset_flows_max,_ = torch.max(torch.cat(offset_flows, dim=1), dim=1, keepdim=True) # K*[B,1,H,W] -> [B,1,H,W] K个warp里面最大的
                    eps = 1e-8
                    offset_flows_max = torch.maximum(offset_flows_max, offset_flows_max.detach().clone().fill_(eps))
                    flow_means_max,_ = torch.max(torch.cat(flow_means, dim=1), dim=1) # K*[B] -> [B]
                    flow_means_max = flow_means_max.squeeze()
                    
                    offset_flow_imgs = []
                    flow_mean_imgs = []
                    flow_weight_imgs = []
                    # flow_weight = torch.ones()
                    for k in range(opt.K):
                        if opt.gmm_pixel_wise:
                            ''' choice 1. constrain flow at each pixel, (dir*(f-fmax)+fmax) * attn
                            '''
                            flow_weight =  (offset_flows[k]-offset_flows_max) * sims[k].unsqueeze(1).unsqueeze(1).unsqueeze(1)+ offset_flows_max  # [B,1,H,W]
                            loss_gmm += torch.sum(flow_weight  * mask_normed[:,k:k+1,...] * g_c  ) / torch.sum(g_c) # F * A / Batchsize / area_of_1
                            offset_flow_imgs += [np.concatenate([offset_flows[k].permute(0,2,3,1)[0].detach().cpu().numpy(),]*3,axis=2)]  # K * [H,W,3]
                            flow_mean_imgs += [ np.concatenate([(offset_flows[k] / offset_flows_max).permute(0,2,3,1)[0].detach().cpu().numpy(),]*3, axis=2) ] # [B,1,H,W] / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]
                            flow_weight_imgs += [ np.concatenate([(flow_weight / offset_flows_max /2).permute(0,2,3,1)[0].detach().cpu().numpy(),]*3, axis=2) ] # [B,1,H,W] / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]
                        
                        elif opt.only_pose:
                            ''' choice 2. only use pose dot
                            '''
                            flow_weight = ((1-sims[k])/2).unsqueeze(1).unsqueeze(1).unsqueeze(1) # [B,1,1,1]
                            loss_gmm += torch.sum( flow_weight * mask_normed[:,k:k+1,...] * g_c ) / torch.sum(g_c) # [B,1,H,W] * [B,1,H,W] 
                            offset_flow_imgs += [np.concatenate([offset_flows[k].permute(0,2,3,1)[0].detach().cpu().numpy(),]*3,axis=2)]  # K * [H,W,3]
                            flow_mean_imgs += [np.ones((256,256,3))] # [B,1,H,W] / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]
                            flow_weight_imgs += [np.ones((256,256,3))] # [B,1,H,W]  / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]

                        else: # region wise
                            ''' choice 3. constrain flow at a whole region, (dir*(f_avg-f_avgmax)+f_avgmax) * attn
                            '''
                            flow_mean = flow_means[k].view(flow_means[k].shape[0]) #  [B]
                            flow_weight = ((flow_mean - flow_means_max) * sims[k]+ flow_means_max).unsqueeze(1).unsqueeze(1).unsqueeze(1) # [B,1,1,1]
                            loss_gmm += torch.sum( flow_weight * mask_normed[:,k:k+1,...] * g_c ) / torch.sum(g_c) # [B,1,H,W] * [B,1,H,W] 
                            offset_flow_imgs += [np.concatenate([offset_flows[k].permute(0,2,3,1)[0].detach().cpu().numpy(),]*3,axis=2)]  # K * [H,W,3]
                            flow_mean_imgs += [np.concatenate([(flow_mean / flow_means_max).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,1,256,256).permute(0,2,3,1)[0].detach().cpu().numpy(), ]*3, axis=2)] # [B,1,H,W] / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]
                            flow_weight_imgs += [np.concatenate([((flow_weight / flow_means_max.unsqueeze(1).unsqueeze(1).unsqueeze(1)/2).repeat(1,1,256,256) ).permute(0,2,3,1)[0].detach().cpu().numpy(), ]*3,axis=2) ] # [B,1,H,W]  / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]
                    omax = eps
                    for k in range(opt.K):
                        if omax < np.max(offset_flow_imgs[k]):
                            omax = np.max(offset_flow_imgs[k])
                    offset_flow_imgs_copy = offset_flow_imgs.copy()
                    
                    for k in range(opt.K):
                        offset_flow_imgs[k] = offset_flow_imgs[k] / omax * 255.0
                        flow_mean_imgs[k] = flow_mean_imgs[k] * 255.0
                        flow_weight_imgs[k] = flow_weight_imgs[k] * 255.0
                    offset_flow_imgs[0][offset_flow_imgs_copy[0]< offset_flow_imgs_copy[1]]  = 64
                    offset_flow_imgs[0][offset_flow_imgs_copy[0]> offset_flow_imgs_copy[1]]  = 200.0
                    offset_flow_imgs[0][offset_flow_imgs_copy[0]== offset_flow_imgs_copy[1]]  = 0
                    offset_flow_imgs[1][offset_flow_imgs_copy[0]> offset_flow_imgs_copy[1]]  = 64
                    offset_flow_imgs[1][offset_flow_imgs_copy[0]< offset_flow_imgs_copy[1]]  = 200.0
                    offset_flow_imgs[1][offset_flow_imgs_copy[0]== offset_flow_imgs_copy[1]]  = 0

                    loss_gmm = loss_gmm / opt.K * opt.lambda_attnflow
                    writer.add_scalar('{0}/lossG_gmm'.format(phase), loss_gmm.item(), global_step=i_batch_total, walltime=None)
                    lossG += loss_gmm
                    epoch_loss_gmm += loss_gmm.item()
                    epoch_loss_gmm_moving = epoch_loss_gmm / (i_batch+1)

                if opt.use_self_flow:
                    loss_sff = 0
                    uniform_grid = gen_uniform_grid(ref_ps[0]) # this grid is [-1,1] [B2HW]
                    offset_flows = []
                    offset_flow_imgs = []
                    flow_mean_imgs = []
                    flow_weight_imgs = []
                    ref_cs = []

                    for k in range(opt.K):
                        ref_pk = ref_ps[k] # [B,20, H,W]
                        ref_ck = ref_pk[:,5:6,...] + ref_pk[:,6:7,...] +ref_pk[:,7:8,...] + ref_pk[:,12:13,...] #[B,1, H,W]

                        offset_flow_xy = 2 * flows[k] /255  # [B2HW]
                        # print(torch.max(offset_flow_xy))
                        # print(torch.min(offset_flow_xy))

                        offset_flow_len = torch.sqrt(torch.pow(offset_flow_xy[:,0:1,...],2) + torch.pow(offset_flow_xy[:,1:2,...],2)) # sqrt(x^2+y^2) [B1HW]
                        offset_flow = offset_flow_len * g_c # [B,1,H,W],get the absolute value of flow at the interest area
                        offset_flows += [offset_flow]
                        ref_cs += [ref_ck]

                    offset_flows_max,_ = torch.max(torch.cat(offset_flows, dim=1), dim=1, keepdim=True) # K*[B,1,H,W] -> [B,1,H,W] K个warp里面最大的
                    eps = 1e-8
                    offset_flows_max = torch.maximum(offset_flows_max, offset_flows_max.detach().clone().fill_(eps))
                    # print(torch.max(offset_flows_max))
                    for k in range(opt.K):
                        
                        ''' choice 1. constrain flow at each pixel, (dir*(f-fmax)+fmax) * attn
                        '''
                        flow_weight =  (offset_flows[k]-offset_flows_max) * sims[k].unsqueeze(1).unsqueeze(1).unsqueeze(1)+ offset_flows_max  # [B,1,H,W]
                        loss_sff += torch.sum(flow_weight  * mask_normed[:,k:k+1,...] * g_c  ) / torch.sum(g_c) # F * A / Batchsize / area_of_1
                        offset_flow_imgs += [np.concatenate([offset_flows[k].permute(0,2,3,1)[0].detach().cpu().numpy(),]*3,axis=2)]  # K * [H,W,3]
                        flow_mean_imgs += [ np.concatenate([(offset_flows[k] / offset_flows_max).permute(0,2,3,1)[0].detach().cpu().numpy(),]*3, axis=2) ] # [B,1,H,W] / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]
                        flow_weight_imgs += [ np.concatenate([(flow_weight / offset_flows_max /2).permute(0,2,3,1)[0].detach().cpu().numpy(),]*3, axis=2) ] # [B,1,H,W] / [B,1,H,W] = [B,1,H,W] -> K * [H,W,3]

                    omax = eps
                    for k in range(opt.K):
                        if omax < np.max(offset_flow_imgs[k]):
                            omax = np.max(offset_flow_imgs[k])
                    offset_flow_imgs_copy = offset_flow_imgs.copy()
                    
                    for k in range(opt.K):
                        offset_flow_imgs[k] = offset_flow_imgs[k] / omax * 255.0
                        flow_mean_imgs[k] = flow_mean_imgs[k] * 255.0
                        flow_weight_imgs[k] = flow_weight_imgs[k] * 255.0
                    offset_flow_imgs[0][offset_flow_imgs_copy[0]< offset_flow_imgs_copy[1]]  = 64
                    offset_flow_imgs[0][offset_flow_imgs_copy[0]> offset_flow_imgs_copy[1]]  = 200.0
                    offset_flow_imgs[0][offset_flow_imgs_copy[0]== offset_flow_imgs_copy[1]]  = 0
                    offset_flow_imgs[1][offset_flow_imgs_copy[0]> offset_flow_imgs_copy[1]]  = 64
                    offset_flow_imgs[1][offset_flow_imgs_copy[0]< offset_flow_imgs_copy[1]]  = 200.0
                    offset_flow_imgs[1][offset_flow_imgs_copy[0]== offset_flow_imgs_copy[1]]  = 0

                    loss_sff = loss_sff / opt.K * opt.lambda_selfflow
                    writer.add_scalar('{0}/lossG_sff'.format(phase), loss_sff.item(), global_step=i_batch_total, walltime=None)
                    lossG += loss_sff
                    epoch_loss_sff += loss_sff.item()
                    epoch_loss_sff_moving = epoch_loss_sff / (i_batch+1)


                epoch_loss_G += lossG.item()
                epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
                if phase=='train':
                    optimizerG.zero_grad()
                    lossG.backward(retain_graph=True)
                    optimizerG.step()
                    if opt.use_scheduler:
                        lr_scheduler.step(epoch_loss_G_moving)
                    lossesG.append(lossG.item())
                
                writer.add_scalar('{0}/lossG'.format(phase), lossG.item(), global_step=i_batch_total, walltime=None)
                i_batch_total += 1
                            
                # post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
                if opt.use_attnflow:
                    post_fix_str = 'Epoch_loss=%.3f,E_gmm=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving,epoch_loss_gmm_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
                elif opt.use_self_flow:
                    post_fix_str = 'Epoch_loss=%.3f,E_sff=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving,epoch_loss_sff_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
                else:
                    post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())

                # post_fix_str = 'Epoch_loss=%.3f'%(epoch_loss_G_moving)
                if opt.use_mask_reg:
                    post_fix_str += ',L_reg=%.3f'%lossAttentionReg
                if opt.use_sample_correctness:
                    post_fix_str += ',L_correctness=%.3f'%loss_correctness
                if opt.use_attnflow:
                    post_fix_str += ',L_gmm=%.3f'%loss_gmm
                if opt.use_self_flow:
                    post_fix_str += ',L_sff=%.3f'%loss_sff
                if opt.use_tv:
                    post_fix_str += ',L_tv=%.3f'%loss_tv
                pbar.set_postfix_str(post_fix_str)
        
        
        
        
            
            '''save image result'''
            final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,ref_ps, g_x, g_y,g_p,g_p, xf_merge, x_hat,\
                        flows, mask_normed, xfs, xfs_warp, xfs_warp_masked)

            
            white = np.ones((256,256,3)) * 255.0
            grid_img = load_image('256.png').unsqueeze(0).repeat(opt.batch_size,1,1,1).to(device)
            if opt.use_attnflow:
                visual_ref_cs = []
                visual_rigid_cs = []
                visual_tps_cs = []

                visual_rigid_gs = []
                visual_ref_gs = [grid_img[0].cpu().permute(1,2,0).numpy() * 255.0]
                visual_tps_gs = []
                for i in range(len(ref_cs)):
                    ref_ci = ref_cs[i] * ref_xs[i]
                    ref_gridi = ref_cs[i] * grid_img
                    visual_ref_c = ref_ci[0].cpu().permute(1,2,0).numpy() * 255.0
                    visual_ref_c[visual_ref_c<0.5]+=128
                    visual_ref_cs += [visual_ref_c]
                    
                    tps_c = F.grid_sample(ref_ci, tps_grids[i], padding_mode='border',align_corners=opt.align_corner)
                    tps_grid = F.grid_sample(ref_gridi, tps_grids[i], padding_mode='border',align_corners=opt.align_corner)

                    visual_tps_c = tps_c[0].cpu().permute(1,2,0).numpy() * 255.0
                    visual_tps_c[visual_tps_c<0.5]+=128

                    visual_tps_cs += [visual_tps_c]
                    visual_tps_gs += [tps_grid[0].cpu().permute(1,2,0).numpy() * 255.0]
                
                for i in range(1, len(ref_cs)):
                    visual_ref_gs += [white]

                gt_c = (g_c * g_x)[0].cpu().permute(1,2,0).numpy() * 255.0
                gt_g = (g_c * grid_img)[0].cpu().permute(1,2,0).numpy() * 255.0

                visual_mask_c = mask_normed * g_c
                visual_mask1_c = visual_mask_c[0,0:1,...].detach().cpu().permute(1,2,0).repeat(1,1,3).numpy()*255.0
                visual_mask2_c = visual_mask_c[0,1:2,...].detach().cpu().permute(1,2,0).repeat(1,1,3).numpy()*255.0

                visual_tps_cs_row = np.concatenate((visual_ref_cs+visual_tps_cs+[white,gt_c]), axis=1).astype(np.uint8)
                visual_tps_grids_row = np.concatenate((visual_ref_gs+visual_tps_gs+[white,gt_g]), axis=1).astype(np.uint8)

                visual_flow_row = np.concatenate((flow_mean_imgs+offset_flow_imgs+[white,gt_g]), axis=1).astype(np.uint8)
                visual_mask_row = np.concatenate(([white * (sims[0][0].detach().cpu().numpy()+1)/2,white * (sims[1][0].detach().cpu().numpy()+1)/2]+ offset_flow_imgs+[white,white]), axis=1).astype(np.uint8)
                visual_flowweight_row = np.concatenate((flow_weight_imgs+[visual_mask1_c, visual_mask2_c]+[white,gt_g]), axis=1).astype(np.uint8)

                final_img = np.concatenate((final_img,visual_tps_cs_row,visual_tps_grids_row,visual_flow_row,visual_mask_row,visual_flowweight_row),axis=0)

            if opt.use_self_flow:
                gt_g = (g_c * grid_img)[0].cpu().permute(1,2,0).numpy() * 255.0
                visual_mask_c = mask_normed * g_c
                visual_mask1_c = visual_mask_c[0,0:1,...].detach().cpu().permute(1,2,0).repeat(1,1,3).numpy()*255.0
                visual_mask2_c = visual_mask_c[0,1:2,...].detach().cpu().permute(1,2,0).repeat(1,1,3).numpy()*255.0

                visual_ref_gs = [grid_img[0].cpu().permute(1,2,0).numpy() * 255.0]
                visual_warp_gs = []
                visual_flow_arrows = []
                uniform_grid = gen_uniform_grid(ref_ps[0]) # this grid is [-1,1] [B2HW]

                for i in range(len(ref_cs)):
                    # warp_grid = F.grid_sample(grid_img, ((2*flows[i].detach()/255)+uniform_grid).permute(0,2,3,1), align_corners=opt.align_corner)
                    warp_grid = warp_flow(grid_img, flows[i].detach(), opt.align_corner)
                    visual_warp_gs += [warp_grid[0].cpu().permute(1,2,0).numpy() * 255.0]
                    flow_arrow = flow2arrow(flows[i][0].detach().cpu().permute(1,2,0).numpy(), arrow_step=(16,16))
                    visual_flow_arrows += [flow_arrow]
                for i in range(1, len(ref_cs)):
                    visual_ref_gs += [white]

                visual_warp_grid_row = np.concatenate((visual_ref_gs + visual_warp_gs + [white,white]), axis=1).astype(np.uint8)
                visual_flow_arrow_row = np.concatenate(([white,white] + visual_flow_arrows + [white,white]), axis=1).astype(np.uint8)

                visual_flow_row = np.concatenate((flow_mean_imgs+offset_flow_imgs+[white,gt_g]), axis=1).astype(np.uint8)
                visual_mask_row = np.concatenate(([white * (sims[0][0].detach().cpu().numpy()+1)/2,white * (sims[1][0].detach().cpu().numpy()+1)/2]+ offset_flow_imgs+[white,white]), axis=1).astype(np.uint8)
                visual_flowweight_row = np.concatenate((flow_weight_imgs+[visual_mask1_c, visual_mask2_c]+[white,gt_g]), axis=1).astype(np.uint8)

                final_img = np.concatenate((final_img,visual_warp_grid_row, visual_flow_arrow_row, visual_flow_row, visual_mask_row, visual_flowweight_row),axis=0)

            plt.imsave(os.path.join(path_to_visualize_dir, phase,"epoch_latest.png"), final_img)
            
            '''save backup results'''
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
            
                plt.imsave(os.path.join(path_to_visualize_dir, phase,"epoch_{}.png".format(epoch)), final_img)


    writer.close()



def test(opt):
    from util.io import load_image, load_skeleton, load_parsing, transform_image
    import time
    import torchvision
    print('-----------TESTING-----------')
    experiment_name = opt.test_id
    if opt.use_parsing:
        parse_name = opt.test_id_parse
    test_source_dataset = opt.test_source_dataset
    test_source = opt.test_source
    test_target_dataset = opt.test_target_dataset
    test_target = opt.test_target

    print('Experim: ', experiment_name)
    print('Source Dataset: ', test_source_dataset)
    print('Source : ', test_source)
    print('Motion Dataset : ', test_target_dataset)
    print('Motion : ', test_target)

    """Create dataset and dataloader"""
    path_to_test_A = '/dataset/ljw/{0}/test_256/train_A/'.format(test_source_dataset)
    path_to_test_kps = '/dataset/ljw/{0}/test_256/train_alphapose/'.format(test_source_dataset)
    path_to_test_parsing = '/dataset/ljw/{0}/test_256/parsing_A/'.format(test_source_dataset)
    if opt.use_clean_pose:
        path_to_test_kps = '/dataset/ljw/{0}/test_256/train_video2d/'.format(test_source_dataset)
    path_to_test_source_imgs = os.path.join(path_to_test_A, test_source)
    path_to_test_source_kps = os.path.join(path_to_test_kps, test_source)
    path_to_test_source_parse = os.path.join(path_to_test_parsing, test_source)

    path_to_test_A = '/dataset/ljw/{0}/test_256/train_A/'.format(test_target_dataset)
    path_to_test_kps = '/dataset/ljw/{0}/test_256/train_alphapose/'.format(test_target_dataset)
    path_to_test_parsing = '/dataset/ljw/{0}/test_256/parsing_A/'.format(test_target_dataset)
    if opt.use_clean_pose:
        path_to_test_kps = '/dataset/ljw/{0}/test_256/train_video2d/'.format(test_target_dataset)
    path_to_test_tgt_motions = os.path.join(path_to_test_A, test_target)
    path_to_test_tgt_kps = os.path.join(path_to_test_kps, test_target)
    path_to_test_tgt_parse = os.path.join(path_to_test_parsing, test_target)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    ref_ids = opt.ref_ids
    ref_ids = [int(i) for i in ref_ids.split(',')]
    total_ids = len(os.listdir(path_to_test_source_imgs))
    total_gts = len(os.listdir(path_to_test_tgt_motions))
    
    assert(max(ref_ids) <= total_ids)
    ref_names = ['{:05d}'.format(ref_id) for ref_id in ref_ids]
    K = len(ref_names)
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(experiment_name)
    path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
    
    if opt.use_parsing:
        path_to_parse_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(parse_name)
        path_to_parse_chkpt = path_to_parse_ckpt_dir + 'seg_model_weights.tar' 
        test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/'.format(experiment_name+';'+parse_name,test_source_dataset, test_source)
    else:
        test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/'.format(experiment_name,test_source_dataset, test_source)

    test_result_vid_dir = test_result_dir + test_target
    for ref_name in ref_names:
        test_result_vid_dir += '_{0}'.format(ref_name)
    print(test_result_vid_dir)
    if not os.path.isdir(test_result_vid_dir):
        os.makedirs(test_result_vid_dir)

    '''Create Model'''

    GF_inc = 43
    if opt.use_parsing:
        GF_inc+= 40 
    if opt.use_simmap:
        GF_inc += 13
    
    GF = nn.DataParallel(FlowGenerator(inc=GF_inc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral, mask_use_sigmoid=opt.mask_sigmoid).to(device)) # dx + dx + dy = 3 + 20 + 20
    GE = nn.DataParallel(AppearanceEncoder(n_layers=opt.n_enc, inc=3, use_spectral_norm=opt.use_spectral).to(device)) # dx = 3
    GD = nn.DataParallel(AppearanceDecoder(n_bottleneck_layers=opt.n_btn, n_decode_layers=opt.n_enc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral).to(device)) # df = 256
    GP_inc = 20+20+3
    GP = nn.DataParallel(ParsingGenerator(inc=GP_inc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral, use_attn=opt.parse_use_attn))
    '''Load freezed eval model'''
    if opt.use_attnflow:
        if opt.move_rigid:
            _, rigid_gmm = load_gmm(opt, True)
            rigid_gmm.eval()

    GF.eval()
    GE.eval()
    GD.eval()
    GP.eval()

    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    GE.module.load_state_dict(checkpoint['GE_state_dict'], strict=False)
    GD.module.load_state_dict(checkpoint['GD_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    print('Epoch:', epochCurrent)

    if opt.use_parsing:
        parse_ckpt = torch.load(path_to_parse_chkpt, map_location=cpu)
        GP.module.load_state_dict(parse_ckpt['GP_state_dict'], strict=False)

    affine_param = None
    if opt.test_target_dataset == 'iper':
        affine_param = {'angle':0,'scale':1.3, 'shift':(0,-15)}

    ori_ref_xs = []
    ori_ref_ys = []
    ori_ref_js = []
    ori_ref_ps = []
    for i, ref_name in enumerate(ref_names):
        ori_ref_xs += [load_image(os.path.join(path_to_test_source_imgs, ref_name+'.png'))]
        y,j = load_skeleton(os.path.join(path_to_test_source_kps, ref_name+'.json'), is_clean_pose=opt.use_clean_pose)
        ori_ref_ys += [y]
        ori_ref_js += [j]
        ori_ref_ps += [load_parsing(os.path.join(path_to_test_source_parse, ref_name+'.png'))]

    assert(len(ori_ref_xs)==len(ori_ref_ys))
    assert(len(ori_ref_xs)==len(ori_ref_ps))

    """ preprocess for input
    """    
    
    

    avg_l1_loss, gt_sum = 0,0

    for gt_id in tqdm(range(0, total_gts, opt.test_freq)):
        start = time.time()

        gt_name = '{:05d}'.format(gt_id)
        gt_sum += 1
        g_x = load_image(os.path.join(path_to_test_tgt_motions, gt_name+'.png'))
        
        g_y, g_j = load_skeleton(os.path.join(path_to_test_tgt_kps, gt_name+'.json'), is_clean_pose=opt.use_clean_pose, affine=affine_param)
        # as we do not have ground truth parsing for testing
        g_p = None
        if not opt.no_gt_parsing:
            g_p = load_parsing(os.path.join(path_to_test_tgt_parse, gt_name+'.png')).unsqueeze(0).to(device)


        """ preprocess input
        """
        ref_xs, ref_ys, ref_ps = ori_ref_xs.copy(), ori_ref_ys.copy(), ori_ref_ps.copy()

        if opt.use_simmap:
            similarity_maps = get_pose_similarity_maps(g_j, ori_ref_js)

        if opt.align_input:
            import torchvision.transforms.functional as TF
            for i in range(0, K):
                dx, dy = get_distance(ori_ref_js[i], g_j)
                fill_color = (255,255,255)
                ref_xs[i] = TF.to_tensor(TF.affine(TF.to_pil_image(ref_xs[i]),angle=0,translate=(dx,dy),scale=1,shear=0, fillcolor=fill_color, resample=Image.BILINEAR))
                ref_ys[i] = TF.affine(ref_ys[i] ,angle=0,translate=(dx,dy),scale=1,shear=0, resample=Image.BILINEAR)
                similarity_maps[i] = F.affine(similarity_maps[i] ,angle=0,translate=(dx,dy),scale=1,shear=0, resample=Image.BILINEAR)
                ref_ps[i] = TF.affine(ref_ps[i], angle=0,translate=(dx,dy),scale=1,shear=0, resample=Image.BILINEAR)

        for i in range(0, K):
            ref_xs[i] = ref_xs[i].unsqueeze(0).to(device)
            ref_ys[i] = ref_ys[i].unsqueeze(0).to(device)
            ref_ps[i] = ref_ps[i].unsqueeze(0).to(device)
            similarity_maps[i] = similarity_maps[i].unsqueeze(0).to(device) #[13,256,256] -> [1,13,256,256]
        g_x = g_x.unsqueeze(0).to(device)
        g_y = g_y.unsqueeze(0).to(device)
        '''Use parsing generator to get target p_hat
        '''
        g_phat = None
        if opt.use_parsing:
            logits = []
            attns = []
            if opt.parse_use_attn:
                for k in range(0, K):
                    logit_k,attn_k = GP(ref_xs[k], ref_ps[k], g_y) # [B, 20, H, W], [B, 1, H, W]
                    logits += [logit_k]
                    attns += [attn_k]
                attns = torch.cat(attns, dim=1)
                attn_norm = torch.softmax(attns, dim=1)
                logit_avg = logits[0] * attn_norm[:,0:1,:,:] #  [B, 20, H, W]
                for k in range(1, K):
                    logit_avg += logits[k] * attn_norm[:,k:k+1,:,:] #  [B, 20, H, W]
            else:
                for k in range(0, K):
                    logit_k = GP(ori_ref_xs[k], ori_ref_ps[k], g_y) # [B, 20, H, W], [B, 1, H, W]
                    logits += [logit_k]
                logit_avg = logits[0] / K #  [B, 20, H, W]
                for k in range(1, K):
                    logit_avg += logits[k] / K #  [B, 20, H, W]            
            
            g_phat = logit_avg[:,0:1,:,:].clone().detach() # [B,1, H, W]
            g_phat = g_phat.repeat(1,20,1,1) # [B, 20, H, W]
            
            for batch in range(logit_avg.shape[0]):
                p_hat_indices = torch.argmax(logit_avg[batch], dim=0) #  [C, H, W] -> [H, W]
                p_hat_bin_map = p_hat_indices.view(-1,256,256).repeat(20,1,1) # [20, H, W]
                for i in range(20):
                    p_hat_bin_map[i, :, :] = (p_hat_indices == i).int()
                g_phat[batch] = p_hat_bin_map
            

        if opt.use_attnflow:
            if opt.move_rigid:
                for i in range(0, K):
                    '''Use parsing generator to get target p_hat'''
                    ref_pk = ref_ps[i] # [B,20, H,W]
                    ref_ck = ref_pk[:,5:6,...] + ref_pk[:,6:7,...] +ref_pk[:,7:8,...] + ref_pk[:,12:13,...] #[B,1, H,W]
                    g_c = g_phat[:,5:6,...] + g_phat[:,6:7,...] + g_phat[:,7:8,...] + g_phat[:,12:13,...] #[B,1, H,W]
                    
                    rigid_grid, _ = rigid_gmm(ref_ck, g_c) # [B,H,W,2] # this grid is [-1,1]
                    ref_xs[i] = F.grid_sample(ref_xs[i], rigid_grid, padding_mode='border', align_corners=opt.align_corner)
                    ref_ys[i] = F.grid_sample(ref_ys[i], rigid_grid, padding_mode='border', align_corners=opt.align_corner)
                    ref_ps[i] = F.grid_sample(ref_ps[i], rigid_grid, padding_mode='border', align_corners=opt.align_corner)
            

        # only for testing 
        # g_phat = g_p

        '''Get flows and masks and features'''
        flows, masks, xfs = [], [], []
        flows_down, xfs_warp = [], []
        for k in range(0, K):
            if opt.use_parsing and not opt.use_simmap:
                flow_k, mask_k = GF(ref_xs[k], torch.cat((ref_ys[k], ref_ps[k]), dim=1), torch.cat((g_y, g_p), dim=1))
            elif opt.use_simmap and opt.use_parsing:
                flow_k, mask_k = GF(ref_xs[k], torch.cat((ref_ys[k], ref_ps[k]), dim=1), torch.cat((g_y, g_p), dim=1), similarity_maps[k])
            else:
                flow_k, mask_k = GF(ref_xs[k], ref_ys[k], g_y)

            xf_k = GE(ref_xs[k])
            flow_k_down = F.interpolate(flow_k * xf_k.shape[2] / flow_k.shape[2], size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
            mask_k_down = F.interpolate(mask_k, size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
            # x_k_down = F.interpolate(ref_xs[:,k,...], size=xf_k.shape[2:], mode='bilinear',align_corners=False)
            # xf_k_warp = warp_flow(x_k_down, flow_k_down)
            xf_k_warp = warp_flow(xf_k, flow_k_down, align_corners=opt.align_corner)

            flows += [flow_k]
            masks += [mask_k]
            xfs += [xf_k]
            flows_down += [flow_k_down]
            xfs_warp += [xf_k_warp]
        
        '''normalize masks to sum to 1'''
        mask_cat = torch.cat(masks, dim=1)
        if opt.mask_norm_type == 'softmax':
            mask_normed = F.softmax(mask_cat, dim=1) # pixel wise sum to 1
        else:
            eps = 1e-12
            mask_normed = mask_cat / (torch.sum(mask_cat, dim=1).unsqueeze(1)+eps) # pixel wise sum to 1
        
        '''merge k features'''
        xfs_warp_masked = None
        xf_merge = None
        for k in range(0, K):
            mask_normed_k_down = F.interpolate(mask_normed[:,k:k+1,...], size=xfs[0].shape[2:], mode='bilinear',align_corners=opt.align_corner)

            if xfs_warp_masked is None:
                xfs_warp_masked = [xfs_warp[k] * mask_normed_k_down]
                xf_merge = xfs_warp[k] * mask_normed_k_down
            else:
                xfs_warp_masked.append(xfs_warp[k] * mask_normed_k_down)
                xf_merge += xfs_warp[k] * mask_normed_k_down

        x_hat = GD(xf_merge) 
        generate_time = time.time()


        l1loss = nn.L1Loss()(x_hat, g_x)
        avg_l1_loss += l1loss.item()

        '''save image result'''
        final_img,simp_img,mid_img = get_visualize_result(opt, ref_xs, ref_ys,ref_ps, g_x, g_y,g_p,g_phat, xf_merge, x_hat,\
                flows, mask_normed, xfs, xfs_warp, xfs_warp_masked)
        get_visual_time = time.time()
        
        ###### PIL save image faster than matplotlib
        Image.fromarray(final_img).save(os.path.join(test_result_vid_dir,"{0}_result.jpg".format(gt_name)))
        Image.fromarray(simp_img).save(os.path.join(test_result_vid_dir,"{0}_result_simp.jpg".format(gt_name)))
        Image.fromarray(mid_img).save(os.path.join(test_result_vid_dir,"{0}_result_mid.jpg".format(gt_name)))
        # plt.imsave(os.path.join(test_result_vid_dir,"{0}_result.png".format(gt_name)), final_img)
        # plt.imsave(os.path.join(test_result_vid_dir,"{0}_result_simp.png".format(gt_name)), simp_img)
        # plt.imsave(os.path.join(test_result_vid_dir,"{0}_result_mid.png".format(gt_name)), mid_img)

        

        save_time = time.time()
        # print('parse time:', parser_time - start)
        # print('generate time:', generate_time - parser_time)
        # print('get visual time:', get_visual_time - generate_time)
        # print('save time:', save_time - get_visual_time)
        # print('')
    
    avg_l1_loss /= gt_sum
    print(avg_l1_loss)

    

    '''save video result'''
    save_video_name = test_target
    img_dir = test_result_vid_dir
    save_video_dir = test_result_dir
    for ref_name in ref_names:
        save_video_name += '_{0}'.format(ref_name)
    save_video_name_simp = save_video_name +'_result_simp.mp4'
    save_video_name_mid = save_video_name +'_result_mid.mp4'
    save_video_name += '_result.mp4'

    # metric_loss_save_path = save_video_dir + save_video_name.replace('.mp4', '_loss.txt')
    # with open(metric_loss_save_path, 'w') as f:
    #     f.write('%03f \n'%avg_l1_loss)

    imgs = os.listdir(img_dir)
    import cv2
    video_out_simp = cv2.VideoWriter(save_video_dir+save_video_name_simp, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (256*(K+2), 256))
    video_out_mid = cv2.VideoWriter(save_video_dir+save_video_name_mid, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (256*(K+2), 256*4))
    video_out = cv2.VideoWriter(save_video_dir+save_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (256*(K*2+2), 256*3))



    for img in tqdm(sorted(imgs)):
        frame = cv2.imread(os.path.join(img_dir, img))
        if img.split('.')[0].split('_')[-1] == 'simp':
            video_out_simp.write(frame)
        elif img.split('.')[0].split('_')[-1] == 'result':
            video_out.write(frame)
        elif img.split('.')[0].split('_')[-1] == 'mid':
            video_out_mid.write(frame)
    video_out_simp.release()
    video_out_mid.release()
    video_out.release()
    pass


if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    if not opt.test:
        # opt.use_parsing = True
        # opt.use_attnflow =True
        # opt.use_rigid=True
        # opt.use_tv=True
        # opt.move_rigid = True
        # opt.use_dot = True
        # opt.lambda_attnflow = 100
        # opt.lr = 0.0001
        # opt.batch_size=1
        # opt.K=2
        # opt.tps_ckpt='Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027'
        # opt.id = '16_add_moverigid_shouderdot_attnflow100'
        for k,v in sorted(vars(opt).items()):
            print(k,':',v)
        set_random_seed(opt.seed)
        today = datetime.today().strftime("%Y%m%d")
        # today = '20201114'
        experiment_name = 'v{0}_{1}shot_gmm_{2}_moverigid_{3}_tv_{4}_lr{5}-{6}'.format(opt.id, opt.K, opt.use_attnflow,opt.move_rigid, opt.use_tv, opt.lr, today)
        # experiment_name = 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025'
        # experiment_name = 'v13_2shot_mask_True_gmm_False_maskNormtype_softmax_parsing_True_lr0.0001-20201025'
        print(experiment_name)
        train(opt, experiment_name)
        print(experiment_name+'#### Training Done')
        for k,v in sorted(vars(opt).items()):
            print(k,':',v)
    else:
        with torch.no_grad():
            test(opt)




