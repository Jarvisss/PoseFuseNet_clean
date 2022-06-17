"""multiview generation without direct generation branch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
from PIL import Image
#matplotlib.use('agg')
from matplotlib import pyplot as plt
matplotlib.use('agg')  # for image save not render
torch.autograd.set_detect_anomaly(True)
# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
# from model.Parsing_net import ParsingGenerator
# from model.flow_generator import FlowGenerator, AppearanceEncoder, AppearanceDecoder
from model.pyramid_flow_generator_with_occlu_attn import FlowGenerator, AppearanceDecoder, AppearanceEncoder, PoseAwareResidualFlowDecoder
from model.discriminator import ResDiscriminator
from model.blocks import warp_flow,_freeze,_unfreeze

from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_pyramid_visualize_result, tensor2im
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashion_dataset import  make_dataset
from loss.loss_generator import PerceptualCorrectness, LossG, MultiAffineRegularizationLoss, FlowAttnLoss, FusingCompactnessLoss
from loss.externel_functions import AdversarialLoss

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
import pandas as pd
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
    parser.add_argument('--phase',  type=str,default='train',  help='train|test')
    parser.add_argument('--id', type=str, default='default', help = 'experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--seed', type=int, default=7, help = 'random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--K', type=int, default=2, help='source image views')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_D',type=float, default=1e-5, help='learning rate of discriminator')
    parser.add_argument('--root_dir',type=str, default='/home/ljw/playground/poseFuseNet/')
    parser.add_argument('--path_to_dataset',type=str, default='/dataset/ljw/deepfashion/GLFA_split/fashion')
    parser.add_argument('--dataset',type=str, default='fashion', help='danceFashion | iper | fashion')
    parser.add_argument('--align_corner', action='store_true', help='behaviour in pytorch grid_sample, before torch=1.2.0 is default True, after 1.2.0 is default False')


    '''Train options'''
    parser.add_argument('--epochs', type=int, default=500, help='num epochs')
    parser.add_argument('--use_scheduler', action='store_true', help='open this to use learning rate scheduler')
    parser.add_argument('--flow_onfly',action='store_true', help='open this if want to train flow generator end-to-end')
    parser.add_argument('--flow_exp_name',type=str, default='', help='if pretrain flow, specify this to use it in final train')
    parser.add_argument('--which_flow_epoch',type=str, default='epoch_latest', help='specify it to use certain epoch checkpoint')
    parser.add_argument('--anno_size', type=int, nargs=2, help='input annotation size')
    parser.add_argument('--model_save_freq', type=int, default=2000, help='save model every N iters')
    parser.add_argument('--img_save_freq', type=int, default=200, help='save image every N iters')
    parser.add_argument('--pretrain_flow', action='store_true')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--load_exp_name', type=str, default=None)
    parser.add_argument('--freeze_F', action='store_true')

    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--categories', type=int, default=9)
    parser.add_argument('--use_parsing', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_input_mask', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--GD_use_gt_mask', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--GD_use_predict_mask', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_logits', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--align_input', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_simmap', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--joints_for_cos_sim',type=int, default=-1, help='joints used for cosine sim computation')
    parser.add_argument('--use_bone_RGB', action='store_true')

    '''Model options'''
    parser.add_argument('--n_enc', type=int, default=2, help='encoder(decoder) layers ')
    parser.add_argument('--n_btn', type=int, default=2, help='bottle neck layers in generator')
    parser.add_argument('--n_res_block', type=int, default=1, help='res blocks in fusing decoder')
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn"')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='"ReLU" or "LeakyReLU"')
    parser.add_argument('--use_pose_decoder', action='store_true', help='use pose in the target decoder')
    parser.add_argument('--use_self_attention', action='store_true', help='use sa in the target decoder')
    parser.add_argument('--use_res_attn', action='store_true')
    
    parser.add_argument('--use_spectral_G', action='store_true', help='open this if use spectral normalization in generator')
    parser.add_argument('--use_spectral_D', action='store_true', help='open this if use spectral normalization in discriminator')


    '''Test options'''
    # if --test is open
    parser.add_argument('--test_id', type=str, default='default', help = 'test experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--test_ckpt_name', type=str, default='model_weights_G', help = 'test checkpoint name.')
    parser.add_argument('--ref_ids', type=str, default='0', help='test ref ids')
    parser.add_argument('--test_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_source', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref images')
    parser.add_argument('--test_target_motion', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref motions')
    parser.add_argument('--output_all', action='store_true', help='open this to output the full image')
    parser.add_argument('--test_samples', type=int, default=-1, help='test how many samples, -1 means all')

    '''Experiment options'''
    parser.add_argument('--use_attn', action='store_true', help='use attention for multi-view parsing generation')
    parser.add_argument('--mask_sigmoid', action='store_true', help='Use Sigmoid() as mask output layer or not')
    parser.add_argument('--mask_norm_type', type=str, default='softmax', help='softmax | divsum')
    parser.add_argument('--use_tps_sim', action='store_true', help='use precomputed tps sim')
    parser.add_argument('--use_label_field_tps', action='store_true', help='use label field tps sim if true, else use dot tps')
    parser.add_argument('--tps_sim_beta1', type=float, default=2.0, help='use precomputed tps sim')
    parser.add_argument('--tps_sim_beta2', type=float, default=40.0, help='use precomputed tps sim')

    '''Loss options'''
    parser.add_argument('--use_adv', action='store_true', help='use adversarial loss in total generation')
    parser.add_argument('--use_bilinear_correctness', action='store_true', help='use bilinear sampling in sample loss')
    parser.add_argument('--G_use_resample', action='store_true', help='use gaussian sampling in the target decoder')
    parser.add_argument('--use_correctness', action='store_true', help='use sample correct loss')
    parser.add_argument('--single_correctness', action='store_true', help='use one sample correct loss for output flow')
    parser.add_argument('--use_single_reg', action='store_true', help='use one sample correct loss for output flow')
    parser.add_argument('--use_sim_attn_loss', action='store_true', help='sim weighted sample correctness')
    parser.add_argument('--use_flow_attn_loss', action='store_true', help='constrain attention by flow')
    parser.add_argument('--use_flow_reg', action='store_true', help='use flow regularization')
    parser.add_argument('--use_attn_reg', action='store_true', help='constrain attention by sim')
    parser.add_argument('--use_fuse_loss', action='store_true', help='constrain attention by sim')


    parser.add_argument('--lambda_style', type=float, default=500.0, help='style loss')
    parser.add_argument('--lambda_content', type=float, default=0.5, help='content loss')
    parser.add_argument('--lambda_rec', type=float, default=5.0, help='L1 loss')
    parser.add_argument('--lambda_adv', type=float, default=2.0, help='GAN loss weight')
    parser.add_argument('--lambda_correctness', type=float, default=5.0, help='sample correctness weight')
    parser.add_argument('--lambda_struct_correctness', type=float, default=20.0, help='sample correctness weight')
    parser.add_argument('--lambda_flow_reg', type=float, default=0.0025, help='regular sample loss weight')
    parser.add_argument('--lambda_attn_reg', type=float, default=1, help='regular sample loss weight')
    parser.add_argument('--lambda_sim_attn', type=float, default=5.0, help='regular sample loss weight')
    parser.add_argument('--lambda_flow_attn', type=float, default=0.2, help='regular sample loss weight')
    parser.add_argument('--lambda_fuse_loss', type=float, default=1, help='regular sample loss weight')
    opt = parser.parse_args()
    return opt   


def create_writer(path_to_log_dir):
    TIMESTAMP = "/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(path_to_log_dir+TIMESTAMP)
    return writer

def make_ckpt_log_vis_dirs(opt, exp_name):
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(exp_name)
    path_to_visualize_dir = opt.root_dir+ 'visualize_result/{0}/'.format(exp_name)
    path_to_log_dir = opt.root_dir+ 'logs/{0}'.format(exp_name)
    

    if not os.path.isdir(path_to_ckpt_dir):
        os.makedirs(path_to_ckpt_dir)
    if not os.path.isdir(path_to_visualize_dir):
        os.makedirs(path_to_visualize_dir)
    if not os.path.isdir(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    
    return path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir

def init_weights(m, init_type='normal'):
    if type(m) == nn.Conv2d:
        if init_type=='xavier':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='normal':
            torch.nn.init.normal_(m.weight,0,0.02)
        elif init_type=='kaiming':
            torch.nn.init.kaiming_normal_(m.weight)

def make_dataloader(opt, dataset):
    is_train = opt.phase == 'train'
    batch_size = 1 if not is_train else opt.batch_size
    shuffle = is_train # 如果在训练阶段则打乱顺序
    drop_last = is_train # 如果在训练阶段则丢掉不完整的batch
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=8, drop_last=drop_last)
    return dataloader

def save_discriminator(parallel, D, optimizerD, path_to_chkpt_D):
    D_state_dict = D.state_dict() if not parallel else D.module.state_dict()
    torch.save({
        'Dis_state_dict': D_state_dict,
        'optimizerD': optimizerD.state_dict(),
    }, path_to_chkpt_D)    
    pass

def save_flow_generator(parallel,epoch,lossesG, GF,i_batch, optimizerG, path_to_chkpt_G):
    GF_state_dict = GF.state_dict() if not parallel else GF.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GF_state_dict': GF_state_dict,
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
    }, path_to_chkpt_G)


def save_generator(parallel, epoch, lossesG, GE, GF, GD, i_batch, optimizerG,path_to_chkpt_G):
    GE_state_dict = GE.state_dict() if not parallel else GE.module.state_dict()
    GF_state_dict = GF.state_dict() if not parallel else GF.module.state_dict()
    GD_state_dict = GD.state_dict() if not parallel else GD.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GE_state_dict': GE_state_dict,
        'GF_state_dict': GF_state_dict,
        'GD_state_dict': GD_state_dict,
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
    }, path_to_chkpt_G)

def init_flowgenerator(opt, path_to_chkpt):
    image_nc = 3
    structure_nc = 21 if opt.use_bone_RGB else 18
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_input_mask:
        structure_nc += 1
    if opt.use_simmap:
        image_nc += 13
    GF = FlowGenerator(inc=image_nc+structure_nc+structure_nc, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
    GF = GF.to(device)

    optimizerG = optim.Adam(params = list(GF.parameters()) ,
                            lr=opt.lr,
                            amsgrad=False)
    if opt.use_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizerG, 'min', factor=np.sqrt(0.1), patience=5, min_lr=5e-7)
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        GF.apply(init_weights)
        
        save_flow_generator(opt.parallel, 0, [], GF, 0, optimizerG, path_to_chkpt)
        print('...Done')

    return GF, optimizerG

def init_discriminator(opt, path_to_chkpt):
    D_inc = 3
    if opt.parallel:
        D = nn.DataParallel(ResDiscriminator(input_nc=D_inc,ndf=32, img_f=128,layers=4, use_spect=opt.use_spectral_D).to(device)) # dx + dx + dy = 3 + 20 + 20
    else:
        D = ResDiscriminator(input_nc=D_inc,ndf=32,img_f=128,layers=4, use_spect=opt.use_spectral_D).to(device) # dx + dx + dy = 3 + 20 + 20

    optimizerD = optim.Adam(params = list(D.parameters()) ,
                            lr=opt.lr_D,
                            amsgrad=False)
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        D.apply(init_weights)
        print('Initiating new Discriminator model checkpoint...')
        save_discriminator(opt.parallel, D, optimizerD, path_to_chkpt)
        print('...Done')
    return D, optimizerD

def init_generator(opt, path_to_chkpt, path_to_flow_chkpt=None):
    '''doc string
    '''
    image_nc = 3
    structure_nc = 21 if opt.use_bone_RGB else 18
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_input_mask:
        structure_nc += 1
    if opt.use_simmap:
        image_nc += 13

    GF = FlowGenerator(inc=image_nc+structure_nc*2, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    
    if not opt.use_pose_decoder:
        GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    else:
        GD = PoseAwareResidualFlowDecoder(structure_nc=structure_nc,n_res_blocks=opt.n_res_block,n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G,use_self_attention=opt.use_self_attention, align_corners=opt.align_corner, use_resample=opt.G_use_resample, use_res_attn=opt.use_res_attn)

    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
        GE = nn.DataParallel(GE)
        GD = nn.DataParallel(GD)

    GF = GF.to(device)
    GE = GE.to(device)
    GD = GD.to(device)

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
        if not opt.flow_onfly:
            if path_to_flow_chkpt is not None:
                checkpoint_flow = torch.load(path_to_flow_chkpt, map_location=cpu)
                print('load flow from existing checkpoint, epoch: {0},batch: {1}'.format(checkpoint_flow['epoch'],checkpoint_flow['i_batch']))
                if opt.parallel:
                    GF.module.load_state_dict(checkpoint_flow['GF_state_dict'], strict=False)
                else:
                    GF.load_state_dict(checkpoint_flow['GF_state_dict'], strict=False)

            else:
                print('Please specify the pretrained flow model path')
        
        save_generator(opt.parallel, 0, [],GE, GF, GD, 0, optimizerG, path_to_chkpt)
        print('...Done')

    return GF, GE, GD, optimizerG

def train_flow_net(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    path_to_chkpt = path_to_ckpt_dir + 'flow_model_weights.tar' 

    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    '''Create dataset and dataloader'''
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    '''Create Model'''
    GF, optimizerG = init_flowgenerator(opt, path_to_chkpt)
    
    '''Loading from past checkpoint'''
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    else:
        GF.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    i_batch_current = checkpoint['i_batch']
    i_batch_total = epochCurrent * dataloader.__len__() // opt.batch_size + i_batch_current
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    # from loss.externel_functions import VGG19
    # GE = VGG19().to(device)
    '''create tensorboard writter'''
    writer = create_writer(path_to_log_dir)
    
    '''Losses'''
    criterionL1 = nn.L1Loss().to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)


    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        if epoch > epochCurrent:
            i_batch_current = 0
        epoch_loss_G = 0
        pbar = tqdm(dataloader, leave=True, initial=0)
        pbar.set_description('[{0:>4}/{1:>4}], lr-{2}'.format(epoch,opt.epochs,optimizerG.param_groups[0]['lr']))
        
        for i_batch, batch_data in enumerate(pbar, start=0):
            ref_xs = batch_data['ref_xs']
            ref_ys = batch_data['ref_ys']
            g_x = batch_data['g_x']
            g_y = batch_data['g_y']
            assert(len(ref_xs)==len(ref_ys))
            assert(len(ref_xs)==opt.K)
            for i in range(len(ref_xs)):
                ref_xs[i] = ref_xs[i].to(device)
                ref_ys[i] = ref_ys[i].to(device)

            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]
            if opt.use_input_mask:
                ref_ms = batch_data['ref_ms']
                g_m = batch_data['g_m']
                assert(len(ref_ms)==opt.K)
                for i in range(opt.K):
                    ref_ms[i] = ref_ms[i].to(device)
                g_m = g_m.to(device)
            
            flows, flow_ones = [], []
            
            for k in range(0, opt.K):
                if opt.use_input_mask:
                    # flow_ks, mask_ks, _ = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), torch.cat((g_y,g_m),dim=1)) # 32, 64
                    flow_ks, mask_ks, _ = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), g_y) # 32, 64
                else:
                    flow_ks, mask_ks, _ = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64

                flow_k = flow_ks[1]
                # print(flow_k.shape)

                flows += [flow_ks]
                flow_ones += [flow_k ]


            # x_hat = GD(xf_merge)
            # x_hat =  warp_flow(F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear',align_corners=opt.align_corner), flow_k, align_corners=opt.align_corner)
            # x_hat = F.interpolate(x_hat, size=g_x.shape[2:],mode='bilinear',align_corners=opt.align_corner)
            down_ref_xk = F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear', align_corners=opt.align_corner)
            x_hat_down =  warp_flow(down_ref_xk, flow_k, align_corners=opt.align_corner)
            x_hat = F.interpolate(x_hat_down, size=ref_xs[k].shape[2:], mode='bilinear', align_corners=opt.align_corner)
            loss_correct=0
            for k in range(opt.K):
                loss_correct += criterionCorrectness(g_x, ref_xs[k], flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness)
                # loss_correct += criterionL1(xfs_warp[k], g_xf)
            loss_correct = loss_correct/opt.K * opt.lambda_correctness
            lossG = loss_correct
            writer.add_scalar('{0}/loss_correct'.format(opt.phase), loss_correct.item(), global_step=i_batch_total, walltime=None)
            
            loss_regular = 0
            if opt.use_flow_reg:
                for k in range(opt.K):
                    loss_regular += criterionReg(flows[k])
                
                loss_regular = loss_regular / opt.K * opt.lambda_flow_reg
            
                lossG += loss_regular
                writer.add_scalar('{0}/loss_regular'.format(opt.phase), loss_regular.item(), global_step=i_batch_total, walltime=None)
            
            loss_struct = 0
            if opt.use_input_mask:
                for k in range(opt.K):
                    flow_ks = flows[k]
                    for l in range(len(flow_ks)):
                        flow_kl = flow_ks[l]
                        ref_m_down = F.interpolate(ref_ms[k], size=flow_kl.shape[2:], mode='bilinear', align_corners=opt.align_corner)
                        warp_m_down = warp_flow(ref_m_down, flow_kl, align_corners=opt.align_corner)
                        g_m_down = F.interpolate(g_m, size=flow_kl.shape[2:], mode='bilinear', align_corners=opt.align_corner)
                        loss_struct += criterionL1(warp_m_down, g_m_down)
                loss_struct /= opt.K * opt.lambda_struct_correctness
                lossG += loss_struct
                writer.add_scalar('{0}/loss_struct'.format(opt.phase), loss_struct.item(), global_step=i_batch_total, walltime=None)
            
            
            optimizerG.zero_grad()
            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            post_fix_str = 'Epoch_loss=%.3f, G=%.3f, Cor=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_correct.item())
            if opt.use_flow_reg:
                post_fix_str += ', Reg=%.3f'%loss_regular.item()
            if opt.use_input_mask:
                post_fix_str += ', MaskCorr=%.3f'%loss_struct.item()
            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            from util.vis_util import get_flow_visualize_result
            if i_batch % opt.img_save_freq == 0: 
                if not opt.use_input_mask:   
                    final_img,_,_ = get_flow_visualize_result(opt, ref_xs, ref_ys,None, g_x, g_y,None,None, x_hat,\
                        flow_ones)
                else:
                    ref_m_down = F.interpolate(ref_ms[0], size=flow_ones[0].shape[2:], mode='bilinear', align_corners=opt.align_corner)
                    m_hat = warp_flow(ref_m_down,flow_ones[0],align_corners=opt.align_corner)
                    final_img,_,_ = get_flow_visualize_result(opt, ref_xs, ref_ys,ref_ms, g_x, g_y,g_m,m_hat, x_hat,\
                        flow_ones)
                
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), final_img)

            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                save_flow_generator(opt.parallel, epoch, lossesG, GF, i_batch, optimizerG, path_to_save_G)
        save_flow_generator(opt.parallel, epoch, lossesG, GF, i_batch, optimizerG, path_to_chkpt)

    writer.close()   

def train(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    
    path_to_chkpt_G = path_to_ckpt_dir + 'model_weights_G.tar' 
    path_to_chkpt_D = path_to_ckpt_dir + 'model_weights_D.tar' 
    if opt.continue_train:
        
        path_to_chkpt_G = opt.root_dir+ 'checkpoints/{0}/'.format(opt.load_exp_name) + 'model_weights_G.tar' 
        path_to_chkpt_D = opt.root_dir+ 'checkpoints/{0}/'.format(opt.load_exp_name) + 'model_weights_D.tar' 
    
    save_pathG = path_to_ckpt_dir + 'model_weights_G.tar' 
    save_pathD = path_to_ckpt_dir + 'model_weights_D.tar' 

    path_to_chkpt_flow = None
    if not opt.flow_onfly:
        path_to_chkpt_flow = opt.root_dir+ 'checkpoints/{0}/{1}.tar'.format(opt.flow_exp_name, opt.which_flow_epoch)

    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    '''Create dataset and dataloader'''
    print('-------Loading Dataset--------')
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    print('-------Creating Model--------')
    '''Create Model, save a new one if not exist'''
    GF, GE, GD, optimizerG = init_generator(opt, path_to_chkpt_G, path_to_chkpt_flow)
    
    '''Loading from past checkpoint_G'''
    # load_chkpt_G = os.path.join('checkpoints','fashion_v7_sc_attn_reg_in_mask_tps_sim_2shot_20210111/model_weights_G.tar')
    # load_chkpt_D = os.path.join('checkpoints','fashion_v7_sc_attn_reg_in_mask_tps_sim_2shot_20210111/model_weights_D.tar')
    checkpoint_G = torch.load(path_to_chkpt_G, map_location=cpu)
    # checkpoint_G = torch.load(load_chkpt_G, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint_G['GF_state_dict'], strict=False)
        GE.module.load_state_dict(checkpoint_G['GE_state_dict'], strict=False)
        GD.module.load_state_dict(checkpoint_G['GD_state_dict'], strict=False)
    else:
        GF.load_state_dict(checkpoint_G['GF_state_dict'], strict=False)
        GE.load_state_dict(checkpoint_G['GE_state_dict'], strict=False)
        GD.load_state_dict(checkpoint_G['GD_state_dict'], strict=False)
    epochCurrent = checkpoint_G['epoch']
    lossesG = checkpoint_G['lossesG']
    i_batch_current = checkpoint_G['i_batch']
    i_batch_total = epochCurrent * dataloader.__len__() // opt.batch_size + i_batch_current
    optimizerG.load_state_dict(checkpoint_G['optimizerG'])
    GF.train()
    GE.train()
    GD.train()

    if opt.freeze_F:
        for param in GF.parameters():
            param.requires_grad = False
        print(GF)
        GF.mask1.train(True)
        GF.mask2.train(True)
        GF.attn1.train(True)
        GF.attn2.train(True)
    if opt.use_adv:
        D, optimizerD = init_discriminator(opt, path_to_chkpt_D)
        checkpoint_D = torch.load(path_to_chkpt_D, map_location=cpu)
        # checkpoint_D = torch.load(load_chkpt_D, map_location=cpu)
        optimizerD.load_state_dict(checkpoint_D['optimizerD'])
        if opt.parallel:
            D.module.load_state_dict(checkpoint_D['Dis_state_dict'], strict=False)
        else:
            D.load_state_dict(checkpoint_D['Dis_state_dict'], strict=False)
        D.train()


    '''create tensorboard writter'''
    writer = create_writer(path_to_log_dir)
    
    '''Losses'''
    criterionG = LossG(device=device)
    criterion_GAN = AdversarialLoss(type='lsgan').to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionFuse = FusingCompactnessLoss(opt.K).to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)
    criterionFlowAttn = FlowAttnLoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    print('-------Training Start--------')
    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        if epoch > epochCurrent:
            i_batch_current = 0
        epoch_loss_G = 0
        pbar = tqdm(dataloader, leave=True, initial=0)
        pbar.set_description('[{0:>4}/{1:>4}]'.format(epoch,opt.epochs))
        
        for i_batch, batch_data in enumerate(pbar, start=0):
            ref_xs = batch_data['ref_xs']
            ref_ys = batch_data['ref_ys']
            g_x = batch_data['g_x']
            g_y = batch_data['g_y']
            if opt.use_sim_attn_loss or opt.use_attn_reg:
                sims = batch_data['sim']
                assert(len(sims)==opt.K)
                for i in range(opt.K):
                    sims[i] = sims[i].to(device)
            
            if opt.use_input_mask or opt.use_attn_reg:
                ref_ms = batch_data['ref_ms']
                g_m = batch_data['g_m']
                assert(len(ref_ms)==opt.K)
                for i in range(opt.K):
                    ref_ms[i] = ref_ms[i].to(device)
                g_m = g_m.to(device)
                    
            assert(len(ref_xs)==len(ref_ys))
            assert(len(ref_xs)==opt.K)
            for i in range(len(ref_xs)):
                ref_xs[i] = ref_xs[i].to(device)
                ref_ys[i] = ref_ys[i].to(device)
                

            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]
            
            flows, masks, attns,  xfs = [], [], [], []
            flows_down,masks_down, xfs_warp = [], [], []

            # flows: K *[tensor[2,32,32] tensor[2,64,64]]
            # masks: K *[tensor[1,32,32] tensor[1,64,64]]
            # xfs: K *[tensor[256,32,32] tensor[128,64,64]]
            with torch.no_grad():
                gf = GE(g_x)[0:2]
          
            for k in range(0, opt.K):
                # get 2 flows, masks and attns at two resolution 32, 64
                if opt.use_input_mask:
                    flow_ks, mask_ks, attn_ks = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), torch.cat((g_y,g_m),dim=1)) # 32, 64
                    # flow_ks, mask_ks, attn_ks = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), g_y) # 32, 64
                else:
                    flow_ks, mask_ks, attn_ks = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64
                
                # get 2 source features at resolution 32, 64
                xf_ks = GE(ref_xs[k])[0:2]
                flows += [flow_ks]
                masks += [mask_ks]
                attns += [attn_ks]
                xfs += [xf_ks]
            
            # 使每个位置，K个attention的和为1
            # mask_norm: [tensor[K,32,32] tensor[K,64,64]]
            
            attn_norm = []
            for i in range(len(attns[0])):
                temp = []
                for k in range(opt.K):
                    temp += [ attns[k][i] ]
                attn_norm += [F.softmax(torch.cat(temp,dim=1), dim=1)]         
            
            # mask_norm_trans -> K *[tensor[1,32,32] tensor[1,64,64]]
            attn_norm_trans = []
            for k in range(0, opt.K):
                temp = []
                for i in range(0, len(attn_norm)):
                    temp += [attn_norm[i][:,k:k+1,...]] # += [1,32,32]
                attn_norm_trans += [temp]

            if opt.use_input_mask:
                mask_warp_merged = []
                for i in range(len(attn_norm)):
                    temp = []
                    for k in range(opt.K):
                        ref_mk = F.interpolate(ref_ms[k], attn_norm[i].shape[-2:], mode='bilinear', align_corners=opt.align_corner)
                        ref_warped_mk = warp_flow(ref_mk, flows[k][i], align_corners=opt.align_corner)
                        temp += [ref_warped_mk * attn_norm[i][:,k:k+1,...] ]
                    temp_merge = sum(temp)
                    temp_merge_resize = F.interpolate(temp_merge, ref_ms[0].shape[-2:], mode='bilinear', align_corners=opt.align_corner)
                
                    mask_warp_merged += [temp_merge_resize]
                mask_merged = sum(mask_warp_merged) / len(attn_norm)
            
            ### GD input is:
            # flows: K * [B,2,32,32][B,2,64,64]
            # masks: K * [B,1,32,32][B,1,64,64]
            # source_features: K * [B,256,32,32][B,128,64,64]
            # print('xf len:',len(xfs))
            # print('xf shape 1:',xfs[0][0].shape)
            # print('xf shape 2:',xfs[0][1].shape)
            # print('flow len:',len(flows))
            # print('flow shape 1:',flows[0][0].shape)
            # print('flow shape 2:',flows[0][1].shape)
            # print('mask len:',len(mask_norm_trans))
            # print('mask shape 1:',mask_norm_trans[0][0].shape)
            # print('mask shape 2:',mask_norm_trans[0][1].shape)
            if opt.use_pose_decoder:
                if opt.use_input_mask:
                    if opt.GD_use_gt_mask:
                        x_hat = GD(torch.cat((g_y, g_m),dim=1), xfs, flows, masks, attn_norm_trans)
                    elif opt.GD_use_predict_mask:
                        x_hat = GD(torch.cat((g_y, mask_merged),dim=1), xfs, flows, masks, attn_norm_trans)
                else:
                    x_hat, out_flows, out_attns, decode_feats = GD(g_y, xfs, flows, masks, attn_norm_trans)
            else:
                x_hat = GD(xfs, flows, attn_norm_trans)

            if opt.use_adv:
                # Discriminator backward
                optimizerD.zero_grad()
                _unfreeze(D)
                D_real = D(g_x)
                D_fake = D(x_hat.detach())
                D_real_loss = criterion_GAN(D_real, True)
                D_fake_loss = criterion_GAN(D_fake, False)

                lossD = (D_real_loss + D_fake_loss) * 0.5
                writer.add_scalar('{0}/lossD'.format(opt.phase), lossD.item(), global_step=i_batch_total, walltime=None)

                lossD.backward()
                optimizerD.step()

            # Generator backward
            optimizerG.zero_grad()
            

            lossG_content, lossG_style, lossG_L1 = criterionG(g_x, x_hat)
            
            lossG_content = lossG_content * opt.lambda_content
            lossG_style = lossG_style * opt.lambda_style
            lossG_L1 = lossG_L1 * opt.lambda_rec
             
            writer.add_scalar('{0}/lossG_content'.format(opt.phase), lossG_content.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('{0}/lossG_style'.format(opt.phase), lossG_style.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('{0}/lossG_L1'.format(opt.phase), lossG_L1.item(), global_step=i_batch_total, walltime=None)
            lossG = lossG_content + lossG_style + lossG_L1
            # L1, L_vgg
            # GAN
            if opt.use_adv:
                _freeze(D)
                D_fake = D(x_hat)
                lossG_adv = criterion_GAN(D_fake, True) * opt.lambda_adv
                writer.add_scalar('{0}/lossG_adv'.format(opt.phase), lossG_adv.item(), global_step=i_batch_total, walltime=None)
                lossG = lossG + lossG_adv
            
            # sample correctness
            if opt.use_correctness:
                loss_correct=0
                for k in range(opt.K):
                    if opt.single_correctness:
                        loss_correct += criterionCorrectness(g_x, ref_xs[k], out_flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness)
                    else:
                        loss_correct += 0.5 * criterionCorrectness(g_x, ref_xs[k], flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness)
                        loss_correct += 0.5 * criterionCorrectness(g_x, ref_xs[k], out_flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness)
                    # loss_correct += criterionL1(xfs_warp[k], g_xf)
                loss_correct = loss_correct/opt.K * opt.lambda_correctness
                writer.add_scalar('{0}/lossG_correct'.format(opt.phase), loss_correct.item(), global_step=i_batch_total, walltime=None)
                lossG = lossG + loss_correct
            if opt.use_fuse_loss:
                # aggregate different residual warped sources vgg features by the attention maps
                # calculate a fuse loss by the |vgg(gt)-sigma(vgg(W(xi)))|
                loss_fuse = criterionFuse(g_x, ref_xs, out_flows, attn_norm_trans, used_layers=[2, 3])
                loss_fuse = loss_fuse * opt.lambda_fuse_loss
                writer.add_scalar('{0}/lossG_fuse'.format(opt.phase), loss_fuse.item(), global_step=i_batch_total, walltime=None)
                lossG = lossG+ loss_fuse
            # weighted sample correctness
            if opt.use_sim_attn_loss:
                loss_sim_attn = 0 #  sum_i(w_i(P_G-P_i)), wi = exp(-attn)
                sim = []
                
                for k in range(0, opt.K):
                    tt = []
                    for i in range(0, len(attns[k])):
                        b,c,h,w = flows[k][i].shape
                        down_sim = F.interpolate(sims[k], size=(h,w), mode='bilinear',align_corners=opt.align_corner)
                        # temp = torch.ones_like(attns[k][i]) * down_sim
                        tt+=[down_sim]
                    sim += [tt]
                for k in range(0, opt.K):
                    loss_sim_attn += criterionFlowAttn(g_x, ref_xs[k], flows[k], sim[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness) #  
                loss_sim_attn = loss_sim_attn / opt.K * opt.lambda_sim_attn

                lossG += loss_sim_attn
                writer.add_scalar('{0}/loss_sim_attn'.format(opt.phase), loss_sim_attn.item(), global_step=i_batch_total, walltime=None)

            # attn_flow
            if opt.use_flow_attn_loss:
                assert(opt.use_input_mask)
                loss_flow_attn = 0
                for k in range(opt.K):
                    for l in range(len(attns[k])):# [B,1,H,W] * [B,1,H,W]
                        b,c,h,w = flows[k][l].shape
                        ref_m_down = F.interpolate(ref_ms[k], (h,w), mode='bilinear', align_corners=opt.align_corner)
                        warp_m_down = warp_flow(ref_m_down, flows[k][l], align_corners=opt.align_corner)
                        # print(torch.sum(torch.norm(flows[k][l],dim=1,keepdim=True)*warp_m_down * attn_norm_trans[k][l])/torch.sum(warp_m_down))
                        loss_flow_attn += torch.sum(torch.norm(flows[k][l],dim=1,keepdim=True)*warp_m_down * attn_norm_trans[k][l])/torch.sum(warp_m_down)
                loss_flow_attn = loss_flow_attn / opt.K * opt.lambda_flow_attn
                lossG += loss_flow_attn
                writer.add_scalar('{0}/loss_flow_attn'.format(opt.phase), loss_flow_attn.item(), global_step=i_batch_total, walltime=None)

            # attn_sim
            if opt.use_attn_reg:
                loss_attn_reg = 0 #  sum(attn_i * sim_i)/2 + attn_GD /2 (sum(attn_i,attn_GD)==1, sum(sim_i)==1)
                for k in range(0, opt.K):
                    for i in range(0, len(attns[k])):
                        b,c,h,w = flows[0][i].shape
                        # ref_m_down = F.interpolate(ref_ms[k], (h,w), mode='bilinear', align_corners=opt.align_corner)
                        # warp_m_down = warp_flow(ref_m_down, flows[k][l], align_corners=opt.align_corner)
                        down_sim = F.interpolate(sims[k], size=(h,w), mode='bilinear',align_corners=opt.align_corner)
                        down_gm = F.interpolate(g_m, size=(h,w), mode='bilinear',align_corners=opt.align_corner)
                        loss_attn_reg += torch.mean(attn_norm_trans[k][i] * (1-down_sim) * down_gm)
                    # loss_attn_reg += torch.mean(attn_normed[i][:,0:1,...] /(opt.K+1))
                loss_attn_reg = loss_attn_reg / opt.K * opt.lambda_attn_reg
                lossG += loss_attn_reg
                writer.add_scalar('{0}/loss_attn_reg'.format(opt.phase), loss_attn_reg.item(), global_step=i_batch_total, walltime=None)

            # flow_reg
            if opt.use_flow_reg:
                loss_regular=0
                for k in range(opt.K):
                    loss_regular += criterionReg(flows[k])
                    if not opt.use_single_reg:
                        loss_regular += criterionReg(out_flows[k])
                
                loss_regular = loss_regular / opt.K * opt.lambda_flow_reg
            
                lossG += loss_regular
                writer.add_scalar('{0}/loss_regular'.format(opt.phase), loss_regular.item(), global_step=i_batch_total, walltime=None)

            

            # flow_mask
            loss_struct = 0
            if opt.use_input_mask:
                for k in range(opt.K):
                    flow_ks = flows[k]
                    for l in range(len(flow_ks)):
                        flow_kl = flow_ks[l]
                        ref_m_down = F.interpolate(ref_ms[k], size=flow_kl.shape[2:], mode='bilinear', align_corners=opt.align_corner)
                        warp_m_down = warp_flow(ref_m_down, flow_kl, align_corners=opt.align_corner)
                        g_m_down = F.interpolate(g_m, size=flow_kl.shape[2:], mode='bilinear', align_corners=opt.align_corner)
                        loss_struct += criterionL1(warp_m_down, g_m_down)
                loss_struct = loss_struct / opt.K * opt.lambda_struct_correctness
                lossG += loss_struct
                writer.add_scalar('{0}/loss_struct'.format(opt.phase), loss_struct.item(), global_step=i_batch_total, walltime=None)
            

            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            post_fix_str = 'Epo_L=%.3f, G=%.2f,L1=%.2f,L_cnt=%.2f,L_sty=%.2f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
            if opt.use_correctness:
                post_fix_str += ', corr=%.2f'%(loss_correct.item())
            if opt.use_adv:
                post_fix_str += ', G_adv=%.2f, D_L=%.2f'%(lossG_adv.item(), lossD.item())
            if opt.use_flow_reg:
                post_fix_str += ', reg=%.2f'%(loss_regular.item())
            if opt.use_sim_attn_loss:
                post_fix_str += ', sim_atn=%.2f'%(loss_sim_attn.item())
            if opt.use_flow_attn_loss:
                post_fix_str += ', flo_atn=%.2f'%(loss_flow_attn.item())
            if opt.use_attn_reg:
                post_fix_str += ', atnreg=%.2f'%(loss_attn_reg.item())
            if opt.use_input_mask:
                post_fix_str += ', struct=%.2f'%(loss_struct.item())
            
            if opt.use_fuse_loss:
                post_fix_str += ', fuse=%.2f'%(loss_fuse.item())
            

            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            if i_batch % opt.img_save_freq == 0:
                if not opt.use_input_mask: 
                    final_img,_,_ = get_pyramid_visualize_result(opt, ref_xs, ref_ys,None, g_x,x_hat, g_y,None,None, \
                                flows, attn_norm_trans,masks, xfs, gf, out_flows, out_attns, decode_feats)
                else:
                    final_img,_,_ = get_pyramid_visualize_result(opt, ref_xs, ref_ys,ref_ms, g_x,x_hat, g_y,g_m,mask_merged, \
                                flows, attn_norm_trans,masks, xfs, gf, out_flows, out_attns, decode_feats)

                if opt.use_sim_attn_loss or opt.use_attn_reg:
                    sims_vis = [tensor2im(sims[i], 0, is_mask=True).type(torch.uint8).to(cpu).numpy() for i in range(opt.K)]
                    masked_sims_vis = [tensor2im(sims[i]*g_m, 0, is_mask=True).type(torch.uint8).to(cpu).numpy() for i in range(opt.K)]
                    
                    for k in range(opt.K):
                        rows = final_img.shape[0]//256
                        final_img[256*(rows-2):256*(rows-1), 256*(opt.K*2+k):256*(opt.K*2+k+1)] = sims_vis[k]
                        final_img[256*(rows-1):256*(rows), 256*(opt.K*2+k):256*(opt.K*2+k+1)] = masked_sims_vis[k]
                Image.fromarray(final_img).save(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)))    

            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                path_to_save_D = path_to_ckpt_dir + 'epoch_{}_batch_{}_D.tar'.format(epoch, i_batch)
                save_generator(opt.parallel, epoch, lossesG, GE, GF, GD, i_batch, optimizerG, path_to_save_G)
                save_discriminator(opt.parallel, D, optimizerD, path_to_save_D) 
        
        save_generator(opt.parallel, epoch+1, lossesG, GE, GF, GD, i_batch, optimizerG, save_pathG)
        save_discriminator(opt.parallel, D, optimizerD, save_pathD)
        '''save image result'''
        if not opt.use_input_mask: 
            final_img,_,_ = get_pyramid_visualize_result(opt, ref_xs, ref_ys,None, g_x,x_hat, g_y,None,None, \
                        flows, attn_norm_trans,masks, xfs, gf, out_flows, out_attns)
        else:
            final_img,_,_ = get_pyramid_visualize_result(opt, ref_xs, ref_ys,ref_ms, g_x,x_hat, g_y,g_m,None, \
                        flows, attn_norm_trans,masks, xfs, gf, out_flows, out_attns)
        
        Image.fromarray(final_img).save(os.path.join(path_to_visualize_dir,"epoch_latest.png"))
        # plt.imsave(os.path.join(path_to_visualize_dir,"epoch_latest.png"), final_img)
        
            
    writer.close()


def test(opt):
    from util.io import load_image, load_skeleton, load_parsing, transform_image
    import time
    

    print('-----------TESTING-----------')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    experiment_name = opt.test_id
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(experiment_name)
    
    path_to_chkpt_G = path_to_ckpt_dir + '{0}.tar'.format(opt.test_ckpt_name) 
    test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}_shot/'.format(experiment_name, opt.test_ckpt_name, opt.K)
    test_result_eval_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}_shot_eval/'.format(experiment_name, opt.test_ckpt_name, opt.K)
    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.isdir(test_result_eval_dir):
        os.makedirs(test_result_eval_dir)
    '''save parser'''
    
    '''Create dataset and dataloader'''
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    '''Create Model'''
    image_nc = 3
    structure_nc = 21 if opt.use_bone_RGB else 18
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_input_mask:
        structure_nc += 1
    if opt.use_simmap:
        image_nc += 13

    GF = FlowGenerator(inc=image_nc+structure_nc*2, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    if not opt.use_pose_decoder:
        GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    else:
        GD = PoseAwareResidualFlowDecoder(structure_nc=structure_nc,n_res_blocks=opt.n_res_block, n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample, use_res_attn=opt.use_res_attn)
    
    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
        GE = nn.DataParallel(GE)
        GD = nn.DataParallel(GD)

    GF = GF.to(device)
    GE = GE.to(device)
    GD = GD.to(device)
    
    '''Loading from past checkpoint_G'''
    checkpoint_G = torch.load(path_to_chkpt_G, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint_G['GF_state_dict'], strict=True)
        GE.module.load_state_dict(checkpoint_G['GE_state_dict'], strict=True)
        GD.module.load_state_dict(checkpoint_G['GD_state_dict'], strict=True)
    else:
        GF.load_state_dict(checkpoint_G['GF_state_dict'], strict=True)
        GE.load_state_dict(checkpoint_G['GE_state_dict'], strict=True)
        GD.load_state_dict(checkpoint_G['GD_state_dict'], strict=True)
    epochCurrent = checkpoint_G['epoch']
    GF.eval()
    GE.eval()
    GD.eval()
    if opt.GD_use_predict_mask:
        from model.Parsing_net import MaskGenerator
        GP = MaskGenerator(inc=image_nc+structure_nc,onc=1, n_layers=5,ngf=64, max_nc=512, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
        if opt.parallel:
            GP = nn.DataParallel(GP) # dx + dx + dy = 3 + 20 + 20
        GP = GP.to(device)
        GP.eval()

        path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format('fashion_pretrain_masknet_2shot_20210107')

        path_to_chkpt = path_to_ckpt_dir + 'epoch_3_batch_5000_G.tar'
        checkpoint = torch.load(path_to_chkpt, map_location=cpu)
        if opt.parallel:
            GP.module.load_state_dict(checkpoint['GP_state_dict'], strict=False)
        else:
            GP.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    '''Losses'''
    criterionG = LossG(device=device)
    criterion_GAN = AdversarialLoss(type='lsgan').to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)

    """ Test start """
    epoch_loss_G = 0
    pbar = tqdm(dataloader, leave=True, initial=0)
    num_test_samples = opt.test_samples if opt.test_samples > 0 else len(dataloader)
    for i_batch, batch_data in enumerate(pbar, start=0):
        if i_batch == num_test_samples:
            break
        model_G_start = time.time()
        froms = batch_data['froms']
        to = batch_data['to']
        ref_xs = batch_data['ref_xs']
        ref_ys = batch_data['ref_ys']
        g_x = batch_data['g_x']
        g_y = batch_data['g_y']
        assert(len(ref_xs)==len(ref_ys))
        assert(len(ref_xs)==opt.K)
        for i in range(len(ref_xs)):
            ref_xs[i] = ref_xs[i].to(device)
            ref_ys[i] = ref_ys[i].to(device)

        # if opt.use_sim_attn_loss or opt.use_attn_reg:
        sims = batch_data['sim']
        assert(len(sims)==opt.K)
        for i in range(opt.K):
            sims[i] = sims[i].to(device)
        if opt.use_input_mask or opt.use_attn_reg:
            ref_ms = batch_data['ref_ms']
            g_m = batch_data['g_m']
            assert(len(ref_ms)==opt.K)
            for i in range(opt.K):
                ref_ms[i] = ref_ms[i].to(device)
            g_m = g_m.to(device)
        g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]
        
        flows, masks, attns,  xfs = [], [], [], []
        flows_down,masks_down, xfs_warp = [], [], []

        # flows: K *[tensor[2,32,32] tensor[2,64,64]]
        # masks: K *[tensor[1,32,32] tensor[1,64,64]]
        # xfs: K *[tensor[256,32,32] tensor[128,64,64]]
        with torch.no_grad():
            gf = GE(g_x)[0:2]
        if opt.GD_use_predict_mask:
            logits = []
            for k in range(0, opt.K):
                logit_k,_ = GP(ref_xs[k], ref_ms[k], g_y) # [B, 1, H, W], [B, 1, H, W]
                logits += [logit_k]
            
            logit_avg = sum(logits) / opt.K #  [B, 1, H, W]
            m_hat = nn.Sigmoid()(logit_avg)
            if not opt.use_logits:
                m_hat[m_hat>0.5]=1
                m_hat[m_hat<=0.5]=0
        
        for k in range(0, opt.K):
            # get 2 flows and masks at two resolution 32, 64
            if opt.use_input_mask:
                if opt.GD_use_predict_mask:
                    flow_ks, mask_ks, attn_ks = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), torch.cat((g_y,m_hat),dim=1)) # 32, 64
                else:
                    flow_ks, mask_ks, attn_ks = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), torch.cat((g_y,g_m),dim=1)) # 32, 64

                # flow_ks, mask_ks, attn_ks = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), g_y) # 32, 64
            else:
                flow_ks, mask_ks, attn_ks = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64

            
            # get 2 source features at resolution 32, 64
            xf_ks = GE(ref_xs[k])[0:2]

            flows += [flow_ks]
            masks += [mask_ks]
            attns += [attn_ks]
            xfs += [xf_ks]
        
        # 使每个位置，K个attention的和为1
        # mask_norm: [tensor[K,32,32] tensor[K,64,64]]
        attn_norm = []
        for i in range(len(attns[0])):
            temp = []
            for k in range(opt.K):
                temp += [ attns[k][i] ]
            attn_norm += [F.softmax(torch.cat(temp,dim=1), dim=1)]         
        
        # mask_norm_trans -> K *[tensor[1,32,32] tensor[1,64,64]]
        attn_norm_trans = []
        for k in range(0, opt.K):
            temp = []
            for i in range(0, len(attn_norm)):
                temp += [attn_norm[i][:,k:k+1,...]] # += [1,32,32]
            attn_norm_trans += [temp]

        if opt.use_input_mask:
            mask_warp_merged = []
            for i in range(len(attn_norm)):
                temp = []
                for k in range(opt.K):
                    ref_mk = F.interpolate(ref_ms[k], attn_norm[i].shape[-2:], mode='bilinear', align_corners=opt.align_corner)
                    ref_warped_mk = warp_flow(ref_mk, flows[k][i], align_corners=opt.align_corner)
                    temp += [ref_warped_mk * attn_norm[i][:,k:k+1,...] ]
                temp_merge = sum(temp)
                temp_merge_resize = F.interpolate(temp_merge, ref_ms[0].shape[-2:], mode='bilinear', align_corners=opt.align_corner)
            
                mask_warp_merged += [temp_merge_resize]
            mask_merged = sum(mask_warp_merged) / len(attn_norm)
            
        ### GD input is:
        # flows: K * [B,2,32,32][B,2,64,64]
        # masks: K * [B,1,32,32][B,1,64,64]
        # source_features: K * [B,256,32,32][B,128,64,64]
        # print('xf len:',len(xfs))
        # print('xf shape 1:',xfs[0][0].shape)
        # print('xf shape 2:',xfs[0][1].shape)
        # print('flow len:',len(flows))
        # print('flow shape 1:',flows[0][0].shape)
        # print('flow shape 2:',flows[0][1].shape)
        # print('mask len:',len(mask_norm_trans))
        # print('mask shape 1:',mask_norm_trans[0][0].shape)
        # print('mask shape 2:',mask_norm_trans[0][1].shape)
        if opt.use_pose_decoder:
            if opt.use_input_mask:
                if opt.GD_use_gt_mask:
                    x_hat = GD(torch.cat((g_y, g_m),dim=1), xfs, flows, masks, attn_norm_trans)
                elif opt.GD_use_predict_mask:
                    # x_hat = GD(torch.cat((g_y, mask_merged),dim=1), xfs, flows, masks, attn_norm_trans)
                    x_hat = GD(torch.cat((g_y, m_hat),dim=1), xfs, flows, masks, attn_norm_trans)
            else:
                x_hat, out_flows, out_attns, decode_feats,merged_feats = GD(g_y, xfs, flows, masks, attn_norm_trans)
        else:
            x_hat = GD(xfs, flows, attn_norm_trans)
        

        # print('model G time:%.3f'%(model_G_end-model_G_start))
        lossG_content, lossG_style, lossG_L1 = criterionG(g_x, x_hat)
            
        lossG_content = lossG_content * opt.lambda_content
        lossG_style = lossG_style * opt.lambda_style
        lossG_L1 = lossG_L1 * opt.lambda_rec
        lossG = lossG_content + lossG_style + lossG_L1

        epoch_loss_G += lossG.item()
        epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
        
        post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
        
        pbar.set_postfix_str(post_fix_str)
        visual_start = time.time()

        if not opt.use_input_mask:
            ref_ps = None
            gp = None
            gp_hat = None
        else:
            ref_ps = ref_ms
            gp = g_m
            gp_hat = mask_merged
        
        full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
            ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=ref_ps, gx=g_x, x_hat=x_hat, gy=g_y,\
                gp=gp,gp_hat=gp_hat,flows=flows, attns_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf, refined_flows=out_flows, refined_attns=out_attns, decode_feats=decode_feats, merged_feats=merged_feats)
        
        # full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
        #     ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=ref_ps, gx=g_x, x_hat=x_hat, gy=g_y,\
        #         gp=gp,gp_hat=gp_hat,flows=flows, attns_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf, refined_flows=None, refined_attns=None)
        
        visual_end = time.time()
        # print('visual time:%.3f'%(visual_end-visual_start))
        
        save_start = time.time()
        test_name = ''
        # print(froms)
        for k in range(opt.K):
            test_name += str(froms[k][0])+'+'
        test_name += str(to[0])
        
        Image.fromarray(out_img).save(os.path.join(test_result_eval_dir,"{0}_vis.jpg".format(test_name)))
        Image.fromarray(simp_img).save(os.path.join(test_result_dir,"{0}_simp.jpg".format(test_name)))
        if opt.output_all:
            
            # if opt.use_sim_attn_loss or opt.use_attn_reg:
            sims_vis = [tensor2im(sims[i], 0, is_mask=True).type(torch.uint8).to(cpu).numpy() for i in range(opt.K)]
            # masked_sims_vis = [tensor2im(sims[i]*g_m, 0, is_mask=True).type(torch.uint8).to(cpu).numpy() for i in range(opt.K)]
            
            for i in range(opt.K):
                full_img[256*3:256*4, 256*(opt.K*2+i):256*(opt.K*2+i+1)] = sims_vis[i]
                # full_img[256*7:256*8, 256*(opt.K+i):256*(opt.K+i+1)] = masked_sims_vis[i]
            Image.fromarray(full_img).save(os.path.join(test_result_dir,"{0}_all.jpg".format(test_name)))
        save_end = time.time()
        # print('save time:%.3f'%(save_end-save_start))
        
    pass



def test_for_arbitrary_pose(opt, source_idx, video_id='91-3003CN5S'):
    from util.io import load_image, load_skeleton, load_parsing, transform_image
    import time
    
    B_dir = '/dataset/ljw/danceFashion/test_256/train_alphapose/'
    B_paths = os.listdir(os.path.join(B_dir, video_id))
    g_ys = [load_skeleton(os.path.join(B_dir, video_id, B_path))[0] for B_path in sorted(B_paths)]

    print('-----------TESTING-----------')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    experiment_name = opt.test_id
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(experiment_name)
    
    path_to_chkpt_G = path_to_ckpt_dir + '{0}.tar'.format(opt.test_ckpt_name) 
    test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/{3}_shot/'.format(experiment_name, opt.test_ckpt_name, source_idx, opt.K)
    test_result_eval_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/{3}_shot_eval/'.format(experiment_name, opt.test_ckpt_name,source_idx, opt.K)
    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.isdir(test_result_eval_dir):
        os.makedirs(test_result_eval_dir)
    '''save parser'''
    
    '''Create dataset and dataloader'''
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    '''Create Model'''
    image_nc = 3
    structure_nc = 21 if opt.use_bone_RGB else 18
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_input_mask:
        structure_nc += 1
    if opt.use_simmap:
        image_nc += 13

    GF = FlowGenerator(inc=image_nc+structure_nc*2, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    if not opt.use_pose_decoder:
        GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    else:
        GD = PoseAwareResidualFlowDecoder(structure_nc=structure_nc,n_res_blocks=opt.n_res_block, n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample, use_res_attn=opt.use_res_attn)
    
    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
        GE = nn.DataParallel(GE)
        GD = nn.DataParallel(GD)

    GF = GF.to(device)
    GE = GE.to(device)
    GD = GD.to(device)
    
    '''Loading from past checkpoint_G'''
    checkpoint_G = torch.load(path_to_chkpt_G, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint_G['GF_state_dict'], strict=True)
        GE.module.load_state_dict(checkpoint_G['GE_state_dict'], strict=True)
        GD.module.load_state_dict(checkpoint_G['GD_state_dict'], strict=True)
    else:
        GF.load_state_dict(checkpoint_G['GF_state_dict'], strict=True)
        GE.load_state_dict(checkpoint_G['GE_state_dict'], strict=True)
        GD.load_state_dict(checkpoint_G['GD_state_dict'], strict=True)
    epochCurrent = checkpoint_G['epoch']
    GF.eval()
    GE.eval()
    GD.eval()
    if opt.GD_use_predict_mask:
        from model.Parsing_net import MaskGenerator
        GP = MaskGenerator(inc=image_nc+structure_nc,onc=1, n_layers=5,ngf=64, max_nc=512, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
        if opt.parallel:
            GP = nn.DataParallel(GP) # dx + dx + dy = 3 + 20 + 20
        GP = GP.to(device)
        GP.eval()

        path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format('fashion_pretrain_masknet_2shot_20210107')

        path_to_chkpt = path_to_ckpt_dir + 'epoch_3_batch_5000_G.tar'
        checkpoint = torch.load(path_to_chkpt, map_location=cpu)
        if opt.parallel:
            GP.module.load_state_dict(checkpoint['GP_state_dict'], strict=False)
        else:
            GP.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    '''Losses'''
    criterionG = LossG(device=device)
    criterion_GAN = AdversarialLoss(type='lsgan').to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)

    """ Test start """
    epoch_loss_G = 0
    pbar = tqdm(dataloader, leave=True, initial=0)
    num_test_samples = opt.test_samples if opt.test_samples > 0 else len(dataloader)
    
    ref_xs = dataset[source_idx]['ref_xs']
    ref_ys = dataset[source_idx]['ref_ys']
    froms = dataset[source_idx]['froms']
    for k in range(opt.K):
        ref_xs[k] = ref_xs[k].unsqueeze(0)
        ref_ys[k] = ref_ys[k].unsqueeze(0)
    assert(len(ref_xs)==len(ref_ys))
    assert(len(ref_xs)==opt.K)
    for i in range(len(ref_xs)):
        ref_xs[i] = ref_xs[i].to(device)
        ref_ys[i] = ref_ys[i].to(device)

    

    i_batch=0
    # for i_batch, batch_data in enumerate(pbar, start=0):
    for i_batch in range(len(g_ys)):
        print(i_batch)
        if i_batch == num_test_samples:
            break
        # froms = batch_data['froms']
        # to = batch_data['to']
        # ref_xs = batch_data['ref_xs']
        # ref_ys = batch_data['ref_ys']
        # g_x = batch_data['g_x']
        g_x = ref_xs[0]
        g_y = g_ys[i_batch].unsqueeze(0)
        # g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]
        
        flows, masks, attns,  xfs = [], [], [], []
        flows_down,masks_down, xfs_warp = [], [], []

        # flows: K *[tensor[2,32,32] tensor[2,64,64]]
        # masks: K *[tensor[1,32,32] tensor[1,64,64]]
        # xfs: K *[tensor[256,32,32] tensor[128,64,64]]
        with torch.no_grad():
            gf = GE(g_x)[0:2]
        
        for k in range(0, opt.K):
            # get 2 flows and masks at two resolution 32, 64
            flow_ks, mask_ks, attn_ks = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64
            xf_ks = GE(ref_xs[k])[0:2]

            flows += [flow_ks]
            masks += [mask_ks]
            attns += [attn_ks]
            xfs += [xf_ks]
        
        # 使每个位置，K个attention的和为1
        # mask_norm: [tensor[K,32,32] tensor[K,64,64]]
        attn_norm = []
        for i in range(len(attns[0])):
            temp = []
            for k in range(opt.K):
                temp += [ attns[k][i] ]
            attn_norm += [F.softmax(torch.cat(temp,dim=1), dim=1)]         
        
        # mask_norm_trans -> K *[tensor[1,32,32] tensor[1,64,64]]
        attn_norm_trans = []
        for k in range(0, opt.K):
            temp = []
            for i in range(0, len(attn_norm)):
                temp += [attn_norm[i][:,k:k+1,...]] # += [1,32,32]
            attn_norm_trans += [temp]

        if opt.use_input_mask:
            mask_warp_merged = []
            for i in range(len(attn_norm)):
                temp = []
                for k in range(opt.K):
                    ref_mk = F.interpolate(ref_ms[k], attn_norm[i].shape[-2:], mode='bilinear', align_corners=opt.align_corner)
                    ref_warped_mk = warp_flow(ref_mk, flows[k][i], align_corners=opt.align_corner)
                    temp += [ref_warped_mk * attn_norm[i][:,k:k+1,...] ]
                temp_merge = sum(temp)
                temp_merge_resize = F.interpolate(temp_merge, ref_ms[0].shape[-2:], mode='bilinear', align_corners=opt.align_corner)
            
                mask_warp_merged += [temp_merge_resize]
            mask_merged = sum(mask_warp_merged) / len(attn_norm)
            
        ### GD input is:
        # flows: K * [B,2,32,32][B,2,64,64]
        # masks: K * [B,1,32,32][B,1,64,64]
        # source_features: K * [B,256,32,32][B,128,64,64]
        # print('xf len:',len(xfs))
        # print('xf shape 1:',xfs[0][0].shape)
        # print('xf shape 2:',xfs[0][1].shape)
        # print('flow len:',len(flows))
        # print('flow shape 1:',flows[0][0].shape)
        # print('flow shape 2:',flows[0][1].shape)
        # print('mask len:',len(mask_norm_trans))
        # print('mask shape 1:',mask_norm_trans[0][0].shape)
        # print('mask shape 2:',mask_norm_trans[0][1].shape)
        if opt.use_pose_decoder:
            if opt.use_input_mask:
                if opt.GD_use_gt_mask:
                    x_hat = GD(torch.cat((g_y, g_m),dim=1), xfs, flows, masks, attn_norm_trans)
                elif opt.GD_use_predict_mask:
                    # x_hat = GD(torch.cat((g_y, mask_merged),dim=1), xfs, flows, masks, attn_norm_trans)
                    x_hat = GD(torch.cat((g_y, m_hat),dim=1), xfs, flows, masks, attn_norm_trans)
            else:
                x_hat, out_flows, out_attns, decode_feats = GD(g_y, xfs, flows, masks, attn_norm_trans)
        else:
            x_hat = GD(xfs, flows, attn_norm_trans)
        

        if not opt.use_input_mask:
            ref_ps = None
            gp = None
            gp_hat = None
        else:
            ref_ps = ref_ms
            gp = g_m
            gp_hat = mask_merged
        
        full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
            ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=ref_ps, gx=g_x, x_hat=x_hat, gy=g_y,\
                gp=gp,gp_hat=gp_hat,flows=flows, attns_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf, refined_flows=out_flows, refined_attns=out_attns, decode_feats=decode_feats, merged_feats=None)
        
        # full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
        #     ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=ref_ps, gx=g_x, x_hat=x_hat, gy=g_y,\
        #         gp=gp,gp_hat=gp_hat,flows=flows, attns_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf, refined_flows=None, refined_attns=None)
        
        visual_end = time.time()
        # print('visual time:%.3f'%(visual_end-visual_start))
        
        save_start = time.time()
        test_name = ''
        # print(froms)
        for k in range(opt.K):
            test_name += str(froms[k])+'+'
        test_name += str(i_batch)
        
        Image.fromarray(out_img).save(os.path.join(test_result_eval_dir,"{0}_vis.jpg".format(test_name)))
        Image.fromarray(simp_img).save(os.path.join(test_result_dir,"{0}_simp.jpg".format(test_name)))
        if opt.output_all:
            Image.fromarray(full_img).save(os.path.join(test_result_dir,"{0}_all.jpg".format(test_name)))
        save_end = time.time()
        # print('save time:%.3f'%(save_end-save_start))
        
    pass

def test_flownet(opt):
    from util.io import load_image, load_skeleton, load_parsing, transform_image
    import time
    

    print('-----------TESTING-----------')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    experiment_name = opt.test_id
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(experiment_name)
    
    path_to_chkpt_G = path_to_ckpt_dir + '{0}.tar'.format(opt.test_ckpt_name) 
    test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}_shot/'.format(experiment_name, opt.test_ckpt_name, opt.K)
    test_result_eval_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}_shot_eval/'.format(experiment_name, opt.test_ckpt_name, opt.K)
    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)
    if not os.path.isdir(test_result_eval_dir):
        os.makedirs(test_result_eval_dir)

    
    '''Create dataset and dataloader'''
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    '''Create Model'''
    image_nc = 3
    structure_nc = 21 if opt.use_bone_RGB else 18
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_input_mask:
        structure_nc += 1
    if opt.use_simmap:
        image_nc += 13
    
    GF = FlowGenerator(inc=3+21+22, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
    GF = GF.to(device)
    checkpoint = torch.load(path_to_chkpt_G, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    else:
        GF.load_state_dict(checkpoint['GF_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    i_batch_current = checkpoint['i_batch']
    from loss.externel_functions import VGG19
    GE = VGG19().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)

    """ Test start """
    epoch_loss_G = 0
    pbar = tqdm(dataloader, leave=True, initial=0)
    num_test_samples = opt.test_samples if opt.test_samples > 0 else len(dataloader)
    for i_batch, batch_data in enumerate(pbar, start=0):
        if i_batch == num_test_samples:
            break
        model_G_start = time.time()
        froms = batch_data['froms']
        to = batch_data['to']
        ref_xs = batch_data['ref_xs']
        ref_ys = batch_data['ref_ys']
        g_x = batch_data['g_x']
        g_y = batch_data['g_y']
        assert(len(ref_xs)==len(ref_ys))
        assert(len(ref_xs)==opt.K)
        for i in range(len(ref_xs)):
            ref_xs[i] = ref_xs[i].to(device)
            ref_ys[i] = ref_ys[i].to(device)
        if opt.use_input_mask:
            ref_ms = batch_data['ref_ms']
            g_m = batch_data['g_m']
            assert(len(ref_ms)==opt.K)
            for i in range(opt.K):
                ref_ms[i] = ref_ms[i].to(device)
            g_m = g_m.to(device)
        if opt.use_sim_attn_loss or opt.use_attn_reg:
            sims = batch_data['sim']
            assert(len(sims)==opt.K)
            for i in range(opt.K):
                sims[i] = sims[i].to(device)


        g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]
        
        flows, flow_ones, masks, xfs = [], [], [], []
        flows_down,masks_down, xfs_warp = [], [], []
        # flows: K *[tensor[2,32,32] tensor[2,64,64]]
        # masks: K *[tensor[1,32,32] tensor[1,64,64]]
        # xfs: K *[tensor[256,32,32] tensor[128,64,64]]
        g_xf = GE(g_x)['relu4_1']
        
        for k in range(0, opt.K):
            if opt.use_input_mask:
                # flow_ks, mask_ks, _ = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), torch.cat((g_y,g_m),dim=1)) # 32, 64
                flow_ks, mask_ks, _ = GF(ref_xs[k], torch.cat((ref_ys[k],ref_ms[k]),dim=1), g_y) # 32, 64
            else:
                flow_ks, mask_ks, _ = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64

            flow_k = flow_ks[1]
            # print(flow_k.shape)
            mask_k = mask_ks[1]
            # print(mask_k.shape)

            xf_k = GE(ref_xs[k])['relu4_1']
            # print(xf_k.shape)
            flow_k_down = F.interpolate(flow_k * xf_k.shape[2] / flow_k.shape[2], size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)

            mask_k_down = F.interpolate(mask_k, size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
            xf_k_warp = warp_flow(xf_k, flow_k_down, align_corners=opt.align_corner)

            flows += [flow_ks]
            flow_ones += [flow_k ]
            masks += [mask_k]
            # attns += [mask_k]
            xfs += [xf_k]
            flows_down += [flow_k_down]
            masks_down += [mask_k_down]
            xfs_warp += [xf_k_warp]
        '''normalize masks to sum to 1'''
        mask_cat = torch.cat(masks_down, dim=1)
        if opt.mask_norm_type == 'softmax':
            mask_normed = F.softmax(mask_cat, dim=1) # pixel wise sum to 1
        else:
            eps = 1e-12
            mask_normed = mask_cat / (torch.sum(mask_cat, dim=1).unsqueeze(1)+eps) # pixel wise sum to 1

        xfs_warp_masked = None
        xf_merge = None
        for k in range(0, opt.K):
            mask_normed_k_down = mask_normed[:,k:k+1,...]

            if xfs_warp_masked is None:
                xfs_warp_masked = [xfs_warp[k] * mask_normed_k_down]
                xf_merge = xfs_warp[k] * mask_normed_k_down
            else:
                xfs_warp_masked.append(xfs_warp[k] * mask_normed_k_down)
                xf_merge += xfs_warp[k] * mask_normed_k_down
            
        down_ref_xk = F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear', align_corners=opt.align_corner)
        x_hat_down =  warp_flow(down_ref_xk, flow_k, align_corners=opt.align_corner)
        x_hat = F.interpolate(x_hat_down, size=ref_xs[k].shape[2:], mode='bilinear', align_corners=opt.align_corner)
        loss_correct=0
        for k in range(opt.K):
            loss_correct += criterionCorrectness(g_x, ref_xs[k], flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness)
            # loss_correct += criterionL1(xfs_warp[k], g_xf)
        loss_correct = loss_correct/opt.K * opt.lambda_correctness
        lossG = loss_correct
        
        loss_regular = 0
        if opt.use_flow_reg:
            for k in range(opt.K):
                loss_regular += criterionReg(flows[k])
            
            loss_regular = loss_regular / opt.K * opt.lambda_flow_reg
            lossG += loss_regular
        
        loss_struct = 0
        if opt.use_input_mask:
            for k in range(opt.K):
                flow_ks = flows[k]
                for l in range(len(flow_ks)):
                    flow_kl = flow_ks[l]
                    ref_m_down = F.interpolate(ref_ms[k], size=flow_kl.shape[2:], mode='bilinear', align_corners=opt.align_corner)
                    warp_m_down = warp_flow(ref_m_down, flow_kl, align_corners=opt.align_corner)
                    g_m_down = F.interpolate(g_m, size=flow_kl.shape[2:], mode='bilinear', align_corners=opt.align_corner)
                    loss_struct += criterionL1(warp_m_down, g_m_down)
            loss_struct /= opt.K * opt.lambda_struct_correctness
            lossG += loss_struct
        
        epoch_loss_G += lossG.item()
        epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
        post_fix_str = 'Epoch_loss=%.3f, G=%.3f, Cor=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_correct.item())
        if opt.use_input_mask:
            post_fix_str += ', Reg=%.3f'%loss_regular.item()
        if opt.use_flow_reg:
            post_fix_str += ', MaskCorr=%.3f'%loss_struct.item()
        pbar.set_postfix_str(post_fix_str)
        
        lossesG.append(lossG.item())
        test_name = ''
        # print(froms)
        for k in range(opt.K):
            test_name += str(froms[k][0])+'+'
        test_name += str(to[0])
        from PIL import Image
        from util.vis_util import get_visualize_result
        if not opt.use_input_mask:   
            final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,None, g_x, g_y,None,None, g_xf, x_hat,\
                flow_ones, F.interpolate(mask_normed, size=g_x.shape[2:], mode='bilinear',align_corners=opt.align_corner), xfs, xfs_warp, xfs_warp_masked)
        else:
            ref_m_down = F.interpolate(ref_ms[0], size=flow_ones[0].shape[2:], mode='bilinear', align_corners=opt.align_corner)
            m_hat = warp_flow(ref_m_down,flow_ones[0],align_corners=opt.align_corner)
            final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,ref_ms, g_x, g_y,g_m,m_hat, g_xf, x_hat,\
                flow_ones, F.interpolate(mask_normed, size=g_x.shape[2:], mode='bilinear',align_corners=opt.align_corner), xfs, xfs_warp, xfs_warp_masked)
        Image.fromarray(final_img).save(os.path.join(test_result_dir,"{0}_all.jpg".format(test_name)))

pass

if __name__ == "__main__":
    opt = get_parser()
    for k,v in sorted(vars(opt).items()):
        print(k,':',v)
    set_random_seed(opt.seed)
    
    if opt.phase == 'train':
        today = datetime.today().strftime("%Y%m%d")
        # today = '20201123'
        # today = '20210106'
        # today = '20210105'
        # today = '20210111'
        # today = '20210321'
        today = '20210331'
        # today = '20210412'
        experiment_name = '{0}_{1}shot_{2}'.format(opt.id,opt.K, today)
        print(experiment_name)
        if opt.pretrain_flow:
            train_flow_net(opt, experiment_name)
        else:
            train(opt, experiment_name)
    else:
        with torch.no_grad():
            if opt.pretrain_flow:
                test_flownet(opt)
            else:
                test(opt)
                # source_idx = 528
                # # video_id = '91-3003CN5S'
                # # test_for_arbitrary_pose(opt, source_idx, video_id)
                # test_for_arbitrary_pose(opt, source_idx)




