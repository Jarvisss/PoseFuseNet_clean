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
# from model.Parsing_net import ParsingGenerator
# from model.pyramid_flow_generator import FlowGenerator
from model.pose_encoder import AutoEncoder
from model.pyramid_part_flow_generator import PartFlowGenerator, PoseAwarePartAppearanceDecoder
from model.pyramid_flow_generator_with_occlu_attn import FlowGenerator,PoseAttnFCNGenerator,AppearanceDecoder, AppearanceEncoder, PoseAwareAppearanceDecoder
from model.discriminator import ResDiscriminator
from model.blocks import warp_flow,_freeze,_unfreeze, gen_uniform_grid

from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_layer_warp_visualize_result,get_layer_warp_K_visualize_result
from util.vis_util import tensor2im
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashion_dataset import FashionDataset
from loss.loss_generator import PerceptualCorrectness, LossG, MultiAffineRegularizationLoss, FlowAttnLoss, GicLoss
from loss.externel_functions import AdversarialLoss,VGG19, VGGLoss

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
    parser.add_argument('--path_to_dataset',type=str, default='/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion')
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

    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_parsing', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--categories', type=int, default=9)
    parser.add_argument('--use_simmap', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')

    '''Model options'''
    parser.add_argument('--n_enc', type=int, default=2, help='encoder(decoder) layers ')
    parser.add_argument('--n_btn', type=int, default=2, help='bottle neck layers in generator')
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn"')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='"ReLU" or "LeakyReLU"')
    parser.add_argument('--use_pose_decoder', action='store_true', help='use pose in the target decoder')

    parser.add_argument('--use_spectral_G', action='store_true', help='open this if use spectral normalization in generator')
    parser.add_argument('--use_spectral_D', action='store_true', help='open this if use spectral normalization in discriminator')
    parser.add_argument('--use_multi_layer_flow', action='store_true', help='open this if generator output multilayer flow')


    '''Test options'''
    # if --test is open
    parser.add_argument('--test_id', type=str, default='default', help = 'test experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--test_ckpt_name', type=str, default='model_weights_G', help = 'test checkpoint name.')
    parser.add_argument('--ref_ids', type=str, default='0', help='test ref ids')
    parser.add_argument('--test_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_source', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref images')
    parser.add_argument('--test_target_motion', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref motions')
    parser.add_argument('--output_all', action='store_true', help='open this to output the full image')

    '''Experiment options'''
    parser.add_argument('--use_attn', action='store_true', help='use attention for multi-view parsing generation')
    parser.add_argument('--attn_avg', action='store_true', help='use average instead of learnt attention')
    parser.add_argument('--mask_sigmoid', action='store_true', help='Use Sigmoid() as mask output layer or not')
    parser.add_argument('--mask_norm_type', type=str, default='softmax', help='softmax | divsum')

    '''Loss options'''
    parser.add_argument('--use_adv', action='store_true', help='use adversarial loss in total generation')
    parser.add_argument('--use_bilinear_correctness', action='store_true', help='use bilinear sampling in sample loss')
    parser.add_argument('--G_use_resample', action='store_true', help='use gaussian sampling in the target decoder')
    parser.add_argument('--use_correctness', action='store_true', help='use sample correct loss')
    parser.add_argument('--use_flow_reg', action='store_true', help='use flow regularization')
    parser.add_argument('--use_flow_attn_loss', action='store_true', help='use flow attention loss')

    parser.add_argument('--lambda_style', type=float, default=500.0, help='style loss')
    parser.add_argument('--lambda_content', type=float, default=0.5, help='content loss')
    parser.add_argument('--lambda_rec', type=float, default=5.0, help='L1 loss')
    parser.add_argument('--lambda_adv', type=float, default=2.0, help='GAN loss weight')
    parser.add_argument('--lambda_correctness', type=float, default=5.0, help='sample correctness weight')
    parser.add_argument('--lambda_flow_attn', type=float, default=1, help='regular sample loss weight')
    
    # roi warp weights
    parser.add_argument('--use_mask_tv',action='store_true', help='use masked total variation flow loss')
    parser.add_argument('--lambda_flow_reg', type=float, default=200, help='regular sample loss weight')
    parser.add_argument('--lambda_struct', type=float, default=10, help='regular sample loss weight')
    parser.add_argument('--lambda_roi_l1', type=float, default=10, help='regular sample loss weight')
    parser.add_argument('--lambda_roi_perc', type=float, default=1, help='regular sample loss weight')

                
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
    
    # path_to_visualize_dir_train = os.path.join(path_to_visualize_dir, 'train')
    # path_to_visualize_dir_val = os.path.join(path_to_visualize_dir, 'val')

    if not os.path.isdir(path_to_ckpt_dir):
        os.makedirs(path_to_ckpt_dir)
    if not os.path.isdir(path_to_visualize_dir):
        os.makedirs(path_to_visualize_dir)
    # if not os.path.isdir(path_to_visualize_dir_train):
    #     os.makedirs(path_to_visualize_dir_train)
    # if not os.path.isdir(path_to_visualize_dir_val):
    #     os.makedirs(path_to_visualize_dir_val)
    if not os.path.isdir(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    
    return path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir

def init_weights(m, init_type='xavier'):
    if type(m) == nn.Conv2d:
        if init_type=='xavier':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='normal':
            torch.nn.init.normal_(m.weight)
        elif init_type=='kaiming':
            torch.nn.init.kaiming_normal_(m.weight)

def make_dataset(opt):
    """Create dataset"""
    path_to_dataset = opt.path_to_dataset
    train_tuples_name = 'fasion-pairs-train.csv' if opt.K==1 else 'fasion-%d_tuples-train.csv'%(opt.K+1)
    test_tuples_name = 'fasion-pairs-test.csv' if opt.K==1 else 'fasion-%d_tuples-test.csv'%(opt.K+1)
    if opt.use_parsing:
        path_to_train_parsing = os.path.join(path_to_dataset, 'train_parsing_merge/')
        path_to_test_parsing = os.path.join(path_to_dataset, 'test_parsing_merge/')
    else:
        path_to_train_parsing = None
        path_to_test_parsing = None
    if path_to_dataset == '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion':
        dataset = FashionDataset(
            phase = opt.phase,
            path_to_train_tuples=os.path.join(path_to_dataset, train_tuples_name), 
            path_to_test_tuples=os.path.join(path_to_dataset, test_tuples_name), 
            path_to_train_imgs_dir=os.path.join(path_to_dataset, 'train_256/'), 
            path_to_test_imgs_dir=os.path.join(path_to_dataset, 'test_256/'),
            path_to_train_anno=os.path.join(path_to_dataset, 'fasion-annotation-train.csv'), 
            path_to_test_anno=os.path.join(path_to_dataset, 'fasion-annotation-test.csv'), 
            path_to_train_parsings_dir=path_to_train_parsing, 
            path_to_test_parsings_dir=path_to_test_parsing, 
            opt=opt)
    else: # '/home/ljw/playground/Multi-source-Human-Image-Generation/data/fasion-dataset'
        dataset = FashionDataset(
            phase = opt.phase,
            path_to_train_tuples=os.path.join(path_to_dataset, train_tuples_name), 
            path_to_test_tuples=os.path.join(path_to_dataset, test_tuples_name), 
            path_to_train_imgs_dir=os.path.join(path_to_dataset, 'train/'), 
            path_to_test_imgs_dir=os.path.join(path_to_dataset, 'test/'),
            path_to_train_anno=os.path.join(path_to_dataset, 'fasion-annotation-train_new_split.csv'), 
            path_to_test_anno=os.path.join(path_to_dataset, 'fasion-annotation-test_new_split.csv'), 
            opt=opt)
    return dataset

def make_dataloader(opt, dataset):
    is_train = opt.phase == 'train'
    batch_size = 1 if not is_train else opt.batch_size
    shuffle = is_train
    drop_last = is_train
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=8, drop_last=drop_last)
    return dataloader

def save_discriminator(parallel, D, optimizerD, path_to_chkpt_D):
    D_state_dict = D.state_dict() if not parallel else D.module.state_dict()
    torch.save({
        'Dis_state_dict': D_state_dict,
        'optimizerD': optimizerD.state_dict(),
    }, path_to_chkpt_D)    
    pass

def save_generator(parallel, epoch, lossesG, GE, GF, GD, i_batch, optimizerG,path_to_chkpt_G,GP=None):
    GE_state_dict = GE.state_dict() if not parallel else GE.module.state_dict()
    GF_state_dict = GF.state_dict() if not parallel else GF.module.state_dict()
    GD_state_dict = GD.state_dict() if not parallel else GD.module.state_dict()
    # GP_state_dict = GP.state_dict() if not parallel else GP.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GE_state_dict': GE_state_dict,
        'GF_state_dict': GF_state_dict,
        'GD_state_dict': GD_state_dict,
        # 'GP_state_dict': GP_state_dict,
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
    }, path_to_chkpt_G)

def save_flow_generator(parallel, epoch, lossesG, GF, i_batch, optimizerG, path_to_chkpt_G):
    GF_state_dict = GF.state_dict() if not parallel else GF.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GF_state_dict': GF_state_dict,
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
    }, path_to_chkpt_G)


def save_poseEncoder(parallel, epoch, losses, AE, i_batch, optimizer, path_to_chkpt):
    AE_state_dict = AE.state_dict() if not parallel else AE.module.state_dict()
    torch.save({
        'epoch': epoch,
        'losses': losses,
        'AE_state_dict': AE_state_dict,
        'i_batch': i_batch,
        'optimizerG': optimizer.state_dict()
    }, path_to_chkpt)
    
def init_poseEncoder(opt, path_to_chkpt):
    structure_nc = 18
    image_nc = 3
    n_layers = 2
    AE = AutoEncoder(input_nc=image_nc, n_layers=n_layers)
    if opt.parallel:
        AE = nn.DataParallel(AE) # dx + dx + dy = 3 + 20 + 20
    AE = AE.to(device)

    optimizer = optim.Adam(params=AE.parameters(), lr=opt.lr, amsgrad=False)
    if not os.path.isfile(path_to_chkpt):
        AE.apply(init_weights)
        print('Initiating new flow model checkpoint...')
        save_poseEncoder(opt.parallel, 0, [], AE, 0, optimizer, path_to_chkpt)
        print('...Done')
    return AE, optimizer


def init_flowgenerator(opt, path_to_chkpt):
    image_nc = 3
    structure_nc = 21
    categories = opt.categories
    flow_layers = [2,3]

    if opt.use_simmap:
        image_nc += 13
    
    GF = PartFlowGenerator(image_nc=image_nc, structure_nc=structure_nc,parsing_nc=categories, n_layers=5, flow_layers=flow_layers, 
    ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G, output_multi_layers=opt.use_multi_layer_flow)
    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
    GF = GF.to(device)

    optimizerG = optim.Adam(params = list(GF.parameters()) ,
                            lr=opt.lr,
                            amsgrad=False)
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        GF.apply(init_weights)

        print('Initiating new flow model checkpoint...')
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
    structure_nc = 21
    categories = opt.categories
    flow_layers = [2,3]

    if opt.use_simmap:
        image_nc += 13

    GF = PartFlowGenerator(image_nc=image_nc, structure_nc=structure_nc, parsing_nc=categories, n_layers=5, flow_layers= flow_layers, 
        ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G, output_multi_layers=opt.use_multi_layer_flow)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    # GP = PoseAttnFCNGenerator(structure_nc=structure_nc,n_layers=5, attn_layers= flow_layers, ngf=32, max_nc=256,norm_type=opt.norm_type, activation=opt.activation,use_spectral_norm=opt.use_spectral_G)

    if not opt.use_pose_decoder:
        GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    else:
        GD = PoseAwarePartAppearanceDecoder(structure_nc=21,parsing_nc=opt.categories, n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)

    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
        GE = nn.DataParallel(GE)
        GD = nn.DataParallel(GD)
        # GP = nn.DataParallel(GP)

    GF = GF.to(device)
    GE = GE.to(device)
    GD = GD.to(device)
    # GP = GP.to(device)

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
        # GP.apply(init_weights)

        print('Initiating new checkpoint...')
        if not opt.flow_onfly:
            if path_to_flow_chkpt is not None:
                print(path_to_flow_chkpt)
                checkpoint_flow = torch.load(path_to_flow_chkpt, map_location=cpu)
                print('load flow from existing checkpoint, epoch: {0}, batch: {1}'.format(checkpoint_flow['epoch'],checkpoint_flow['i_batch']))
                if opt.parallel:
                    GF.module.load_state_dict(checkpoint_flow['GF_state_dict'], strict=False)
                else:
                    GF.load_state_dict(checkpoint_flow['GF_state_dict'], strict=False)

            else:
                print('Please specify the pretrained flow model path')
        
        save_generator(opt.parallel, 0, [],GE, GF, GD, 0, optimizerG, path_to_chkpt)
        print('...Done')

    return GF, GE, GD, optimizerG


def train_autoencoder(opt, exp_name):
    opt.K = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    path_to_chkpt = path_to_ckpt_dir + 'model_weights_AE.tar' 
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    '''Create dataset and dataloader'''
    print('-------Loading Dataset--------')
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    print('-------Creating Model--------')
    '''Create Model'''
    AE, optimizer = init_poseEncoder(opt, path_to_chkpt)

    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    if opt.parallel:
        AE.module.load_state_dict(checkpoint['AE_state_dict'], strict=False)
    else:
        AE.load_state_dict(checkpoint['AE_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']

    AE.train()
    '''create tensorboard writter'''
    writer = create_writer(path_to_log_dir)

    criterionL1 = nn.L1Loss().to(device)
    criterionL2 = nn.MSELoss().to(device)

    print('-------Training Start--------')
    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        pbar = tqdm(dataloader, leave=True, initial=0)
        pbar.set_description('[{0:>4}/{1:>4}]'.format(epoch,opt.epochs))
        epoch_loss = 0
        for i_batch, batch_data in enumerate(pbar, start=0):
            ref_ys = batch_data['ref_ys']
            ref_xs = batch_data['ref_xs']
            g_y = batch_data['g_y']
            g_x = batch_data['g_x']
            ref_y1 = ref_ys[0]
            ref_y2 = ref_ys[1]
            ref_x1 = ref_xs[0]
            ref_x2 = ref_xs[1]
            g_y = g_y.to(device) # [B, 21, 256, 256]
            g_x = g_x.to(device) # [B, 21, 256, 256]
            ref_y1 = ref_y1.to(device) # [B, 21, 256, 256]
            ref_y2 = ref_y2.to(device) # [B, 21, 256, 256]
            ref_x1 = ref_x1.to(device) # [B, 3, 256, 256]
            ref_x2 = ref_x2.to(device) # [B, 3, 256, 256]
            g_y_image = g_y[:,-3:,...]
            ref_y1_image = ref_y1[:,-3:,...]
            ref_y2_image = ref_y2[:,-3:,...]
            
            gy_recon, g_latent = AE(g_y[:,-3:,...])

            # print(g_latent.shape)
            loss_recon = criterionL1(gy_recon, g_y[:,-3:,...])

            optimizer.zero_grad()
            # loss_recon.backward()
            # optimizer.step()
            epoch_loss += loss_recon.item()
            epoch_loss_moving = epoch_loss / (i_batch+1)
            post_fix_str = 'Epoch_loss=%.3f, L1=%.3f'%(epoch_loss_moving, loss_recon.item())
            pbar.set_postfix_str(post_fix_str)

            if i_batch % opt.img_save_freq == 0:
                # visualize latent and g_y image
                with torch.no_grad():
                    refy1_recon,ref1_latent = AE(ref_y1[:,-3:,...])
                    refy2_recon,ref2_latent = AE(ref_y2[:,-3:,...])
                    # ref2_latent = g_latent.clone()
                b,c,h,w = ref1_latent.shape
                ref_latent = torch.cat((ref1_latent.unsqueeze(1),ref2_latent.unsqueeze(1)),dim=1)
                attn_key = ref_latent.view(b,2, c,-1).permute(0, 1, 3, 2).contiguous().view(b, -1, c) # [B,2HW,C]
                attn_query = g_latent.view(b,c,-1) #[B,C,HW]
                energy = torch.bmm(attn_key, attn_query)
                attntion = nn.Softmax(dim=1)(energy) # [B,NHW,HW]
                atn_vis = attntion.view(b, 2, h * w, h * w).sum(2).view(b, 2, h, w)  # B X 2 X H X W
                g_latent_vis = []
                ref1_latent_vis = []
                ref2_latent_vis = []
                # diff_latent_vis = []
                for i in range(64):
                    g_latent_c = tensor2im(g_latent[:,i:i+1,...],display_batch=0,is_mask=True, out_size=(256//8, 256//8))
                    ref1_latent_c = tensor2im(ref_latent[:,0,i:i+1,...],display_batch=0,is_mask=True, out_size=(256//8, 256//8))
                    ref2_latent_c = tensor2im(ref_latent[:,1,i:i+1,...],display_batch=0,is_mask=True, out_size=(256//8, 256//8))
                    # diff1_latent_c = tensor2im(g_latent[:,i:i+1,...] - ref_latent[:,i:i+1,...],display_batch=0,is_mask=True, out_size=(256//8, 256//8))
                    # diff2_latent_c = tensor2im(g_latent[:,i:i+1,...] - ref_latent[:,i:i+1,...],display_batch=0,is_mask=True, out_size=(256//8, 256//8))
                    
                    g_latent_vis += [g_latent_c]
                    ref1_latent_vis += [ref1_latent_c]
                    ref2_latent_vis += [ref2_latent_c]
                    # diff_latent_vis += [diff_latent_c]

                g_latent_rows = []
                ref1_latent_rows = []
                ref2_latent_rows = []
                # diff_latent_rows = []

                for i in range(8):
                    g_latent_row = torch.cat(g_latent_vis[i*8:i*8+8],dim=1)
                    ref1_latent_row = torch.cat(ref1_latent_vis[i*8:i*8+8],dim=1)
                    ref2_latent_row = torch.cat(ref2_latent_vis[i*8:i*8+8],dim=1)
                    # diff_latent_row = torch.cat(diff_latent_vis[i*8:i*8+8],dim=1)
                    g_latent_rows += [g_latent_row]
                    ref1_latent_rows += [ref1_latent_row]
                    ref2_latent_rows += [ref2_latent_row]
                    # diff_latent_rows += [diff_latent_row]
                
                g_latent_vis = torch.cat(g_latent_rows, dim=0)
                ref1_latent_vis = torch.cat(ref1_latent_rows, dim=0)
                ref2_latent_vis = torch.cat(ref2_latent_rows, dim=0)
                # diff_latent_vis = torch.cat(diff_latent_rows, dim=0)

                g_y_vis = g_y_image[0].detach().permute(1,2,0) * 255
                g_x_vis = tensor2im(g_x)
                ref_y1_vis = ref_y1_image[0].detach().permute(1,2,0) * 255
                ref_y2_vis = ref_y2_image[0].detach().permute(1,2,0) * 255
                ref_x1_vis = tensor2im(ref_x1)
                ref_x2_vis = tensor2im(ref_x2)
                non_local_vis1 = tensor2im(atn_vis[:,0:1,...],display_batch=0,is_mask=True,out_size=(256, 256))
                non_local_vis2 = tensor2im(atn_vis[:,1:2,...],display_batch=0,is_mask=True,out_size=(256, 256))

                white = torch.ones(g_y_vis.shape).to(device) * 255.0

                g_vis = torch.cat((g_y_vis, g_latent_vis),dim=0)
                g_xvis = torch.cat((g_x_vis, white),dim=0)
                ref1_vis = torch.cat((ref_y1_vis, ref1_latent_vis),dim=0)
                ref2_vis = torch.cat((ref_y2_vis, ref2_latent_vis),dim=0)
                diff1_vis = torch.cat((ref_x1_vis, non_local_vis1),dim=0)
                diff2_vis = torch.cat((ref_x2_vis, non_local_vis2),dim=0)
                final_vis = torch.cat((ref1_vis,diff1_vis, ref2_vis, diff2_vis, g_vis,g_xvis),dim=1)

                final_vis = final_vis.type(torch.uint8).to(cpu).numpy()
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), final_vis)

            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                save_poseEncoder(opt.parallel, epoch, [], AE, i_batch, optimizer, path_to_save_G)





            

if __name__ == "__main__":
    opt = get_parser()
    for k,v in sorted(vars(opt).items()):
        print(k,':',v)
    set_random_seed(opt.seed)
    
    if opt.phase == 'train':
        today = datetime.today().strftime("%Y%m%d")
        # experiment_name = 'fashion_pretrain_flow_v{0}_{1}shot_{2}'.format(opt.id,opt.K, today)
        experiment_name = 'train_pose_autoencoder_{0}_{1}'.format(opt.id, today)
        print(experiment_name)
        train_autoencoder(opt, experiment_name)
    else:
        # with torch.no_grad():
        #     test(opt)
        
        with torch.no_grad():
            test_flow_net(opt)




