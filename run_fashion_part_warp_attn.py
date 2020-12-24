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
from model.pyramid_part_flow_generator import PartFlowGenerator
from model.pyramid_flow_generator_with_occlu_attn import FlowGenerator,PoseAttnFCNGenerator,AppearanceDecoder, AppearanceEncoder, PoseAwareAppearanceDecoder
from model.discriminator import ResDiscriminator
from model.blocks import warp_flow,_freeze,_unfreeze, gen_uniform_grid

from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_pyramid_visualize_result
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
    parser.add_argument('--path_to_dataset',type=str, default='/home/ljw/playground/Multi-source-Human-Image-Generation/data/fasion-dataset')
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
    parser.add_argument('--lambda_flow_reg', type=float, default=20, help='regular sample loss weight')
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
    if opt.use_parsing:
        path_to_train_parsing = os.path.join(path_to_dataset, 'train_parsing_merge/')
        path_to_test_parsing = os.path.join(path_to_dataset, 'test_parsing_merge/')
    else:
        path_to_train_parsing = None
        path_to_test_parsing = None
    if path_to_dataset == '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion':
        dataset = FashionDataset(
            phase = opt.phase,
            path_to_train_tuples=os.path.join(path_to_dataset, 'fasion-3_tuples-train.csv'), 
            path_to_test_tuples=os.path.join(path_to_dataset, 'fasion-3_tuples-test.csv'), 
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
            path_to_train_tuples=os.path.join(path_to_dataset, 'fasion-3_tuples-train.csv'), 
            path_to_test_tuples=os.path.join(path_to_dataset, 'fasion-4_tuples-test.csv'), 
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

def save_generator(parallel, epoch, lossesG, GE, GF, GD,GP, i_batch, optimizerG,path_to_chkpt_G):
    GE_state_dict = GE.state_dict() if not parallel else GE.module.state_dict()
    GF_state_dict = GF.state_dict() if not parallel else GF.module.state_dict()
    GD_state_dict = GD.state_dict() if not parallel else GD.module.state_dict()
    GP_state_dict = GP.state_dict() if not parallel else GP.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GE_state_dict': GE_state_dict,
        'GF_state_dict': GF_state_dict,
        'GD_state_dict': GD_state_dict,
        'GP_state_dict': GP_state_dict,
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

def init_flowgenerator(opt, path_to_chkpt):
    image_nc = 3
    structure_nc = 21
    categories = opt.categories
    flow_layers = [2,3]

    if opt.use_simmap:
        image_nc += 13
    
    GF = PartFlowGenerator(image_nc=image_nc, structure_nc=structure_nc,parsing_nc=categories, n_layers=5, flow_layers=flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
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
    flow_layers = [2,3]
    categories = opt.categories

    if opt.use_parsing:
        structure_nc += 20
    if opt.use_simmap:
        image_nc += 13

    GF = PartFlowGenerator(image_nc=image_nc, structure_nc=structure_nc, parsing_nc=categories, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GP = PoseAttnFCNGenerator(structure_nc=structure_nc,n_layers=5, attn_layers= flow_layers, ngf=32, max_nc=256,norm_type=opt.norm_type, activation=opt.activation,use_spectral_norm=opt.use_spectral_G)

    if not opt.use_pose_decoder:
        GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    else:
        GD = PoseAwareAppearanceDecoder(structure_nc=21, n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)

    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
        GE = nn.DataParallel(GE)
        GD = nn.DataParallel(GD)
        GP = nn.DataParallel(GP)

    GF = GF.to(device)
    GE = GE.to(device)
    GD = GD.to(device)
    GP = GP.to(device)

    optimizerG = optim.Adam(params = list(GF.parameters()) + list(GE.parameters()) + list(GD.parameters()) + list(GP.parameters()) ,
                            lr=opt.lr,
                            amsgrad=False)
    if opt.use_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizerG, 'min', factor=np.sqrt(0.1), patience=5, min_lr=5e-7)
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        GF.apply(init_weights)
        GE.apply(init_weights)
        GD.apply(init_weights)
        GP.apply(init_weights)

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
        
        save_generator(opt.parallel, 0, [],GE, GF, GD,GP, 0, optimizerG, path_to_chkpt)
        print('...Done')

    return GF, GE, GD,GP, optimizerG

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
    GE = VGG19().to(device)
    _freeze(GE)
    '''create tensorboard writter'''
    writer = create_writer(path_to_log_dir)
    
    '''Losses'''
    criterionG = LossG(device=device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionReg = GicLoss().to(device)
    criterionVgg = VGGLoss().to(device)

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
            ref_ps = batch_data['ref_ps']
            g_x = batch_data['g_x']
            g_y = batch_data['g_y']
            g_p = batch_data['g_p']
            assert(len(ref_xs)==opt.K)
            assert(len(ref_ys)==opt.K)
            assert(len(ref_ps)==opt.K)

            for i in range(len(ref_xs)):
                ref_xs[i] = ref_xs[i].to(device)
                ref_ys[i] = ref_ys[i].to(device)
                ref_ps[i] = ref_ps[i].to(device)

            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]
            g_p = g_p.to(device) # [B, 20, 256, 256]
            
            flows, masks, attns = [], [], []
            for k in range(0, opt.K):
                flow_ks, mask_ks, attn_ks = GF(ref_xs[k], ref_ys[k],ref_ps[k], g_y, g_p)

                flows += [flow_ks]
                masks += [mask_ks]
                attns += [attn_ks]

            loss_reg = 0
            loss_roi_perc = 0
            loss_struct = 0
            loss_roi_l1 = 0
            for k in range(0, opt.K):
                for i in range(2):
                    size = (flows[k][i].shape[2], flows[k][i].shape[3])
                    ref_x_down = F.interpolate(ref_xs[k], size, mode='bilinear', align_corners=opt.align_corner)
                    g_x_down = F.interpolate(g_x, size, mode='bilinear', align_corners= opt.align_corner)
                    warp_refx_down = warp_flow(ref_x_down, flows[k][i], align_corners=opt.align_corner)
                    uniform_grid = gen_uniform_grid(flows[k][i])
                    for c in range(opt.categories):
                        if torch.sum(ref_ps[k][:,c:c+1,...]) > 0 and torch.sum(g_p[:,c:c+1,...]) > 0: # if source and target both have this category
                            ref_pc_down = F.interpolate(ref_ps[k][:,c:c+1,...], size, mode='bilinear', align_corners=opt.align_corner)
                            g_pc_down = F.interpolate(g_p[:,c:c+1,...], size, mode='bilinear', align_corners=opt.align_corner)
                            warp_refpc_down = warp_flow(ref_pc_down, flows[k][i], align_corners=opt.align_corner)
                            
                            loss_struct += criterionL1(warp_refpc_down, g_pc_down)
                            content, _ = criterionVgg(warp_refx_down * warp_refpc_down, g_x_down * g_pc_down)
                            loss_roi_perc += content
                            loss_roi_l1 += criterionL1(warp_refx_down * warp_refpc_down, g_x_down*g_pc_down)
                            if opt.use_mask_tv:
                                mask = warp_refpc_down
                            else:
                                mask = None
                            loss_reg += criterionReg((flows[k][i]*2/flows[k][i].shape[2]+uniform_grid).permute(0,2,3,1), mask=mask)

            loss_struct = loss_struct / opt.K * opt.lambda_struct
            loss_roi_l1 = loss_roi_l1 / opt.K * opt.lambda_roi_l1
            loss_roi_perc = loss_roi_perc / opt.K * opt.lambda_roi_perc
            loss_reg = loss_reg / opt.K * opt.lambda_flow_reg
            '''normalize masks to sum to 1'''

            writer.add_scalar('{0}/loss_struct'.format(opt.phase), loss_struct.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('{0}/loss_roi_l1'.format(opt.phase), loss_roi_l1.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('{0}/loss_roi_perc'.format(opt.phase), loss_roi_perc.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('{0}/loss_reg'.format(opt.phase), loss_reg.item(), global_step=i_batch_total, walltime=None)
            lossG = loss_struct + loss_roi_l1 + loss_roi_perc + loss_reg

            optimizerG.zero_grad()
            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            post_fix_str = 'Epo_loss=%.3f, G=%.3f,L1=%.3f,L_cnt=%.3f,L_reg=%.3f,L_struct=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_roi_l1.item(), loss_roi_perc.item(), loss_reg.item(), loss_struct.item())

            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            ref_xfs = []
            
            for k in range(opt.K):
                ref_xfs += [ [GE(ref_xs[k])['relu4_1'],GE(ref_xs[k])['relu3_1']] ]

            gf = [GE(g_x)['relu4_1'],GE(g_x)['relu3_1']]
            
            vis_layer = 1
            size = (flows[0][vis_layer].shape[2], flows[0][vis_layer].shape[3])
            ori_size = (ref_xs[0].shape[2], ref_xs[0].shape[3])
            ref_down = F.interpolate(ref_xs[0], size, mode='bilinear', align_corners=opt.align_corner)
            ref_p_down = F.interpolate(ref_ps[0], size, mode='bilinear', align_corners=opt.align_corner)
            ref_warp = warp_flow(ref_down, flows[0][vis_layer],align_corners=opt.align_corner)
            p_warp = warp_flow(ref_p_down, flows[0][vis_layer], align_corners=opt.align_corner)

            x_hat = F.interpolate(ref_warp, ori_size, mode='bilinear', align_corners=opt.align_corner)
            phat = F.interpolate(p_warp, ori_size, mode='bilinear', align_corners=opt.align_corner)

            if i_batch % opt.img_save_freq == 0:    
                final_img,_,_ = get_pyramid_visualize_result(opt, ref_xs, ref_ys, ref_ps, g_x,x_hat, g_y, g_p, phat, flows, attns, masks, ref_xfs, gf)
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), final_img)
            
            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                save_flow_generator(opt.parallel, epoch, lossesG, GF, i_batch, optimizerG, path_to_save_G)
        save_flow_generator(opt.parallel, epoch+1, lossesG, GF, i_batch, optimizerG, path_to_chkpt)

def train(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    path_to_chkpt_G = path_to_ckpt_dir + 'model_weights_G.tar' 
    path_to_chkpt_D = path_to_ckpt_dir + 'model_weights_D.tar' 

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
    '''Create Model'''
    GF, GE, GD, GP, optimizerG = init_generator(opt, path_to_chkpt_G, path_to_chkpt_flow)
    
    '''Loading from past checkpoint_G'''
    checkpoint_G = torch.load(path_to_chkpt_G, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint_G['GF_state_dict'], strict=False)
        GE.module.load_state_dict(checkpoint_G['GE_state_dict'], strict=False)
        GD.module.load_state_dict(checkpoint_G['GD_state_dict'], strict=False)
        GP.module.load_state_dict(checkpoint_G['GP_state_dict'], strict=False)
    else:
        GF.load_state_dict(checkpoint_G['GF_state_dict'], strict=False)
        GE.load_state_dict(checkpoint_G['GE_state_dict'], strict=False)
        GD.load_state_dict(checkpoint_G['GD_state_dict'], strict=False)
        GP.load_state_dict(checkpoint_G['GP_state_dict'], strict=False)
    epochCurrent = checkpoint_G['epoch']
    lossesG = checkpoint_G['lossesG']
    i_batch_current = checkpoint_G['i_batch']
    i_batch_total = epochCurrent * dataloader.__len__() // opt.batch_size + i_batch_current
    optimizerG.load_state_dict(checkpoint_G['optimizerG'])
    GF.train()
    GE.train()
    GD.train()
    GP.train()

    if opt.use_adv:
        D, optimizerD = init_discriminator(opt, path_to_chkpt_D)
        checkpoint_D = torch.load(path_to_chkpt_D, map_location=cpu)
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
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)
    criterionFlowAttn = FlowAttnLoss().to(device)

    print('-------Training Start--------')
    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        if epoch > epochCurrent:
            i_batch_current = 0
        epoch_loss_G = 0
        pbar = tqdm(dataloader, leave=True, initial=0)
        pbar.set_description('[{0:>4}/{1:>4}], G_lr-{2},D_lr-{3}'.format(epoch,opt.epochs,optimizerG.param_groups[0]['lr'],optimizerD.param_groups[0]['lr']))
        
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
            
            flows, masks, attns,  xfs = [], [], [], []
            flows_down,masks_down, xfs_warp = [], [], []

            # flows: K *[tensor[2,32,32] tensor[2,64,64]]
            # masks: K *[tensor[1,32,32] tensor[1,64,64]]
            # xfs: K *[tensor[256,32,32] tensor[128,64,64]]
            with torch.no_grad():
                gf = GE(g_x)[0:2]
            for k in range(0, opt.K):
                # get 2 flows, masks and attns at two resolution 32, 64
                flow_ks, mask_ks, _ = GF(ref_xs[k], ref_ys[k], g_y)
                # flow_ks, mask_ks = GF(ref_xs[k], ref_ys[k], g_y)
                attn_ks = GP(ref_ys[k], g_y)
                # get 2 source features at resolution 32, 64
                xf_ks = GE(ref_xs[k])[0:2]
                flows += [flow_ks]
                masks += [mask_ks]
                attns += [attn_ks]
                xfs += [xf_ks]
            
            # 如果是平均，则每个attn的值都为1，然后做softmax变为1/K
            if opt.attn_avg:
                for k in range(0, opt.K):
                    for i in range(0, len(attns[k])):
                        attns[k][i] = torch.ones_like(attns[k][i])
            
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
                x_hat = GD(g_y, xfs, flows, masks, attn_norm_trans)
            else:
                x_hat = GD(xfs, flows, masks, attn_norm_trans)

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
            if opt.use_adv:
                _freeze(D)
                D_fake = D(x_hat)
                lossG_adv = criterion_GAN(D_fake, True) * opt.lambda_adv
                writer.add_scalar('{0}/lossG_adv'.format(opt.phase), lossG_adv.item(), global_step=i_batch_total, walltime=None)
                lossG = lossG + lossG_adv
            
            if opt.use_correctness:
                loss_correct=0
                for k in range(opt.K):
                    loss_correct += criterionCorrectness(g_x, ref_xs[k], flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness)
                    # loss_correct += criterionL1(xfs_warp[k], g_xf)
                loss_correct = loss_correct/opt.K * opt.lambda_correctness
                writer.add_scalar('{0}/lossG_correct'.format(opt.phase), loss_correct.item(), global_step=i_batch_total, walltime=None)
                lossG = lossG + loss_correct

            if opt.use_flow_reg:
                loss_regular=0
                for k in range(opt.K):
                    loss_regular += criterionReg(flows[k])
                
                loss_regular = loss_regular / opt.K * opt.lambda_flow_reg
            
                lossG += loss_regular
                writer.add_scalar('{0}/loss_regular'.format(opt.phase), loss_regular.item(), global_step=i_batch_total, walltime=None)
            
            if opt.use_flow_attn_loss:
                loss_flow_attn = 0 #  sum_i(w_i(P_G-P_i)), wi = exp(-attn)
                for k in range(0, opt.K):
                    
                    loss_flow_attn += criterionFlowAttn(g_x, ref_xs[k], flows[k], attn_norm_trans[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear_correctness) #  
                loss_flow_attn = loss_flow_attn / opt.K * opt.lambda_flow_attn
                lossG += loss_flow_attn
                writer.add_scalar('{0}/loss_flow_attn'.format(opt.phase), loss_flow_attn.item(), global_step=i_batch_total, walltime=None)

            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
            if opt.use_adv:
                post_fix_str += ', G_adv=%.3f, D_loss=%.3f'%(lossG_adv.item(), lossD.item())
            if opt.use_correctness:
                post_fix_str += ', G_cor=%.3f'%(loss_correct.item())
            if opt.use_flow_reg:
                post_fix_str += ', G_reg=%.3f'%(loss_regular.item())
            if opt.use_flow_attn_loss:
                post_fix_str += ', G_flo=%.3f'%(loss_flow_attn.item())
            
            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            if i_batch % opt.img_save_freq == 0:
                full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
                    ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=None, gx=g_x, x_hat=x_hat, gy=g_y,\
                        gp=None,gp_hat=None,flows=flows, masks_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf)
                
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), full_img)

            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                path_to_save_D = path_to_ckpt_dir + 'epoch_{}_batch_{}_D.tar'.format(epoch, i_batch)
                save_generator(opt.parallel, epoch, lossesG, GE, GF, GD,GP, i_batch, optimizerG, path_to_save_G)
                save_discriminator(opt.parallel, D, optimizerD, path_to_save_D) 
        
        save_generator(opt.parallel, epoch+1, lossesG, GE, GF, GD,GP, i_batch, optimizerG, path_to_chkpt_G)
        save_discriminator(opt.parallel, D, optimizerD, path_to_chkpt_D)
        '''save image result'''
        full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
            ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=None, gx=g_x, x_hat=x_hat, gy=g_y,\
                gp=None,gp_hat=None,flows=flows, masks_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf)

        plt.imsave(os.path.join(path_to_visualize_dir,"epoch_latest.png"), full_img)
        
            
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
    structure_nc = 21
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_simmap:
        image_nc += 13

    GF = FlowGenerator(image_nc=image_nc, structure_nc=structure_nc, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GP = PoseAttnFCNGenerator(structure_nc=structure_nc,n_layers=5, attn_layers= flow_layers, ngf=32, max_nc=256,norm_type=opt.norm_type, activation=opt.activation,use_spectral_norm=opt.use_spectral_G)

    if not opt.use_pose_decoder:
        GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    else:
        GD = PoseAwareAppearanceDecoder(structure_nc=21, n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
            activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)

    if opt.parallel:
        GF = nn.DataParallel(GF) # dx + dx + dy = 3 + 20 + 20
        GE = nn.DataParallel(GE)
        GD = nn.DataParallel(GD)
        GP = nn.DataParallel(GP)

    GF = GF.to(device)
    GE = GE.to(device)
    GD = GD.to(device)
    GP = GP.to(device)
    
    '''Loading from past checkpoint_G'''
    checkpoint_G = torch.load(path_to_chkpt_G, map_location=cpu)
    if opt.parallel:
        GF.module.load_state_dict(checkpoint_G['GF_state_dict'], strict=False)
        GE.module.load_state_dict(checkpoint_G['GE_state_dict'], strict=False)
        GD.module.load_state_dict(checkpoint_G['GD_state_dict'], strict=False)
        GP.module.load_state_dict(checkpoint_G['GP_state_dict'], strict=False)
    else:
        GF.load_state_dict(checkpoint_G['GF_state_dict'], strict=False)
        GE.load_state_dict(checkpoint_G['GE_state_dict'], strict=False)
        GD.load_state_dict(checkpoint_G['GD_state_dict'], strict=False)
        GP.load_state_dict(checkpoint_G['GP_state_dict'], strict=False)
    epochCurrent = checkpoint_G['epoch']
    GF.eval()
    GE.eval()
    GD.eval()
    GP.eval()

    '''Losses'''
    criterionG = LossG(device=device)
    criterion_GAN = AdversarialLoss(type='lsgan').to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)

    """ Test start """
    epoch_loss_G = 0
    pbar = tqdm(dataloader, leave=True, initial=0)
    
    for i_batch, batch_data in enumerate(pbar, start=0):
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

        g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]
        
        flows, masks, attns, xfs = [], [], [], []
        flows_down,masks_down, xfs_warp = [], [], []

        # flows: K *[tensor[2,32,32] tensor[2,64,64]]
        # masks: K *[tensor[1,32,32] tensor[1,64,64]]
        # xfs: K *[tensor[256,32,32] tensor[128,64,64]]
        with torch.no_grad():
            gf = GE(g_x)[0:2]
        
        for k in range(0, opt.K):
            # get 2 flows, masks and attns at two resolution 32, 64
            flow_ks, mask_ks, _ = GF(ref_xs[k], ref_ys[k], g_y)
            # flow_ks, mask_ks = GF(ref_xs[k], ref_ys[k], g_y)
            attn_ks = GP(ref_ys[k], g_y)
            # get 2 source features at resolution 32, 64
            xf_ks = GE(ref_xs[k])[0:2]
            attns += [attn_ks]
            flows += [flow_ks]
            masks += [mask_ks]
            xfs += [xf_ks]
        
        if opt.attn_avg:
            for k in range(0, opt.K):
                for i in range(0, len(attns[k])):
                    attns[k][i] = torch.ones_like(attns[k][i])
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
            x_hat = GD(g_y, xfs, flows, masks, attn_norm_trans)
        else:
            x_hat = GD(xfs, flows, masks, attn_norm_trans)
        model_G_end = time.time()
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

        full_img,simp_img, out_img = get_pyramid_visualize_result(opt, \
                    ref_xs=ref_xs, ref_ys=ref_ys,ref_ps=None, gx=g_x, x_hat=x_hat, gy=g_y,\
                        gp=None,gp_hat=None,flows=flows, masks_normed=attn_norm_trans, occlusions=masks, ref_features=xfs, g_features=gf)
        visual_end = time.time()
        # print('visual time:%.3f'%(visual_end-visual_start))
        
        save_start = time.time()
        test_name = ''
        # print(froms)
        for k in range(opt.K):
            test_name += str(froms[k][0])+'+'
        test_name += str(to[0])
        from PIL import Image
        Image.fromarray(simp_img).save(os.path.join(test_result_dir,"{0}_simp.jpg".format(test_name)))
        Image.fromarray(out_img).save(os.path.join(test_result_eval_dir,"{0}_vis.jpg".format(test_name)))

        if opt.output_all:
            Image.fromarray(full_img).save(os.path.join(test_result_dir,"{0}_all.jpg".format(test_name)))
        save_end = time.time()
        # print('save time:%.3f'%(save_end-save_start))
        
    pass


if __name__ == "__main__":
    opt = get_parser()
    for k,v in sorted(vars(opt).items()):
        print(k,':',v)
    set_random_seed(opt.seed)
    
    if opt.phase == 'train':
        today = datetime.today().strftime("%Y%m%d")
        experiment_name = 'fashion_pretrain_flow_v{0}_{1}shot_{2}'.format(opt.id,opt.K, today)
        print(experiment_name)
        # train(opt, experiment_name)
        train_flow_net(opt, experiment_name)
    else:
        with torch.no_grad():
            test(opt)




