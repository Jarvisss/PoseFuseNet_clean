"""multiview generation without direct generation branch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from loss.externel_functions import VGG19
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
import itertools
matplotlib.use('agg')  # for image save not render

# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
# from model.Parsing_net import ParsingGenerator
from model.flow_generator import  AppearanceEncoder, AppearanceDecoder
from model.pyramid_flow_generator import FlowGenerator
from model.discriminator import ResDiscriminator
from model.blocks import warp_flow,_freeze,_unfreeze

from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_visualize_result
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashion_dataset import FashionDataset
from loss.loss_generator import PerceptualCorrectness, LossG, AffineRegularizationLoss, MultiAffineRegularizationLoss
from loss.externel_functions import AdversarialLoss

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
    parser.add_argument('--anno_size', type=int, nargs=2, help='input annotation size')
    parser.add_argument('--model_save_freq', type=int, default=2000, help='save model every N iters')
    parser.add_argument('--img_save_freq', type=int, default=200, help='save image every N iters')

    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_parsing', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_simmap', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')

    '''Model options'''
    parser.add_argument('--n_enc', type=int, default=2, help='encoder(decoder) layers ')
    parser.add_argument('--n_btn', type=int, default=2, help='bottle neck layers in generator')
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn"')
    parser.add_argument('--use_spectral_G', action='store_true', help='open this if use spectral normalization in generator')
    parser.add_argument('--use_spectral_D', action='store_true', help='open this if use spectral normalization in discriminator')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='"ReLU" or "LeakyReLU"')
    parser.add_argument('--use_pose_decoder', action='store_true', help='use pose in the target decoder')

    '''Test options'''
    # if --test is open
    parser.add_argument('--test_id', type=str, default='default', help = 'test experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--ref_ids', type=str, default='0', help='test ref ids')
    parser.add_argument('--test_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_source', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref images')
    parser.add_argument('--test_target_motion', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref motions')
    

    '''Experiment options'''
    parser.add_argument('--use_attn', action='store_true', help='use attention for multi-view parsing generation')
    parser.add_argument('--mask_sigmoid', action='store_true', help='Use Sigmoid() as mask output layer or not')
    parser.add_argument('--mask_norm_type', type=str, default='softmax', help='softmax | divsum')

    '''Loss options'''
    parser.add_argument('--use_adv', action='store_true', help='use adversarial loss for full pipeline')
    parser.add_argument('--use_flow_reg', action='store_true', help='use regularization loss for flow')
    parser.add_argument('--use_bilinear', action='store_true', help='use bilinear sampling in sample loss')
    parser.add_argument('--lambda_style', type=float, default=500.0, help='style loss')
    parser.add_argument('--lambda_content', type=float, default=0.5, help='content loss')
    parser.add_argument('--lambda_rec', type=float, default=5.0, help='L1 loss')
    parser.add_argument('--lambda_adv', type=float, default=2.0, help='GAN loss weight')

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
        elif init_type=='orthogonal':
            torch.nn.init.orthogonal_(m.weight)

def make_dataset(opt):
    """Create dataset"""
    path_to_dataset = opt.path_to_dataset
    train_tuples_name = 'fasion-pairs-train.csv' if opt.K==1 else 'fasion-%d_tuples-train.csv'%(opt.K+1)
    test_tuples_name = 'fasion-pairs-test.csv' if opt.K==1 else 'fasion-%d_tuples-test.csv'%(opt.K+1)
    path_to_train_label = os.path.join(path_to_dataset,'train_tps_field')
    path_to_test_label = os.path.join(path_to_dataset,'test_tps_field')
    path_to_train_sim = os.path.join(path_to_dataset,'train_sim')
    path_to_test_sim = os.path.join(path_to_dataset,'test_sim')
    path_to_train_parsing = os.path.join(path_to_dataset, 'train_parsing_merge/')
    path_to_test_parsing = os.path.join(path_to_dataset, 'test_parsing_merge/')

    if path_to_dataset == '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion':
        dataset = FashionDataset(
            phase = opt.phase,
            path_to_train_tuples=os.path.join(path_to_dataset, train_tuples_name), 
            path_to_test_tuples=os.path.join(path_to_dataset, test_tuples_name), 
            path_to_train_imgs_dir=os.path.join(path_to_dataset, 'train_256/'), 
            path_to_test_imgs_dir=os.path.join(path_to_dataset, 'test_256/'),
            path_to_train_anno=os.path.join(path_to_dataset, 'fasion-annotation-train.csv'), 
            path_to_test_anno=os.path.join(path_to_dataset, 'fasion-annotation-test.csv'), 
            path_to_train_label_dir=path_to_train_label,
            path_to_test_label_dir=path_to_test_label,
            path_to_train_sim_dir=path_to_train_sim,
            path_to_test_sim_dir=path_to_test_sim,
            path_to_train_parsings_dir=path_to_train_parsing, 
            path_to_test_parsings_dir=path_to_test_parsing, opt=opt)
    else: # '/dataset/ljw/deepfashion/GLFA_split/fashion'
        dataset = FashionDataset(
            phase = opt.phase,
            path_to_train_tuples=os.path.join(path_to_dataset, train_tuples_name), 
            path_to_test_tuples=os.path.join(path_to_dataset, test_tuples_name), 
            path_to_train_imgs_dir=os.path.join(path_to_dataset, 'train/'), 
            path_to_test_imgs_dir=os.path.join(path_to_dataset, 'test/'),
            path_to_train_anno=os.path.join(path_to_dataset, 'fasion-annotation-train.csv'), 
            path_to_test_anno=os.path.join(path_to_dataset, 'fasion-annotation-test.csv'), 
            path_to_train_label_dir=path_to_train_label,
            path_to_test_label_dir=path_to_test_label,
            path_to_train_sim_dir=path_to_train_sim,
            path_to_test_sim_dir=path_to_test_sim,
            path_to_train_parsings_dir=path_to_train_parsing, 
            path_to_test_parsings_dir=path_to_test_parsing, opt=opt)
    return dataset
    
def make_dataloader(opt, dataset):
    is_train = opt.phase == 'train'
    batch_size = 1 if not is_train else opt.batch_size
    shuffle = is_train
    drop_last = is_train
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=8, drop_last=drop_last)
    return dataloader

def init_discriminator(opt, path_to_chkpt):
    D_inc = 3
    if opt.parallel:
        D = nn.DataParallel(ResDiscriminator(input_nc=D_inc,ndf=32, img_f=128,layers=4, use_spect=opt.use_spectral_D).to(device)) # dx + dx + dy = 3 + 20 + 20
    else:
        D = ResDiscriminator(input_nc=D_inc,ndf=32,img_f=128,layers=4, use_spect=opt.use_spectral_D).to(device) # dx + dx + dy = 3 + 20 + 20

    optimizerD = optim.Adam(params = list(D.parameters()) ,
                            lr=opt.lr_D,
                            amsgrad=False, betas=(0,0.999))
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        D.apply(init_weights)
        print('Initiating new Discriminator model checkpoint...')
        if opt.parallel:
            torch.save({
                    'Dis_state_dict': D.module.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    }, path_to_chkpt)
        else:
            torch.save({
                    'Dis_state_dict': D.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    }, path_to_chkpt)
        print('...Done')
    return D, optimizerD

def save_generator(parallel, epoch, lossesG, GE, GF, GD, i_batch, optimizerG,path_to_chkpt_G):
    GE_state_dict = GE.state_dict() if not parallel else GE.module.state_dict()
    GF_state_dict = GF.state_dict() if not parallel else GF.module.state_dict()
    GD_state_dict = GD.state_dict() if not parallel else GD.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GE_state_dict': None,
        'GF_state_dict': GF_state_dict,
        'GD_state_dict': None,
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
    }, path_to_chkpt_G)

def init_generator(opt, path_to_chkpt):
    image_nc = 3
    structure_nc = 21
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_simmap:
        image_nc += 13

    GF = FlowGenerator(image_nc=image_nc, structure_nc=structure_nc, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    
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
        
        save_generator(opt.parallel, 0, [],GE, GF, GD, 0, optimizerG, path_to_chkpt)
        print('...Done')

    return GF, GE, GD, optimizerG

def init_generator_with_occ(opt, path_to_chkpt):
    image_nc = 3
    structure_nc = 21
    flow_layers = [2,3]
    if opt.use_parsing:
        structure_nc += 20
    if opt.use_simmap:
        image_nc += 13
    from model.pyramid_flow_generator_with_occlu_attn import FlowGenerator, FlowGenerator2
    GF = FlowGenerator2(inc=image_nc+structure_nc*2, n_layers=5, flow_layers= flow_layers, ngf=32, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    GE = AppearanceEncoder(n_layers=opt.n_enc, inc=3, use_spectral_norm=opt.use_spectral_G)
    # GE = AppearanceEncoder(n_layers=3, inc=3, ngf=64, max_nc=256, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    
    # if not opt.use_pose_decoder:
    #     GD = AppearanceDecoder(n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
    #         activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    # else:
    #     GD = PoseAwareAppearanceDecoder(structure_nc=21, n_decode_layers=3, output_nc=3, flow_layers=flow_layers, ngf=64, max_nc=256, norm_type=opt.norm_type,
    #         activation=opt.activation, use_spectral_norm=opt.use_spectral_G, align_corners=opt.align_corner, use_resample=opt.G_use_resample)
    GD = AppearanceDecoder(n_bottleneck_layers=opt.n_btn, n_decode_layers=opt.n_enc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral_G)
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
    GF, GE, GD, optimizerG = init_generator(opt, path_to_chkpt)
    
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
    _freeze(GD)
    '''create tensorboard writter'''
    writer = create_writer(path_to_log_dir)
    
    '''Losses'''
    criterionG = LossG(device=device)
    criterionL1 = nn.L1Loss().to(device)
    criterionCorrectness = PerceptualCorrectness().to(device)
    criterionReg = MultiAffineRegularizationLoss(kz_dic={2:5, 3:3}).to(device)

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
            
            flows, flow_ones, masks, xfs = [], [], [], []
            flows_down,masks_down, xfs_warp = [], [], []
            with torch.no_grad():
                g_xf = GE(g_x)['relu4_1']
            for k in range(0, opt.K):
                flow_ks, mask_ks = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64
                flow_k = flow_ks[1]
                # print(flow_k.shape)
                mask_k = mask_ks[1]
                # print(mask_k.shape)
                with torch.no_grad():
                    xf_k = GE(ref_xs[k])['relu4_1']
                # print(xf_k.shape)
                flow_k_down = F.interpolate(flow_k * xf_k.shape[2] / flow_k.shape[2], size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)

                mask_k_down = F.interpolate(mask_k, size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
                xf_k_warp = warp_flow(xf_k, flow_k_down, align_corners=opt.align_corner)

                flows += [flow_ks]
                flow_ones += [flow_k ]
                masks += [mask_k]
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

            # x_hat = GD(xf_merge)
            # x_hat =  warp_flow(F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear',align_corners=opt.align_corner), flow_k, align_corners=opt.align_corner)
            # x_hat = F.interpolate(x_hat, size=g_x.shape[2:],mode='bilinear',align_corners=opt.align_corner)
            down_ref_xk = F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear', align_corners=opt.align_corner)
            x_hat_down =  warp_flow(down_ref_xk, flow_k, align_corners=opt.align_corner)
            x_hat = F.interpolate(x_hat_down, size=ref_xs[k].shape[2:], mode='bilinear', align_corners=opt.align_corner)
            loss_correct=0
            for k in range(opt.K):
                loss_correct += criterionCorrectness(g_x, ref_xs[k], flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear)
                # loss_correct += criterionL1(xfs_warp[k], g_xf)
            loss_correct = loss_correct/opt.K * 5
            lossG = loss_correct
            writer.add_scalar('{0}/loss_correct'.format(opt.phase), loss_correct.item(), global_step=i_batch_total, walltime=None)
            
            loss_regular = 0
            if opt.use_flow_reg:
                for k in range(opt.K):
                    loss_regular += criterionReg(flows[k])
                
                loss_regular = loss_regular / opt.K * 0.0025
            
                lossG += loss_regular
                writer.add_scalar('{0}/loss_regular'.format(opt.phase), loss_regular.item(), global_step=i_batch_total, walltime=None)
            optimizerG.zero_grad()
            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            if opt.use_flow_reg:
                post_fix_str = 'Epoch_loss=%.3f, G=%.3f, Cor=%.3f, Reg=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_correct.item(), loss_regular.item())
            else:
                post_fix_str = 'Epoch_loss=%.3f, G=%.3f, Cor=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_correct.item())
            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            if i_batch * opt.batch_size % 1000 == 0:
                final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,None, g_x, g_y,None,None, g_xf, x_hat,\
                    flow_ones, F.interpolate(mask_normed, size=g_x.shape[2:], mode='bilinear',align_corners=opt.align_corner), xfs, xfs_warp, xfs_warp_masked)
                
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_iter_{}.png".format(epoch, i_batch)), final_img)
                torch.save({
                    'epoch': epoch+1,
                    'lossesG': lossesG,
                    'GE_state_dict': GE.state_dict(),
                    'GF_state_dict': GF.state_dict(),
                    'GD_state_dict': GD.state_dict(),
                    'i_batch': i_batch,
                    'optimizerG': optimizerG.state_dict(),
                    }, path_to_chkpt)
    writer.close()
           
def train(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    
    '''Set logging,checkpoint,vis dir'''
    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    path_to_chkpt_G = path_to_ckpt_dir + 'model_weights_G.tar' 
    path_to_chkpt_D = path_to_ckpt_dir + 'model_weights_D.tar' 

    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    '''Create dataset and dataloader'''
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)

    '''Create Model'''
    GF, GE, GD, optimizerG = init_generator(opt, path_to_chkpt_G)
    '''Loading from past checkpoint_G'''
    checkpoint_G = torch.load(path_to_chkpt_G, map_location=cpu)
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
            
            flows, masks, xfs = [], [], []
            flows_down,masks_down, xfs_warp = [], [], []
            for k in range(0, opt.K):
                flow_k, mask_k = GF(ref_xs[k], ref_ys[k], g_y)
                # print(ref_xs[k].shape)
                xf_k = GE(ref_xs[k])
                flow_k_down = F.interpolate(flow_k *xf_k.shape[2] / flow_k.shape[2], size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
                mask_k_down = F.interpolate(mask_k, size=xf_k.shape[2:], mode='bilinear',align_corners=opt.align_corner)
                xf_k_warp = warp_flow(xf_k, flow_k_down, align_corners=opt.align_corner)

                flows += [flow_k]
                masks += [mask_k]
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

            x_hat = GD(xf_merge)

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
                lossG += lossG_adv

            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())
            if opt.use_adv:
                post_fix_str += ', G_adv=%.3f, D_loss=%.3f'%(lossG_adv.item(), lossD.item())

            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            if i_batch * opt.batch_size % 1000 == 0:
                final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,None, g_x, g_y,None,None, xf_merge, x_hat,\
                    flows, F.interpolate(mask_normed, size=g_x.shape[2:], mode='bilinear',align_corners=opt.align_corner), xfs, xfs_warp, xfs_warp_masked)
                
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), final_img)
                if opt.parallel:
                    torch.save({
                        'epoch': epoch+1,
                        'lossesG': lossesG,
                        'GE_state_dict': GE.module.state_dict(),
                        'GF_state_dict': GF.module.state_dict(),
                        'GD_state_dict': GD.module.state_dict(),
                        'i_batch': i_batch,
                        'optimizerG': optimizerG.state_dict(),
                        }, path_to_chkpt_G)
                    torch.save({
                        'Dis_state_dict': D.module.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        }, path_to_chkpt_D)
                else:
                    torch.save({
                        'epoch': epoch+1,
                        'lossesG': lossesG,
                        'GE_state_dict': GE.state_dict(),
                        'GF_state_dict': GF.state_dict(),
                        'GD_state_dict': GD.state_dict(),
                        'i_batch': i_batch,
                        'optimizerG': optimizerG.state_dict(),
                        }, path_to_chkpt_G)

                    torch.save({
                        'Dis_state_dict': D.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        }, path_to_chkpt_D)

        '''save image result'''
        final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,None, g_x, g_y,None,None, xf_merge, x_hat,\
                    flows, F.interpolate(mask_normed, size=g_x.shape[2:], mode='bilinear',align_corners=opt.align_corner), xfs, xfs_warp, xfs_warp_masked)

        plt.imsave(os.path.join(path_to_visualize_dir,"epoch_latest.png"), final_img)
        if epoch % save_freq == 0:
            plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}.png".format(epoch)), final_img)
            if opt.parallel:
                torch.save({
                    'epoch': epoch+1,
                    'lossesG': lossesG,
                    'GE_state_dict': GE.module.state_dict(),
                    'GF_state_dict': GF.module.state_dict(),
                    'GD_state_dict': GD.module.state_dict(),
                    'i_batch': i_batch,
                    'optimizerG': optimizerG.state_dict(),
                    }, path_to_chkpt_G)
                torch.save({
                    'Dis_state_dict': D.module.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    }, path_to_chkpt_D)
            else:
                torch.save({
                    'epoch': epoch+1,
                    'lossesG': lossesG,
                    'GE_state_dict': GE.state_dict(),
                    'GF_state_dict': GF.state_dict(),
                    'GD_state_dict': GD.state_dict(),
                    'i_batch': i_batch,
                    'optimizerG': optimizerG.state_dict(),
                    }, path_to_chkpt_G)

                torch.save({
                    'Dis_state_dict': D.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    }, path_to_chkpt_D)
            
  
    writer.close()

def train_flow_net_with_occ(opt, exp_name):
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
    GF, GE, GD, optimizerG = init_generator_with_occ(opt, path_to_chkpt)
    
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
            
            flows, flow_ones, masks, xfs = [], [], [], []
            flows_down,masks_down, xfs_warp = [], [], []

            with torch.no_grad():
                g_xf = GE(g_x)['relu4_1']
            for k in range(0, opt.K):
                flow_ks, mask_ks, _ = GF(ref_xs[k], ref_ys[k], g_y) # 32, 64
                flow_k = flow_ks[1]
                # print(flow_k.shape)
                mask_k = mask_ks[1]
                # print(mask_k.shape)
                with torch.no_grad():
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

            # x_hat = GD(xf_merge)
            # x_hat =  warp_flow(F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear',align_corners=opt.align_corner), flow_k, align_corners=opt.align_corner)
            # x_hat = F.interpolate(x_hat, size=g_x.shape[2:],mode='bilinear',align_corners=opt.align_corner)
            down_ref_xk = F.interpolate(ref_xs[k], size=flow_k.shape[2:], mode='bilinear', align_corners=opt.align_corner)
            x_hat_down =  warp_flow(down_ref_xk, flow_k, align_corners=opt.align_corner)
            x_hat = F.interpolate(x_hat_down, size=ref_xs[k].shape[2:], mode='bilinear', align_corners=opt.align_corner)
            loss_correct=0
            for k in range(opt.K):
                loss_correct += criterionCorrectness(g_x, ref_xs[k], flows[k], used_layers=[2, 3], use_bilinear_sampling=opt.use_bilinear)
                # loss_correct += criterionL1(xfs_warp[k], g_xf)
            loss_correct = loss_correct/opt.K * 5
            lossG = loss_correct
            writer.add_scalar('{0}/loss_correct'.format(opt.phase), loss_correct.item(), global_step=i_batch_total, walltime=None)
            
            loss_regular = 0
            if opt.use_flow_reg:
                for k in range(opt.K):
                    loss_regular += criterionReg(flows[k])
                
                loss_regular = loss_regular / opt.K * 0.0025
            
                lossG += loss_regular
                writer.add_scalar('{0}/loss_regular'.format(opt.phase), loss_regular.item(), global_step=i_batch_total, walltime=None)
            optimizerG.zero_grad()
            lossG.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            if opt.use_flow_reg:
                post_fix_str = 'Epoch_loss=%.3f, G=%.3f, Cor=%.3f, Reg=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_correct.item(), loss_regular.item())
            else:
                post_fix_str = 'Epoch_loss=%.3f, G=%.3f, Cor=%.3f'%(epoch_loss_G_moving, lossG.item(), loss_correct.item())
            pbar.set_postfix_str(post_fix_str)
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(lossG.item())
            
            if i_batch % opt.img_save_freq == 0:    
                final_img,_,_ = get_visualize_result(opt, ref_xs, ref_ys,None, g_x, g_y,None,None, g_xf, x_hat,\
                    flow_ones, F.interpolate(mask_normed, size=g_x.shape[2:], mode='bilinear',align_corners=opt.align_corner), xfs, xfs_warp, xfs_warp_masked)
                
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), final_img)

            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                save_generator(opt.parallel, epoch, lossesG, GE, GF, GD, i_batch, optimizerG, path_to_save_G)

    writer.close()

def test(opt):
    from util.io import load_image, load_skeleton, load_parsing, transform_image

    print('-----------TESTING-----------')
    experiment_name = opt.test_id
    test_dataset = opt.test_dataset
    test_source = opt.test_source
    test_motion = opt.test_target_motion

    print('Experim: ', experiment_name)
    print('Dataset: ', test_dataset)
    print('Source : ', test_source)
    print('Motion : ', test_motion)

    """Create dataset and dataloader"""
    path_to_test_A = '/dataset/ljw/{0}/test_256/train_A/'.format(test_dataset)
    path_to_test_kps = '/dataset/ljw/{0}/test_256/train_alphapose/'.format(test_dataset)
    path_to_test_parsing = '/dataset/ljw/{0}/test_256/parsing_A/'.format(test_dataset)
    if opt.use_clean_pose:
        path_to_train_kps = '/dataset/ljw/{0}/test_256/train_video2d/'.format(test_dataset)
    path_to_test_source_imgs = os.path.join(path_to_test_A, test_source)
    path_to_test_source_kps = os.path.join(path_to_test_kps, test_source)
    path_to_test_source_parse = os.path.join(path_to_test_parsing, test_source)

    path_to_test_tgt_motions = os.path.join(path_to_test_A, test_motion)
    path_to_test_tgt_kps = os.path.join(path_to_test_kps, test_motion)
    path_to_test_tgt_parse = os.path.join(path_to_test_parsing, test_motion)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    ref_ids = opt.ref_ids
    ref_ids = [int(i) for i in ref_ids.split(',')]
    total_ids = len(os.listdir(path_to_test_source_imgs))
    total_gts = len(os.listdir(path_to_test_tgt_motions))
    
    assert(max(ref_ids) <= total_ids)
    ref_names = ['{:05d}'.format(ref_id) for ref_id in ref_ids]

    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(experiment_name)
    path_to_chkpt = path_to_ckpt_dir + 'seg_model_weights.tar' 

    test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/'.format(experiment_name,test_dataset, test_source)
    test_result_vid_dir = test_result_dir + test_motion
    for ref_name in ref_names:
        test_result_vid_dir += '_{0}'.format(ref_name)
    if not os.path.isdir(test_result_vid_dir):
        os.makedirs(test_result_vid_dir)

    '''Create Model'''
    P_inc = 43 # 20 source parsing + 20 target pose + 3 image
    GP = nn.DataParallel(ParsingGenerator(inc=P_inc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral, use_attn=opt.use_attn).to(device)) # dx + dx + dy = 3 + 20 + 20
    GP.eval()

    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GP.module.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    print('Epoch:', epochCurrent)

    ref_xs = []
    ref_ys = []
    ref_ps = []
    for i, ref_name in enumerate(ref_names):
        ref_xs += [load_image(os.path.join(path_to_test_source_imgs, ref_name+'.png')).unsqueeze(0).to(device)]
        ref_ys += [load_skeleton(os.path.join(path_to_test_source_kps, ref_name+'.json')).unsqueeze(0).to(device)]
        ref_ps += [load_parsing(os.path.join(path_to_test_source_parse, ref_name+'.png')).unsqueeze(0).to(device)]

    K = len(ref_xs)
    assert(len(ref_xs)==len(ref_ys))
    assert(len(ref_xs)==len(ref_ps))

    for gt_id in tqdm(range(0, total_gts, 5)):
        gt_name = '{:05d}'.format(gt_id)
        
        g_x = load_image(os.path.join(path_to_test_tgt_motions, gt_name+'.png')).unsqueeze(0).to(device)
        g_y = load_skeleton(os.path.join(path_to_test_tgt_kps, gt_name+'.json')).unsqueeze(0).to(device)
        g_p = load_parsing(os.path.join(path_to_test_tgt_parse, gt_name+'.png')).unsqueeze(0).to(device)


        '''Get pixel wise logits'''
        logits = []
        attns = []
        if opt.use_attn:
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
                logit_k = GP(ref_xs[k], ref_ps[k], g_y) # [B, 20, H, W], [B, 1, H, W]
                logits += [logit_k]
            logit_avg = logits[0] / K #  [B, 20, H, W]
            for k in range(1, K):
                logit_avg += logits[k] / K #  [B, 20, H, W]            
        
        p_hat_bin_maps = logit_avg[:,0:1,:,:].clone().detach() # [N,1, H, W]
        p_hat_bin_maps = p_hat_bin_maps.repeat(1,20,1,1) # [N, 20, H, W]
        
        for batch in range(logit_avg.shape[0]):
            p_hat_indices = torch.argmax(logit_avg[batch], dim=0) #  [C, H, W] -> [H, W]
            p_hat_bin_map = p_hat_indices.view(-1,256,256).repeat(20,1,1) # [20, H, W]
            for i in range(20):
                p_hat_bin_map[i, :, :] = (p_hat_indices == i).int()
            p_hat_bin_maps[batch] = p_hat_bin_map
        
        final_img = get_parse_visual_result(opt, ref_xs, ref_ys,ref_ps, g_x, g_y, g_p, p_hat_bin_maps)
        
        plt.imsave(os.path.join(test_result_vid_dir,"{0}_result.png".format(gt_name)), final_img)
        # plt.imsave(os.path.join(test_result_vid_dir,"{0}_result_simp.png".format(gt_name)), simp_img)

    '''save video result'''
    save_video_name = test_motion
    img_dir = test_result_vid_dir
    save_video_dir = test_result_dir
    for ref_name in ref_names:
        save_video_name += '_{0}'.format(ref_name)
    save_video_name_simp = save_video_name +'_result_simp.mp4'
    save_video_name += '_result.mp4'
    imgs = os.listdir(img_dir)
    import cv2
    # video_out_simp = cv2.VideoWriter(save_video_dir+save_video_name_simp, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (256*(K+2), 256))
    video_out = cv2.VideoWriter(save_video_dir+save_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (256*(K+2), 256*3))
    for img in tqdm(sorted(imgs)):
        # if img.split('.')[0].split('_')[-1] == 'simp':
        #     frame = cv2.imread(os.path.join(img_dir, img))
        #     video_out_simp.write(frame)
        if img.split('.')[0].split('_')[-1] == 'result':
            frame = cv2.imread(os.path.join(img_dir, img))
            video_out.write(frame)

    # video_out_simp.release()
    video_out.release()
    pass


if __name__ == "__main__":
    opt = get_parser()
    for k,v in sorted(vars(opt).items()):
        print(k,':',v)
    set_random_seed(opt.seed)
    
    if opt.phase == 'train':
        today = datetime.today().strftime("%Y%m%d")
        # today = '20201123'
        # fashion_v0_flow_pretrain_GLFA_lsGAN_1shot_20201123
        experiment_name = 'fashion_pretrain_flow_v{0}_{1}shot_{2}'.format(opt.id,opt.K, today)
        print(experiment_name)
        # train(opt, experiment_name)
        # train_flow_net(opt, experiment_name)
        train_flow_net_with_occ(opt, experiment_name)
    else:
        test(opt)




