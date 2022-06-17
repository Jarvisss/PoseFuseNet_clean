import argparse
import torch
import os
from tensorboardX import SummaryWriter
from datetime import datetime
from dataset.fashion_dataset import FashionDataset
from torch.utils.data import DataLoader

def get_default_parser():
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
    parser.add_argument('--use_simmap', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--joints_for_cos_sim',type=int, default=-1, help='joints used for cosine sim computation')

    '''Model options'''
    parser.add_argument('--n_enc', type=int, default=2, help='encoder(decoder) layers ')
    parser.add_argument('--n_btn', type=int, default=2, help='bottle neck layers in generator')
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn"')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='"ReLU" or "LeakyReLU"')
    parser.add_argument('--use_pose_decoder', action='store_true', help='use pose in the target decoder')
    parser.add_argument('--use_self_attention', action='store_true', help='use sa in the target decoder')

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
    parser.add_argument('--tps_sim_beta1', type=float, default=5.0, help='use precomputed tps sim')
    parser.add_argument('--tps_sim_beta2', type=float, default=40.0, help='use precomputed tps sim')

    '''Loss options'''
    parser.add_argument('--use_adv', action='store_true', help='use adversarial loss in total generation')
    parser.add_argument('--use_bilinear_correctness', action='store_true', help='use bilinear sampling in sample loss')
    parser.add_argument('--G_use_resample', action='store_true', help='use gaussian sampling in the target decoder')
    parser.add_argument('--use_correctness', action='store_true', help='use sample correct loss')
    parser.add_argument('--use_flow_attn_loss', action='store_true', help='use flow attention loss')
    parser.add_argument('--use_flow_reg', action='store_true', help='use flow regularization')
    parser.add_argument('--use_attn_reg', action='store_true', help='use attn regularization')

    parser.add_argument('--lambda_style', type=float, default=500.0, help='style loss')
    parser.add_argument('--lambda_content', type=float, default=0.5, help='content loss')
    parser.add_argument('--lambda_rec', type=float, default=5.0, help='L1 loss')
    parser.add_argument('--lambda_adv', type=float, default=2.0, help='GAN loss weight')
    parser.add_argument('--lambda_correctness', type=float, default=5.0, help='sample correctness weight')
    parser.add_argument('--lambda_attn_reg', type=float, default=1.0, help='regular sample loss weight')
    parser.add_argument('--lambda_flow_reg', type=float, default=0.0025, help='regular sample loss weight')
    parser.add_argument('--lambda_flow_attn', type=float, default=5.0, help='regular sample loss weight')
    
    return parser


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
    if type(m) == torch.nn.Conv2d:
        if init_type=='xavier':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='normal':
            torch.nn.init.normal_(m.weight,0,0.02)
        elif init_type=='kaiming':
            torch.nn.init.kaiming_normal_(m.weight)


def make_dataloader(opt, dataset):
    is_train = opt.phase == 'train'
    batch_size = 1 if not is_train else opt.batch_size
    shuffle = is_train
    drop_last = is_train
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=8, drop_last=drop_last)
    return dataloader