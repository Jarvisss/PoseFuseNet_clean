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
from model.Parsing_net import MaskGenerator
from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_mask_visual_result
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashion_dataset import FashionDataset
from loss.loss_generator import PerceptualCorrectness, LossG, ParsingLoss

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
    parser.add_argument('--pretrain_flow', action='store_true')
    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--categories', type=int, default=9)
    parser.add_argument('--use_parsing', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_input_mask', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_input_y', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--align_input', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_simmap', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')

    '''Model options'''
    parser.add_argument('--norm_type', type=str, default='in', help='normalization type in network, "in" or "bn"')
    parser.add_argument('--activation', type=str, default='LeakyReLU', help='"ReLU" or "LeakyReLU"')

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

    opt = parser.parse_args()
    return opt   


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


def init_weights(m, init_type='normal'):
    if type(m) == nn.Conv2d:
        if init_type=='xavier':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='normal':
            torch.nn.init.normal_(m.weight,0,0.02)
        elif init_type=='kaiming':
            torch.nn.init.kaiming_normal_(m.weight)

def make_dataset(opt):
    """Create dataset"""
    path_to_dataset = opt.path_to_dataset
    train_tuples_name = 'fasion-pairs-train.csv' if opt.K==1 else 'fasion-%d_tuples-train.csv'%(opt.K+1)
    test_tuples_name = 'fasion-pairs-test.csv' if opt.K==1 else 'fasion-%d_tuples-test.csv'%(opt.K+1)
    path_to_train_label = '/dataset/ljw/deepfashion/GLFA_split/fashion/train_tps_field'
    path_to_test_label = '/dataset/ljw/deepfashion/GLFA_split/fashion/test_tps_field' 
    path_to_train_sim = '/dataset/ljw/deepfashion/GLFA_split/fashion/train_sim'
    path_to_test_sim = '/dataset/ljw/deepfashion/GLFA_split/fashion/test_sim' 
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
    else: # '/home/ljw/playground/Multi-source-Human-Image-Generation/data/fasion-dataset'
        dataset = FashionDataset(
            phase = opt.phase,
            path_to_train_tuples=os.path.join(path_to_dataset, 'fasion-%d_tuples-train.csv'%(opt.K+1)), 
            path_to_test_tuples=os.path.join(path_to_dataset, 'fasion-%d_tuples-test.csv'%(opt.K+1)), 
            path_to_train_imgs_dir=os.path.join(path_to_dataset, 'train/'), 
            path_to_test_imgs_dir=os.path.join(path_to_dataset, 'test/'),
            path_to_train_anno=os.path.join(path_to_dataset, 'fasion-annotation-train_new_split.csv'), 
            path_to_test_anno=os.path.join(path_to_dataset, 'fasion-annotation-test_new_split.csv'), 
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
def init_model(path_to_chkpt, GP, optimizerG):
    GP.apply(init_weights)
    print('Initiating new checkpoint...')
    torch.save({
            'epoch': 0,
            'lossesG': [],
            'GP_state_dict': GP.module.state_dict(),
            'i_batch': 0,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)
    print('...Done')

def save_generator(parallel, epoch, lossesG, GP, i_batch, optimizerG,path_to_chkpt_G):
    GP_state_dict = GP.state_dict() if not parallel else GP.module.state_dict()
    torch.save({
        'epoch': epoch,
        'lossesG': lossesG,
        'GP_state_dict': GP_state_dict,
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
    }, path_to_chkpt_G)

def save_eval_loss(epoch, batch, eval_loss, path_to_save):
    with open(path_to_save, 'a') as f:
        f.write('Epoch: %d, Batch: %d, eval loss: %.4f\r\n'%(epoch, batch, eval_loss))


def init_generator(opt, path_to_chkpt):
    image_nc = 3
    structure_nc = 21
    mask_nc = 1
    n_layers = 5

    if opt.use_input_y:
        GP = MaskGenerator(inc=image_nc+structure_nc*2+mask_nc,onc=mask_nc, n_layers=n_layers,ngf=64, max_nc=512, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    else:
        GP = MaskGenerator(inc=image_nc+structure_nc+mask_nc,onc=mask_nc, n_layers=n_layers,ngf=64, max_nc=512, norm_type=opt.norm_type, activation=opt.activation, use_spectral_norm=opt.use_spectral_G)
    
    
    if opt.parallel:
        GP = nn.DataParallel(GP) # dx + dx + dy = 3 + 20 + 20

    GP = GP.to(device)

    optimizerG = optim.Adam(params = list(GP.parameters()),
                            lr=opt.lr,
                            amsgrad=False)
    if opt.use_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizerG, 'min', factor=np.sqrt(0.1), patience=5, min_lr=5e-7)
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        GP.apply(init_weights)
        print('Initiating new checkpoint...')
        save_generator(opt.parallel, 0, [],GP, 0, optimizerG, path_to_chkpt)
        print('...Done')

    return GP, optimizerG

def train(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    '''Set experiment name and logging,checkpoint,vis dir'''

    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, exp_name)
    path_to_chkpt = path_to_ckpt_dir + 'masknet_model_weights.tar' 
    

    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    """Create dataset and dataloader"""
    print('-------Loading Dataset--------')
    dataset = make_dataset(opt)
    dataloader = make_dataloader(opt, dataset)
    
    opt.phase='test'
    # test_dataset = make_dataset(opt)
    eval_dataset = make_dataset(opt)
    # test_dataloader = make_dataloader(opt, dataset)
    opt.phase='train'

    
    '''Create Model'''
    GP, optimizerG = init_generator(opt, path_to_chkpt)

    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    if opt.parallel:
        GP.module.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    else:
        GP.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    GP.train()
    
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    i_batch_current = checkpoint['i_batch']
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    i_batch_total = epochCurrent * dataloader.__len__() // opt.batch_size + i_batch_current

    """
    create tensorboard writter
    """
    writer = create_writer(path_to_log_dir)
    
    '''Losses'''
    criterion = nn.BCEWithLogitsLoss().to(device)


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
            if opt.use_input_mask:
                ref_ms = batch_data['ref_ms']
                g_m = batch_data['g_m']
                assert(len(ref_ms)==opt.K)
                for i in range(opt.K):
                    ref_ms[i] = ref_ms[i].to(device)
                g_m = g_m.to(device)

            for i in range(len(ref_xs)):
                ref_xs[i] = ref_xs[i].to(device)
                ref_ys[i] = ref_ys[i].to(device)


            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]

        
            optimizerG.zero_grad()

            '''Get pixel wise logits'''
            logits = []
            attns = []
            for k in range(0, opt.K):
                if opt.use_input_y:
                    logit_k, attn_k = GP(ref_xs[k], ref_ms[k], g_y, pose_1=ref_ys[k]) # [B, 1, H, W], [B, 1, H, W]
                else:
                    logit_k, attn_k = GP(ref_xs[k], ref_ms[k], g_y) # [B, 1, H, W], [B, 1, H, W]

                logits += [logit_k]
                attns += [attn_k]
            
            attn_norm = torch.cat(attns, dim=1)
            attn_norm = nn.Softmax(dim=1)(attn_norm)

            logit_weighted = []
            for k in range(0, opt.K):
                logit_weighted += [logits[k] * attn_norm[:,k:k+1,...]]

            # logit_avg = sum(logits) / opt.K #  [B, 1, H, W]
            logit_avg = sum(logit_weighted) #  [B, 1, H, W]
            m_hat = nn.Sigmoid()(logit_avg)
            m_hat[m_hat>0.5]=1
            m_hat[m_hat<=0.5]=0
            loss = criterion(logit_avg, g_m)
            loss.backward(retain_graph=False)
            optimizerG.step()
            epoch_loss_G += loss.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            i_batch_total += 1
            
            post_fix_str = 'Epoch_loss=%.3f, batch_loss=%.3f'%(epoch_loss_G_moving, loss.item())
            pbar.set_postfix_str(post_fix_str)
        
            if opt.use_scheduler:
                lr_scheduler.step(epoch_loss_G_moving)
            lossesG.append(loss.item())
            
            '''save image result'''
            if i_batch % opt.img_save_freq == 0:
                final_img = get_mask_visual_result(opt, ref_xs, ref_ys,ref_ms, g_x, g_y, g_m, m_hat)
                plt.imsave(os.path.join(path_to_visualize_dir,"epoch_{}_batch_{}.png".format(epoch, i_batch)), final_img)

                
            if i_batch % opt.model_save_freq == 0:
                path_to_save_G = path_to_ckpt_dir + 'epoch_{}_batch_{}_G.tar'.format(epoch, i_batch)
                save_generator(opt.parallel, 0, [], GP, 0, optimizerG, path_to_save_G)
                path_to_loss_log = path_to_ckpt_dir + 'eval_loss.txt'

                '''eval the model when save'''
                with torch.no_grad():
                    
                    eval_loss = evaluate(opt, eval_dataset, path_to_save_G)
                    save_eval_loss(epoch, i_batch, eval_loss, path_to_loss_log)
                
        save_generator(opt.parallel, epoch+1, lossesG, GP, i_batch, optimizerG, path_to_chkpt)
    writer.close()

def evaluate(opt,dataset,path_to_chkpt, save_img=False, save_dir=None):
    criterion = nn.BCEWithLogitsLoss().to(device)
    GP, optimizerG = init_generator(opt, path_to_chkpt)
    dataloader = DataLoader(dataset, 1, False, num_workers=8, drop_last=False)
    
    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    if opt.parallel:
        GP.module.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    else:
        GP.load_state_dict(checkpoint['GP_state_dict'], strict=False)
    GP.eval()
    epoch_loss_G = 0
    pbar = tqdm(dataloader, leave=True, initial=0)
    for i_batch, batch_data in enumerate(pbar, start=0):
        froms = batch_data['froms']
        to = batch_data['to']
        ref_xs = batch_data['ref_xs']
        ref_ys = batch_data['ref_ys']
        g_x = batch_data['g_x']
        g_y = batch_data['g_y']
        if opt.use_input_mask:
            ref_ms = batch_data['ref_ms']
            g_m = batch_data['g_m']
            assert(len(ref_ms)==opt.K)
            for i in range(opt.K):
                ref_ms[i] = ref_ms[i].to(device)
            g_m = g_m.to(device)

        for i in range(len(ref_xs)):
            ref_xs[i] = ref_xs[i].to(device)
            ref_ys[i] = ref_ys[i].to(device)

        g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]
        # logits = []
        # for k in range(0, opt.K):
        #     logit_k = GP(ref_xs[k], ref_ms[k], g_y) # [B, 1, H, W], [B, 1, H, W]
        #     logits += [logit_k]
        
        # logit_avg = sum(logits) / opt.K #  [B, 1, H, W]
        logits = []
        attns = []
        for k in range(0, opt.K):
            if opt.use_input_y:
                logit_k, attn_k = GP(ref_xs[k], ref_ms[k], g_y, pose_1=ref_ys[k]) # [B, 1, H, W], [B, 1, H, W]
            else:
                logit_k, attn_k = GP(ref_xs[k], ref_ms[k], g_y) # [B, 1, H, W], [B, 1, H, W]
            logits += [logit_k]
            attns += [attn_k]
        
        attn_norm = torch.cat(attns, dim=1)
        attn_norm = nn.Softmax(dim=1)(attn_norm)

        logit_weighted = []
        for k in range(0, opt.K):
            logit_weighted += [logits[k] * attn_norm[:,k:k+1,...]]

        # logit_avg = sum(logits) / opt.K #  [B, 1, H, W]
        logit_avg = sum(logit_weighted) #  [B, 1, H, W]
        m_hat = nn.Sigmoid()(logit_avg)
        m_hat[m_hat>0.5]=1
        m_hat[m_hat<=0.5]=0
        loss = criterion(logit_avg, g_m)
        epoch_loss_G += loss.item()
        if save_img and not save_dir==None:
            test_name = ''
            # print(froms)
            for k in range(opt.K):
                test_name += str(froms[k][0])+'+'
            test_name += str(to[0])
            final_img = get_mask_visual_result(opt, ref_xs, ref_ys,ref_ms, g_x, g_y, g_m, m_hat)
            from PIL import Image
            Image.fromarray(final_img).save(os.path.join(save_dir,"{}_all.png".format(test_name)))

    epoch_loss_G = epoch_loss_G / len(dataloader)
    print('eval loss: %.3f'%(epoch_loss_G))
    return epoch_loss_G

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
        experiment_name = '{0}_{1}shot_{2}'.format(opt.id,opt.K, today)
        print(experiment_name)
        train(opt, experiment_name)
    else:
        with torch.no_grad():
            experiment_name = opt.test_id
            test_result_dir = './test_result/{0}/{1}/{2}_shot/'.format(experiment_name, opt.test_ckpt_name, opt.K)
            test_result_eval_dir = './test_result/{0}/{1}/{2}_shot_eval/'.format(experiment_name, opt.test_ckpt_name, opt.K)
            
            path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(experiment_name)
        
            path_to_chkpt_G = path_to_ckpt_dir + '{0}.tar'.format(opt.test_ckpt_name) 
            if not os.path.isdir(test_result_dir):
                os.makedirs(test_result_dir)
            if not os.path.isdir(test_result_eval_dir):
                os.makedirs(test_result_eval_dir)
            dataset = make_dataset(opt)
            evaluate(opt,dataset, path_to_chkpt_G,save_img=True, save_dir=test_result_dir)
            # test(opt)




