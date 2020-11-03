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
from model.Parsing_net import ParsingGenerator
from util.vis_util import visualize_feature, visualize_feature_group, visualize_parsing, get_parse_visual_result
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashionvideo_dataset import FashionVideoDataset
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
    parser.add_argument('--test',  action='store_true', help='open this to test')
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


    '''Test options'''
    # if --test is open
    parser.add_argument('--test_id', type=str, default='default', help = 'test experiment ID. the experiment dir will be set as "./checkpoint/id/"')
    parser.add_argument('--ref_ids', type=str, default='0', help='test ref ids')
    parser.add_argument('--test_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_source', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref images')
    parser.add_argument('--test_target_motion', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref motions')
    

    '''Experiment options'''
    parser.add_argument('--use_attn', action='store_true', help='use attention for multi-view parsing generation')
    '''Loss options'''


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
    path_to_chkpt = path_to_ckpt_dir + 'seg_model_weights.tar' 
    

    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    """Create dataset and dataloader"""
    path_to_train_A = '/dataset/ljw/{0}/train_256/train_A/'.format(opt.dataset)
    path_to_train_kps = '/dataset/ljw/{0}/train_256/train_alphapose/'.format(opt.dataset)
    path_to_train_parsing = '/dataset/ljw/{0}/train_256/parsing_A/'.format(opt.dataset)
    if opt.use_clean_pose:
        path_to_train_kps = '/dataset/ljw/{0}/train_256/train_video2d/'.format(opt.dataset)

    dataset = FashionVideoDataset(path_to_train_A=path_to_train_A, path_to_train_kps=path_to_train_kps,path_to_train_parsing=path_to_train_parsing, opt=opt)
    print(dataset.__len__())

    fashionVideoDataLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

    '''Create Model'''
    P_inc = 43 # 20 source parsing + 20 target pose + 3 image
    GP = nn.DataParallel(ParsingGenerator(inc=P_inc, norm_type=opt.norm_type, use_spectral_norm=opt.use_spectral, use_attn=opt.use_attn).to(device)) # dx + dx + dy = 3 + 20 + 20
    GP.train()

    optimizerG = optim.Adam(params = list(GP.parameters()), lr=opt.lr, amsgrad=False)
    if opt.use_scheduler:
        lr_scheduler = ReduceLROnPlateau(optimizerG, 'min', factor=np.sqrt(0.1), patience=5, min_lr=5e-7)
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        init_model(path_to_chkpt, GP, optimizerG)


    '''Losses'''
    criterion = ParsingLoss(device=device)

    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GP.module.load_state_dict(checkpoint['GP_state_dict'], strict=False)
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
        for i_batch, (ref_xs, ref_ys,ref_ps, g_x, g_y,g_p, vid_path) in enumerate(pbar, start=0):
            assert(len(ref_xs)==len(ref_ys))
            assert(len(ref_xs)==len(ref_ps))
            assert(len(ref_xs)==opt.K)
            

            for i in range(len(ref_xs)):
                ref_xs[i] = ref_xs[i].to(device)
                ref_ys[i] = ref_ys[i].to(device)
                ref_ps[i] = ref_ps[i].to(device)


            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]
            g_p = g_p.to(device) # [B, 20, 256, 256]
            g_p_label = g_p.clone().detach().to(device)
            
            # from binary map (C=20) to label map(C=1)
            for i in range(20):
                g_p_label[:,i,:,:] = (g_p[:,i,:,:] == 1).int() * i 
            
            g_p_label = g_p_label.sum(dim=1) #[B, 256, 256]

            optimizerG.zero_grad()

            '''Get pixel wise logits'''
            logits = []
            attns = []
            if opt.use_attn:
                for k in range(0, opt.K):
                    logit_k,attn_k = GP(ref_xs[k], ref_ps[k], g_y) # [B, 20, H, W], [B, 1, H, W]
                    logits += [logit_k]
                    attns += [attn_k]
                attns = torch.cat(attns, dim=1)
                attn_norm = torch.softmax(attns, dim=1)
                logit_avg = logits[0] * attn_norm[:,0:1,:,:] #  [B, 20, H, W]
                for k in range(1, opt.K):
                    logit_avg += logits[k] * attn_norm[:,k:k+1,:,:] #  [B, 20, H, W]
            else:
                for k in range(0, opt.K):
                    logit_k,attn_k = GP(ref_xs[k], ref_ps[k], g_y) # [B, 20, H, W], [B, 1, H, W]
                    logits += [logit_k]
                
                logit_avg = logits[0] / opt.K #  [B, 20, H, W]
                for k in range(1, opt.K):
                    logit_avg += logits[k] / opt.K #  [B, 20, H, W]


            loss = criterion(logit_avg, g_p_label.long())
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
        if epoch % save_freq == 0:
            torch.save({
            'epoch': epoch+1,
            'lossesG': lossesG,
            'GP_state_dict': GP.module.state_dict(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)

            p_hat_bin_maps = logit_avg[:,0:1,:,:].clone().detach() # [N,1, H, W]
            p_hat_bin_maps = p_hat_bin_maps.repeat(1,20,1,1) # [N, 20, H, W]
            
            for batch in range(logit_avg.shape[0]):
                p_hat_indices = torch.argmax(logit_avg[batch], dim=0) #  [C, H, W] -> [H, W]
                p_hat_bin_map = p_hat_indices.view(-1,256,256).repeat(20,1,1) # [20, H, W]
                for i in range(20):
                    p_hat_bin_map[i, :, :] = (p_hat_indices == i).int()
                p_hat_bin_maps[batch] = p_hat_bin_map

            final_img = get_parse_visual_result(opt, ref_xs, ref_ys,ref_ps, g_x, g_y, g_p, p_hat_bin_maps)
            plt.imsave(visualize_result_dir+"epoch_{}.png".format(epoch), final_img)
            

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
    
    if not opt.test:
        today = datetime.today().strftime("%Y%m%d")
        experiment_name = 'parsing_v{0}_{1}shot_{2}'.format(opt.id,opt.K, today)
        print(experiment_name)
        train(opt, experiment_name)
    else:
        test(opt)




