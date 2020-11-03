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

from model.GMM import GMM

from util.vis_util import visualize_feature, visualize_feature_group,\
     visualize_cloth_parsing, get_visualize_result, visualize_merge_cloth_parsing, visualize_gt_cloth_parsing
from util.io import load_image
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from dataset.fashionvideo_dataset import FashionVideoGeoMatchingDataset
from loss.loss_generator import GicLoss

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
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--K', type=int, default=1, help='source image views')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--root_dir',type=str, default='/home/ljw/playground/poseFuseNet/')
    parser.add_argument('--dataset',type=str, default='danceFashion', help='"danceFashion" or "iper"')

    '''Train options'''
    parser.add_argument('--epochs', type=int, default=2000, help='num epochs')
    parser.add_argument('--use_scheduler', action='store_true', help='open this to use learning rate scheduler')

    '''Dataset options'''
    
    '''Test options'''
    parser.add_argument('--test_id',  type=str, default='default')
    parser.add_argument('--ref_id', type=int, default=0)
    parser.add_argument('--test_dataset', type=str, default='danceFashion', help='"danceFashion" or "iper"')
    parser.add_argument('--test_source', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref images')
    parser.add_argument('--test_target_motion', type=str, default='A15Ei5ve9BS', help='a test video in dataset as ref motions')

    # if --test is open
    '''GMM options'''
    parser.add_argument('--rigid', action='store_true')
    parser.add_argument('--input_nc', type=int, default=1) 
    parser.add_argument('--fine_width', type=int, default=256)
    parser.add_argument('--fine_height', type=int, default=256)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--radius', type=int, default = 5)
    parser.add_argument('--n_layers', type=int, default=3)

    '''Loss options'''
    parser.add_argument('--use_tvloss', action='store_true')

    return parser   

def writer_create(path_to_log_dir):
    TIMESTAMP = "/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(path_to_log_dir+TIMESTAMP)
    return writer

def init_weights(m, init_type='normal'):
    if type(m) == nn.Conv2d:
        if init_type=='xavier':
            torch.nn.init.xavier_uniform_(m.weight)
        elif init_type=='normal':
            torch.nn.init.normal_(m.weight)
        elif init_type=='kaiming':
            torch.nn.init.kaiming_normal_(m.weight)

def init_model(path_to_chkpt, gmm, optimizerG):

    gmm.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': 0,
            'gmm_state_dict': gmm.module.state_dict(),
            'i_batch': 0,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)
    print('...Done')


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h + w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

"""
train gmm, using source and target parsings to estimate a TPS transform, with 6 parameters to present warping
"""
def train_gmm(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(exp_name)
    visualize_result_dir = opt.root_dir+ 'visualize_result/{0}/'.format(exp_name)
    path_to_log_dir = opt.root_dir+ 'logs/{0}'.format(exp_name)
    
    if not os.path.isdir(path_to_ckpt_dir):
        os.makedirs(path_to_ckpt_dir)
    if not os.path.isdir(visualize_result_dir):
        os.makedirs(visualize_result_dir)
    if not os.path.isdir(path_to_log_dir):
        os.makedirs(path_to_log_dir)
    '''save parser'''
    save_parser(opt, path_to_ckpt_dir+'config.json')
    
    # criterion
    criterionL1 = nn.L1Loss()
    criterionTv = GicLoss()
    # optimizerG
    path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar'

    gmm = nn.DataParallel(GMM(opt, rigid=False).to(device))
    gmm.train()

    # if opt.use_rigid_affine and opt.use_tps:
    rigid_gmm = nn.DataParallel(GMM(opt, rigid=True).to(device))
    rigid_gmm.eval()
    gmm_checkpoint = torch.load( opt.root_dir+ 'checkpoints/{0}/'.format('Geo_v2_loss_parse_lr0.0001-20201023') + 'model_weights.tar')
    rigid_gmm.module.load_state_dict(gmm_checkpoint['gmm_state_dict'], strict=False)
    rigid_epoch = gmm_checkpoint['epoch']
    print('rigid gmm: trained epochs:', rigid_epoch)
    print('rigid gmm: Load Success')


    optimizerG = torch.optim.Adam(gmm.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    if not os.path.isfile(path_to_chkpt):
        # initiate checkpoint if inexist
        init_model(path_to_chkpt, gmm, optimizerG)
    
    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    gmm.module.load_state_dict(checkpoint['gmm_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    i_batch_current = checkpoint['i_batch']
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    
    '''Create dataset and dataloader'''
    path_to_train_A = '/dataset/ljw/{0}/train_256/train_A/'.format(opt.dataset)
    path_to_train_parsing = '/dataset/ljw/{0}/train_256/parsing_A/'.format(opt.dataset)

    dataset = FashionVideoGeoMatchingDataset(path_to_train_A=path_to_train_A, path_to_train_parsing=path_to_train_parsing, opt=opt)
    print(dataset.__len__())
    fashionVideoDataLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)
    i_batch_total = epochCurrent * fashionVideoDataLoader.__len__() // opt.batch_size + i_batch_current

    """ create tensorboard writter
    """
    writer = writer_create(path_to_log_dir)
    
    save_freq = 1


    """ Training start """
    for epoch in range(epochCurrent, epochCurrent + opt.epochs):
        if epoch >= 100:
            save_freq=5
        if epoch > epochCurrent:
            i_batch_current = 0
        epoch_loss_G = 0
        pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
        pbar.set_description('epoch[{0}/{1}], lr-{2}'.format(epoch,opt.epochs,optimizerG.param_groups[0]['lr']))
        img_grid = load_image('256.png').unsqueeze(0).to(device).repeat(4,1,1,1)
        
        for i_batch, (ref_x, ref_p, g_x, g_p, vid_path) in enumerate(pbar, start=0):
            ref_x = ref_x.to(device) # [B, 3, 256, 256]
            ref_p = ref_p.to(device) # [B, 1, 256, 256]
            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_p = g_p.to(device) # [B, 1, 256, 256]

            optimizerG.zero_grad()
            ref_c = ref_x * (ref_p > 0.5).float()
            gt_c = g_x * (g_p > 0.5).float()


            '''
            step 1: get affine warped grid, cloth, parsing
            '''
            with torch.no_grad():
                rigid_grid,_ = rigid_gmm(ref_p, g_p)
                rigid_warped_parsing = F.grid_sample(ref_p, rigid_grid, padding_mode='zeros')
                rigid_warped_grid = F.grid_sample(img_grid, rigid_grid, padding_mode='zeros') 
                rigid_warp_c = F.grid_sample(ref_c, rigid_grid, padding_mode='border') 

            '''
            step 2: use affine warped parsing and ground truth parsing to predict tps transorm grid
            '''
            grid, theta = gmm(rigid_warped_parsing, g_p)

            '''
            step 3: use tps transorm grid to warp the affine image, grid, parsing and cloth
            '''
            # warped_img = F.grid_sample(ref_x, grid, padding_mode='border')
            warped_grids = F.grid_sample(rigid_warped_grid, grid, padding_mode='zeros') # for visualize
            warp_c = F.grid_sample(rigid_warp_c, grid, padding_mode='border') 

            warped_parsing = F.grid_sample(rigid_warped_parsing, grid, padding_mode='zeros')
            # lossG = criterionL1(warped_img, g_x)
            # lossG = criterionL1(warp_c, gt_c)
            lossl1 = criterionL1(warped_parsing, g_p)
            # print('l1,',lossl1.item())
            loss_tv = 0
            if opt.use_tvloss:
                loss_tv = criterionTv(grid) * 100
                # print('tv,',loss_tv.item())
                

            lossG = loss_tv  + lossl1
            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            lossG.backward()
            optimizerG.step()

            i_batch_total += 1
                        
            post_fix_str = 'Epoch_loss=%.3f, G=%.3f, l1=%.3f, tv=%.3f'%(epoch_loss_G_moving, lossG.item(), lossl1.item(), loss_tv.item())
            pbar.set_postfix_str(post_fix_str)
            writer.add_scalar('loss/lossG', lossl1.item(), global_step=i_batch_total, walltime=None)
            if opt.use_tvloss:
                writer.add_scalar('loss/lossG', loss_tv.item(), global_step=i_batch_total, walltime=None)

        if epoch % save_freq == 0:
            torch.save({
            'epoch': epoch+1,
            'gmm_state_dict': gmm.module.state_dict(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)

            gx = g_x[0].permute(1,2,0) * 255.0
            rx = ref_x[0].permute(1,2,0) * 255.0
            # wx = warped_img[0].permute(1,2,0) * 255.0

            rgrid = img_grid[0].permute(1,2,0) * 255.0 # the original grid 
            rigid_wgrid = rigid_warped_grid[0].permute(1,2,0) * 255.0 # affine grid
            tps_wgrid = warped_grids[0].permute(1,2,0) * 255.0 # tps grid

            gc = gt_c[0].permute(1,2,0) * 255.0 # ground truth cloth
            rc = ref_c[0].permute(1,2,0) * 255.0 # ref cloth
            rigid_wc = rigid_warp_c[0].permute(1,2,0) * 255.0 # affine warped cloth 
            wc = warp_c[0].permute(1,2,0) * 255.0 # tps warped cloth

            # change cloth bg to gray for better visualization
            gc[gc<0.5] += 128
            rc[rc<0.5] += 128
            wc[wc<0.5] += 128
            rigid_wc[rigid_wc<0.5] += 128
            
            # visualize parsing
            gp = visualize_gt_cloth_parsing(g_p.cpu(), 0)
            wp = visualize_cloth_parsing(warped_parsing.detach().cpu(), 0)
            rigid_wp = visualize_cloth_parsing(rigid_warped_parsing.detach().cpu(), 0)
            rp = visualize_cloth_parsing(ref_p.cpu(), 0)
            
            # visualize merged parsing of two parsings
            gp_wp_merge = visualize_merge_cloth_parsing(g_p.cpu(),warped_parsing.detach().cpu(), 0)
            gp_rwp_merge = visualize_merge_cloth_parsing(g_p.cpu(),rigid_warped_parsing.detach().cpu(), 0)
            gp_rp_merge = visualize_merge_cloth_parsing(g_p.cpu(),ref_p.cpu(), 0)
            
            visual_gp = torch.from_numpy(gp).to(device).float()
            visual_wp = torch.from_numpy(wp).to(device).float()
            visual_rigid_wp = torch.from_numpy(rigid_wp).to(device).float()
            visual_rp = torch.from_numpy(rp).to(device).float()


            visual_gp_wp_merge = torch.from_numpy(gp_wp_merge).to(device).float()
            visual_gp_rwp_merge = torch.from_numpy(gp_rwp_merge).to(device).float()
            visual_gp_rp_merge = torch.from_numpy(gp_rp_merge).to(device).float()


            white = torch.ones(gc.size()).to(device) * 255.0
            
            r_col = torch.cat((rx,  white,      white,              rgrid),dim=0)
            rc_col = torch.cat((rc, visual_rp,  visual_gp_rp_merge, rgrid), dim=0)
            rigid_wc_col = torch.cat((rigid_wc, visual_rigid_wp, visual_gp_rwp_merge, rigid_wgrid), dim=0)
            wc_col = torch.cat((wc, visual_wp, visual_gp_wp_merge,tps_wgrid), dim=0)
            gc_col = torch.cat((gc, visual_gp, white, white), dim=0)
            g_col = torch.cat((gx, white, white, white), dim=0)

            # g_col = torch.cat((gx, ), dim=0)
            # w_col = torch.cat((wx, ), dim=0)
            # r_col = torch.cat((rx, ), dim=0)


            final = torch.cat((r_col, rc_col,rigid_wc_col, wc_col, gc_col, g_col), dim=1).type(torch.uint8).to(cpu).numpy()
            plt.imsave(visualize_result_dir+"epoch_{}.png".format(epoch), final)
        
    writer.close()
    pass


'''
Given certain refer images "test_source", test motions appear in "test_target",
Output predict images generate by "test_source" in target motions.
'''
def test(opt, rigid_ckpt_name, tps_ckpt_name):
    from util.io import load_image, load_skeleton, load_parsing, transform_image
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    test_dataset = opt.test_dataset
    test_source = opt.test_source
    test_target = opt.test_target_motion

    print('rigid ckpt name: ', rigid_ckpt_name)
    print('tps ckpt name: ', tps_ckpt_name)
    print('Dataset: ', test_dataset)
    print('Source : ', test_source)
    print('Target : ', test_target)

    """Create dataset and dataloader"""
    path_to_test_source_imgs = '/dataset/ljw/{0}/test_256/train_A/{1}'.format(test_dataset, test_source)
    path_to_test_tgt_imgs = '/dataset/ljw/{0}/test_256/train_A/{1}'.format(test_dataset, test_target)
    path_to_test_source_parse = '/dataset/ljw/{0}/test_256/parsing_A/{1}'.format(test_dataset, test_source)
    path_to_test_tgt_parse = '/dataset/ljw/{0}/test_256/parsing_A/{1}'.format(test_dataset, test_target)


    
    '''Create Model'''
    '''Rigid GMM
    '''
    rigid_gmm = nn.DataParallel(GMM(opt,rigid=True).to(device))
    rigid_gmm.eval()
    rigid_checkpoint = torch.load( opt.root_dir+ 'checkpoints/{0}/'.format(rigid_ckpt_name) + 'model_weights.tar', map_location=cpu)
    rigid_gmm.module.load_state_dict(rigid_checkpoint['gmm_state_dict'], strict=False)
    rigid_epoch = rigid_checkpoint['epoch']
    print('rigid gmm: trained epochs:', rigid_epoch)
    print('rigid gmm: Load Success')

    '''Tps GMM
    '''
    tps_gmm = nn.DataParallel(GMM(opt,rigid=False).to(device))
    tps_gmm.eval()
    checkpoint = torch.load(opt.root_dir+ 'checkpoints/{0}/'.format(tps_ckpt_name) + 'model_weights.tar', map_location=cpu)
    tps_gmm.module.load_state_dict(checkpoint['gmm_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    print('Epoch:', epochCurrent)    

    '''Where to save
    '''
    test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/{2}/'.format(tps_ckpt_name,test_dataset, test_source)
    ref_name = '{:05d}'.format(opt.ref_id)
    test_result_vid_dir = test_result_dir + test_target + '_{0}'.format(ref_name)
    if not os.path.isdir(test_result_vid_dir):
        os.makedirs(test_result_vid_dir)


    ''' globals of for loop
    '''
    avg_l1_loss, gt_sum = 0,0
    img_grid = load_image('256.png').unsqueeze(0).to(device)
    total_gts = len(os.listdir(path_to_test_tgt_imgs))

    ''' for loop
    '''
    for gt_id in tqdm(range(0, total_gts, 5)):
        gt_name = '{:05d}'.format(gt_id)
        gt_sum += 1
        g_x = load_image(os.path.join(path_to_test_tgt_imgs, gt_name+'.png')).unsqueeze(0).to(device)
        g_p = load_parsing(os.path.join(path_to_test_tgt_parse, gt_name+'.png'))
        g_p = (g_p[5] + g_p[6] + g_p[7] + g_p[12]).unsqueeze(0).unsqueeze(0).to(device)

        ref_x = load_image(os.path.join(path_to_test_source_imgs, ref_name+'.png')).unsqueeze(0).to(device)
        ref_p = load_parsing(os.path.join(path_to_test_source_parse, ref_name+'.png'))
        ref_p = (ref_p[5] + ref_p[6] + ref_p[7] + ref_p[12]).unsqueeze(0).unsqueeze(0).to(device)

        ref_c = ref_x * (ref_p > 0.5).float()
        gt_c = g_x * (g_p > 0.5).float()
        '''step 1: get affine warped grid
        '''
        rigid_grid,_ = rigid_gmm(ref_p, g_p)

        #warp the source
        rigid_warped_parsing = F.grid_sample(ref_p, rigid_grid, padding_mode='zeros')
        rigid_warped_grid = F.grid_sample(img_grid, rigid_grid, padding_mode='zeros') 
        rigid_warp_c = F.grid_sample(ref_c, rigid_grid, padding_mode='border') 

        '''step 2: use affine warped parsing and ground truth parsing to predict tps transorm grid
        '''
        grid, theta = tps_gmm(rigid_warped_parsing, g_p)

        '''step 3: use tps transorm grid to warp the affine image, grid, parsing and cloth
        '''
        # warped_img = F.grid_sample(ref_x, grid, padding_mode='border')
        warped_grids = F.grid_sample(img_grid, grid, padding_mode='zeros') # for visualize
        warp_c = F.grid_sample(rigid_warp_c, grid, padding_mode='border') 
        warped_parsing = F.grid_sample(rigid_warped_parsing, grid, padding_mode='zeros')


        l1loss = nn.L1Loss()(warped_parsing, g_p)
        avg_l1_loss += l1loss.item()

        '''save image result'''
        gx = g_x[0].permute(1,2,0) * 255.0
        rx = ref_x[0].permute(1,2,0) * 255.0
        # wx = warped_img[0].permute(1,2,0) * 255.0

        rgrid = img_grid[0].permute(1,2,0) * 255.0
        rigid_wgrid = rigid_warped_grid[0].permute(1,2,0) * 255.0
        tps_wgrid = warped_grids[0].permute(1,2,0) * 255.0 # tps grid


        gc = gt_c[0].permute(1,2,0) * 255.0
        rc = ref_c[0].permute(1,2,0) * 255.0
        rigid_wc = rigid_warp_c[0].permute(1,2,0) * 255.0 # affine warped cloth 
        wc = warp_c[0].permute(1,2,0) * 255.0

        # change bg to gray
        gc[gc<0.5] += 128
        rc[rc<0.5] += 128
        wc[wc<0.5] += 128
        rigid_wc[rigid_wc<0.5] += 128
        
        gp = visualize_cloth_parsing(g_p.cpu().numpy(), 0)
        wp = visualize_cloth_parsing(warped_parsing.detach().cpu().numpy(), 0)
        rigid_wp = visualize_cloth_parsing(rigid_warped_parsing.detach().cpu(), 0)
        rp = visualize_cloth_parsing(ref_p.cpu().numpy(), 0)


        visual_gp = torch.from_numpy(gp).to(device).float()
        visual_wp = torch.from_numpy(wp).to(device).float()
        visual_rigid_wp = torch.from_numpy(rigid_wp).to(device).float()
        visual_rp = torch.from_numpy(rp).to(device).float()

        gp_wp_merge = visualize_merge_cloth_parsing(g_p.cpu(),warped_parsing.detach().cpu(), 0)
        gp_rwp_merge = visualize_merge_cloth_parsing(g_p.cpu(),rigid_warped_parsing.detach().cpu(), 0)
        gp_rp_merge = visualize_merge_cloth_parsing(g_p.cpu(),ref_p.cpu(), 0)

        visual_gp_wp_merge = torch.from_numpy(gp_wp_merge).to(device).float()
        visual_gp_rwp_merge = torch.from_numpy(gp_rwp_merge).to(device).float()
        visual_gp_rp_merge = torch.from_numpy(gp_rp_merge).to(device).float()
        
        white = torch.ones(gc.size()).to(device) * 255.0

        r_col =         torch.cat((rx,          white,              white,                  rgrid),dim=0)
        rc_col =        torch.cat((rc,          visual_rp,          visual_gp_rp_merge,     rgrid), dim=0)
        rigid_wc_col =  torch.cat((rigid_wc,    visual_rigid_wp,    visual_gp_rwp_merge,    rigid_wgrid), dim=0)
        wc_col =        torch.cat((wc,          visual_wp,          visual_gp_wp_merge,     tps_wgrid), dim=0)
        gc_col =        torch.cat((gc,          visual_gp,          white,                  white), dim=0)
        g_col =         torch.cat((gx,          white,              white,                  white), dim=0)


        # g_col = torch.cat((gx, ), dim=0)
        # w_col = torch.cat((wx, ), dim=0)
        # r_col = torch.cat((rx, ), dim=0)

        final = torch.cat((r_col, rc_col,rigid_wc_col, wc_col, gc_col, g_col), dim=1).type(torch.uint8).to(cpu).numpy()

        plt.imsave(os.path.join(test_result_vid_dir,"{0}_result.png".format(gt_name)), final)

    
    avg_l1_loss /= gt_sum
    print(avg_l1_loss)

    '''save video result'''
    img_dir = test_result_vid_dir
    save_video_dir = test_result_dir
    save_video_name = 'target-{0}_ref-{1}_result.mp4'.format(test_target, ref_name)
    # save_video_name_simp = save_video_name +'_result_simp.mp4'

    metric_loss_save_path = save_video_dir + save_video_name.replace('.mp4', '_loss.txt')
    with open(metric_loss_save_path, 'w') as f:
        f.write('%03f \n'%avg_l1_loss)

    imgs = os.listdir(img_dir)
    import cv2
    # video_out_simp = cv2.VideoWriter(save_video_dir+save_video_name_simp, cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (256*(K+2), 256))
    video_out = cv2.VideoWriter(save_video_dir+save_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (256*6, 256*4))



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
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    
    if not opt.test:
        print('---------------Train---------------')
        for k,v in sorted(vars(opt).items()):
            print(k,':',v)
        set_random_seed(opt.seed)
        today = datetime.today().strftime("%Y%m%d")
        experiment_name = 'Geo_v{0}_lr{1}-{2}'.format(opt.id, opt.lr, '20201027')
        print(experiment_name)
        train_gmm(opt, experiment_name)
    else:
        with torch.no_grad():
            parser.add_argument('--rigid_ckpt_name',type=str)
            parser.add_argument('--tps_ckpt_name',type=str)
            opt, unknown = parser.parse_known_args()
            print('---------------Test---------------')
            for k,v in sorted(vars(opt).items()):
                print(k,':',v)
            test(opt,rigid_ckpt_name=opt.rigid_ckpt_name,tps_ckpt_name=opt.tps_ckpt_name)




