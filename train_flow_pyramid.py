import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt

# from model.DirectE import DirectEmbedder
# from model.DirectG import DirectGenerator
from model.flow_pyramid import FlowPyramid
from util.flow_utils import flow2img

from dataset.fashionvideo_dataset import FashionVideoDataset
from loss.loss_generator import PerceptualCorrectness, LossG

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
from skimage.transform import resize
import argparse
# from options import base_options

device = torch.device("cuda:0") 
cpu = torch.device("cpu")

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

# if __name__ == "__main__":
#     base_option = base_options.BaseOptions()
#     opt = base_option.parse()

#     # base_option.print_options(opt)
    
#     pass

"""Training Parameters"""
# cuda visible devices, related to batchsize, defalut the batchsize should be 2 times cuda devices
# default for cuda:0 to use gpu
device = torch.device("cuda:0") 
cpu = torch.device("cpu")


def make_ckpt_log_vis_dirs(opt, exp_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    path_to_ckpt_dir = opt.root_dir+ 'checkpoints/{0}/'.format(exp_name)
    path_to_visualize_dir = opt.root_dir+ 'visualize_result/{0}/'.format(exp_name)
    path_to_log_dir = opt.root_dir+ 'logs/{0}'.format(exp_name)
    
    path_to_visualize_dir_train = os.path.join(path_to_visualize_dir, 'train')
    path_to_visualize_dir_val = os.path.join(path_to_visualize_dir, 'val')

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

def train(opt, experiment_name):

    

    path_to_ckpt_dir, path_to_log_dir, path_to_visualize_dir = make_ckpt_log_vis_dirs(opt, experiment_name)
    path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 

    """Create dataset and net"""
    dataset = make_dataset(opt)
    fashionVideoDataLoader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

    GF = nn.DataParallel(FlowPyramid(source_nc=23,target_nc=20, mode='bilinear', align_corners=opt.align_corner).to(device))
    # GA = nn.DataParallel(AttentionGenerator(inc=40).to(device))
    GF.train()
    # GA.train()
    optimizerG = optim.Adam(params = list(GF.parameters()),
                            lr=opt.lr)
    criterionG = LossG(device=device)
    # criterionGF = PerceptualCorrectness()

    matplotlib.use('agg') 


    """Training init"""
    lossesG = []

    """initiate checkpoint if inexistant"""
    if not os.path.isfile(path_to_chkpt):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
        GF.apply(init_weights)

        print('Initiating new checkpoint...')
        torch.save({
                'epoch': 0,
                'lossesG': lossesG,
                'G_state_dict': GF.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': 0,
                'optimizerG': optimizerG.state_dict(),
                }, path_to_chkpt)
        print('...Done')


    """Loading from past checkpoint"""
    checkpoint = torch.load(path_to_chkpt, map_location=cpu)
    GF.module.load_state_dict(checkpoint['G_state_dict'], strict=False)
    epochCurrent = checkpoint['epoch']
    lossesG = checkpoint['lossesG']
    num_vid = checkpoint['num_vid']
    i_batch_current = checkpoint['i_batch'] + 1
    optimizerG.load_state_dict(checkpoint['optimizerG'])

    i_batch_total = epochCurrent * fashionVideoDataLoader.__len__() + i_batch_current

    """
    create tensorboard writter
    """
    TIMESTAMP = "/{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(path_to_log_dir+TIMESTAMP)

    pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)

    """ Training start """
    for epoch in range(epochCurrent, opt.epochs):
        if epoch > epochCurrent:
            i_batch_current = 0
            pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
        
        pbar.set_postfix(epoch=epoch)
        epoch_loss_G = 0
        for i_batch, (ref_xs, ref_ys,ref_ps, g_x, g_y,g_p,sims,sim_maps, vid_path) in enumerate(pbar, start=0):
            ref_x = ref_xs[0].to(device) # [B, 3, 256, 256]
            ref_y = ref_ys[0].to(device) # [B, 20, 256, 256]
            g_x = g_x.to(device) # [B, 3, 256, 256]
            g_y = g_y.to(device) # [B, 20, 256, 256]

            optimizerG.zero_grad()

            # print(g_y.shape, e_hat.shape)
            flows, x_hat = GF(ref_x, ref_y, g_y)

            lossG_content, lossG_style, lossG_L1 = criterionG(g_x, x_hat)
            lossG_content = lossG_content * opt.lambda_content
            lossG_style = lossG_style * opt.lambda_style
            lossG_L1 = lossG_L1 * opt.lambda_rec
            lossG = lossG_content + lossG_style + lossG_L1

            lossG.backward(retain_graph=False)
            optimizerG.step()

            epoch_loss_G += lossG.item()
            epoch_loss_G_moving = epoch_loss_G / (i_batch+1)
            writer.add_scalar('loss/lossG', lossG.item(), global_step=i_batch_total, walltime=None)
            # writer.add_scalar('loss/lossG_vggface', loss_face.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('loss/lossG_content', lossG_content.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('loss/lossG_style', lossG_style.item(), global_step=i_batch_total, walltime=None)
            writer.add_scalar('loss/lossG_L1', lossG_L1.item(), global_step=i_batch_total, walltime=None)
            i_batch_total +=1
            post_fix_str = 'Epoch_loss=%.3f, G=%.3f,L1=%.3f,L_content=%.3f,L_sytle=%.3f'%(epoch_loss_G_moving, lossG.item(), lossG_L1.item(), lossG_content.item(), lossG_style.item())

            pbar.set_postfix_str(post_fix_str)

        lossesG.append(lossG.item())
        torch.save({
                'epoch': epoch+1,
                'lossesG': lossesG,
                'G_state_dict': GF.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': i_batch,
                'optimizerG': optimizerG.state_dict(),
                }, path_to_chkpt)
        

        # ref_group = get_group_ref_imgs(ref_xs, batch_dim=0)
        ref_group = (ref_x[0]*255).permute(1,2,0)
        ref_pose = (ref_y[0][17:20,...]*255.0).permute(1,2,0)
        flowimg = flow2img(flows[0][0].detach().permute(1,2,0).to(cpu).numpy())
        flowimg = torch.from_numpy(flowimg).to(device).float()
        # maskimg = torch.cat((maskimg,)*3, dim=2)
        
        # for img_no in range(1,batch_size):
            # ref_group = torch.cat((ref_group, get_group_ref_imgs(ref_xs, img_no)), dim = 0)

        out = (x_hat[0]*255).permute(1,2,0)
        # for img_no in range(1,batch_size):
        #     out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 0)
        white = flowimg.new_tensor(torch.ones(flowimg.size()) * 255)
        pose = (g_y[0][17:20,...]*255.0).permute(1,2,0)
        # for img_no in range(1,batch_size):
        #     pose = torch.cat((pose, (g_y[img_no][17:20,...]*scale_pose).transpose(0,2)), dim = 0)

        gtruth = (g_x[0]*255).permute(1,2,0)
        # for img_no in range(1,batch_size):
        #     gtruth = torch.cat((gtruth, (g_x[img_no]*255).transpose(0,2)), dim = 0)

        ref = torch.cat((ref_group, ref_pose), dim =0)
        gt = torch.cat((gtruth, pose), dim=0)
        mid = torch.cat((out, flowimg), dim=0)

        out = torch.cat((ref, gt, mid), dim=1)

        out = out.type(torch.uint8).to(cpu).numpy()
        # out = out.transpose()
        # out = np.transpose(out, (1, 0, 2))
        plt.imsave(path_to_visualize_dir+"epoch_{}_batch_{}.png".format(epoch, i_batch), out)

    writer.close()




if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    opt.K = 1
    experiment_name = 'FuseG_v3_wosim_feature_pyramid_lr2e_4_biup-1119'
    
    train(opt,experiment_name)