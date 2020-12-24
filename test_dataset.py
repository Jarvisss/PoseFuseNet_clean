from dataset.fashion_dataset import FashionDataset
import argparse
import os
from util.vis_util import *
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import numpy as np
import matplotlib.pyplot as plt

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
    parser.add_argument('--anno_size', type=int, nargs=2,default=[256, 176], help='input annotation size')
    parser.add_argument('--model_save_freq', type=int, default=0, help='save model every epoch')
    parser.add_argument('--img_save_freq', type=int, default=200, help='save image every N iters')

    '''Dataset options'''
    parser.add_argument('--use_clean_pose', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--use_parsing', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--categories', type=int, default=9)
    parser.add_argument('--use_simmap', action='store_true', help='use clean pose, only for fashionVideo and iPER dataset')
    parser.add_argument('--joints_for_cos_sim', type=int, default=-1)

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

if __name__ == '__main__':
    opt = get_parser()
    opt.K = 2
    path_to_dataset = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'
    train_tuples_name = 'fasion-pairs-train.csv' if opt.K==1 else 'fasion-%d_tuples-train.csv'%(opt.K+1)
    test_tuples_name = 'fasion-pairs-test.csv' if opt.K==1 else 'fasion-%d_tuples-test.csv'%(opt.K+1)
    if opt.use_parsing:
        path_to_train_parsing = os.path.join(path_to_dataset, 'train_parsing_merge/')
        path_to_test_parsing = os.path.join(path_to_dataset, 'test_parsing_merge/')
    else:
        path_to_train_parsing = None
        path_to_test_parsing = None
    dataset = FashionDataset(
            phase = 'train',
            path_to_train_tuples=os.path.join(path_to_dataset, train_tuples_name), 
            path_to_test_tuples=os.path.join(path_to_dataset, test_tuples_name), 
            path_to_train_imgs_dir=os.path.join(path_to_dataset, 'train_256/'), 
            path_to_test_imgs_dir=os.path.join(path_to_dataset, 'test_256/'),
            path_to_train_anno=os.path.join(path_to_dataset, 'fasion-annotation-train.csv'), 
            path_to_test_anno=os.path.join(path_to_dataset, 'fasion-annotation-test.csv'), 
            path_to_train_parsings_dir=path_to_train_parsing, 
            path_to_test_parsings_dir=path_to_test_parsing, 
            opt=opt)

    save_dir = './test_sim_result/all_bones_normed/'
    # save_dir = './test_sim_result/4_bones_sho_hip_normed/'
    # save_dir = './test_sim_result/4_bones_sho_hip/'
    # save_dir = './test_sim_result/2_bones_sho/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, batch_data in enumerate(dataset):
        print(i)
        out_size = (256,256)
        ref_xs = batch_data['ref_xs']
        ref_ys = batch_data['ref_ys']
        sims = batch_data['cos_sim']
        g_x = batch_data['g_x']
        g_y = batch_data['g_y']
        froms = batch_data['froms']
        to = batch_data['to']
        
        visual_g_x = tensor2im(g_x.unsqueeze(0), out_size=out_size).type(torch.uint8).numpy()
        visual_g_y = tensor2im(g_y.unsqueeze(0), out_size=out_size).type(torch.uint8).numpy()
        visual_refy1 = tensor2im(ref_ys[0].unsqueeze(0), out_size=out_size).type(torch.uint8).numpy()
        visual_refy2 = tensor2im(ref_ys[1].unsqueeze(0), out_size=out_size).type(torch.uint8).numpy()
        visual_refx1 = tensor2im(ref_xs[0].unsqueeze(0), out_size=out_size).type(torch.uint8).numpy()
        visual_refx2 = tensor2im(ref_xs[1].unsqueeze(0), out_size=out_size).type(torch.uint8).numpy()
        
        texted_refy1 = Image.fromarray(visual_refy1.astype(np.uint8))
        texted_refy2 = Image.fromarray(visual_refy2.astype(np.uint8))
        
        # texted_refy1 = cv2.putText(img=np.array(visual_refy1), text=sims[0].numpy(), org=(200,200),fontFace=3, fontScale=3, color=(255,255,255), thickness=5)
        # texted_refy2 = cv2.putText(img=np.array(visual_refy2), text=sims[1].numpy(), org=(200,200),fontFace=3, fontScale=3, color=(255,255,255), thickness=5)
        draw1 = ImageDraw.Draw(texted_refy1)
        draw2 = ImageDraw.Draw(texted_refy2)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw1.text((0, 0),str(sims[0].numpy()),(255,255,255))
        draw2.text((0, 0),str(sims[1].numpy()),(255,255,255))
        
        imgs = np.concatenate((visual_refx1, visual_refx2, visual_g_x),axis=1)
        ys = np.concatenate((np.asarray(texted_refy1), np.asarray(texted_refy2), visual_g_y), axis=1)
        final = np.concatenate((imgs,ys),axis=0)
        final = Image.fromarray(final)
        final.save(save_dir+'{}.jpg'.format(i))