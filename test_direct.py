import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt

from model.DirectE import DirectEmbedder
from model.DirectG import DirectGenerator
from dataset.fashionvideo_dataset import FashionVideoDataset,FashionVideoTestDataset, FashionFrameTestDataset
from loss.loss_generator import LossG

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
import os
from skimage.transform import resize


"""
Test parameters
"""
# test_datset = 'iPER'
test_datset = 'danceFashion'

# cuda visible devices, related to batchsize, defalut the batchsize should be 2 times cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# default for cuda:0 to use gpu
device = torch.device("cuda:0") 
cpu = torch.device("cpu") 

experiment_name = 'DirectG_pose_normalize_255-0903'
test_result_dir = '/home/ljw/playground/poseFuseNet/test_result/{0}/{1}/'.format(experiment_name,test_datset)
path_to_ckpt_dir = '/home/ljw/playground/poseFuseNet/checkpoints/{0}/'.format(experiment_name)

if not os.path.isdir(test_result_dir):
    os.makedirs(test_result_dir)

path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
path_to_backup = path_to_ckpt_dir + 'backup_model_weights.tar'
batch_size = 1
K = 5
pose_normalize = True
scale_pose = 255 if pose_normalize else 1


path_to_test_A = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/{0}/test_256/train_A/'.format(test_datset)
# path_to_test_A = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/iPER/test_256/train_A/'

is_clean_pose = False
path_to_test_kps = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/{0}/test_256/train_alphapose/'.format(test_datset)
if is_clean_pose:
    path_to_test_kps = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/{0}/test_256/train_video2d/'.format(test_datset)


dataset = FashionVideoTestDataset(path_to_test_A=path_to_test_A, path_to_test_kps=path_to_test_kps, K=K, is_clean_pose=is_clean_pose, pose_scale=scale_pose)


fashionVideoDataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

E = nn.DataParallel(DirectEmbedder(inc=23).to(device))
G = nn.DataParallel(DirectGenerator(inc=20).to(device))

E.eval()
G.eval()

"""Loading from past checkpoint"""
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.module.load_state_dict(checkpoint['E_state_dict'])
G.module.load_state_dict(checkpoint['G_state_dict'], strict=False)

def dir_imgs_to_video(imgs_dir, result_dir):
    os.system("ffmpeg -r 30 -i {0}/frame_%05d.png -vcodec mpeg4 -y {1}/{2}.mp4".format(imgs_dir,result_dir, imgs_dir.split('/')[-1]))

    
    pass

def get_group_ref_imgs(ref_xs, batch_dim):
        ref0 = (ref_xs[batch_dim][0]*255).transpose(0,2)
        ref1 = (ref_xs[batch_dim][1]*255).transpose(0,2)
        ref2 = (ref_xs[batch_dim][2]*255).transpose(0,2)
        ref3 = (ref_xs[batch_dim][3]*255).transpose(0,2)

        refup = torch.cat((ref0, ref1), dim=0)
        refdown = torch.cat((ref2, ref3), dim=0)

        ref_group = torch.cat((refup, refdown), dim=1)
        ref_group_numpy = ref_group.to(cpu).numpy()
        ref_group_numpy = resize(ref_group_numpy, (256,256))
        ref_group = torch.from_numpy(ref_group_numpy).to(device)
        return ref_group

pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
with torch.no_grad():

    for _, (ref_xs, ref_ys, idx) in enumerate(pbar, start=0):
        
        ref_xs = ref_xs.to(device) # [1, 4, 3, 256, 256]
        ref_ys = ref_ys.to(device) # [1, 4, 20, 256, 256]
        
        
        
        # [B*4, 3, 256, 256]
        ref_xs_compact = ref_xs.view(-1, ref_xs.shape[-3], ref_xs.shape[-2], ref_xs.shape[-1])
        # [B*4, 20, 256, 256]
        ref_ys_compact = ref_ys.view(-1, ref_ys.shape[-3], ref_ys.shape[-2], ref_ys.shape[-1])
        
        e_vectors = E(ref_xs_compact, ref_ys_compact) #[BK * 512 * 1]
        e_vectors = e_vectors.view(-1, ref_xs.shape[1], 512, 1) # [BK, 512, 1]

        e_hat = e_vectors.mean(dim=1) # B * 512 * 1

        ref_group = get_group_ref_imgs(ref_xs, batch_dim=0)
        
        for img_no in range(1,batch_size):
            ref_group = torch.cat((ref_group, get_group_ref_imgs(ref_xs, img_no)), dim = 0)

        vid_paths = sorted(os.listdir(path_to_test_A))
        video_name = vid_paths[idx]
        path_to_test_imgs = os.path.join(path_to_test_A, video_name)
        path_to_test_img_kps = os.path.join(path_to_test_kps, video_name)
        frameset = FashionFrameTestDataset(path_to_test_imgs=path_to_test_imgs, path_to_test_kps=path_to_test_img_kps, K=K, is_clean_pose=is_clean_pose, pose_scale=scale_pose)
        fashionFrameDataLoader = DataLoader(frameset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        for _, (g_x, g_y, idx) in enumerate(fashionFrameDataLoader):
            g_x = g_x.to(device) # [1, N, 3, 256, 256]
            g_y = g_y.to(device) # [1, N, 20, 256, 256]

            x_hat = G(g_y, e_hat)
            
            out = (x_hat[0]*255).transpose(0,2)
            for img_no in range(1,batch_size):
                out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 0)

            pose = (g_y[0][17:20,...]*scale_pose).transpose(0,2)
            for img_no in range(1,batch_size):
                pose = torch.cat((pose, (g_y[img_no][17:20,...]*scale_pose).transpose(0,2)), dim = 0)

            gtruth = (g_x[0]*255).transpose(0,2)
            for img_no in range(1,batch_size):
                gtruth = torch.cat((gtruth, (g_x[img_no]*255).transpose(0,2)), dim = 0)
        
            out = torch.cat((ref_group, out, gtruth, pose), dim=1)

            out = out.type(torch.uint8).to(cpu).numpy()
            out = np.transpose(out, (1, 0, 2))
            if not os.path.isdir(os.path.join(test_result_dir,video_name)):
                os.makedirs(os.path.join(test_result_dir,video_name))
            plt.imsave(os.path.join(test_result_dir,video_name)+"/frame_%05d.png"% idx, out)

        dir_imgs_to_video(os.path.join(test_result_dir,video_name), test_result_dir)
        pass








