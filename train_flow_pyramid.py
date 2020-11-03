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
# from options import base_options


# if __name__ == "__main__":
#     base_option = base_options.BaseOptions()
#     opt = base_option.parse()

#     # base_option.print_options(opt)
    
#     pass

"""Training Parameters"""
# cuda visible devices, related to batchsize, defalut the batchsize should be 2 times cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# default for cuda:0 to use gpu
device = torch.device("cuda:0") 
cpu = torch.device("cpu")

experiment_name = 'FuseG_v3_wosim_feature_pyramid_lr2e_4_biup-0924'
visualize_result_dir = '/home/ljw/playground/poseFuseNet/visualize_result/{0}/'.format(experiment_name)
path_to_ckpt_dir = '/home/ljw/playground/poseFuseNet/checkpoints/{0}/'.format(experiment_name)
path_to_log_dir = '/home/ljw/playground/poseFuseNet/logs/{0}'.format(experiment_name)

path_to_chkpt = path_to_ckpt_dir + 'model_weights.tar' 
path_to_backup = path_to_ckpt_dir + 'backup_model_weights.tar'
batch_size = 2
K = 2
lambda_style = 500.0
lambda_content = 0.5
lambda_rec = 5.0
pose_normalize = True
scale_pose = 255 if pose_normalize else 1


path_to_train_A = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/danceFashion/train_256/train_A/'

is_clean_pose = False
path_to_train_kps = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/danceFashion/train_256/train_alphapose/'
if is_clean_pose:
    path_to_train_kps = '/home/ljw/playground/Global-Flow-Local-Attention/dataset/danceFashion/train_256/train_video2d/'



if not os.path.isdir(path_to_ckpt_dir):
    os.makedirs(path_to_ckpt_dir)
if not os.path.isdir(visualize_result_dir):
    os.makedirs(visualize_result_dir)
if not os.path.isdir(path_to_log_dir):
    os.makedirs(path_to_log_dir)

"""Create dataset and net"""

dataset = FashionVideoDataset(path_to_train_A=path_to_train_A, path_to_train_kps=path_to_train_kps, K=K, is_clean_pose=is_clean_pose, pose_scale=scale_pose)
print(dataset.__len__())

fashionVideoDataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

GF = nn.DataParallel(FlowPyramid(source_nc=23,target_nc=20, mode='bilinear').to(device))
# GA = nn.DataParallel(AttentionGenerator(inc=40).to(device))

GF.train()
# GA.train()

optimizerG = optim.Adam(params = list(GF.parameters()),
                        lr=2e-4)

criterionG = LossG(device=device)
# criterionGF = PerceptualCorrectness()

matplotlib.use('agg') 


"""Training init"""
epochCurrent = epoch = i_batch = 0
lossesG = []
i_batch_current = 0
i_batch_total = 0
num_epochs = 15000

"""initiate checkpoint if inexistant"""
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    GF.apply(init_weights)
    # E.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': lossesG,
            # 'E_state_dict': E.module.state_dict(),
            'G_state_dict': GF.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
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
for epoch in range(epochCurrent, num_epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(fashionVideoDataLoader, leave=True, initial=0)
    
    pbar.set_postfix(epoch=epoch)
    for i_batch, (ref_xs, ref_ys, g_x, g_y, idx) in enumerate(pbar, start=0):


        ref_x = ref_xs.squeeze().to(device) # [B, 3, 256, 256]
        ref_y = ref_ys.squeeze().to(device) # [B, 20, 256, 256]
        g_x = g_x.to(device) # [B, 3, 256, 256]
        g_y = g_y.to(device) # [B, 20, 256, 256]

        optimizerG.zero_grad()

        # print(g_y.shape, e_hat.shape)
        flows, x_hat = GF(ref_x, ref_y, g_y)

        lossG_content, lossG_style, lossG_L1 = criterionG(g_x, x_hat)
        lossG_content = lossG_content * lambda_content
        lossG_style = lossG_style * lambda_style
        lossG_L1 = lossG_L1 * lambda_rec
        lossG = lossG_content + lossG_style + lossG_L1

        lossG.backward(retain_graph=False)
        optimizerG.step()

        writer.add_scalar('loss/lossG', lossG.item(), global_step=i_batch_total, walltime=None)
        # writer.add_scalar('loss/lossG_vggface', loss_face.item(), global_step=i_batch_total, walltime=None)
        writer.add_scalar('loss/lossG_content', lossG_content.item(), global_step=i_batch_total, walltime=None)
        writer.add_scalar('loss/lossG_style', lossG_style.item(), global_step=i_batch_total, walltime=None)
        writer.add_scalar('loss/lossG_L1', lossG_L1.item(), global_step=i_batch_total, walltime=None)
        i_batch_total +=1
        pbar.set_postfix(epoch=epoch, G_loss=lossG.item())

    lossesG.append(lossG.item())
    torch.save({
            'epoch': epoch+1,
            'lossesG': lossesG,
            'G_state_dict': GF.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            }, path_to_chkpt)
    
    # def get_group_ref_imgs(ref_xs, batch_dim):
    #     ref0 = (ref_xs[batch_dim][0]*255).transpose(0,2)
    #     ref1 = (ref_xs[batch_dim][1]*255).transpose(0,2)
    #     ref2 = (ref_xs[batch_dim][2]*255).transpose(0,2)
    #     ref3 = (ref_xs[batch_dim][3]*255).transpose(0,2)

    #     refup = torch.cat((ref0, ref1), dim=0)
    #     refdown = torch.cat((ref2, ref3), dim=0)

    #     ref_group = torch.cat((refup, refdown), dim=1)
    #     ref_group_numpy = ref_group.to(cpu).numpy()
    #     ref_group_numpy = resize(ref_group_numpy, (256,256))
    #     ref_group = torch.from_numpy(ref_group_numpy).to(device)
    #     return ref_group

    # ref_group = get_group_ref_imgs(ref_xs, batch_dim=0)
    ref_group = (ref_x[0]*255).permute(1,2,0)
    ref_pose = (ref_y[0][17:20,...]*scale_pose).permute(1,2,0)
    flowimg = flow2img(flows[0][0].detach().permute(1,2,0).to(cpu).numpy())
    flowimg = torch.from_numpy(flowimg).to(device).float()
    # maskimg = mask[0].detach().permute(1,2,0) * 255.0
    # maskimg = torch.cat((maskimg,)*3, dim=2)
    
    # for img_no in range(1,batch_size):
        # ref_group = torch.cat((ref_group, get_group_ref_imgs(ref_xs, img_no)), dim = 0)

    out = (x_hat[0]*255).permute(1,2,0)
    # for img_no in range(1,batch_size):
    #     out = torch.cat((out, (x_hat[img_no]*255).transpose(0,2)), dim = 0)
    white = flowimg.new_tensor(torch.ones(flowimg.size()) * 255)
    pose = (g_y[0][17:20,...]*scale_pose).permute(1,2,0)
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
    plt.text(20,20, 'assfd')
    # out = np.transpose(out, (1, 0, 2))
    plt.imsave(visualize_result_dir+"epoch_{}_batch_{}.png".format(epoch, i_batch), out)

writer.close()


