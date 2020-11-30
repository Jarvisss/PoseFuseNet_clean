import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import cv2
import torchvision.transforms.functional as F
from util import openpose_utils
from PIL import Image, ImageDraw
import random
import json
import sys

class FashionVideoDataset(data.Dataset):
    """
    @ path_to_train_A
        the path to train image dataset
    @ path_to_train_kps
        the path to train keypoints
    @ path_to_train_parsing
        the path to parsings
    @ K
        K ref poses and ref images
    @ is_clean_pose
        If True: kps in openpose 17 format, more stable
        If False: kps in alphapose 18 format, more jitter
    """
    def __init__(self, path_to_train_A, path_to_train_kps, path_to_train_parsing, opt, load_size=256, pose_scale=255.0):
        super(FashionVideoDataset, self).__init__()
        self.path_to_train_A = path_to_train_A
        self.path_to_train_kps = path_to_train_kps
        self.path_to_train_parsing = path_to_train_parsing
        self.K = opt.K
        self.step = opt.step
        self.use_dot = opt.use_dot
        self.is_clean_pose = opt.use_clean_pose
        self.align_input = opt.align_input

        self.name = opt.dataset
        self.load_size = (load_size, load_size)
        self.no_bone_RGB = False
        self.angle = (-10, 10)
        self.shift = (30, 3)
        self.scale = (0.8, 1.2)        
        # self.angle = None
        # self.shift = None
        # self.scale = None
        self.img_size = (0,0)
        self.pose_scale = pose_scale
        self.align_corner = opt.align_corner


    def getRandomAffineParam(self):
        if not self.angle and not self.scale and not self.shift:
            affine_param = None
            return affine_param
        else:
            affine_param=dict()
            affine_param['angle'] = np.random.uniform(low=self.angle[0], high=self.angle[1]) if self.angle is not False else 0
            affine_param['scale'] = np.random.uniform(low=self.scale[0], high=self.scale[1]) if self.scale is not False else 1
            shift_x = np.random.uniform(low=-self.shift[0], high=self.shift[0]) if self.shift is not False else 0
            shift_y = np.random.uniform(low=-self.shift[1], high=self.shift[1]) if self.shift is not False else 0
            affine_param['shift']=(shift_x, shift_y)

            return affine_param


    def transform_image(self, image, resize_param, method=Image.BICUBIC, affine=None, normalize=False, toTensor=True, fillWhiteColor=False):
        image = F.resize(image, resize_param, interpolation=method)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            fillcolor = None if not fillWhiteColor else (255,255,255)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
        if toTensor:
            image = F.to_tensor(image)
        if normalize:
            image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        return image  

    def load_image(self, A_path):
        A_img = Image.open(A_path)
        self.img_size = A_img.size
        # padding white color after affine transformation  
        fillWhiteColor = True if self.name =='danceFashion' else False
        Ai = self.transform_image(A_img, self.load_size, affine=self.affine_param, fillWhiteColor=fillWhiteColor)
        return Ai


    def load_skeleton(self, B_path):
        B_coor = json.load(open(B_path))["people"]

        B_coor = B_coor[0]
        pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.img_size, affine=self.affine_param)
        alpha_pose_body = pose_dict['body']
        if not self.is_clean_pose:
            pose_body = openpose_utils.openpose18_to_coco17(alpha_pose_body) #np(2,17)

        gaussian_bump = openpose_utils.obtain_map(pose_body, self.load_size) 
        pose_gaussian = np.transpose(gaussian_bump,(2, 0, 1))
        pose_gaussian = torch.Tensor(pose_gaussian)
        Bi = pose_gaussian
        if not self.no_bone_RGB:
            color = np.zeros(shape=self.load_size + (3, ), dtype=np.uint8)
            LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if self.is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
            color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
            color = np.transpose(color,(2,0,1))
            color = color / self.pose_scale # normalize to 0-1
            color = torch.Tensor(color)
            Bi = torch.cat((Bi, color), dim=0)
        
        return Bi,torch.Tensor(alpha_pose_body)
    
    def load_parsing(self, parsing_path):
        parsing_img = Image.open(parsing_path)
        self.img_size = parsing_img.size
        parsing_img = np.array(self.transform_image(parsing_img, self.load_size, affine=self.affine_param, fillWhiteColor=False, toTensor=False)) # 0-19 one channel map
        # [H W 1 ] -> [H W 20]
        parsing_bin_map = np.stack((parsing_img,)*20, axis=2)
        
        # to 20 channel binary map
        for i in range(20):
            parsing_bin_map[:,:, i] = (parsing_img == i).astype(np.uint8) * 255 
        
        # [H W 20] -> [20 H W]
        parsing_bin_map = F.to_tensor(parsing_bin_map) 

        return parsing_bin_map


    def select_frames_kp(self, video_frame_path, video_kp_path,video_parsing_path, K):
        '''
        select K frames and corresponding keypoints from video
        '''
        video_frame_names = sorted(os.listdir(video_frame_path))
        video_kp_names = sorted(os.listdir(video_kp_path))
        video_parsing_names = sorted(os.listdir(video_parsing_path))
        video_parsing_names_ = []
        for v in video_parsing_names:
            if v.split('.')[-1] == 'png' and v.split('.')[0].split('_')[-1]!='vis':
                video_parsing_names_ += [v]

        assert len(video_frame_names) == len(video_kp_names)
        assert len(video_frame_names) == len(video_parsing_names_)
        n_frames = len(video_frame_names) // self.step
        if n_frames <= K: #There are not enough frames in the video
            rand_frames_idx = [1]*n_frames
        else:
            rand_frames_idx = [0]*n_frames
            i = 0

            """choose K frames by index """
            while(i < K):
                idx = random.randint(0, n_frames-1)
                if rand_frames_idx[idx ] == 0:
                    rand_frames_idx[idx ] = 1
                    i += 1

        # rand_frames_idx = [0]*n_frames
        # rand_frames_idx[0] = 1
        # rand_frames_idx[100] = 1
        # rand_frames_idx[200] = 1
        selected_imgs = []
        selected_kps = []
        selected_joints = []
        selected_parsings = []

        # get K sample from the dataset
        for i, is_chosen in enumerate(rand_frames_idx):
            if is_chosen:
                i = i * self.step
                frame_img = self.load_image(os.path.join(video_frame_path,video_frame_names[i]))
                frame_kps, joints = self.load_skeleton(os.path.join(video_kp_path, video_kp_names[i]))
                frame_parsings = self.load_parsing(os.path.join(video_parsing_path, video_parsing_names_[i])) # [20, H, W]

                selected_imgs = selected_imgs + [frame_img]
                selected_kps = selected_kps + [frame_kps]
                selected_joints = selected_joints + [joints]
                selected_parsings = selected_parsings + [frame_parsings]
        return rand_frames_idx, selected_imgs, selected_kps,selected_joints, selected_parsings

    '''Calculate shoulder direction vector cosine similarity
    @ g_j: [2, 17] ndarray,  y,x order
    @ ref_js: K * [2, 17] ndarray, y,x order
    '''
    def pose_similarity(self, g_j, ref_js):
        K = len(ref_js)
        similarities = []
        LShoulder_idx = openpose_utils.OPENPOSE_18['LShoulder']
        RShoulder_idx = openpose_utils.OPENPOSE_18['RShoulder']
        
        g_shoud_dir = g_j[:, RShoulder_idx] - g_j[:, LShoulder_idx]
        dot_max = 1e-8
        for ref_j in ref_js:
            ref_shoud_dir = ref_j[:, RShoulder_idx] - ref_j[:, LShoulder_idx]        
            if not self.use_dot:
                cosine_sim = torch.cosine_similarity(g_shoud_dir, ref_shoud_dir, dim=0)
                similarities += [cosine_sim]
            else:
                dot = torch.dot(g_shoud_dir, ref_shoud_dir)
                if torch.abs(dot) > dot_max:
                    dot_max = torch.abs(dot)
                similarities += [dot]
        if self.use_dot:
            similarities = [i/dot_max for i in similarities]
        return similarities

    '''
    Input: @ref [2, 18] coords y,x order, @length [13] scale value,
    Output: [13,256,256] tensor , each channel is one bone
    '''
    def to_map(self, ref, length, length_max, line_width=5):
        LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], 
            [8,9], [9,10], [1,11], [11,12], [12,13], [1,0]]
        OPENPOSE_18 = { "Nose":0,"Neck":1,"RShoulder":2,"RElbow":3,"RWrist":4,"LShoulder":5,"LElbow":6,
                "LWrist":7,"RHip":8,"RKnee":9,"RAnkle":10,"LHip":11,"LKnee":12,"LAnkle":13,
                "REye":14,"LEye":15,"REar":16,"LEar":17 }  
        
        images = []
        for i, limb in enumerate(LIMB_SEQ):
            intensity =(255 * length[i]/ length_max).int().item()
            image = Image.new('RGB', (2**8, 2**8), 'black')
            draw = ImageDraw.Draw(image, 'RGB')
            draw.line([tuple(ref[:,limb[1]].numpy()[::-1].tolist()), tuple(ref[:,limb[0]].numpy()[::-1].tolist())], fill=(intensity,intensity,intensity), width=line_width)
            image_np = np.array(image)[:,:,0:1] # 256,256,1
            images += [image_np]
        
        full_image = np.concatenate(images, axis=2) # 256,256,13
        full_image = F.to_tensor(full_image) # 13,256,256
        # print(full_image[0][0][0].item())
        # image_np = np.array(image)
        return full_image
    '''
    get pose similarity map by calculate [dx, dy] for each bone segment, except head's
    @ g_j: [2, 18] tensor,  y,x order
    @ ref_js: K * [2, 18] tensor, y,x order
    '''
    def get_pose_similarity_maps(self, gt, refs):
        LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], 
            [8,9], [9,10], [1,11], [11,12], [12,13], [1,0]]  # 13 limbs , 计算向量，平移不变
        
        g_dirs = []
        for limb in LIMB_SEQ:
            g_dir = gt[:,limb[1]] - gt[:, limb[0]]
            g_dirs += [g_dir]
        
        similarity_maps = [] # K * [256,256,3]
        k_lengths = [] # K * [13]
        length_max = 0 # max length in K samples
        for ref in refs:
            lengths = torch.zeros(len(LIMB_SEQ))
            for i,limb in enumerate(LIMB_SEQ):
                j_dir = ref[:,limb[1]] - ref[:, limb[0]]
                lengths[i] = torch.sqrt(torch.sum((g_dirs[i] - j_dir)**2))
                if length_max < lengths[i]:
                    length_max = lengths[i]
            k_lengths += [lengths]
        
        for i,ref in enumerate(refs):
            similarity_maps += [self.to_map(refs[i], k_lengths[i], length_max)]
        
        return similarity_maps
    

    def __len__(self):
        vid_num = 0
        for video in os.listdir(self.path_to_train_A):
            vid_num+=1
        return vid_num

    def __getitem__(self, idx): 
        # get random affine augmentation
        self.affine_param = self.getRandomAffineParam()
        
        vid_idx = kp_idx = parsing_idx = idx
        vid_paths = sorted(os.listdir(self.path_to_train_A))
        vid_path = os.path.join(self.path_to_train_A, vid_paths[vid_idx])

        kp_paths = sorted(os.listdir(self.path_to_train_kps))
        kp_path = os.path.join(self.path_to_train_kps, kp_paths[kp_idx])

        parsing_paths = sorted(os.listdir(self.path_to_train_parsing))
        parsing_path = os.path.join(self.path_to_train_parsing, parsing_paths[parsing_idx])

        assert len(vid_paths) == len(kp_paths)
        assert len(vid_paths) == len(parsing_paths)

        # select [K+1, 3, H, W], [K+1, 20, H, W], [K+1, 20, H, W]

        rand_frames_idx, selected_imgs, selected_kps,selected_joints, selected_parsings = \
            self.select_frames_kp(vid_path, kp_path,parsing_path, self.K+1) 
        
        # print('rand_frames_idx:', np.where(np.array(rand_frames_idx)==1))
        g_idx = random.randint(0, self.K)
        # print('g_idx:', g_idx)
        # g_idx = 1
        g_x = selected_imgs[g_idx]
        g_y = selected_kps[g_idx]
        g_j = selected_joints[g_idx]
        g_parse = selected_parsings[g_idx]

        ref_xs = selected_imgs[0:g_idx] + selected_imgs[g_idx+1:]
        ref_ys = selected_kps[0:g_idx] + selected_kps[g_idx+1:]
        ref_js = selected_joints[0:g_idx] + selected_joints[g_idx+1:]
        ref_parsings = selected_parsings[0:g_idx] + selected_parsings[g_idx+1:]

        similarities = self.pose_similarity(g_j,ref_js)
        similarity_maps = self.get_pose_similarity_maps(g_j, ref_js) # K * [13,256,256] tensor
        # refs = np.concatenate((ref_ys[0][17:20].permute(1,2,0).numpy() * 255,similarity_maps[0]), axis=1)
        # for i in range(1, self.K):
        #     refs = np.concatenate((refs, ref_ys[i][17:20].permute(1,2,0).numpy() * 255,similarity_maps[i]), axis=1)
        # cv2.imwrite('/dataset/ljw/ref_simi_gt.png', np.concatenate((refs, g_y[17:20].permute(1,2,0).numpy()* 255), axis=1))
        if self.align_input:
            for i in range(len(ref_xs)):
                dx, dy = openpose_utils.get_distance(ref_js[i], g_j)
                fill_color = (255,255,255)
                ref_xs[i] = F.to_tensor(F.affine(F.to_pil_image(ref_xs[i]),angle=0,translate=(dx,dy),scale=1,shear=0, fillcolor=fill_color, resample=Image.BILINEAR))
                ref_ys[i] = F.affine(ref_ys[i] ,angle=0,translate=(dx,dy),scale=1,shear=0, resample=Image.BILINEAR)
                similarity_maps[i] = F.affine(similarity_maps[i] ,angle=0,translate=(dx,dy),scale=1,shear=0, resample=Image.BILINEAR)
                ref_parsings[i] = F.affine(ref_parsings[i], angle=0,translate=(dx,dy),scale=1,shear=0, resample=Image.BILINEAR)

        return ref_xs, ref_ys,ref_parsings, g_x, g_y, g_parse, similarities,similarity_maps, vid_path


class FashionVideoGeoMatchingDataset(data.Dataset):
    """
    @ path_to_train_A
        the path to train image dataset
    @ path_to_train_kps
        the path to train keypoints
    @ path_to_train_parsing
        the path to parsings
    @ K
        K ref poses and ref images
    @ is_clean_pose
        If True: kps in openpose 17 format, more stable
        If False: kps in alphapose 18 format, more jitter
    """
    def __init__(self, path_to_train_A, path_to_train_kps, path_to_train_parsing, opt, load_size=256, pose_scale=255.0):
        super(FashionVideoGeoMatchingDataset, self).__init__()
        self.path_to_train_A = path_to_train_A
        self.path_to_train_kps = path_to_train_kps
        self.path_to_train_parsing = path_to_train_parsing
        self.step = opt.step
        self.is_clean_pose = opt.use_clean_pose
        self.align_parsing = opt.align_parsing
        self.name = opt.dataset
        self.load_size = (load_size, load_size)
        self.no_bone_RGB = False
        self.angle = (-10, 10)
        self.shift = (30, 3)
        self.scale = (0.8, 1.2)        
        # self.angle = None
        # self.shift = None
        # self.scale = None
        self.img_size = (0,0)
        self.pose_scale = pose_scale


    def getRandomAffineParam(self):
        if not self.angle and not self.scale and not self.shift:
            affine_param = None
            return affine_param
        else:
            affine_param=dict()
            affine_param['angle'] = np.random.uniform(low=self.angle[0], high=self.angle[1]) if self.angle is not False else 0
            affine_param['scale'] = np.random.uniform(low=self.scale[0], high=self.scale[1]) if self.scale is not False else 1
            shift_x = np.random.uniform(low=-self.shift[0], high=self.shift[0]) if self.shift is not False else 0
            shift_y = np.random.uniform(low=-self.shift[1], high=self.shift[1]) if self.shift is not False else 0
            affine_param['shift']=(shift_x, shift_y)

            return affine_param


    def transform_image(self, image, resize_param, method=Image.BICUBIC, affine=None, normalize=False, toTensor=True, fillWhiteColor=False):
        image = F.resize(image, resize_param, interpolation=method)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            fillcolor = None if not fillWhiteColor else (255,255,255)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
        if toTensor:
            image = F.to_tensor(image)
        if normalize:
            image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        return image  

    def load_image(self, A_path):
        A_img = Image.open(A_path)
        self.img_size = A_img.size
        # padding white color after affine transformation  
        fillWhiteColor = True if self.name =='danceFashion' else False
        Ai = self.transform_image(A_img, self.load_size, affine=self.affine_param, fillWhiteColor=fillWhiteColor)
        return Ai


    def load_skeleton(self, B_path):
        B_coor = json.load(open(B_path))["people"]

        B_coor = B_coor[0]
        pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.img_size, affine=self.affine_param)
        alpha_pose_body = pose_dict['body']
        if not self.is_clean_pose:
            pose_body = openpose_utils.openpose18_to_coco17(alpha_pose_body) #np(2,17)

        gaussian_bump = openpose_utils.obtain_map(pose_body, self.load_size) 
        pose_gaussian = np.transpose(gaussian_bump,(2, 0, 1))
        pose_gaussian = torch.Tensor(pose_gaussian)
        Bi = pose_gaussian
        if not self.no_bone_RGB:
            color = np.zeros(shape=self.load_size + (3, ), dtype=np.uint8)
            LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if self.is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
            color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
            color = np.transpose(color,(2,0,1))
            color = color / self.pose_scale # normalize to 0-1
            color = torch.Tensor(color)
            Bi = torch.cat((Bi, color), dim=0)
        
        return Bi,torch.Tensor(alpha_pose_body) #tensor(2,18)
    
    def load_parsing(self, parsing_path):
        parsing_img = Image.open(parsing_path)
        self.img_size = parsing_img.size
        parsing_img = np.array(self.transform_image(parsing_img, self.load_size, affine=self.affine_param, fillWhiteColor=False, toTensor=False)) # 0-19 one channel map
        # [H W 1 ] -> [H W 20]
        parsing_bin_map = np.stack((parsing_img,)*20, axis=2)
        
        # to 20 channel binary map
        for i in range(20):
            parsing_bin_map[:,:, i] = (parsing_img == i).astype(np.uint8) * 255 
        
        cloth_indices = [5,6,7,12]
        parsing_cloth_bin_map = parsing_img
        a = parsing_img == 5
        b = parsing_img == 6
        c = parsing_img == 7
        d = parsing_img == 12

        parsing_cloth_bin_map = (a+b+c+d).astype(np.uint8) 
        parsing_cloth_bin_map = parsing_cloth_bin_map * 255

        # [H W 20] -> [20 H W]
        parsing_bin_map = F.to_tensor(parsing_bin_map) 
        parsing_cloth_bin_map = F.to_tensor(parsing_cloth_bin_map) 
        return parsing_cloth_bin_map


    def select_frames_kp(self, video_frame_path, video_kp_path,video_parsing_path, K=2):
        '''
        select K frames and corresponding keypoints from video
        '''
        video_frame_names = sorted(os.listdir(video_frame_path))
        video_kp_names = sorted(os.listdir(video_kp_path))
        video_parsing_names = sorted(os.listdir(video_parsing_path))
        video_parsing_names_ = []
        for v in video_parsing_names:
            if v.split('.')[-1] == 'png' and v.split('.')[0].split('_')[-1]!='vis':
                video_parsing_names_ += [v]

        assert len(video_frame_names) == len(video_kp_names)
        assert len(video_frame_names) == len(video_parsing_names_)
        n_frames = len(video_frame_names) // self.step
        if n_frames <= K: #There are not enough frames in the video
            rand_frames_idx = [1]*n_frames
        else:
            rand_frames_idx = [0]*n_frames
            i = 0

            """choose K frames by index """
            while(i < K):
                idx = random.randint(0, n_frames-1)
                if rand_frames_idx[idx ] == 0:
                    rand_frames_idx[idx ] = 1
                    i += 1

        # rand_frames_idx = [0]*n_frames
        # rand_frames_idx[0] = 1
        # rand_frames_idx[100] = 1
        # rand_frames_idx[200] = 1
        selected_imgs = []
        selected_kps = []
        selected_joints = []
        selected_parsings = []

        # get K sample from the dataset
        for i, is_chosen in enumerate(rand_frames_idx):
            if is_chosen:
                i = i * self.step
                frame_img = self.load_image(os.path.join(video_frame_path,video_frame_names[i]))
                frame_kps, joints = self.load_skeleton(os.path.join(video_kp_path, video_kp_names[i])) # joints is in openpose 18 format
                frame_parsings = self.load_parsing(os.path.join(video_parsing_path, video_parsing_names_[i])) # [20, H, W]

                selected_imgs = selected_imgs + [frame_img]
                selected_kps = selected_kps + [frame_kps]
                selected_joints = selected_joints + [joints]
                selected_parsings = selected_parsings + [frame_parsings]
        return rand_frames_idx, selected_imgs, selected_kps,selected_joints, selected_parsings

    def __len__(self):
        vid_num = 0
        for video in os.listdir(self.path_to_train_A):
            vid_num+=1
        return vid_num

    def __getitem__(self, idx): 
        # get random affine augmentation
        self.affine_param = self.getRandomAffineParam()
        
        vid_idx = kp_idx = parsing_idx = idx
        vid_paths = sorted(os.listdir(self.path_to_train_A))
        vid_path = os.path.join(self.path_to_train_A, vid_paths[vid_idx])

        kp_paths = sorted(os.listdir(self.path_to_train_kps))
        kp_path = os.path.join(self.path_to_train_kps, kp_paths[kp_idx])

        parsing_paths = sorted(os.listdir(self.path_to_train_parsing))
        parsing_path = os.path.join(self.path_to_train_parsing, parsing_paths[parsing_idx])

        assert len(vid_paths) == len(kp_paths)
        assert len(vid_paths) == len(parsing_paths)

        # select [K+1, 3, H, W], [K+1, 20, H, W], [K+1, 20, H, W]

        rand_frames_idx, selected_imgs, selected_kps,selected_joints, selected_parsings = \
            self.select_frames_kp(vid_path, kp_path,parsing_path, 2) 
        
        # print('rand_frames_idx:', np.where(np.array(rand_frames_idx)==1))
        # print('g_idx:', g_idx)
        LHip_idx = openpose_utils.OPENPOSE_18['LHip']
        RHip_idx = openpose_utils.OPENPOSE_18['RHip']
        g_idx = 1
        g_x = selected_imgs[g_idx]
        g_j = selected_joints[g_idx] # [2,18]
        g_j_root = (g_j[:,LHip_idx] + g_j[:, RHip_idx])/2
        g_parse = selected_parsings[g_idx]

        ref_x = selected_imgs[0]
        ref_j = selected_joints[0] 
        ref_j_root = (ref_j[:,LHip_idx] + ref_j[:, RHip_idx])/2
        ref_parsing = selected_parsings[0]
        
        if self.align_parsing:
            shift_delta = g_j_root - ref_j_root #[2] in y,x order
            import torchvision.transforms.functional as F
            ref_x = F.affine(ref_x, angle=0, translate=(shift_delta[1], shift_delta[0]), scale=1, shear=0)
            ref_parsing = F.affine(ref_parsing, angle=0, translate=(shift_delta[1], shift_delta[0]), scale=1, shear=0)
        return ref_x,ref_parsing, g_x, g_parse, vid_path

class FashionVideoTestDataset(data.Dataset):
    """
    @ path_to_test_A
        the path to test image dataset
    @ path_to_test_kps
        the path to test keypoints
    @ K
        1 for target pose and image, K-1 for ref poses and ref images
    @ is_clean_pose
        If True: kps in openpose 17 format, more stable
        If False: kps in alphapose 18 format, more jitter
    """
    def __init__(self, path_to_test_A, path_to_test_kps, K=5, is_clean_pose=False, name='fashion', load_size=256, pose_scale=255.0):
        super(FashionVideoTestDataset, self).__init__()
        self.path_to_test_A = path_to_test_A
        self.path_to_test_kps = path_to_test_kps
        self.K = K
        self.is_clean_pose = is_clean_pose
        
        self.name = name
        self.load_size = (load_size, load_size)
        self.y_nc = 17+3
        self.no_bone_RGB = False
        self.angle = False
        self.shift = False
        self.scale = False
        self.img_size = (0,0)
        self.pose_scale = pose_scale
        self.affine_param = None


    def getRandomAffineParam(self):
        if not self.angle and not self.scale and not self.shift:
            affine_param = None
            return affine_param
        else:
            affine_param=dict()
            affine_param['angle'] = np.random.uniform(low=self.angle[0], high=self.angle[1]) if self.angle is not False else 0
            affine_param['scale'] = np.random.uniform(low=self.scale[0], high=self.scale[1]) if self.scale is not False else 1
            shift_x = np.random.uniform(low=-self.shift[0], high=self.shift[0]) if self.shift is not False else 0
            shift_y = np.random.uniform(low=-self.shift[1], high=self.shift[1]) if self.shift is not False else 0
            affine_param['shift']=(shift_x, shift_y)

            return affine_param


    def transform_image(self, image, resize_param, method=Image.BICUBIC, affine=None, normalize=False, toTensor=True, fillWhiteColor=False):
        image = F.resize(image, resize_param, interpolation=method)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            fillcolor = None if not fillWhiteColor else (255,255,255)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
        if toTensor:
            image = F.to_tensor(image)
        if normalize:
            image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        return image  

    def load_image(self, A_path):
        A_img = Image.open(A_path)
        self.img_size = A_img.size
        # padding white color after affine transformation  
        fillWhiteColor = True if self.name =='fashion' else False
        Ai = self.transform_image(A_img, self.load_size, affine=self.affine_param, fillWhiteColor=fillWhiteColor)
        return Ai


    def load_skeleton(self, B_path):
        B_coor = json.load(open(B_path))["people"]
        if len(B_coor)==0:
            pose = torch.zeros(self.y_nc, self.load_size[0], self.load_size[1])
        else:
            B_coor = B_coor[0]
            pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.img_size, affine=self.affine_param)
            pose_body = pose_dict['body']
            if not self.is_clean_pose:
                pose_body = openpose_utils.openpose18_to_coco17(pose_body)

            pose_numpy = openpose_utils.obtain_map(pose_body, self.load_size) 
            pose = np.transpose(pose_numpy,(2, 0, 1))
            pose = torch.Tensor(pose)
            Bi = pose
            if not self.no_bone_RGB:
                color = np.zeros(shape=self.load_size + (3, ), dtype=np.uint8)
                LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if self.is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
                color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
                color = np.transpose(color,(2,0,1))
                color = color / self.pose_scale # normalize to 0-1
                color = torch.Tensor(color)
                Bi = torch.cat((Bi, color), dim=0)
        
        return Bi
    

    def select_frames_kp(self, video_frame_path, video_kp_path, K):
        video_frame_names = sorted(os.listdir(video_frame_path))
        video_kp_names = sorted(os.listdir(video_kp_path))

        assert len(video_kp_names) == len(video_frame_names)
        n_frames = len(video_frame_names)

        if n_frames <= K: #There are not enough frames in the video
            rand_frames_idx = [1]*n_frames
        else:
            rand_frames_idx = [0]*n_frames
            i = 0

            """choose K frames by index """
            while(i < K):
                idx = random.randint(0, n_frames-1)
                if rand_frames_idx[idx] == 0:
                    rand_frames_idx[idx] = 1
                    i += 1
        
        selected_imgs = None
        selected_kps = None
        for i, is_chosen in enumerate(rand_frames_idx):
            if is_chosen:
                frame_img = self.load_image(os.path.join(video_frame_path,video_frame_names[i]))
                frame_kps = self.load_skeleton(os.path.join(video_kp_path, video_kp_names[i]))
                
                frame_img = frame_img.unsqueeze(0)
                frame_kps = frame_kps.unsqueeze(0)
                if selected_imgs is None:
                    selected_imgs = frame_img
                    selected_kps = frame_kps
                else:
                    selected_imgs = torch.cat([selected_imgs, frame_img])
                    selected_kps = torch.cat([selected_kps, frame_kps])
        return selected_imgs, selected_kps

    # def initialize(self, opt):
    #     self.opt = opt
        
    #     self.frame_kp, self.target_pose, self.


    def __len__(self):
        vid_num = 0
        for video in os.listdir(self.path_to_test_A):
            vid_num+=1
        return vid_num

    def __getitem__(self, idx): 
        # get random affine augmentation
        
        vid_idx = kp_idx = idx
        vid_paths = sorted(os.listdir(self.path_to_test_A))
        vid_path = os.path.join(self.path_to_test_A, vid_paths[vid_idx])

        kp_paths = sorted(os.listdir(self.path_to_test_kps))
        kp_path = os.path.join(self.path_to_test_kps, kp_paths[kp_idx])
        
        ref_xs, ref_ys = self.select_frames_kp(vid_path, kp_path, self.K-1)

        return ref_xs, ref_ys, idx

class FashionFrameTestDataset(data.Dataset):
    """
    @ path_to_test_imgs
        the path to test image dir
    @ path_to_test_kps
        the path to test keypoints dir
    @ K
        1 for target pose and image, K-1 for ref poses and ref images
    @ is_clean_pose
        If True: kps in openpose 17 format, more stable
        If False: kps in alphapose 18 format, more jitter
    """
    def __init__(self, path_to_test_imgs, path_to_test_kps, K=5, is_clean_pose=False, name='fashion', load_size=256, pose_scale=255.0):
        super(FashionFrameTestDataset, self).__init__()
        self.path_to_test_imgs = path_to_test_imgs
        self.path_to_test_kps = path_to_test_kps
        self.K = K
        self.is_clean_pose = is_clean_pose
        
        self.name = name
        self.load_size = (load_size, load_size)
        self.y_nc = 17+3
        self.no_bone_RGB = False
        self.angle = False
        self.shift = False
        self.scale = False
        self.img_size = (0,0)
        self.pose_scale = pose_scale
        self.affine_param = None


    def getRandomAffineParam(self):
        if not self.angle and not self.scale and not self.shift:
            affine_param = None
            return affine_param
        else:
            affine_param=dict()
            affine_param['angle'] = np.random.uniform(low=self.angle[0], high=self.angle[1]) if self.angle is not False else 0
            affine_param['scale'] = np.random.uniform(low=self.scale[0], high=self.scale[1]) if self.scale is not False else 1
            shift_x = np.random.uniform(low=-self.shift[0], high=self.shift[0]) if self.shift is not False else 0
            shift_y = np.random.uniform(low=-self.shift[1], high=self.shift[1]) if self.shift is not False else 0
            affine_param['shift']=(shift_x, shift_y)

            return affine_param


    def transform_image(self, image, resize_param, method=Image.BICUBIC, affine=None, normalize=False, toTensor=True, fillWhiteColor=False):
        image = F.resize(image, resize_param, interpolation=method)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            fillcolor = None if not fillWhiteColor else (255,255,255)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
        if toTensor:
            image = F.to_tensor(image)
        if normalize:
            image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        return image

    def load_image(self, A_path):
        A_img = Image.open(A_path)
        self.img_size = A_img.size
        # padding white color after affine transformation  
        fillWhiteColor = True if self.name =='fashion' else False
        Ai = self.transform_image(A_img, self.load_size, affine=self.affine_param, fillWhiteColor=fillWhiteColor)
        return Ai


    def load_skeleton(self, B_path):
        B_coor = json.load(open(B_path))["people"]
        if len(B_coor)==0:
            pose = torch.zeros(self.y_nc, self.load_size[0], self.load_size[1])
        else:
            B_coor = B_coor[0]
            pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=self.load_size, org_size=self.img_size, affine=self.affine_param)
            pose_body = pose_dict['body']
            if not self.is_clean_pose:
                pose_body = openpose_utils.openpose18_to_coco17(pose_body)

            pose_numpy = openpose_utils.obtain_map(pose_body, self.load_size) 
            pose = np.transpose(pose_numpy,(2, 0, 1))
            pose = torch.Tensor(pose)
            Bi = pose
            if not self.no_bone_RGB:
                color = np.zeros(shape=self.load_size + (3, ), dtype=np.uint8)
                LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if self.is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
                color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
                color = np.transpose(color,(2,0,1))
                color = color / self.pose_scale # normalize to 0-1
                color = torch.Tensor(color)
                Bi = torch.cat((Bi, color), dim=0)
        
        return Bi

    def __len__(self):
        frame_num = 0
        for frame in os.listdir(self.path_to_test_imgs):
            frame_num+=1
        return frame_num

    def __getitem__(self, idx): 
        # get random affine augmentation
        
        frame_idx = kp_idx = idx
        frame_paths = sorted(os.listdir(self.path_to_test_imgs))
        frame_path = os.path.join(self.path_to_test_imgs, frame_paths[frame_idx])

        kp_paths = sorted(os.listdir(self.path_to_test_kps))
        kp_path = os.path.join(self.path_to_test_kps, kp_paths[kp_idx])
        frame_img = self.load_image(frame_path)
        frame_kps = self.load_skeleton(kp_path)

        return frame_img, frame_kps, idx