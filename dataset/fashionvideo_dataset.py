import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import cv2
import torchvision.transforms.functional as F
from util import openpose_utils
from PIL import Image
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
        pose_body = pose_dict['body']
        if not self.is_clean_pose:
            pose_body = openpose_utils.openpose18_to_coco17(pose_body) #np(2,17)

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
        
        return Bi,torch.Tensor(pose_body)
    
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
        LShoulder_idx = openpose_utils.COCO_17['LShoulder']
        RShoulder_idx = openpose_utils.COCO_17['RShoulder']
        
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

        return ref_xs, ref_ys,ref_parsings, g_x, g_y, g_parse, similarities, vid_path


class FashionVideoGeoMatchingDataset(data.Dataset):
    """
    @ path_to_train_A
        the path to train image dataset
    @ path_to_train_parsing
        the path to parsings
    """
    def __init__(self, path_to_train_A, path_to_train_parsing, opt, load_size=256, pose_scale=255.0):
        super(FashionVideoGeoMatchingDataset, self).__init__()
        self.path_to_train_A = path_to_train_A
        self.path_to_train_parsing = path_to_train_parsing
        
        self.name = opt.dataset
        self.load_size = (load_size, load_size)
        self.angle = (-10, 10)
        self.shift = (30, 3)
        self.scale = (0.8, 1.2)        
        # self.angle = None
        # self.shift = None
        # self.scale = None
        self.img_size = (0,0)

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

    def select_frames_kp(self, video_frame_path,video_parsing_path, K=2):
        '''
        select K frames and corresponding keypoints from video
        '''
        video_frame_names = sorted(os.listdir(video_frame_path))
        video_parsing_names = sorted(os.listdir(video_parsing_path))
        video_parsing_names_ = []
        for v in video_parsing_names:
            if v.split('.')[-1] == 'png' and v.split('.')[0].split('_')[-1]!='vis':
                video_parsing_names_ += [v]

        assert len(video_frame_names) == len(video_parsing_names_)
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
        
        selected_imgs = []
        selected_parsings = []


        for i, is_chosen in enumerate(rand_frames_idx):
            if is_chosen:
                frame_img = self.load_image(os.path.join(video_frame_path,video_frame_names[i]))
                frame_parsings = self.load_parsing(os.path.join(video_parsing_path, video_parsing_names_[i])) # [20, H, W]

                selected_imgs = selected_imgs + [frame_img]
                selected_parsings = selected_parsings + [frame_parsings]
        return selected_imgs, selected_parsings

    def __len__(self):
        vid_num = 0
        for video in os.listdir(self.path_to_train_A):
            vid_num+=1
        return vid_num

    def __getitem__(self, idx): 
        # get random affine augmentation
        self.affine_param = self.getRandomAffineParam()
        
        vid_idx = parsing_idx = idx
        vid_paths = sorted(os.listdir(self.path_to_train_A))
        vid_path = os.path.join(self.path_to_train_A, vid_paths[vid_idx])

        parsing_paths = sorted(os.listdir(self.path_to_train_parsing))
        parsing_path = os.path.join(self.path_to_train_parsing, parsing_paths[parsing_idx])

        assert len(vid_paths) == len(parsing_paths)

        # select [K+1, 3, H, W], [K+1, 20, H, W], [K+1, 20, H, W]
        selected_imgs, selected_parsings = self.select_frames_kp(vid_path, parsing_path) 

        # g_idx = np.random.randint(low=0, high=2)
        g_idx = 1
        g_x = selected_imgs[g_idx]
        g_parse = selected_parsings[g_idx]

        ref_x = selected_imgs[0]
        ref_parsing = selected_parsings[0]

        # print('gx_shape:', g_x.shape)
        # print('gy_shape:', g_y.shape)
        # print('ref_xs_shape:', ref_xs.shape)
        # print('ref_ys_shape:', ref_ys.shape)
        return ref_x, ref_parsing, g_x, g_parse, vid_path


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