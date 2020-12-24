import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import cv2
import torchvision.transforms.functional as F
from util import openpose_utils, pose_utils
from PIL import Image, ImageDraw
from numpy import dot
from numpy.linalg import norm
import random
import json
import sys
import math
import numbers

class FashionDataset(data.Dataset):
    """
    @ phase
        train|test|val
    @ path_to_train_tuples
        the path to fasion-x_tuples-train.csv
    @ path_to_test_tuples
        the path to fasion-x_tuples-test.csv
    @ path_to_train_root
        the path to train image root
    @ path_to_train_anno
        the path to train image keypoint annotations
    @ path_to_test_root
        the path to test image root
    @ path_to_test_anno
        the path to test image keypoint annotations
    """
    def __init__(self, phase,  path_to_train_tuples, path_to_test_tuples, 
    path_to_train_imgs_dir, path_to_train_anno, 
    path_to_test_imgs_dir, path_to_test_anno, opt,
    path_to_train_label_dir=None, path_to_test_label_dir=None,
    path_to_train_parsings_dir=None,path_to_test_parsings_dir=None, 
    load_size=256, pose_scale=255.0):
        super(FashionDataset, self).__init__()
        
        self._is_train = phase == 'train'
        self._train_tuples = pd.read_csv(path_to_train_tuples)
        self._train_anno = pd.read_csv(path_to_train_anno, sep=':')
        self._train_img_dir = path_to_train_imgs_dir
        self._train_parsing_dir = path_to_train_parsings_dir
        self._train_label_dir = path_to_train_label_dir
        print ("--------------- Dataset Info: ---------------")
        print ("Phase: %s" % phase)
        print ("Number of tuples train: %s" % len(self._train_tuples))

        self._test_tuples = pd.read_csv(path_to_test_tuples)
        self._test_anno = pd.read_csv(path_to_test_anno, sep=':')
        self._test_img_dir = path_to_test_imgs_dir
        self._test_label_dir = path_to_test_label_dir
        print ("Number of tuples test: %s" % len(self._test_tuples))
        # self._test_tuples = pd.read_csv(path_to_train_tuples)
        # self._test_anno = pd.read_csv(path_to_test_anno, sep=':')
        # self._test_img_dir = path_to_train_imgs_dir
        # print ("Number of tuples test: %s" % len(self._test_tuples))        

        self._nb_inputs = opt.K
        if opt.use_parsing:
            self._parsing_categories = opt.categories
        self._annotation_file = pd.concat([self._train_anno, self._test_anno], axis=0, ignore_index=True)
        self._annotation_file = self._annotation_file.set_index('name')
        print ("Number of total images: %s" % len(self._annotation_file))

        self.size = len(self._train_tuples) if self._is_train else len(self._test_tuples)

        # self.use_dot = opt.use_dot
        # self.align_input = opt.align_input

        self.name = 'fashion'
        self.load_size = (load_size, load_size)
        self.anno_size = tuple(opt.anno_size) if opt.anno_size else (load_size,load_size)
        self.no_bone_RGB = False
        # self.angle = (-10, 10)
        # self.shift = (30, 3)
        # self.scale = (0.8, 1.2)        
        self.angle = None
        self.shift = None
        self.scale = None
        self.img_size = (0,0)
        self.pose_scale = pose_scale
        self.align_corner = opt.align_corner
        self.use_parsing = opt.use_parsing
        self.use_tps_sim = opt.use_tps_sim
        self.tps_sim_beta1 = opt.tps_sim_beta1
        self.tps_sim_beta2 = opt.tps_sim_beta2
        self.affine_param = self.getRandomAffineParam()

        self.joints_for_cos_sim = opt.joints_for_cos_sim if hasattr(opt, 'joints_for_cos_sim') else 4



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

    def transform_image(self, image, resize_param, method=Image.BILINEAR, affine=None, normalize=True, toTensor=True, fillWhiteColor=False):
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

    def load_image(self, tuple_df, direction='', affine=None):
        assert direction in ['to'] + ['from_' + str(i) for i in range(self._nb_inputs)] + ['from']
        
        if self._is_train: 
            A_path = os.path.join(self._train_img_dir, tuple_df[direction])
        else:
            A_path = os.path.join(self._test_img_dir, tuple_df[direction])
        A_img = Image.open(A_path)
        self.img_size = A_img.size
        fillWhiteColor = True if self.name =='fashion' else False
        Ai = self.transform_image(A_img, self.load_size, affine=affine, fillWhiteColor=fillWhiteColor, normalize=True)

        return Ai

    def load_keypoints(self, tuple_df, direction='', affine_matrix=None):
        assert direction in ['to'] + ['from_' + str(i) for i in range(self._nb_inputs)]  + ['from']
        row = self._annotation_file.loc[tuple_df[direction]]
        kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x']) #[18,2]

        return kp_array.transpose()

    def load_skeleton(self, tuple_df, direction='', affine_matrix=None):
        assert direction in ['to'] + ['from_' + str(i) for i in range(self._nb_inputs)]  + ['from']
        row = self._annotation_file.loc[tuple_df[direction]]
        kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x']) #[18,2]

        # kp_resize = pose_utils.resize_cords(kp_array, self.load_size, self.anno_size, affine_matrix)
        # gaussian_bump = pose_utils.cords_to_map(kp_resize, self.load_size) #[H,W,18] resize keypoints to load_size
        gaussian_bump = pose_utils.cords_to_map(kp_array, self.load_size, self.anno_size, affine_matrix ) #[H,W,18] resize keypoints to load_size
        
        pose_gaussian = np.transpose(gaussian_bump,(2, 0, 1)) 
        pose_gaussian = torch.Tensor(pose_gaussian) #[18,H,W]
        Bi = pose_gaussian
        if not self.no_bone_RGB:
            color = pose_utils.draw_pose_from_map(gaussian_bump)[0]
            # color = pose_utils.draw_pose_from_cords(kp_resize, gaussian_bump.shape[:2])[0]
            color = np.transpose(color,(2,0,1))
            color = color / self.pose_scale # normalize to 0-1
            color = torch.Tensor(color)
            Bi = torch.cat((Bi, color), dim=0)
        
        return Bi
    
    def load_tps_label(self,tuple_df, direction=''):
        assert direction in ['to'] + ['from_' + str(i) for i in range(self._nb_inputs)] + ['from']
        if self._is_train: 
            label_path = os.path.join(self._train_label_dir, tuple_df[direction].replace('.jpg','_sim.npy'))
        else:
            label_path = os.path.join(self._test_label_dir, tuple_df[direction].replace('.jpg','_sim.npy'))
        
        label = np.load(label_path)
        # label = torch.Tensor(label).permute(2,0,1)
        return label

    def load_parsing(self, tuple_df, direction='' ,categories=8, affine=None):
        assert direction in ['to'] + ['from_' + str(i) for i in range(self._nb_inputs)] + ['from']
        if self._is_train: 
            parsing_path = os.path.join(self._train_parsing_dir, tuple_df[direction].replace('.jpg','_merge.png'))
        else:
            parsing_path = os.path.join(self._test_parsing_dir, tuple_df[direction].replace('.jpg','_merge.png'))
        parsing_img = Image.open(parsing_path)
        parsing_img = np.array(self.transform_image(parsing_img, self.load_size, affine=self.affine_param, fillWhiteColor=False, toTensor=False, normalize=False)) # 0-19 one channel map
        # [H W 1 ] -> [H W 20]
        parsing_bin_map = np.stack((parsing_img,)*categories, axis=2)
        # print(parsing_bin_map.shape)
        # to 20 channel binary map
        for i in range(categories):
            parsing_bin_map[:,:, i] = (parsing_img == i).astype(np.uint8) * 255 
        
        # [H W 20] -> [20 H W]
        # [0-255] -> [0-1]
        parsing_bin_map = F.to_tensor(parsing_bin_map) 
        return parsing_bin_map

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
    
    # input two
    def get_cosine_similarity(self, vector1, vector2):
        cos = torch.nn.CosineSimilarity(dim=0)
        cos_sim = cos(vector1,vector2)
        return cos_sim

    def get_warp_grid(self, scale, trans, grid_size):
        H, W = grid_size
        theta = torch.zeros((2,3))
        theta[0,0] = 1/scale
        theta[0,2] = -2 * trans[0]/ W
        theta[1,1] = 1/scale
        theta[1,2] = -2 * trans[1]/ H
        theta = theta.unsqueeze(0)
        grid = torch.nn.functional.affine_grid(theta,[1,3,H,W],align_corners=self.align_corner)
        return grid

    def get_tps_similarity(self, gt_y, ref_ys, gt_label, ref_labels):
        '''
        gt_y: [2,18] tensor
        ref_ys: K * [2,18] tensor
        gt_label: [anno_size , 3] tensor
        ref_labels: K * [anno_size , 3] tensor

        return: K*[1, load_size] tensor
        '''
        sims = []
        
        for i in range(len(ref_ys)):
            scale, trans = pose_utils.get_scale_trans(ref_ys[i][::-1,:],gt_y[::-1,:]) # y,x to x,y
            grid = self.get_warp_grid(scale, trans, self.anno_size)

            ref_affine_tensor = torch.nn.functional.grid_sample(torch.Tensor(ref_labels[i]).unsqueeze(0).permute(0,3,1,2), grid, padding_mode='border',align_corners=self.align_corner)
            ref_affine = ref_affine_tensor.permute(0,2,3,1)[0].numpy()
            ref_affine_norm = np.linalg.norm(ref_affine[:,:,0:2], axis=2)
            ref_affine[:,:,0] /= (ref_affine_norm+np.finfo(float).eps)
            ref_affine[:,:,1] /= (ref_affine_norm+np.finfo(float).eps)

            _,_,sim = pose_utils.get_dot_sim(ref_affine, gt_label, norm_value=17, beta1=self.tps_sim_beta1, beta2=self.tps_sim_beta2)
            sims += [sim] # [anno_size]
        sims_sum = sum(sims)
        sims = [torch.Tensor(sim / (sims_sum+np.finfo(float).eps)) for sim in sims] # normlize to sum 1
        sims = [F.resize(sim.unsqueeze(0), self.load_size, interpolation=Image.BILINEAR) for sim in sims]
        sims = [sim for sim in sims]
        assert(sims[0].shape == (1,256,256))
        return sims 

    def get_similarity_value(self, gt, refs):
        '''
        gt: [2,18] tensor
        refs: K * [2,18] tensor
        '''
        # LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
        #    [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
        #    [0,15], [15,17], [2,16], [5,17]]  # fasion limbs , 计算向量

        # LIMB_SEQ = [[1,2], [1,5],  [1,8], [1,11]]  # fasion limbs , 计算向量
        LIMB_SEQ = [[1,2], [1,5]]  # fasion limbs , 计算向量
        g_dirs = None
        for limb in LIMB_SEQ:
            g_dir = gt[:,limb[1]] - gt[:, limb[0]]
            if g_dirs ==None:
                g_dirs = g_dir
            else:
                g_dirs = torch.cat((g_dirs, g_dir),dim=0)
        cosine_similarities = []
        for ref in refs:
            ref_dirs = None
            for i,limb in enumerate(LIMB_SEQ):
                ref_dir = ref[:,limb[1]] - ref[:, limb[0]]
                
                if ref_dirs ==None:
                    ref_dirs = ref_dir
                else:
                    ref_dirs = torch.cat((ref_dirs, ref_dir),dim=0)
            cos = self.get_cosine_similarity(g_dirs, ref_dirs) /2 + 0.5
            cosine_similarities += [cos]
        return cosine_similarities
    
    def get_dot_sim(self, source:np.ndarray, target:np.ndarray, norm_value:float, beta1:float, beta2:float):
        '''
        @source: H by W by 3 input flow and label features
        @target: H by W by 3 input flow and label features
        
        '''
        label_diff = np.abs(source[:,:,2]- target[:,:,2])
        label_diff = np.minimum(label_diff, beta2)/ norm_value
        label_term = np.exp(- beta1 * label_diff) # H,W
        
        flow_term = source[:,:,:2] * target[:,:,:2]
        flow_term = (np.sum(flow_term, axis=2)+1)/2 # H,W
        return flow_term,label_term,flow_term*label_term
        

    def get_normalized_Cosinesimilarity_value(self, gt, refs):
        '''
        gt: [2,18] tensor
        refs: K * [2,18] tensor
        return: K *[1, load_size ] tensor
        '''
        gt = torch.Tensor(gt)
        refs = [torch.Tensor(ref) for ref in refs]
        if self.joints_for_cos_sim == -1:
            LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
            [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
            [0,15], [15,17], [2,16], [5,17]]  # fasion limbs , 计算向量
        elif self.joints_for_cos_sim == 4:
            LIMB_SEQ = [[1,2], [1,5],  [1,8], [1,11]]  # fasion limbs , 计算向量
        else:
            raise NotImplementedError
        # LIMB_SEQ = [[1,2], [1,5]]  # fasion limbs , 计算向量
        cos = torch.nn.CosineSimilarity(dim=0)
        g_dirs = []
        MISSING_VALUE = -1
        MISSING_CORD = torch.Tensor([MISSING_VALUE,MISSING_VALUE])
        MISSING_DIR = torch.Tensor([self.load_size[0]+1, self.load_size[1]+1])
        for limb in LIMB_SEQ:
            g_dir = gt[:,limb[1]] - gt[:, limb[0]]
            if torch.allclose(gt[:,limb[1]], MISSING_CORD) or torch.allclose(gt[:,limb[0]], MISSING_CORD):
                g_dir = MISSING_DIR
            
            g_dirs += [g_dir]
        
        cosine_similarities = []
        for ref in refs:
            ref_dirs = None
            cosines = []
            for i,limb in enumerate(LIMB_SEQ):
                ref_dir = ref[:,limb[1]] - ref[:, limb[0]]
                if torch.allclose(ref[:,limb[1]], MISSING_CORD) or torch.allclose(ref[:,limb[0]], MISSING_CORD):
                    ref_dir = MISSING_DIR

                if torch.allclose(g_dirs[i], MISSING_DIR) or torch.allclose(ref_dir, MISSING_DIR):
                    continue
                else:
                    cosine = cos(g_dirs[i], ref_dir)

                cosines += [(cosine+1)/2]
                if ref_dirs ==None:
                    ref_dirs = ref_dir
                else:
                    ref_dirs = torch.cat((ref_dirs, ref_dir),dim=0)
            avg_cosine = sum(cosines) / len(cosines)
            cosine_similarities += [avg_cosine]
        
        # normalize to 1
        sum_cosine_sim = 0
        order = 3
        for a in cosine_similarities:
            sum_cosine_sim += a ** order 
        normed_cosine = [a ** order / sum_cosine_sim * torch.ones(1, self.load_size[0],self.load_size[1]) for a in cosine_similarities]
        return normed_cosine 

    def __len__(self):
        return self.size

    def __getitem__(self, idx): 
        import time
        start = time.time()
        if self._is_train:
            tuple_df = self._train_tuples.iloc[idx]
        else:
            tuple_df = self._test_tuples.iloc[idx]
        
        if self._nb_inputs == 1:
            from_names = [tuple_df['from'].split('.')[0]]
            from_indices = ['from']
        else:
            from_names = [tuple_df[from_idx].split('.')[0] for from_idx in ['from_'+str(i) for i in range(self._nb_inputs)] ]
            from_indices = ['from_'+str(i) for i in range(self._nb_inputs)]

        to_name = tuple_df['to'].split('.')[0]
        
        # center = (self.load_size[0] * 0.5 + 0.5, self.load_size[1] * 0.5 + 0.5)
        # affine_param = self.getRandomAffineParam()
        # affine_matrix = self.get_affine_matrix(center=center, angle=affine_param['angle'], translate=affine_param['shift'], scale=affine_param['scale'], shear=0)
        ref_xs = [self.load_image(tuple_df, from_idx) for from_idx in from_indices]
        g_x = self.load_image(tuple_df, 'to')
        image_end = time.time()

        ref_skeletons = [self.load_skeleton(tuple_df, from_idx) for from_idx in from_indices]
        ref_js = [self.load_keypoints(tuple_df, from_idx) for from_idx in from_indices]
        g_skeleton = self.load_skeleton(tuple_df, 'to')
        g_j = self.load_keypoints(tuple_df, 'to')

        ref_parsings = [self.load_parsing(tuple_df, from_idx, categories=self._parsing_categories) for from_idx in from_indices] if self.use_parsing else None
        g_parsing = self.load_parsing(tuple_df, 'to', categories=self._parsing_categories) if self.use_parsing else None

        ref_tps_labels = [self.load_tps_label(tuple_df, from_idx) for from_idx in from_indices]  if self.use_tps_sim else None
        g_tps_label = self.load_tps_label(tuple_df, 'to') if self.use_tps_sim else None

        # print('get image item time:%.3f'%(image_end-start))
        ref_ys = [ref_skeletons[i] for i in range(len(ref_skeletons))]
        
        g_y = g_skeleton
        # print(g_j.data.numpy())
        # print(to_name)
        structure_end = time.time()
        # print('get structure item time:%.3f'%(structure_end-image_end))
        # affine_param = self.getRandomAffineParam()
        # affine_matrix = self.get_affine_matrix(center=center, angle=affine_param['angle'], translate=affine_param['shift'], scale=affine_param['scale'], shear=0)
        # similarities = self.get_similarity_value(g_j, ref_js)
        if not self.use_tps_sim:
            similarities = self.get_normalized_Cosinesimilarity_value(g_j, ref_js) 
        else: 
            similarities = self.get_tps_similarity(g_j, ref_js, g_tps_label,ref_tps_labels)
        end = time.time()
        # print('get item time:%.3f'%(end-start))
        if self.use_parsing:
            return {'ref_xs':ref_xs, 'ref_ys':ref_ys, 'ref_ps':ref_parsings, 'g_x':g_x, 'g_y':g_y,  'g_p':g_parsing, 'froms':from_names, 'to':to_name, 'sim':similarities}
        else:
            return {'ref_xs':ref_xs, 'ref_ys':ref_ys, 'g_x':g_x, 'g_y':g_y, 'froms':from_names, 'to':to_name, 'sim':similarities}

    def get_affine_matrix(self, center, angle, translate, scale, shear):
        matrix_inv = self.get_inverse_affine_matrix(center, angle, translate, scale, shear)

        matrix_inv = np.matrix(matrix_inv).reshape(2,3)
        pad = np.matrix([0,0,1])
        matrix_inv = np.concatenate((matrix_inv, pad), 0)
        matrix = np.linalg.inv(matrix_inv)
        return matrix

    def get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # code from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#affine
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        #       RSS is rotation with scale and shear matrix
        #       RSS(a, scale, shear) = [ cos(a + shear_y)*scale    -sin(a + shear_x)*scale     0]
        #                              [ sin(a + shear_y)*scale    cos(a + shear_x)*scale     0]
        #                              [     0                  0          1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1


        angle = math.radians(angle)
        if isinstance(shear, (tuple, list)) and len(shear) == 2:
            shear = [math.radians(s) for s in shear]
        elif isinstance(shear, numbers.Number):
            shear = math.radians(shear)
            shear = [shear, 0]
        else:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " +
                "two values. Got {}".format(shear))
        scale = 1.0 / scale

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear[0]) * math.cos(angle + shear[1]) + \
            math.sin(angle + shear[0]) * math.sin(angle + shear[1])
        matrix = [
            math.cos(angle + shear[0]), math.sin(angle + shear[0]), 0,
            -math.sin(angle + shear[1]), math.cos(angle + shear[1]), 0
        ]
        matrix = [scale / d * m for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]
        return matrix



