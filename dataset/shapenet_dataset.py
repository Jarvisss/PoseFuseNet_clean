import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
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
import h5py

def make_dataset(opt, sub_dataset_model='chair'):
    """Create dataset"""
    root = opt.path_to_dataset

    if opt.phase == 'train':
        dataset = ShapeNetDataset(
            phase = opt.phase,
            image_id_file=os.path.join(root, 'id_%s_%s.txt' %(sub_dataset_model,opt.phase)), 
            image_name_file  = os.path.join(root, 'name_%s_%s.txt' %(sub_dataset_model,opt.phase)),
            test_id_file = None,
            hdf5_file = os.path.join(root, 'data_%s.hdf5'%sub_dataset_model),
            opt=opt)
    else:
        dataset = ShapeNetDataset(
            phase = opt.phase,
            image_id_file=os.path.join(root, 'id_%s_%s.txt' %(sub_dataset_model,opt.phase)), 
            image_name_file  = os.path.join(root, 'name_%s_%s.txt' %(sub_dataset_model,opt.phase)),
            test_id_file = os.path.join(root, 'id_%s_elevation_0.txt'%sub_dataset_model),
            hdf5_file = os.path.join(root, 'data_%s.hdf5'%sub_dataset_model),
            opt=opt)
    
    return dataset

class ShapeNetDataset(data.Dataset):
    """
    @ phase
        train | test | val
    @ path_to_train_tuples
        the path to fasion-x_tuples-train.csv
    @ path_to_test_tuples
        the path to fasion-x_tuples-test.csv
    @ path_to_train_anno
        the path to train image keypoint annotations
    @ path_to_test_anno
        the path to test image keypoint annotations
    """
    def __init__(self, phase,  image_id_file, image_name_file, test_id_file,
    hdf5_file,opt,
    load_size=256):
        super(ShapeNetDataset, self).__init__()
        
        self._is_train = phase == 'train'
        self._is_eval = phase=='eval'
        self.image_id_file = image_id_file
        
        self.image_name_file = image_name_file
        self.hdf5_file = hdf5_file
        self.image_ids = np.genfromtxt(self.image_id_file, dtype=np.str)
        self.size = len(self.image_ids)

        if not self._is_train:
            with open(test_id_file, 'r') as id_list_path:
                id_list = id_list_path.readlines()
            self.id_list = [ids.strip().split(' ') for ids in id_list]
            
            print('test tuples : %d' % len(self.id_list))
            self.size = len(self.id_list)
            
        self._nb_inputs = opt.K

        print ("--------------- Dataset Info: ---------------")
        print ("Phase: %s" % phase)
        print ("Number of samples %s: %s" % (phase, str(self.size)))
        # self._test_tuples = pd.read_csv(path_to_train_tuples)
        # self._test_anno = pd.read_csv(path_to_test_anno, sep=':')
        # self._test_img_dir = path_to_train_imgs_dir
        # print ("Number of tuples test: %s" % len(self._test_tuples))        

        self.angle_list = range(0, 36, 2)
    
        self.name = 'shapenet'
        self.load_size = (256, 256)
        self.angle = None
        self.shift = None
        self.scale = None
        self.hdf5_data = None
        self.img_size = (0,0)
        self.align_corner = opt.align_corner

    def transform_image(self, image, resize_param, method=Image.BILINEAR, affine=None, normalize=True, toTensor=True, fillWhiteColor=False):
        image = F.resize(image, resize_param, interpolation=method)
        if affine is not None:
            angle, translate, scale = affine['angle'], affine['shift'], affine['scale']
            fillcolor = (128,128,128) if not fillWhiteColor else (255,255,255)
            image = F.affine(image, angle=angle, translate=translate, scale=scale, shear=0, fillcolor=fillcolor)  
        if toTensor:
            image = F.to_tensor(image)
        if normalize:
            image = F.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        return image  


    def __len__(self):
        return self.size
    
    def get_random_source_ids(self, target_id, K):
        angles = list(self.angle_list)
        random.shuffle(angles)
        source_angles = angles[:K]
        id_base = target_id.split('_')[0]
        h = target_id.split('_')[-1]
        source_ids = ['_'.join([id_base, str(source_angles[i]), str(h)]) for i in range(K) ]
        return source_ids
    
    def __getitem__(self, index):
        if self.hdf5_data is None:
            # follow the solution at 
            # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
            # to sovle multi-thread read of HDF5 file 
            self.hdf5_data = h5py.File(self.hdf5_file, 'r')
        ref_xs = []
        ref_ys = []
        if self._is_train:
            target_id = self.image_ids[index]
            target_id = target_id.decode("utf-8") if isinstance(target_id, bytes) else target_id
            source_ids = self.get_random_source_ids(target_id, self._nb_inputs)

            
            for i in range(self._nb_inputs):

                P1_img = self.hdf5_data[source_ids[i]]['image'][()]
                P1_img = Image.fromarray(np.uint8(P1_img))
                P1 = self.transform_image(P1_img, self.load_size, fillWhiteColor=True)
                BP1 = torch.tensor(self.hdf5_data[source_ids[i]]['pose'][()])
                BP1_onehot = torch.zeros(len(self.angle_list)+3)
                BP1_onehot[ BP1[0]//2 ] = 1
                BP1_onehot[ len(self.angle_list)+BP1[1]//10 ] = 1
                BP1_onehot = BP1_onehot.view(-1,1,1) # to C*H*W
                ref_xs += [P1]
                ref_ys += [BP1_onehot]
            
            P2_img = self.hdf5_data[target_id]['image'][()]
            P2_img = Image.fromarray(np.uint8(P2_img))
            P2 = self.transform_image(P2_img, self.load_size, fillWhiteColor=True)
            BP2 = torch.tensor(self.hdf5_data[target_id]['pose'][()])
            BP2_onehot = torch.zeros(len(self.angle_list)+3)
            BP2_onehot[ BP2[0]//2 ] = 1
            BP2_onehot[ len(self.angle_list)+BP2[1]//10 ] = 1
            BP2_onehot = BP2_onehot.view(-1,1,1) # to C*H*W

        else:
            test_tuple = self.id_list[index]
            if isinstance(test_tuple[0], bytes):
                test_tuple = [id.decode("utf-8") for id in test_tuple]
            target_id = test_tuple[0]
            source_ids = test_tuple[1:self._nb_inputs+1]
            for id_source  in source_ids:
                P1_img = self.hdf5_data[id_source]['image'][()]
                P1_img = Image.fromarray(np.uint8(P1_img))
                P1 = self.transform_image(P1_img, self.load_size, fillWhiteColor=True)
                BP1 = torch.tensor(self.hdf5_data[id_source]['pose'][()])
                BP1_onehot = torch.zeros(len(self.angle_list)+3)
                BP1_onehot[ BP1[0]//2 ] = 1
                BP1_onehot[ len(self.angle_list)+BP1[1]//10 ] = 1
                BP1_onehot = BP1_onehot.view(-1,1,1) # to C*H*W
                ref_xs += [P1]
                ref_ys += [BP1_onehot]
            
            P2_img = self.hdf5_data[target_id]['image'][()]
            P2_img = Image.fromarray(np.uint8(P2_img))
            P2 = self.transform_image(P2_img, self.load_size, fillWhiteColor=True)
            BP2 = torch.tensor(self.hdf5_data[target_id]['pose'][()])
            BP2_onehot = torch.zeros(len(self.angle_list)+3)
            BP2_onehot[ BP2[0]//2 ] = 1
            BP2_onehot[ len(self.angle_list)+BP2[1]//10 ] = 1
            BP2_onehot = BP2_onehot.view(-1,1,1) # to C*H*W
            pass

        return {'ref_xs': ref_xs, 'ref_ys': ref_ys, 'g_x': P2, 'g_y': BP2_onehot, 
                'ref_ids': source_ids, 'g_id': target_id}

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



