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

def make_dataset(opt, sub_dataset_model='kitti'):
    """Create dataset"""
    root = opt.path_to_dataset
    test_id_file = os.path.join(root, 'id_%s.txt'%sub_dataset_model) if opt.phase == 'test' else None
    dataset = SceneDataset(
            phase = opt.phase,
            image_id_file=os.path.join(root, 'id_%s_%s.txt' %(sub_dataset_model,opt.phase)), 
            test_id_file = test_id_file,
            hdf5_file = os.path.join(root, 'data_%s.hdf5'%sub_dataset_model),
            opt=opt)
    
    return dataset

class SceneDataset(data.Dataset):
    """
    @ phase
        train | test | val
    @ image_id_file
        the path to fasion-x_tuples-train.csv
    @ test_id_file
        the path to fasion-x_tuples-test.csv
    @ hdf5_file
        the path to train image keypoint annotations
    @ path_to_test_anno
        the path to test image keypoint annotations
    """
    def __init__(self, phase,  image_id_file, test_id_file,
    hdf5_file,opt):
        super(SceneDataset, self).__init__()
        
        self._is_train = phase == 'train'
        self._is_eval = phase=='eval'
        self.image_id_file = image_id_file
        
        self.hdf5_file = hdf5_file
        self.image_ids = np.genfromtxt(self.image_id_file, dtype=np.str)
        self.size = len(self.image_ids)

        if not self._is_train:
            with open(test_id_file, 'r') as test_id_list_path:
                test_id_list = test_id_list_path.readlines()
            self.test_id_list = [ids.strip().split(' ') for ids in test_id_list]
            
            print('test tuples : %d' % len(self.test_id_list))
            self.size = len(self.test_id_list)
            
        self._nb_inputs = opt.K

        print ("--------------- Dataset Info: ---------------")
        print ("Phase: %s" % phase)
        print ("Number of samples %s: %s" % (phase, str(self.size)))
        # self._test_tuples = pd.read_csv(path_to_train_tuples)
        # self._test_anno = pd.read_csv(path_to_test_anno, sep=':')
        # self._test_img_dir = path_to_train_imgs_dir
        # print ("Number of tuples test: %s" % len(self._test_tuples))        

        self.name = 'shapenet'
        self.load_size = (256, 256)
        self.angle = None
        self.shift = None
        self.scale = None
        self.hdf5_data = None
        self.img_size = (0,0)
        self.bound = 10
        self.num_digit = 4
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
    
    def get_random_source_ids(self, target_id, K=2, bound=10, num_digit=4):
        # angles = list(self.angle_list)
        decks = list(range(-bound, bound))
        random.shuffle(decks)
        id_num = int(target_id[-num_digit:])
        source_ids = []
        valid_count = 0
        for i in range(len(decks)):
            if valid_count < K:
                source_id = target_id[:-num_digit] + str(id_num + decks[i]).zfill(num_digit)
                if source_id in self.hdf5_data:
                    source_ids.append(source_id)
                    valid_count+=1
        return source_ids
    
    # @profile
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
                BP1 = BP1.view(-1,1,1) # to C*H*W, in this case [6,1,1]
                ref_xs += [P1]
                ref_ys += [BP1]
            
            P2_img = self.hdf5_data[target_id]['image'][()]
            P2_img = Image.fromarray(np.uint8(P2_img))
            P2 = self.transform_image(P2_img, self.load_size, fillWhiteColor=True)
            BP2 = torch.tensor(self.hdf5_data[target_id]['pose'][()])
            BP2 = BP2.view(-1,1,1) # to C*H*W

        else:
            test_tuple = self.test_id_list[index][:1+self._nb_inputs]
            if isinstance(test_tuple[0], bytes):
                test_tuple = [id.decode("utf-8") for id in test_tuple]
            target_id = test_tuple[0]
            source_ids = test_tuple[1:]
            for id_source  in source_ids:
                P1_img = self.hdf5_data[id_source]['image'][()]
                P1_img = Image.fromarray(np.uint8(P1_img))
                P1 = self.transform_image(P1_img, self.load_size, fillWhiteColor=True)
                BP1 = torch.tensor(self.hdf5_data[id_source]['pose'][()])
                BP1 = BP1.view(-1,1,1) # to C*H*W
                ref_xs += [P1]
                ref_ys += [BP1]
            
            P2_img = self.hdf5_data[target_id]['image'][()]
            P2_img = Image.fromarray(np.uint8(P2_img))
            P2 = self.transform_image(P2_img, self.load_size, fillWhiteColor=True)
            BP2 = torch.tensor(self.hdf5_data[target_id]['pose'][()])
            BP2 = BP2.view(-1,1,1) # to C*H*W
            pass

        return {'ref_xs': ref_xs, 'ref_ys': ref_ys, 'g_x': P2, 'g_y': BP2, 
                'ref_ids': source_ids, 'g_id': target_id}




