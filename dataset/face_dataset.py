import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import cv2
import torchvision.transforms.functional as F
import random
from PIL import Image

random.seed(7)

def make_dataset(opt):
    path_to_test_tuples = '/dataset/ljw/voxceleb2/test_target_%dshot.txt' % opt.K if opt.phase == 'test' else None
    dataset = FaceDataset(
        phase = opt.phase,
        total_K = 8,
        K = opt.K, 
        path_to_preprocess=opt.path_to_dataset,
        path_to_test_tuples=path_to_test_tuples)
    
    return dataset

def transform_image(image, resize_param, method=Image.BILINEAR, affine=None, normalize=True, toTensor=True, fillWhiteColor=False):
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

def select_preprocess_frames(frames_path, K, size=(224,224)):
    img = np.array(Image.open(frames_path))
    # img = cv2.imread(frames_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    images_list =  [transform_image(Image.fromarray(np.uint8(img[:, i*224:(i+1)*224, :])), size) for i in range(K)]
    landmark_list = [transform_image(Image.fromarray(np.uint8(img[:, i*224:(i+1)*224, :])), size) for i in range(K,2*K)]
    
    return images_list, landmark_list
    

class FaceDataset(data.Dataset):
    def __init__(self,phase, total_K, K, path_to_preprocess, path_to_test_tuples=None):
        self.is_train = True if phase == 'train' else False
        self.total_K = total_K
        self.K = K
        self.path_to_preprocess = path_to_preprocess
        self.person_id_list = os.listdir(path_to_preprocess)
        self.test_tuples = None
        if path_to_test_tuples is not None and not self.is_train:
            with open(path_to_test_tuples) as f:

                self.test_tuples = f.readlines()
    def __len__(self):
        vid_num = 0
        for person_id in self.person_id_list:
            vid_num += len(os.listdir(os.path.join(self.path_to_preprocess, person_id)))
        
        return vid_num - 1


        
    def __getitem__(self, index):
        path = os.path.join(self.path_to_preprocess,
                            str(index//256),
                            str(index)+".png")

        im, mark = select_preprocess_frames(path, self.total_K)
        # frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3 / value ~ [0,255]
        # frame_mark = frame_mark.transpose(2,4)/255 #K,2,3,224,224 / value ~ [0,1]

        # g_idx = torch.randint(low = 0, high = self.total_K, size = (1,1))
        if self.test_tuples is None:
            g_idx = random.randint(0, self.total_K-1)
            deck = list(set(range(self.total_K)) - {g_idx})
            random.shuffle(deck)

        else:
            g_idx = int(self.test_tuples[index].strip().split()[-1])
            deck = [int(i) for i in (self.test_tuples[index].strip().split()[:-1])]
        

        ref_xs = [im[deck[i]] for i in range(self.K)]
        ref_ys = [mark[deck[i]] for i in range(self.K)]

        g_x = im[g_idx].squeeze()
        g_y = mark[g_idx].squeeze()

        return {'ref_xs':ref_xs, 'ref_ys':ref_ys, 'g_x':g_x, 'g_y':g_y}

path_to_preprocess = '/dataset/ljw/voxceleb2/preprocess_test_k8'
person_id_list = os.listdir(path_to_preprocess)
vid_num = 0
for person_id in person_id_list:
    vid_num += len(os.listdir(os.path.join(path_to_preprocess, person_id)))
print(vid_num)
random.seed(7)
k=2
with open('/dataset/ljw/voxceleb2/test_target_%dshot.txt'%k, 'w') as f:
    for index in range(vid_num):
        path = os.path.join(path_to_preprocess,
                                str(index//256),
                                str(index)+".png")
        g_idx = random.randint(0, 7)
        sources = list(set(range(0,8)) - {g_idx})
        random.shuffle(sources)
        source_ids = sources[:k]
        for source_id in source_ids:
            f.write(str(source_id)+' ')
        f.write(str(g_idx)+'\r\n')