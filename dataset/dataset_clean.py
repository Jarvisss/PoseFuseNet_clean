'''
这个文件用于数据清洗，得到train_pair, test_pair
'''

import numpy as np
import pandas as pd
import os

root_dir = '/dataset/ljw/danceFashoin/'
train_dir = root_dir + 'train_256/' 
test_dir = root_dir + 'test_256/'

COCO_17 =  { "Nose":0, "LEye":1, "REye":2, "LEar":3, "REar":4, "LShoulder":5, 
             "RShoulder":6, "LElbow":7, "RElbow":8, "LWrist":9, "RWrist":10, "LHip":11, 
             "RHip":12, "LKnee":13, "RKnee":14, "LAnkle":15, "RAnkle":16} 

LIMB_SEQ_COCO_17 = [[0,1],[1,3],[0,2],[2,4],[5,7],[7,9],
                        [6,8],[8,10],[11,12],[5,6],[11,13],[12,14],
                        [13,15],[14,16],[5,11],[6,12]]  


def openpose18_to_coco17(pose_joints_18):
    pose_joints_17 = np.zeros((2,17)).astype(pose_joints_18.dtype)
    i = 0
    for key in COCO_17:
        pose_joints_17[:, i] = pose_joints_18[:, OPENPOSE_18[key]]
        i = i+1
    return pose_joints_17  

def obtain_2d_cords(B_coor, resize_param=None, org_size=None, affine=None):
    pose = B_coor["pose_keypoints_2d"]
    coor_x = [pose[3*i]   for i in range(int(len(pose)/3))]
    coor_y = [pose[3*i+1] for i in range(int(len(pose)/3))]
    score  = [pose[3*i+2] for i in range(int(len(pose)/3))]
    coor_body = np.array([coor_y, coor_x])

    return coor_body

def load_skeleton(path):
    B_coor = json.load(open(path))["people"]
    B_coor = B_coor[0]
    pose_body = obtain_2d_cords(B_coor) #[2, 18]
    pose_body = openpose_utils.openpose18_to_coco17(pose_body) #np(2,17)

    return pose_body

def load_parsing(parsing_path):
    parsing_img = Image.open(parsing_path)
    # [H W 1 ] -> [H W 20]
    parsing_bin_map = np.stack((parsing_img,)*20, axis=2)
    # to 20 channel binary map
    for i in range(20):
        parsing_bin_map[:,:, i] = (parsing_img == i).astype(np.uint8) * 255 
    
    return parsing_bin_map # np(H,W,20)

# # 骨骼位于衣服parsing内，可能是遮挡。
# 怎么判断：骨骼连线位置的像素，如果是衣服类别，则不遮挡，如果是手、腿、皮肤的类别，则是遮挡
def occlude(kp_path, parsing_path):
    kp = load_skeleton(kp_path) # [2,17]
    parse = load_parsing(parsing_path) # [H,W,20]

    


    return True


# 得到有遮挡的数据
def get_occlude_data(train_test='train'):
    
    train_test_dir = train_dir if train_test == 'train' else test_dir
    
    person_ids = os.listdir(os.path.join(train_test_dir, 'train_A'))
    
    for p in person_ids:
        p_kps_names = os.listdir(os.path.join(train_test_dir, 'train_alphapose', p))
        parsings = os.listdir(os.path.join(train_test_dir, 'parsing_A', p))
        p_parsing_names = []
        for p in parsings:
            p_parsing_names += [p] if p.endswith('.png') and not p.endswith('vis.png')
        
        assert(len(p_kps_names) == len(p_parsing_names))





    pass



def clean():
    pass