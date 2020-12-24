import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import disk, line_aa, polygon
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import skimage.measure, skimage.transform
import sys

LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]


LIMB_SEQ_HUMAN36M_17 = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],
                        [0,7],[7,8],[8,9],[9,10],[8,11],[11,12],
                        [12,13],[8,14],[14,15],[15,16]]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def labelcolormap(N):
    if N == 18: # CelebAMask-HQ
        cmap = np.array([[255, 0, 0],   [255, 85, 0], [255, 170, 0], [255, 255, 0], 
                         [170, 255, 0], [85, 255, 0], [0, 255, 0],
                         [0, 255, 85], [0, 255, 170], [0, 255, 255], 
                         [0, 170, 255], [0, 85, 255], [0, 0, 255],   [85, 0, 255],
                         [170, 0, 255], [255, 0, 255],[255, 0, 170], [255, 0, 85]],
                         dtype=np.uint8) 
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']

MISSING_VALUE = -1


def map_to_cord(pose_map, threshold=0.1):
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[..., :18]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

def resize_cords(cords, img_size, old_size=None, affine_matrix=None):
    old_size = img_size if old_size is None else old_size
    output = cords.copy()
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        point[0] = point[0]/old_size[0] * img_size[0]
        point[1] = point[1]/old_size[1] * img_size[1]
        if affine_matrix is not None:
            point_ =np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3,1))
            point[0] = int(point_[1])
            point[1] = int(point_[0])
        else:
            point[0] = int(point[0])
            point[1] = int(point[1])
        output[i] = point
    return output
        
# def cords_to_map(cords,img_size, sigma=6):
#     result = np.zeros(img_size + cords.shape[0:1], dtype='float32') # [256,256,18]
#     xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
#     for i, point in enumerate(cords):
#         result[..., i] = np.exp(-((yy - int(point[0])) ** 2 + (xx - int(point[1])) ** 2) / (2 * sigma ** 2))
#     return result


def cords_to_map(cords, img_size, old_size=None, affine_matrix=None, sigma=6):
    old_size = img_size if old_size is None else old_size
    cords = cords.astype(float)
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))

    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        point[0] = point[0]/old_size[0] * img_size[0]
        point[1] = point[1]/old_size[1] * img_size[1]
        if affine_matrix is not None:
            point_ =np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3,1))
            point_0 = int(point_[1])
            point_1 = int(point_[0])
        else:
            point_0 = int(point[0])
            point_1 = int(point[1])
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result


def draw_pose_from_cords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = disk((joint[0], joint[1]), radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def draw_pose_from_map(pose_map, threshold=0.1, **kwargs):
    cords = map_to_cord(pose_map, threshold=threshold)
    return draw_pose_from_cords(cords, pose_map.shape[:2], **kwargs)


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def mean_inputation(X):
    X = X.copy()
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            val = np.mean(X[:, i, j][X[:, i, j] != -1])
            X[:, i, j][X[:, i, j] == -1] = val
    return X

def draw_legend():
    handles = [mpatches.Patch(color=np.array(color) / 255.0, label=name) for color, name in zip(COLORS, LABELS)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def get_scale_trans(source_kp:np.ndarray, target_kp:np.ndarray):

    '''
    get scale and translation from source to target keypoint,
    use 4 points to compute
    '''
    KP = [2,5,8,11] # rs, ls, rh, lh
    source_central = np.average(source_kp[:,KP],axis=1)
    target_central = np.average(target_kp[:,KP],axis=1)
    source_lenth = np.average(np.linalg.norm((source_kp[:,KP] - source_central[:,np.newaxis]),axis=0))
    target_lenth = np.average(np.linalg.norm((target_kp[:,KP] - target_central[:,np.newaxis]),axis=0))
    scale = target_lenth / (source_lenth + np.finfo(float).eps)

    # print('scale: ',scale)

    center = np.array([(175+0)/2, (256+0)/2]) # the center of image
    
    trans = (target_kp[:,1] - center)/scale + center - source_kp[:,1]
    # print('trans: ',trans)
    return scale, trans

def get_dot_sim(source:np.ndarray, target:np.ndarray, norm_value:float, beta1:float, beta2:float):
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

if __name__ == "__main__":
    import pandas as pd
    from skimage.io import imread
    import pylab as plt
    import os
    import json
    # df = pd.read_csv('data/market-annotation-train.csv', sep=':')

    # for index, row in df.iterrows():
    #     pose_cords = load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])

    #     colors, mask = draw_pose_from_cords(pose_cords, (128, 64))

    #     mmm = produce_ma_mask(pose_cords, (128, 64)).astype(float)[..., np.newaxis].repeat(3, axis=-1)
    #     print(mmm.shape)
    #     img = imread('data/market-dataset/train/' + row['name'])

    #     mmm[mask] = colors[mask]

    #     print (mmm)
    #     plt.subplot(1, 1, 1)
    #     plt.imshow(mmm)
    #     plt.show()
    
    # fp = '../dataset/danceFashion/test_256/train_video2d/91-3003CN5S/00056.json'
    # img_fp = '../dataset/danceFashion/test_256/train_A/91-3003CN5S/00056.png'
    
    kpdir = '../dataset/danceFashion/test_256/train_video2d/'
    outimgdir = '../dataset/danceFashion/test_256/train_video2d_img/'
    if not os.path.isdir(outimgdir):
        os.mkdir(outimgdir)
    vnames = os.listdir(kpdir)
    for v in vnames:
        print(v)
        vp = os.path.join(kpdir,v)
        outp = os.path.join(outimgdir, v)
        if not os.path.isdir(outp):
            os.mkdir(outp)

        fnames = os.listdir(vp)
        for f in fnames:
            fp = os.path.join(vp, f)
            outfp = os.path.join(outp, f.replace('.json','.png'))

            kp = json.load(open(fp))['people'][0]
            kp3d = kp['pose_keypoints_2d']
            kp2d = np.array([[0, 0]] * (len(kp3d)//3)) 

            for i in range(len(kp3d)//3):
                kp2d[i][0] = int(kp3d[i*3+1]+0.5)
                kp2d[i][1] = int(kp3d[i*3+0]+0.5)

            # print(kp2d)

            img_size = (256, 256)
            colors, mask = draw_pose_from_cords(kp2d, img_size)
            colors = colors / 255

            # img = plt.imread(img_fp)

            # save = np.vstack((img,colors))
            # print(save.shape)
            # plt.subplot()
            plt.imsave(outfp,colors)
            # plt.show()
            # print(mask.shape)

    # kp = json.load(open(fp))['people'][0]
    # kp3d = kp['pose_keypoints_2d']
    # kp2d = np.array([[0, 0]] * (len(kp3d)//3)) 

    # for i in range(len(kp3d)//3):
    #     kp2d[i][0] = int(kp3d[i*3+1]+0.5)
    #     kp2d[i][1] = int(kp3d[i*3+0]+0.5)

    # # print(kp2d)

    # img_size = (256, 256)
    # colors, mask = draw_pose_from_cords(kp2d, img_size)
    # colors = colors / 255

    # # img = plt.imread(img_fp)

    # # save = np.vstack((img,colors))
    # # print(save.shape)
    # # plt.subplot()
    # plt.imsave('aa.png',colors)
    # plt.show()
    # print(mask.shape)