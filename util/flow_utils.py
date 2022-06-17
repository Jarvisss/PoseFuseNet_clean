import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import cv2

# def flow2img(flow_data):
#     """
#     convert optical flow into color image
#     :param flow_data:
#     :return: color image
#     """
#     # print(flow_data.shape)
#     # print(type(flow_data))
#     u = flow_data[:, :, 0]
#     v = flow_data[:, :, 1]

#     UNKNOW_FLOW_THRESHOLD = 1e7
#     pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
#     pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
#     idx_unknown = (pr1 | pr2)
#     u[idx_unknown] = v[idx_unknown] = 0

#     # get max value in each direction
#     maxu = -999.
#     maxv = -999.
#     minu = 999.
#     minv = 999.
#     maxu = max(maxu, np.max(u))
#     maxv = max(maxv, np.max(v))
#     minu = min(minu, np.min(u))
#     minv = min(minv, np.min(v))

#     rad = np.sqrt(u ** 2 + v ** 2)
#     maxrad = max(-1, np.max(rad))
#     u = u / maxrad + np.finfo(float).eps
#     v = v / maxrad + np.finfo(float).eps

#     img = compute_color(u, v)

#     idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
#     img[idx] = 0

#     return np.uint8(img)

def flow2arrow(flow_data, arrow_step=(2,2)): 
    """
    convert optical flow into arrow image, flow is sample on [0,W-1] grid
    :param flow_data, sample_step
    :return: flow arrow image
    """
    h, w = flow_data.shape[0],flow_data.shape[1]
    u = flow_data[:, :, 0]
    v = flow_data[:, :, 1]
    arrow_img = np.ones((h,w,3)) * 255

    print(np.max(u), np.max(v))    
    for i in range(arrow_step[0]//2, h-arrow_step[0]//2, arrow_step[0]):
        for j in range(arrow_step[1]//2, w-arrow_step[1]//2, arrow_step[1]):
            arrow_img  = cv2.arrowedLine(arrow_img, (j, i), (j+int(u[i,j]), i+int(v[i,j])),line_type=cv2.LINE_AA, color=(0,0,0))

    return np.uint8(arrow_img)

class flow2img():
    # code from: https://github.com/tomrunia/OpticalFlow_Visualization
# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03
    def __init__(self):
        self.colorwheel = make_colorwheel()


    def flow_compute_color(self, u, v, convert_to_bgr=False):
        '''
        Applies the flow color wheel to (possibly clipped) flow components u and v.
        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun
        :param u: np.ndarray, input horizontal flow
        :param v: np.ndarray, input vertical flow
        :param convert_to_bgr: bool, whether to change ordering and output BGR instead of RGB
        :return:
        '''
        flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
        ncols = self.colorwheel.shape[0]

        rad = np.sqrt(np.square(u) + np.square(v))
        a = np.arctan2(-v, -u)/np.pi
        fk = (a+1) / 2*(ncols-1)
        k0 = np.floor(fk).astype(np.int32)
        k1 = k0 + 1
        k1[k1 == ncols] = 0
        f = fk - k0

        for i in range(self.colorwheel.shape[1]):

            tmp = self.colorwheel[:,i]
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1-f)*col0 + f*col1

            idx = (rad <= 1)
            col[idx]  = 1 - rad[idx] * (1-col[idx])
            col[~idx] = col[~idx] * 0.75   # out of range?

            # Note the 2-i => BGR instead of RGB
            ch_idx = 2-i if convert_to_bgr else i
            flow_image[:,:,ch_idx] = np.floor(255 * col)

        return flow_image


    def __call__(self, flow_uv, clip_flow=None, convert_to_bgr=False):
        '''
        Expects a two dimensional flow image of shape [H,W,2]
        According to the C++ source code of Daniel Scharstein
        According to the Matlab source code of Deqing Sun
        :param flow_uv: np.ndarray of shape [H,W,2]
        :param clip_flow: float, maximum clipping value for flow
        :return:
        '''
        if len(flow_uv.shape) != 3:
            flow_uv = flow_uv[0]
            flow_uv = flow_uv.permute(1,2,0).cpu().detach().numpy()    

        assert flow_uv.ndim == 3, 'input flow must have three dimensions'
        assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

        if clip_flow is not None:
            flow_uv = np.clip(flow_uv, 0, clip_flow)

        u = flow_uv[:,:,1]
        v = flow_uv[:,:,0]


        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)

        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        image = self.flow_compute_color(u, v, convert_to_bgr) 
        # image = torch.tensor(image).float().permute(2,0,1)/255.0 * 2 - 1
        return image

def make_colorwheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel