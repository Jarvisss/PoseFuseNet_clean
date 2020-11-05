import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from util.flow_utils import flow2img
from itertools import product
from time import time

'''
Visualize the feature (channel >= 3)
'''
def visualize_feature(feat_tensor, batch, out_shape):
    feat_tensor = F.interpolate(feat_tensor, size=out_shape, mode='bilinear',align_corners=False)[batch].permute(1,2,0)
    visualize_feat_tensor = feat_tensor[...,0:3]
    visualize_feat_tensor = (visualize_feat_tensor - torch.min(visualize_feat_tensor))* 255.0/(torch.max(visualize_feat_tensor) - torch.min(visualize_feat_tensor))
    return visualize_feat_tensor


'''
Visualize the feature group(N features with channel >= 3)
'''
def visualize_feature_group(feat_tensor_group, batch, out_shape):
    feat_min = 1e5
    feat_max = -1e5
    visualize_feature_list = None
    for feat_tensor in feat_tensor_group:
        feat_tensor = F.interpolate(feat_tensor, size=out_shape, mode='bilinear',align_corners=False)[batch].permute(1,2,0)
        feat_tensor = feat_tensor[...,0:3]

        feat_min = feat_tensor.min() if feat_min > feat_tensor.min() else feat_min
        feat_max = feat_tensor.max() if feat_max < feat_tensor.max() else feat_max

    for feat_tensor in feat_tensor_group:
        feat_tensor = F.interpolate(feat_tensor, size=out_shape, mode='bilinear',align_corners=False)[batch].permute(1,2,0)
        visualize_feat_tensor = feat_tensor[...,0:3]
        visualize_feat_tensor = (visualize_feat_tensor - feat_min)* 255.0/(feat_max - feat_min)
        if visualize_feature_list is None:
            visualize_feature_list = [visualize_feat_tensor]  
        else:
            visualize_feature_list.append(visualize_feat_tensor)
        

    return visualize_feature_list

'''
parsing with shape [B, 20, H, W]
'''
def visualize_parsing(parsing, batch):
    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    visual_batch = parsing[batch]
    c,h,w = visual_batch.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for cc in range(c):
        vc = visual_batch[cc] #[h, w]
        output[vc>0.5] = label_colours[cc]
    
    return output

'''
parsing with shape [B, 20, H, W]
'''
def visualize_cloth_parsing(parsing, batch):
    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    visual_batch = parsing[batch]
    c,h,w = visual_batch.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for cc in range(c):
        vc = visual_batch[cc] #[h, w]
        output[vc>0.5] = label_colours[cc+2]
    
    return output

'''
parsing with shape [B, 20, H, W]
'''
def visualize_gt_cloth_parsing(parsing, batch):
    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    visual_batch = parsing[batch]
    c,h,w = visual_batch.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for cc in range(c):
        vc = visual_batch[cc] #[h, w]
        output[vc>0.5] = label_colours[cc+1]
    
    return output

'''
parsing with shape [B, 20, H, W]
'''
def visualize_merge_cloth_parsing(parsing,parsing2, batch):
    label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]
    visual_batch = parsing[batch]
    visual_batch2 = parsing2[batch]
    c,h,w = visual_batch.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    for cc in range(c):
        vc = visual_batch[cc] #[h, w]
        v2c = visual_batch2[cc] #[h, w]
        intersect = (vc>0.5) & (v2c>0.5)
        gonly = (vc>0.5) & (v2c<=0.5)
        ronly = (vc<=0.5) & (v2c>0.5)
        output[intersect] = label_colours[cc+3]
        output[gonly] = label_colours[cc+1]
        output[ronly] = label_colours[cc+2]
    
    return output

def get_parse_visual_result(opt, ref_xs, ref_ys,ref_ps,gx, gy, gp, p_hat_bin_map):
    cpu = torch.device("cpu")
    device = torch.device("cuda:0")
    DISPLAY_BATCH = 0

    K = len(ref_xs)
    assert(len(ref_ys)==K)
    assert(len(ref_ps)==K)
    
    g_x = gx[DISPLAY_BATCH].permute(1,2,0) * 255.0
    g_y = gy[DISPLAY_BATCH][17:20,...].permute(1,2,0)* 255.0
    g_p = gp.to(cpu).numpy()
    p_hat_bin_map = p_hat_bin_map.to(cpu).numpy()
    white = torch.ones(g_x.size()).to(device) * 255.0

    visual_ref_xs = ref_xs.copy()
    visual_ref_ys = ref_xs.copy()
    visual_ref_ps = ref_xs.copy()
    ref = ref_xs.copy()


    img_shape = white.shape[0:2]
    visual_gp = visualize_parsing(g_p, DISPLAY_BATCH)
    visual_gp = torch.from_numpy(visual_gp).to(device).float()

    visual_p_out = visualize_parsing(p_hat_bin_map, DISPLAY_BATCH)
    visual_p_out = torch.from_numpy(visual_p_out).to(device).float()

    for i in range(K):
        visual_ref_xs[i] = ref_xs[i][DISPLAY_BATCH].permute(1,2,0) * 255.0
        visual_ref_ys[i] = ref_ys[i][DISPLAY_BATCH][17:20].permute(1,2,0) * 255.0
        visual_ref_ps[i] = visualize_parsing(ref_ps[i].to(cpu).numpy(), DISPLAY_BATCH)
        visual_ref_ps[i] = torch.from_numpy(visual_ref_ps[i]).to(device).float()

    '''Each col of result image'''    
    for i in range(K):
        ref[i] = torch.cat((visual_ref_xs[i], visual_ref_ys[i], visual_ref_ps[i]), dim=0)

    refs = torch.cat(ref, dim=1)

    out_col = torch.cat((g_x, g_y, visual_p_out),dim=0)
    gt = torch.cat((g_x, g_y, visual_gp), dim=0)

    final_img = torch.cat((refs, out_col, gt), dim=1)
    # print(simp_img.shape)
    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    return final_img


def get_visualize_result(opt, ref_xs, ref_ys, gx, gy, gp, gp_hat, xf_merge, x_hat, flows, masks_normed, features, features_warped, features_warped_masked ):
    cpu = torch.device("cpu")
    device = torch.device("cuda:0")
    DISPLAY_BATCH = 0

    K = len(ref_xs)
    assert(len(ref_ys)==K)
    assert(masks_normed.shape[1]==K)
    assert(len(flows)==K)
    assert(len(features)==K)
    assert(len(features_warped)==K)
    assert(len(features_warped_masked)==K)

    g_x = gx[DISPLAY_BATCH].permute(1,2,0) * 255.0
    g_y = gy[DISPLAY_BATCH][17:20,...].permute(1,2,0)* 255.0
    out = x_hat[DISPLAY_BATCH].permute(1,2,0) * 255.0 # (256,256,3)
    
    white = torch.ones(out.size()).to(device) * 255.0

    masks = flows.copy()
    visualize_feat = flows.copy()
    visualize_feat_warp = flows.copy()
    ref = flows.copy()
    feat = flows.copy()
    visual_ref_xs = flows.copy()
    visual_ref_ys = flows.copy()


    img_shape = out.shape[0:2]
    if gp is not None:
        g_p = gp.to(cpu).numpy()
        visual_gp = visualize_parsing(g_p, DISPLAY_BATCH)
        visual_gp = torch.from_numpy(visual_gp).to(device).float()
    else:
        visual_gp = white

    if gp_hat is not None:
        g_phat = gp_hat.to(cpu).numpy()
        visual_gphat = visualize_parsing(g_phat, DISPLAY_BATCH)
        visual_gphat = torch.from_numpy(visual_gphat).to(device).float()
    else:
        visual_gphat = white

    features_warped_masked.append(xf_merge)
    visualize_feat_warp_masked = visualize_feature_group(features_warped_masked, DISPLAY_BATCH, out_shape=img_shape)
    visualize_feat_merged = visualize_feat_warp_masked[-1]
    visualize_feat_warp_masked = visualize_feat_warp_masked[:-1]
    for i in range(K):
        visual_ref_xs[i] = ref_xs[i][DISPLAY_BATCH].permute(1,2,0) * 255.0
        visual_ref_ys[i] = ref_ys[i][DISPLAY_BATCH][17:20].permute(1,2,0) * 255.0
        flows[i] =  torch.from_numpy(flow2img(flows[i][DISPLAY_BATCH].permute(1,2,0).detach().to(cpu).numpy())).to(device).float()
        masks[i] = masks_normed[:,i:i+1,...][DISPLAY_BATCH].permute(1,2,0)
        masks[i] = torch.cat((masks[i],)*3, dim=2) * 255.0
        visualize_feat[i] = visualize_feature(features[i], DISPLAY_BATCH, out_shape=img_shape)
        visualize_feat_warp[i] = visualize_feature(features_warped[i], DISPLAY_BATCH, out_shape=img_shape)

    '''Each col of result image'''    
    for i in range(K):
        ref[i] = torch.cat((visual_ref_xs[i], visual_ref_ys[i], visualize_feat[i]), dim=0)
        feat[i] = torch.cat((visualize_feat_warp[i], masks[i], visualize_feat_warp_masked[i]), dim=0)

    

    refs = torch.cat(ref, dim=1)
    feats = torch.cat(feat, dim=1)

    out_col = torch.cat((out, visualize_feat_merged, visual_gphat),dim=0)
    gt = torch.cat((g_x, g_y, visual_gp), dim=0)

    final_img = torch.cat((refs, feats, out_col,gt), dim=1)
    simp_img = torch.cat((final_img[0:256,0:256*K], final_img[0:256,-2*256:]),dim=1)
    # print(simp_img.shape)
    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    simp_img = simp_img.type(torch.uint8).to(cpu).numpy()

    # out = np.transpose(out, (1, 0, 2))
    return final_img,simp_img


