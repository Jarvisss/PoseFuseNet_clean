import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
from util.flow_utils import flow2img
from itertools import product
from time import time
from model.blocks import warp_flow

'''
Visualize the feature (channel >= 3)
'''
def visualize_feature(feat_tensor, batch, out_shape):
    feat_tensor = F.interpolate(feat_tensor, size=out_shape, mode='bilinear',align_corners=True)[batch].permute(1,2,0)
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
        feat_tensor = F.interpolate(feat_tensor, size=out_shape, mode='bilinear',align_corners=True)[batch].permute(1,2,0)
        feat_tensor = feat_tensor[...,0:3]

        feat_min = feat_tensor.min() if feat_min > feat_tensor.min() else feat_min
        feat_max = feat_tensor.max() if feat_max < feat_tensor.max() else feat_max

    for feat_tensor in feat_tensor_group:
        feat_tensor = F.interpolate(feat_tensor, size=out_shape, mode='bilinear',align_corners=True)[batch].permute(1,2,0)
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

#-1 1 to 0 255
def to_image(tensor):
    return (tensor+1)/2*255

def get_visualize_result(opt, ref_xs, ref_ys,ref_ps, gx, gy, gp, gp_hat, xf_merge, x_hat, flows, masks_normed, features, features_warped, features_warped_masked ):
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

    g_x = to_image(gx[DISPLAY_BATCH].permute(1,2,0))
    g_y = gy[DISPLAY_BATCH][-3:,...].permute(1,2,0)* 255.0
    out = to_image(x_hat[DISPLAY_BATCH].permute(1,2,0)) # (256,256,3)
    
    white = torch.ones(out.size()).to(device) * 255.0

    masks = flows.copy()
    visualize_feat = flows.copy()
    visualize_feat_warp = flows.copy()
    ref = flows.copy()
    feat = flows.copy()
    visual_ref_xs = flows.copy()
    visual_ref_ys = flows.copy()
    visual_ref_ps = flows.copy()
    visual_flows = flows.copy()

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
        visual_ref_xs[i] = to_image(ref_xs[i][DISPLAY_BATCH].permute(1,2,0))
        visual_ref_ys[i] = ref_ys[i][DISPLAY_BATCH][-3:].permute(1,2,0) * 255.0
        if ref_ps is not None:
            visual_ref_ps[i] = visualize_parsing(ref_ps[i].to(cpu).numpy(), DISPLAY_BATCH)
            visual_ref_ps[i] = torch.from_numpy(visual_ref_ps[i]).to(device).float()
        else:
            visual_ref_ps[i] = white

        visual_flows[i] =  torch.from_numpy(flow2img(flows[i][DISPLAY_BATCH].permute(1,2,0).detach().to(cpu).numpy())).to(device).float()
        masks[i] = masks_normed[:,i:i+1,...][DISPLAY_BATCH].permute(1,2,0)
        masks[i] = torch.cat((masks[i],)*3, dim=2) * 255.0
        visualize_feat[i] = visualize_feature(features[i], DISPLAY_BATCH, out_shape=img_shape)
        visualize_feat_warp[i] = visualize_feature(features_warped[i], DISPLAY_BATCH, out_shape=img_shape)


    '''Each col of result image'''    
    for i in range(K):
        ref[i] = torch.cat((visual_ref_xs[i], visual_ref_ys[i], visualize_feat[i]), dim=0)
        feat[i] = torch.cat((visualize_feat_warp[i], masks[i], visualize_feat_warp_masked[i]), dim=0)

    

    refs = torch.cat(ref, dim=1)
    ps = torch.cat(visual_ref_ps, dim=1)
    feats = torch.cat(feat, dim=1)

    out_col = torch.cat((out, visualize_feat_merged, visual_gphat),dim=0)
    gt = torch.cat((g_x, g_y, visual_gp), dim=0)

    final_img = torch.cat((refs, feats, out_col,gt), dim=1)
    simp_img = torch.cat((final_img[0:256,0:256*K], final_img[0:256,-2*256:]),dim=1)
    attns = torch.cat((final_img[256:512,256*K:256*K*2], white,white ),dim=1)
    ys = torch.cat((final_img[256:512,0:256*K],g_y,g_y), dim=1)
    ps = torch.cat((ps, visual_gphat,visual_gp ),dim=1)
    parsings = torch.cat((final_img[256:512,256*K:256*K*2], white,white ),dim=1)
    mid_img = torch.cat((simp_img, attns, ys, ps), dim=0)

    # print(simp_img.shape)
    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    simp_img = simp_img.type(torch.uint8).to(cpu).numpy()
    mid_img = mid_img.type(torch.uint8).to(cpu).numpy()
    # out = np.transpose(out, (1, 0, 2))
    return final_img,simp_img, mid_img




def tensor2im(x, display_batch=0, is_parsing=False,is_mask=False, out_size=(256,256)):
    '''
    take 4-d tensor as input, [B C H W]
    output 3-d image tensor, range(0,255), [H W C]
    output white image if tensor is None
    '''
    
    device = torch.device("cuda:0")
    cpu = torch.device("cpu")
    img = torch.ones((out_size[0],out_size[1],3)).to(device) * 255.0
    if x is None:
        return img
    tensor = x.clone()
    assert len(tensor.shape)==4
    channel = tensor.shape[1]
    if channel == 3:
        # image
        tensor = F.interpolate(tensor, size=out_size, mode='bilinear',align_corners=True)
        img = tensor[display_batch].permute(1,2,0)
        img = to_image(img)
    elif channel == 21:
        # bone
        img = tensor[display_batch][-3:,...].permute(1,2,0) * 255.0
    elif channel == 1 and is_parsing:
        tensor = tensor.to(cpu).numpy()
        img = visualize_parsing(tensor, display_batch)
        img = torch.from_numpy(img).to(device).float()
    elif channel == 1 and is_mask:
        tensor = F.interpolate(tensor, size=out_size, mode='bilinear',align_corners=True)
        img = tensor[display_batch].permute(1,2,0)
        img = torch.cat((img,)*3, dim=2) * 255.0
    elif channel >= 32:
        # feature map
        img = visualize_feature(tensor, display_batch, out_size)
    else:
        print('Unknown Shape:',x.shape)

    return img


def get_pyramid_visualize_result(opt, ref_xs, ref_ys, ref_ps, gx, x_hat,  gy, gp, gp_hat, flows, masks_normed,occlusions, ref_features, g_features):
    '''
    ref_xs: K[HW]
    ref_ys: K[HW]
    ref_ps: K[HW]
    gx: [HW]
    xhat: [HW]
    gy: [HW]
    gp: [HW]
    gphat: [HW]
    flows: KL[H`W`]
    masks_normed: KL[H`W`]
    ref_features: KL[H`W`]
    g_features: L[H`W`]
    '''

    cpu = torch.device("cpu")
    device = torch.device("cuda:0")
    DISPLAY_BATCH = 0

    K = len(ref_xs)
    assert(len(ref_ys)==K)
    assert(len(masks_normed)==K)
    assert(len(flows)==K)
    assert(len(ref_features)==K)

    layers = len(flows[0])
    # print(layers)
    # print(len(masks_normed[0]))
    # print(len(ref_features[0]))
    # print(len(g_features))
    assert(len(masks_normed[0])==layers)
    assert(len(ref_features[0])==layers)
    assert(len(g_features)==layers)
    out_size = ref_xs[0].shape[2:]

    rows = (3 + layers * 2)*out_size[0]
    cols = (2 * K + 2)*out_size[1]

    white = tensor2im(None, out_size=out_size)
    visual_g_x = tensor2im(gx, out_size=out_size)
    visual_g_y = tensor2im(gy, out_size=out_size)
    visual_out = tensor2im(x_hat, out_size=out_size)
    
    visual_gp = tensor2im(gp, is_parsing=True, out_size=out_size) 
    visual_gphat = tensor2im(gp_hat, is_parsing=True, out_size=out_size) 
    visual_ref_xs = [0] * K
    visual_ref_ys = [0] * K
    visual_ref_ps = [0] * K
    for i in range(K):
        visual_ref_xs[i] = tensor2im(ref_xs[i], out_size=out_size)
        visual_ref_ys[i] = tensor2im(ref_ys[i], out_size=out_size)
        if ref_ps is None:
            visual_ref_ps[i] = tensor2im(None, is_parsing=True, out_size=out_size)
        else:
            visual_ref_ps[i] = tensor2im(ref_ps[i], is_parsing=True, out_size=out_size)
    
    visual_ref_xs_tensor = torch.cat(visual_ref_xs, dim=1)
    simp_img = torch.cat((visual_ref_xs_tensor,visual_out,visual_g_x),dim=1).type(torch.uint8).to(cpu).numpy()
    if not opt.output_all and not opt.phase=='train':
        return None, simp_img, None

    visual_feats = [0] * K * layers
    visual_warp_feats = [0] * K * layers
    visual_masks = [0] * K * layers
    visual_occlusions = [0] * K * layers
    visual_masked_feats = [0] * K * layers
    visual_warp_imgs = [0] * K * layers
    
    xf_merge = [0] * layers
    for i in range(layers):
        for k in range(K):
            visual_feats[i*K + k] = tensor2im(ref_features[k][i], out_size=out_size)
            visual_masks[i*K + k] = tensor2im(masks_normed[k][i], is_mask=True, out_size=out_size)
            if occlusions is None:
                visual_occlusions[i*K+k] = tensor2im(None, is_mask=True, out_size=out_size)
            else:
                visual_occlusions[i*K+k] = tensor2im(occlusions[k][i], is_mask=True, out_size=out_size)

            warp_feature = warp_flow(ref_features[k][i], flows[k][i])
            ref_img_down = F.interpolate(ref_xs[k], flows[k][i].shape[2:],mode='bilinear',align_corners=True)
            warp_img = warp_flow(ref_img_down, flows[k][i])
            masked_feature = warp_feature * masks_normed[k][i]
            xf_merge[i] += masked_feature
            visual_warp_feats[i*K + k] = tensor2im(warp_feature, out_size=out_size)
            visual_warp_imgs[i*K + k] = tensor2im(warp_img, out_size=out_size)
            visual_masked_feats[i*K + k] = tensor2im(masked_feature, out_size=out_size)

        

    ''' 可视化网络输出和GT两列, 每列有 2*layer+3 行'''
    visual_feature_merged = [0] * layers
    visual_feature_gt = [0] * layers
    for i in range(layers):
        visual_feature_merged[i] = tensor2im(xf_merge[i], out_size=out_size)
        visual_feature_gt[i] = tensor2im(g_features[i], out_size=out_size)

    layers_white  = torch.cat((white,)*layers, dim=0)
    visual_feature_gt  = torch.cat(visual_feature_gt, dim=0)
    visual_feature_merged  = torch.cat(visual_feature_merged, dim=0)
    gt_col = torch.cat((visual_g_x, visual_g_y, layers_white, visual_feature_gt, visual_gp), dim=0)
    out_col = torch.cat((visual_out, visual_g_y, layers_white, visual_feature_merged, visual_gphat), dim=0)

    assert gt_col.shape[0] == rows
    assert out_col.shape[0] == rows

    ''' 可视化网络输入和中间结果 2*K 列 '''
    ref = [0] * K
    feat = [0] * K
    for k in range(K):
        visual_features = [0]*layers
        visual_warp_features = [0]*layers
        visual_warp_images = [0]*layers
        visual_masked_features = [0]*layers
        visual_maskss = [0]*layers
        visual_occlusionss = [0] * layers
        for l in range(layers):
            visual_features[l] = visual_feats[l*K+k]
            visual_warp_features[l] = visual_warp_feats[l*K+k]
            visual_warp_images[l] = visual_warp_imgs[l*K+k]
            visual_masked_features[l] = visual_masked_feats[l*K+k]
            visual_maskss[l] = visual_masks[l*K+k]
            visual_occlusionss[l] = visual_occlusions[l*K+k]
        
        visual_features  = torch.cat(visual_features, dim=0)
        visual_warp_features  = torch.cat(visual_warp_features, dim=0)
        visual_warp_images  = torch.cat(visual_warp_images, dim=0)
        visual_masked_features  = torch.cat(visual_masked_features, dim=0)
        visual_maskss  = torch.cat(visual_maskss, dim=0)
        visual_occlusionss = torch.cat(visual_occlusionss, dim=0)
        # print(visual_occlusionss.shape)

        ref[k] = torch.cat((visual_ref_xs[k], visual_ref_ys[k], visual_features,visual_warp_features, white), dim=0)
        if occlusions is None:
            feat[k] = torch.cat((white,white,visual_maskss, visual_masked_features, white ),dim=0)
        else:
            feat[k] = torch.cat((visual_occlusionss, visual_maskss, visual_masked_features, white), dim=0)

    refs = torch.cat(ref, dim=1)
    # ps = torch.cat(visual_ref_ps, dim=1)
    feats = torch.cat(feat, dim=1)
    assert refs.shape[0] == rows
    assert feats.shape[0] == rows

    final_img = torch.cat((refs, feats, out_col,gt_col), dim=1)
    final_img = final_img.type(torch.uint8).to(cpu).numpy()
    return final_img, simp_img, None
    
    # attns = torch.cat((final_img[256:512,256*K:256*K*2], white,white ),dim=1)
    # ys = torch.cat((final_img[256:512,0:256*K],g_y,g_y), dim=1)
    # ps = torch.cat((ps, visual_gphat,visual_gp ),dim=1)
    # parsings = torch.cat((final_img[256:512,256*K:256*K*2], white,white ),dim=1)
    # mid_img = torch.cat((simp_img, attns, ys, ps), dim=0)

    # mid_img = mid_img.type(torch.uint8).to(cpu).numpy()
    # out = np.transpose(out, (1, 0, 2))


