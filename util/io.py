import json
import os
import shutil
from PIL import Image
import torch
import numpy as np
import torchvision.transforms.functional as F
from time import time
#io functions of SCRC
def load_str_list(filename, end = '\n'):
    with open(filename, 'r') as f:
        str_list = f.readlines()
    str_list = [s[:-len(end)] for s in str_list]
    return str_list

def save_str_list(str_list, filename, end = '\n'):
    str_list = [s+end for s in str_list]
    with open(filename, 'w') as f:
        f.writelines(str_list)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(json_obj, filename):
    with open(filename, 'w') as f:
        # json.dump(json_obj, f, separators=(',\n', ':\n'))
        json.dump(json_obj, f, indent = 0, separators = (',', ': '))

def mkdir_if_missing(output_dir):
  """
  def mkdir_if_missing(output_dir)
  """
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def copy(fn_src, fn_tar):
    shutil.copyfile(fn_src, fn_tar)

def transform_image(image, resize_param, method=Image.BICUBIC, affine=None, normalize=False, toTensor=True, fillWhiteColor=False):
    
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

def load_image(A_path, name='fashion', load_size=(256,256)):
    A_img = Image.open(A_path)
    # padding white color after affine transformation  
    fillWhiteColor = True if name =='fashion' else False
    Ai = transform_image(A_img, load_size, fillWhiteColor=fillWhiteColor)
    if Ai.shape[0]>3:
        Ai = Ai[:3,...]
    return Ai

def load_parsing(parsing_path, load_size=(256,256)):
    parsing_img = Image.open(parsing_path)
    parsing_img = np.array(transform_image(parsing_img, load_size, fillWhiteColor=False, toTensor=False))
    # [H W 1 ]
    parsing_img_map = np.stack((parsing_img,)*20, axis=2)
    for i in range(20):
        parsing_img_map[:,:, i] = (parsing_img == i).astype(np.uint8) * 255
    parsing_img_map = F.to_tensor(parsing_img_map) # [20, H, W]
    return parsing_img_map

def getAffineParam(angle, scale, shift):
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

def load_skeleton(B_path, load_size=(256,256), is_clean_pose=False, no_bone_RGB=True, pose_scale=255, affine=None):
    
    from util import openpose_utils
    
    B_coor = json.load(open(B_path))["people"]
    B_coor = B_coor[0]
    pose_dict = openpose_utils.obtain_2d_cords(B_coor, resize_param=load_size, org_size=load_size, affine=affine)
    alpha_pose_body = pose_dict['body']
    if not is_clean_pose:
        pose_body = alpha_pose_body
        # pose_body = openpose_utils.openpose18_to_coco17(alpha_pose_body)
        pass
    pose_numpy = openpose_utils.obtain_map(pose_body, load_size) 
    pose = np.transpose(pose_numpy,(2, 0, 1))
    pose = torch.Tensor(pose)
    Bi = pose
    if not no_bone_RGB:
        color = np.zeros(shape=load_size + (3, ), dtype=np.uint8)
        LIMB_SEQ = openpose_utils.LIMB_SEQ_HUMAN36M_17 if is_clean_pose else openpose_utils.LIMB_SEQ_COCO_17
        color = openpose_utils.draw_joint(color, pose_body.astype(np.int), LIMB_SEQ)
        color = np.transpose(color,(2,0,1))
        color = color / pose_scale # normalize to 0-1
        color = torch.Tensor(color)
        Bi = torch.cat((Bi, color), dim=0)
    return Bi, torch.Tensor(alpha_pose_body)



