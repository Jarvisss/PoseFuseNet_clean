#!/bin/sh

cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v2_pose_aware_pyramid_pretrain_GLFA_split_occ_attn_highres_3shot_20210403' \
 --K=3\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_pose_decoder\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_39_batch_10000_G'\
 --align_input
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v2_pose_aware_pyramid_pretrain_GLFA_split_occ_attn_highres_3shot_20210403/epoch_39_batch_10000_G/3_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v2_highres_align_input_ep39_1w_3shot
