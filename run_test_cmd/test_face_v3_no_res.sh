#!/bin/sh

cd ..

python run_face_residual_flow.py \
 --test_id='face_v3_no_res_flow_flow_ep14_4shot_20210825' \
 --K=4\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --path_to_dataset '/dataset/ljw/voxceleb2/preprocess_test_k8'\
 --test_ckpt_name 'epoch_2_batch_20000_G'\
 --use_pose_decoder
