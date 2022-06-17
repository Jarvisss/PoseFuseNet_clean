#!/bin/sh

cd ..

python run_face_residual_flow.py \
 --test_id='face_v3_residual_flow_single_correctness_loss_flow_ep14_4shot_20210701' \
 --K=7\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --path_to_dataset '/dataset/ljw/voxceleb2/preprocess_test_k8'\
 --test_ckpt_name 'epoch_3_batch_25000_G'\
 --use_pose_decoder
