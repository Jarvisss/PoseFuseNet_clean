#!/bin/sh

cd ..

python run_kitti_residual_flow.py \
 --test_id='kitti_resflow_2shot_20210708' \
 --K=2\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/kitti'\
 --test_ckpt_name 'epoch_50_batch_0_G'
