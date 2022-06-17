#!/bin/sh

cd ..

python run_face_residual_flow.py \
 --test_id='face_v3_residual_flow_single_correctness_loss_flow_ep14_4shot_20210701' \
 --K=2\
 --gpu=3\
 --phase 'test' \
 --align_corner  \
 --use_res_flow\
 --path_to_dataset '/dataset/ljw/voxceleb2/preprocess_test_k8'\
 --test_ckpt_name 'epoch_3_batch_25000_G'\
 --use_pose_decoder \
 --test_samples 100\
 --test_video\
 --driving_video_path '/home/ljw/playground/Realistic-Neural-Talking-Head-Models/examples/fine_tuning/00479.mp4'
