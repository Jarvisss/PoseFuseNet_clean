#!/bin/sh

cd ..
python run_fashion_pyramid_decoder_residual_flow.py \
 --test_id='fashion_v3_residual_flow_single_correctness_loss_non_align_input_2shot_20210331' \
 --K=3\
 --gpu=3\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_29_batch_10000_G'\
 --use_pose_decoder\
 --test_samples 400
