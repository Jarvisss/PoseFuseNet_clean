#!/bin/sh

cd ..

# python run_shapenet_residual_flow.py \
#  --test_id='shapenet_resflow_4shot_20210619' \
#  --K=2\
#  --gpu=3 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/shapenet/chair'\
#  --test_ckpt_name 'epoch_10_batch_5000_G'

# python run_shapenet_residual_flow.py \
#  --test_id='shapenet_resflow_4shot_20210619' \
#  --K=2\
#  --gpu=3 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/shapenet/chair'\
#  --test_ckpt_name 'epoch_13_batch_5000_G'


python run_shapenet_residual_flow.py \
 --test_id='shapenet_resflow_4shot_20210619' \
 --K=2\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/shapenet/chair'\
 --test_ckpt_name 'epoch_16_batch_5000_G'




# python run_shapenet_residual_flow.py \
#  --test_id='shapenet_resflow_4shot_20210619' \
#  --K=4\
#  --gpu=3 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/shapenet/chair'\
#  --test_ckpt_name 'epoch_2_batch_5000_G'\

#  python run_shapenet_residual_flow.py \
#  --test_id='shapenet_resflow_4shot_20210619' \
#  --K=4\
#  --gpu=2 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/shapenet/chair'\
#  --test_ckpt_name 'epoch_3_batch_5000_G'\

#  python run_shapenet_residual_flow.py \
#  --test_id='shapenet_resflow_4shot_20210619' \
#  --K=4\
#  --gpu=2 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/shapenet/chair'\
#  --test_ckpt_name 'epoch_4_batch_5000_G'\