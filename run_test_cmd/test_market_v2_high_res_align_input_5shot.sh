#!/bin/sh

cd ..

python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
 --K=5\
 --gpu=2 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_1_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_1_batch_15000_G/5_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_ep1_5shot

python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
 --K=5\
 --gpu=2 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_2_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_2_batch_15000_G/5_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_ep2_5shot


python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
 --K=5\
 --gpu=2 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_3_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_3_batch_15000_G/5_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_ep3_5shot

# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_4_batch_15000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_4_batch_15000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_ep4_2shot


# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_5_batch_15000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_5_batch_15000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_ep5_2shot

# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_6_batch_15000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_6_batch_15000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_ep6_2shot

# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_7_batch_15000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_7_batch_15000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_ep7_2shot


# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --use_pose_decoder\
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_8_batch_15000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_8_batch_15000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_ep8_2shot