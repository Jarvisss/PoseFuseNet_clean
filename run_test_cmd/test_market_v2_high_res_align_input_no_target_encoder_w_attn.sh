#!/bin/sh

cd ..

python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_no_tge_2shot_20210513' \
 --K=3\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_1_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_no_tge_2shot_20210513/epoch_1_batch_15000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_no_tge_ep1_3shot


python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_no_tge_2shot_20210513' \
 --K=3\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_2_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_no_tge_2shot_20210513/epoch_2_batch_15000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_no_tge_ep2_3shot


python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_no_tge_2shot_20210513' \
 --K=3\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_3_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_no_tge_2shot_20210513/epoch_3_batch_15000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_no_tge_ep3_3shot

python run_market_pyramid_occ+attn.py \
 --test_id='market_v2_no_tge_2shot_20210513' \
 --K=3\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_4_batch_15000_G'\
 --align_input


CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_no_tge_2shot_20210513/epoch_4_batch_15000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_no_tge_ep4_3shot

# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_no_tge_2shot_20210513' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_4_batch_10000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210510/epoch_4_batch_10000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_no_tge_ep4-1w-_2shot

# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_no_tge_2shot_20210513' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_4_batch_20000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210510/epoch_4_batch_20000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_no_tge_attn_avg_ep4-2w-_2shot




# python run_market_pyramid_occ+attn.py \
#  --test_id='market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210510' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --align_corner  \
#  --attn_avg\
#  --path_to_dataset '/dataset/ljw/market'\
#  --test_ckpt_name 'epoch_5_batch_15000_G'\
#  --align_input


# CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
# --gt_path /dataset/ljw/market/test  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210510/epoch_5_batch_15000_G/2_shot_eval/ \
# --fid_real_path /dataset/ljw/market/train  \
# --name market_v2_highres_align_input_no_tge_attn_avg_ep5_2shot

