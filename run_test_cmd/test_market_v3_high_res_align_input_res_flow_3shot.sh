#!/bin/sh

cd ..


python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_1_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_1_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep1_1w_3shot


python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_2_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_2_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep2_1w_3shot

python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_2_batch_20000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_2_batch_20000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep2_2w5_3shot


python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_3_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_3_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep3_3shot


python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_4_batch_10000_G'\
 --align_input
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_4_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep4_3shot

python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_5_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_5_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep5_3shot

python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_6_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_6_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep6_3shot

python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_7_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_7_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep7_3shot

python run_market_pyramid_decoder_residual_flow.py \
 --test_id='market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --test_ckpt_name 'epoch_8_batch_10000_G'\
 --align_input

CUDA_VISIBLE_DEVICES=0 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v3_residual_flow_single_correctness_loss_align_input_3shot_20210412/epoch_8_batch_10000_G/3_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v3_res_flow_highres_align_input_ep8_3shot