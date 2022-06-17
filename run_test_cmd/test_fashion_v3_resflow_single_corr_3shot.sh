#!/bin/sh

cd ..
# python run_fashion_pyramid_decoder_residual_flow.py \
#  --test_id='fashion_v3_residual_flow_single_correctness_loss_align_input_3shot_3shot_20210412' \
#  --K=3\
#  --gpu=0\
#  --phase 'test' \
#  --align_corner  \
#  --align_input \
#  --use_tps_sim\
#  --use_attn_reg\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_27_batch_5000_G'\
#  --use_pose_decoder
# #  --use_bone_RGB\

# #  --output_all \
# #  --test_samples 400

# CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_3shot_3shot_20210412/epoch_27_batch_5000_G/3_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v3_highres_align_input_res_flow_single_corr_ep27_5k_3shot

python run_fashion_pyramid_decoder_residual_flow.py \
 --test_id='fashion_v3_residual_flow_single_correctness_loss_align_input_3shot_3shot_20210412' \
 --K=3\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --align_input \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_29_batch_5000_G'\
 --use_pose_decoder
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_3shot_3shot_20210412/epoch_29_batch_5000_G/3_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v3_highres_align_input_res_flow_single_corr_ep29_5k_3shot


python run_fashion_pyramid_decoder_residual_flow.py \
 --test_id='fashion_v3_residual_flow_single_correctness_loss_align_input_3shot_3shot_20210412' \
 --K=3\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --align_input \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_31_batch_5000_G'\
 --use_pose_decoder
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_3shot_3shot_20210412/epoch_31_batch_5000_G/3_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v3_highres_align_input_res_flow_single_corr_ep31_5k_3shot


