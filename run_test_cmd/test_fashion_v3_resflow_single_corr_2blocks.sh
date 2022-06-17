#!/bin/sh

cd ..
python run_fashion_pyramid_decoder_residual_flow.py \
 --test_id='fashion_v3_residual_flow_2_blocks_single_correctness_loss_align_input_2shot_20210412' \
 --K=2\
 --n_res_block=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --align_input \
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_11_batch_15000_G'\
 --use_pose_decoder\
 --test_samples 2
#  --use_bone_RGB\

# #  --output_all \
# #  --test_samples 400

# CUDA_VISIBLE_DEVICES=3 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321/epoch_30_batch_15000_G/2_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v3_highres_align_input_res_flow_single_corr_ep30_2shot


# python run_fashion_pyramid_decoder_residual_flow.py \
#  --test_id='fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321' \
#  --K=2\
#  --gpu=3\
#  --phase 'test' \
#  --align_corner  \
#  --align_input \
#  --use_tps_sim\
#  --use_attn_reg\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_33_batch_15000_G'\
#  --use_pose_decoder
# #  --use_bone_RGB\

# #  --output_all \
# #  --test_samples 400

# CUDA_VISIBLE_DEVICES=3 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321/epoch_33_batch_15000_G/2_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v3_highres_align_input_res_flow_single_corr_ep33_2shot



# python run_fashion_pyramid_decoder_residual_flow.py \
#  --test_id='fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321' \
#  --K=2\
#  --gpu=3\
#  --phase 'test' \
#  --align_corner  \
#  --align_input \
#  --use_tps_sim\
#  --use_attn_reg\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_32_batch_15000_G'\
#  --use_pose_decoder
# #  --use_bone_RGB\

# #  --output_all \
# #  --test_samples 400

# CUDA_VISIBLE_DEVICES=3 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321/epoch_32_batch_15000_G/2_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v3_highres_align_input_res_flow_single_corr_ep32_2shot

# python run_fashion_pyramid_decoder_residual_flow.py \
#  --test_id='fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321' \
#  --K=2\
#  --gpu=0\
#  --phase 'test' \
#  --align_corner  \
#  --align_input \
#  --use_tps_sim\
#  --use_attn_reg\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_24_batch_15000_G'\
#  --use_pose_decoder
# #  --use_bone_RGB\

# #  --output_all \
# #  --test_samples 400

# CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321/epoch_24_batch_15000_G/2_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v3_highres_align_input_res_flow_single_corr_ep24_2shot