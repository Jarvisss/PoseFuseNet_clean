#!/bin/sh

cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v4_sim_attn_non_align_input_2shot_20210403' \
 --K=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_7_batch_15000_G'\
 --use_pose_decoder
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v4_sim_attn_non_align_input_2shot_20210403/epoch_7_batch_15000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v4_highres_non_align_input_sim_attn_ep7_2shot


python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v4_sim_attn_non_align_input_2shot_20210403' \
 --K=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_attn_reg\
 --use_tps_sim\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_15_batch_15000_G'\
 --use_pose_decoder
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v4_sim_attn_non_align_input_2shot_20210403/epoch_15_batch_15000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v4_highres_non_align_input_sim_attn_ep15_2shot

python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v4_sim_attn_non_align_input_2shot_20210403' \
 --K=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_24_batch_15000_G'\
 --use_pose_decoder
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v4_sim_attn_non_align_input_2shot_20210403/epoch_24_batch_15000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v4_highres_non_align_input_sim_attn_ep24_2shot