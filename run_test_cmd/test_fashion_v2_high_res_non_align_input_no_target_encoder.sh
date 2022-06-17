#!/bin/sh

cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v2_non_align_baseline_no_target_encoder_2shot_20210317' \
 --K=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_25_batch_10000_G'

#  --use_pose_decoder\
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v2_non_align_baseline_no_target_encoder_2shot_20210317/epoch_25_batch_10000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v2_highres_nonalign_input_no_target_encoder_ep25_1w_2shot


python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v2_non_align_baseline_no_target_encoder_2shot_20210317' \
 --K=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_25_batch_15000_G'

#  --use_pose_decoder\
#  --use_bone_RGB\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v2_non_align_baseline_no_target_encoder_2shot_20210317/epoch_25_batch_15000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v2_highres_nonalign_input_no_target_encoder_ep25_1w5_2shot

python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v2_non_align_baseline_no_target_encoder_2shot_20210317' \
 --K=2\
 --gpu=0\
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_27_batch_10000_G'

#  --use_bone_RGB\
#  --use_pose_decoder\

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v2_non_align_baseline_no_target_encoder_2shot_20210317/epoch_27_batch_10000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v2_highres_nonalign_input_no_target_encoder_ep27_2shot
