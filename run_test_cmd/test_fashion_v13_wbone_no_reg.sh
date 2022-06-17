cd ..
# python run_fashion_pyramid_decoder_residual_flow.py \
#  --test_id='fashion_v13_residual_flow_double_correctness_loss_with_bone_2shot_20210120' \
#  --K=2\
#  --gpu=0 \
#  --phase 'test' \
#  --align_corner  \
#  --use_bone_RGB\
#  --use_tps_sim\
#  --use_pose_decoder\
#  --use_attn_reg\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_15_batch_15000_G' 

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v13_residual_flow_double_correctness_loss_with_bone_2shot_20210120/epoch_15_batch_15000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v13_w_bone_no_reg_ep15_2shot
#  --output_all \
#  --test_samples 400
