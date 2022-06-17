cd ..
# python run_fashion_pyramid_decoder_residual_flow.py \
#  --test_id='fashion_v13_resattn_residual_flow_double_correctness_loss_align_input_2shot_20210222' \
#  --K=2\
#  --gpu=1 \
#  --phase 'test' \
#  --use_res_attn\
#  --align_corner  \
#  --align_input\
#  --use_tps_sim\
#  --use_pose_decoder\
#  --use_attn_reg\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_27_batch_10000_G' 

#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=1 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v13_resattn_residual_flow_double_correctness_loss_align_input_2shot_20210222/epoch_27_batch_10000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v13_no_bone_w_reg_align_input_ep27_1w_2shot_0222
