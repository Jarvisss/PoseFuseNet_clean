cd ..
python run_fashion_pyramid_decoder_residual_flow.py \
 --test_id='fashion_v13_residual_flow_double_correctness_loss_align_input_2shot_20210125' \
 --K=3\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --align_input\
 --use_tps_sim\
 --use_pose_decoder\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_19_batch_5000_G' 

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v13_residual_flow_double_correctness_loss_align_input_2shot_20210125/epoch_19_batch_5000_G/3_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v13_no_bone_w_reg_align_input_ep19_3shot
#  --output_all \
#  --test_samples 400
