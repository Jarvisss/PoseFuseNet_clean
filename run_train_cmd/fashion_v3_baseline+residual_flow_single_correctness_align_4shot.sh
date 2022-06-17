cd ..

python run_fashion_pyramid_decoder_residual_flow.py \
 --id=fashion_v3_residual_flow_single_correctness_loss_align_input \
 --K=4 \
 --gpu=3\
 --phase 'train'\
 --align_corner\
 --batch_size 5 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_flow_reg\
 --align_input\
 --single_correctness\
 --use_adv \
 --use_spectral_D \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --anno_size 256 176 \
 --use_correctness \
 --lambda_flow_reg 0.0025\
 --lambda_attn_reg 1\
 --use_tps_sim\
 --model_save_freq 5000 \
 --continue_train\
 --load_exp_name 'fashion_v3_residual_flow_single_correctness_loss_align_input_2shot_20210321'\
 --flow_exp_name 'fashion_pretrain_flow_no_bone_RGB_align_input_1shot_20210111'\
 --which_flow_epoch 'epoch_15_batch_5000_G'
