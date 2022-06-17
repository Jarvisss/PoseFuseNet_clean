cd ..
python run_market_pyramid_decoder_residual_flow.py \
 --id=market_v3_residual_flow_single_correctness_loss_non_align_input \
 --K=2 \
 --gpu=0\
 --phase 'train'\
 --align_corner\
 --batch_size 8 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_flow_reg\
 --single_correctness\
 --use_adv \
 --use_spectral_D \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/market'\
 --use_correctness \
 --lambda_flow_reg 0.0025\
 --lambda_attn_reg 1\
 --model_save_freq 5000 \
 --flow_exp_name 'market_pretrain_flow_no_bone_RGB_non_align_input_1shot_20220117'\
 --which_flow_epoch 'epoch_15_batch_5000_G'

#  --use_tps_sim\

#  --use_flow_attn_loss\
#  --lambda_flow_attn 1\
# fashion_pretrain_flow_no_bone_RGB_1shot_20210110
# fashion_pretrain_flow_v0_with_occ_1shot_20201201
