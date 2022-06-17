cd ..
python run_fashion_joint_train_residual_flow.py \
 --id=fashion_v14_residual_flow_joint_train \
 --K=2 \
 --gpu=1\
 --phase 'train'\
 --align_corner\
 --batch_size 5 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_bone_RGB\
 --use_flow_reg\
 --use_adv \
 --use_spectral_D \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --anno_size 256 176 \
 --use_correctness \
 --lambda_flow_reg 0.0025\
 --lambda_attn_reg 1\
 --use_tps_sim\
 --model_save_freq 2000 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_cascade_1shot_20210119'\
 --which_flow_epoch 'epoch_25_batch_10000_G'

#  --use_flow_attn_loss\
#  --lambda_flow_attn 1\
# fashion_pretrain_flow_no_bone_RGB_1shot_20210110
# fashion_pretrain_flow_v0_with_occ_1shot_20201201
