cd ..
python run_fashion_pyramid_self_attn.py \
 --id=fashion_v12_decoder_attn \
 --K=2 \
 --gpu=3\
 --phase 'train'\
 --align_corner\
 --batch_size 5 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_adv \
 --use_spectral_D \
 --use_pose_decoder\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_correctness \
 --use_flow_reg\
 --lambda_flow_reg 0.0025\
 --lambda_attn_reg 1\
 --use_tps_sim\
 --model_save_freq 5000 \
 --flow_exp_name 'fashion_pretrain_flow_no_bone_RGB_1shot_20210110'\
 --which_flow_epoch 'epoch_15_batch_5000_G'

#  --use_flow_attn_loss\
#  --lambda_flow_attn 1\
