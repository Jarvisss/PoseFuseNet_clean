cd ..
python run_market_pyramid_occ+attn.py \
 --id=market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input \
 --K=2 \
 --gpu=2 \
 --phase 'train' \
 --align_corner\
 --batch_size 8 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_adv \
 --use_spectral_D \
 --use_correctness\
 --path_to_dataset '/dataset/ljw/market'\
 --use_pose_decoder \
 --align_input\
 --model_save_freq 5000 \
 --flow_exp_name 'market_pretrain_flow_no_bone_RGB_align_input_1shot_20210505'\
 --which_flow_epoch 'epoch_15_batch_5000_G'\

#  --use_bone_RGB\
#  --which_flow_epoch 'epoch_5_batch_4000_G'
