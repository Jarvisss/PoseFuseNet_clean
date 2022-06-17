cd ..
python run_fashion_pyramid_occ+attn.py \
 --id=fashion_v2_non_align_baseline_attn_avg \
 --K=2 \
 --gpu=1 \
 --phase 'train' \
 --align_corner\
 --batch_size 5 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_adv \
 --use_spectral_D \
 --use_correctness\
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --anno_size 256 176 \
 --attn_avg\
 --use_pose_decoder \
 --model_save_freq 5000 \
 --flow_exp_name 'fashion_pretrain_flow_no_bone_RGB_1shot_20210110'\
 --which_flow_epoch 'epoch_24_batch_5000_G'

#  --use_bone_RGB\
#  --which_flow_epoch 'epoch_5_batch_4000_G'
