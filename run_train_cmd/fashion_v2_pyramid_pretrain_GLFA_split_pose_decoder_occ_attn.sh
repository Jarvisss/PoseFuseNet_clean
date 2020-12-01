cd ..
python run_fashion_pyramid_occ+attn.py --id=2_pose_aware_pyramid_pretrain_GLFA_split_occ_attn --K=2 --gpu=2 \
 --phase 'train' --align_corner\
 --batch_size 5 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_correctness\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_5_batch_4000_G'
