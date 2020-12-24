cd ..
python run_fashion_pyramid_occ+attn_different_net.py --test_id='fashion_v3_pose_aware_pyramid_pretrain_GLFA_split_occ_attn_different_net_flow22_lsGAN_2shot_20201202' --K=2 --gpu=1 \
 --phase 'test' --align_corner  --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_14000_G'\
 #--output_all
