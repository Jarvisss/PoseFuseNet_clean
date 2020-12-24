cd ..
python run_fashion_pyramid_occ+attn.py --test_id='fashion_v2_pose_aware_pyramid_pretrain_GLFA_split_occ_attn_flow35_lsGAN_2shot_20201204' \
 --K=2 --gpu=0 \
 --phase 'test' --align_corner  --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_5_batch_10000_G' --output_all
