cd ..
python run_fashion_pyramid.py --test_id='fashion_v1_pose_aware_pyramid_pretrain_GLFA_split_lsGAN_2shot_20201124' --K=2 --gpu=2 \
 --phase 'test' --align_corner  --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --output_all\
 --test_ckpt_name 'epoch_7_batch_12000_G'
