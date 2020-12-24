cd ..
python run_fashion_pyramid_occ+attn_precompute.py --test_id='fashion_v6_attn_precompute_visible_joints_lsGAN_2shot_20201206' \
 --K=2 --gpu=0 \
 --phase 'test' --align_corner  --use_pose_decoder \
 --anno_size 256 176 \
 --joints_for_cos_sim -1\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_20000_G' --output_all
