cd ..
python run_fashion_pyramid_occ+attn_different_net.py --test_id='fashion_v4_flow_attn_loss_pose_aware_pyramid_pretrain_occ_attn_different_net_flow35_lsGAN_2shot_20201206' \
 --K=1 --gpu=3 \
 --phase 'test' --align_corner  --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_8000_G' --output_all
