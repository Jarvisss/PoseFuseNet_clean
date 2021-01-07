cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v7_wsc_4joints_sc_original_lsGAN_2shot_20201223' \
 --K=2\
 --gpu=2 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --anno_size 256 176 \
 --joints_for_cos_sim 4\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_8_batch_5000_G' \
#  --output_all
