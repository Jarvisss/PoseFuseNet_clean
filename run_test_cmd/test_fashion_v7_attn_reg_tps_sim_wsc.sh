cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v7_4joints_wsc_attn_reg_lamda_2_2shot_20201230' \
 --K=2\
 --gpu=2 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_15000_G' \
 --use_tps_sim\
 --output_all \
 --test_samples 400

