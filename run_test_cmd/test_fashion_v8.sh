cd ..
python run_fashion_pyramid_K+1attn.py \
 --test_id='fashion_v8_GD_self_attn_lsGAN_2shot_20201227' \
 --K=2\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --anno_size 256 176 \
 --joints_for_cos_sim 4\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_8_batch_5000_G' \
 --output_all \
 --test_samples 400
