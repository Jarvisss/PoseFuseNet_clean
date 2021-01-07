cd ..
python run_fashion_pyramid_K+1attn.py \
 --test_id='fashion_v8_GD_self_attn_reg3_lsGAN_2shot_20201228' \
 --K=2\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_0_G' \
 --output_all \
 --test_samples 400
#  --joints_for_cos_sim 4\

