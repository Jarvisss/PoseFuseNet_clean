cd ..
python run_fashion_pyramid_self_attn.py \
 --test_id='fashion_v12_decoder_attn_2shot_20210111' \
 --K=2\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_pose_decoder\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_15_batch_0_G' 
#  --output_all \
#  --test_samples 400
