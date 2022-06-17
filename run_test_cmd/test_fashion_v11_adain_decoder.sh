cd ..
python run_fashion_pyramid_K+1attn_adain.py \
 --test_id='fashion_v11_adain_decoder_lsGAN_2shot_20210113' \
 --K=2\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_pose_decoder\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_15000_G' 
#  --output_all \
#  --test_samples 400
