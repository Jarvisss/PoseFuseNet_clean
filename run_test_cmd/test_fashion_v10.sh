cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v10_train_stage1_2shot_20210111' \
 --K=2\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_tps_sim\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_5_batch_10000_G' \
 --output_all \
 --test_samples 400
