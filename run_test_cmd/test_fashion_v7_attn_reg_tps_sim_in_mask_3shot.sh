cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v7_sc_attn_reg_in_mask_tps_sim_3shot_20210111' \
 --K=3\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_13_batch_15000_G' \
 --use_tps_sim\
 --use_bone_RGB
#  --output_all \
#  --test_samples 400
 
