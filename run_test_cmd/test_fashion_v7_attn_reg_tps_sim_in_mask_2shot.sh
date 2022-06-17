cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v7_sc_attn_reg_in_mask_tps_sim_2shot_20210111' \
 --K=2\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_15_batch_15000_G' \
 --use_bone_RGB\
 --use_tps_sim
#  --output_all \
#  --test_samples 400
 
