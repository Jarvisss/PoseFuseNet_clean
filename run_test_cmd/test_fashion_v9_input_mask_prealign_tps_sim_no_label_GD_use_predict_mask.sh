cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v9_input_mask_prealign_attn_reg_tps_sim_no_label_sc_use_predict_mask_2shot_20210105' \
 --K=2\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_pose_decoder \
 --align_input\
 --use_tps_sim\
 --use_logits\
 --GD_use_predict_mask\
 --use_input_mask\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_7_batch_15000_G' 
#  --output_all \
#  --test_samples 400
