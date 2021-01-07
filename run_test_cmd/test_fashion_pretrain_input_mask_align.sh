cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_pretrain_flow_input_mask_align_input_1shot_20210102' \
 --pretrain_flow\
 --K=1\
 --gpu=1\
 --phase 'test'\
 --align_corner\
 --align_input\
 --use_input_mask\
 --use_flow_reg\
 --use_pose_decoder\
 --anno_size 256 176\
 --joints_for_cos_sim 4\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_11_batch_5000_G' \
 --test_samples 400
#  --output_all
