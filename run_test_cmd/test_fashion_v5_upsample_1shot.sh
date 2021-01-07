cd ..
python run_fashion_part_layer_warp_attn.py --test_id='fashion_v5_layer_flow_noextra_posesimnet_use_pretrained_15epoch_decoder_upsample_2shot_20201215' \
 --K=1 --gpu=2 \
 --use_parsing \
 --categories 9\
 --phase 'test' --align_corner  --use_pose_decoder \
 --use_multi_layer_flow\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_6_G' 
 #--output_all
