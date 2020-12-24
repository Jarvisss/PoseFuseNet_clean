cd ..
python run_fashion_part_layer_warp_attn.py \
 --test_id='fashion_pretrain_flow_v2_layer_flow_mask_tv_armleg_reg200_1shot_20201211' \
 --K=1 --gpu=0 \
 --phase 'test' --align_corner \
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --use_parsing \
 --use_mask_tv \
 --categories 9 \
 --test_ckpt_name 'epoch_6_batch_18000_G' --output_all\
  --use_multi_layer_flow
