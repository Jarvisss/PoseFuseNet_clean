cd ..
python run_fashion_part_layer_warp_attn.py --id=2_layer_flow_reg200 --K=1 --gpu=3 \
 --phase 'train' --align_corner\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --batch_size 5 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_flow_reg \
 --use_parsing \
 --categories 9 \
 --lambda_flow_reg 200\
 --lambda_roi_l1 30\
 --lambda_struct 40\
 --use_multi_layer_flow
