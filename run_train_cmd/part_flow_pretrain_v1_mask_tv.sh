cd ..
python run_fashion_part_warp_attn.py --id=1_part_flow_pretrain_mask_tv --K=1 --gpu=1 \
 --phase 'train' --align_corner\
 --anno_size 256 176 \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --batch_size 5 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_flow_reg \
 --use_parsing \
 --use_mask_tv \
 --categories 9 
