cd ..
python run_fashion_part_layer_warp_attn.py --id=5_layer_flow_noextra_posesimnet_use_pretrained_15epoch_decoder_upsample --K=2 --gpu=2 \
 --phase 'train' --align_corner --use_parsing\
 --batch_size 4 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --categories 9 \
 --use_multi_layer_flow\
 --use_mask_tv \
 --lambda_flow_reg 200\
 --lambda_roi_l1 30\
 --lambda_struct 40\
 --flow_exp_name 'fashion_pretrain_flow_v2_layer_flow_mask_tv_armleg_reg200_1shot_20201211'\
 --which_flow_epoch 'epoch_14_batch_4000_G'
