cd ..
python run_fashion_pyramid_occ+attn_different_net.py --id=4_flow_attn_loss_lambda5_pose_aware_pyramid_pretrain_occ_attn_different_net_flow35 --K=2 --gpu=3 \
 --phase 'train' --align_corner\
 --batch_size 4 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_correctness\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --use_flow_attn_loss \
 --lambda_flow_attn 5 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_35_batch_4000_G'
