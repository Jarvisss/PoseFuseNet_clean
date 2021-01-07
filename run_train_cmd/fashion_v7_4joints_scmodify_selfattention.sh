cd ..
python run_fashion_pyramid_occ+attn.py \
 --id=fashion_v7_4joints_wsc_original_self_attention \
 --K=2 \
 --gpu=1\
 --phase 'train'\
 --align_corner\
 --batch_size 4 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_adv \
 --use_spectral_D \
 --use_self_attention \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --use_flow_reg\
 --lambda_flow_reg 0.0025\
 --lambda_flow_attn 5\
 --use_flow_attn_loss \
 --joints_for_cos_sim 4\
 --model_save_freq 5000 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_35_batch_4000_G'

 #  --use_correctness \ 不用 correctness loss
