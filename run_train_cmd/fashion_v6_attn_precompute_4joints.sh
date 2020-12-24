cd ..
python run_fashion_pyramid_occ+attn_precompute.py --id=fashion_v6_attn_precompute_4joints\
 --K=2 --gpu=3 \
 --phase 'train' --align_corner\
 --batch_size 4 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_correctness\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --joints_for_cos_sim 4\
 --anno_size 256 176 \
 --use_pose_decoder \
 --use_flow_attn_loss \
 --model_save_freq 5000 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_35_batch_4000_G'
