cd ..
python run_shapenet_residual_flow.py \
 --id=shapenet_resflow \
 --K=4 \
 --gpu=1\
 --phase 'train'\
 --align_corner\
 --batch_size 5 \
 --use_adv\
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_correctness\
 --single_correctness\
 --use_spectral_D \
 --use_flow_reg\
 --path_to_dataset '/dataset/ljw/shapenet/chair'\
 --use_pose_decoder \
 --lambda_flow_reg 0.0025\
 --model_save_freq 5000 \
 --img_save_freq 500\
 --flow_exp_name 'shapenet_pretrain_flow_1shot_20210616'\
 --which_flow_epoch 'epoch_45_batch_0_G'\

