cd ..
python run_shapenet_residual_flow.py \
 --id=shapenet_pretrain_flow \
 --pretrain_flow\
 --K=1 \
 --gpu=1\
 --phase 'train'\
 --align_corner\
 --batch_size 8 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_correctness\
 --use_flow_reg\
 --path_to_dataset '/dataset/ljw/shapenet/chair'\
 --use_pose_decoder \
 --lambda_flow_reg 0.0025\
 --model_save_freq 5000 \
 --img_save_freq 500

