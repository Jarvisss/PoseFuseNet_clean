cd ..
python run_fashion_pyramid_occ+attn.py \
 --id=fashion_pretrain_flow_no_bone_RGB \
 --pretrain_flow\
 --K=7 \
 --gpu=2\
 --phase 'train'\
 --align_corner\
 --batch_size 8 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_correctness\
 --use_flow_reg\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --lambda_flow_reg 0.0025\
 --model_save_freq 5000 \
 --img_save_freq 500

