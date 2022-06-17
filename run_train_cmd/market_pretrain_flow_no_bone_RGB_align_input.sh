cd ..
python run_market_pyramid_occ+attn.py \
 --id=market_pretrain_flow_no_bone_RGB_align_input \
 --pretrain_flow\
 --align_input\
 --K=1 \
 --gpu=0\
 --phase 'train'\
 --align_corner\
 --batch_size 8 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_correctness\
 --use_flow_reg\
 --path_to_dataset '/dataset/ljw/market'\
 --use_pose_decoder \
 --lambda_flow_reg 0.0025\
 --model_save_freq 5000 \
 --img_save_freq 500

