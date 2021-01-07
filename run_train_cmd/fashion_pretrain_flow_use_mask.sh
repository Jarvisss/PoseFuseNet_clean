cd ..
python run_fashion_pyramid_occ+attn.py \
 --id=fashion_pretrain_flow_input_mask_GF_no_GT_mask \
 --pretrain_flow\
 --K=1 \
 --gpu=2\
 --phase 'train'\
 --align_corner\
 --use_parsing\
 --use_input_mask\
 --batch_size 8 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_correctness\
 --use_flow_reg\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --lambda_flow_reg 0.0025\
 --lambda_flow_attn 5\
 --model_save_freq 5000 \
 --img_save_freq 500 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_35_batch_4000_G'

 #  --use_correctness \ 不用 correctness loss
