cd ..
python run_fashion_pyramid_K+1attn_adain.py \
 --id=fashion_v11_adain_decoder_k+1_no_attnreg_align_input \
 --K=2 \
 --gpu=2\
 --phase 'train'\
 --align_corner\
 --align_input\
 --batch_size 5 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_adv \
 --use_pose_decoder\
 --use_spectral_D \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --anno_size 256 176 \
 --use_correctness \
 --use_flow_reg\
 --lambda_flow_reg 0.0025\
 --lambda_attn_reg 1\
 --use_tps_sim\
 --model_save_freq 5000 \
 --flow_exp_name 'fashion_pretrain_flow_no_bone_RGB_align_input_1shot_20210111'\
 --which_flow_epoch 'epoch_15_batch_5000_G'

#  --use_flow_attn_loss\
#  --lambda_flow_attn 1\
