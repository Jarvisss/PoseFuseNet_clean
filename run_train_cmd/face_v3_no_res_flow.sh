cd ..

python run_face_residual_flow.py \
 --id=face_v3_no_res_flow_flow_ep14 \
 --K=4 \
 --gpu=2\
 --phase 'train'\
 --align_corner\
 --batch_size 5 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_flow_reg\
 --align_input\
 --single_correctness\
 --use_adv \
 --use_spectral_D \
 --use_pose_decoder\
 --path_to_dataset '/dataset/ljw/voxceleb2/preprocess'\
 --use_correctness \
 --lambda_flow_reg 0.0025\
 --lambda_attn_reg 1\
 --use_tps_sim\
 --model_save_freq 5000 \
 --flow_exp_name 'face_pretrain_flow_1shot_20210614'\
 --which_flow_epoch 'epoch_14_batch_0_G'
