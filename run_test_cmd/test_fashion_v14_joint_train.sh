cd ..
python run_fashion_joint_train_residual_flow.py \
 --test_id='fashion_v14_residual_flow_joint_train_2shot_20210121' \
 --K=2\
 --gpu=1 \
 --phase 'test' \
 --align_corner  \
 --use_bone_RGB\
 --use_tps_sim\
 --use_pose_decoder\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_7_batch_14000_G'\
 --output_all \
 --test_samples 400

CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v14_residual_flow_joint_train_2shot_20210121/epoch_7_batch_14000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v14_w_bone_w_reg_ep7_2shot

