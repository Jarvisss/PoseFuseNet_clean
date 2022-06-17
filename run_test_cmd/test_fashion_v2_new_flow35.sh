cd ..
python run_fashion_pyramid_occ+attn.py \
 --test_id='fashion_v2_pose_aware_pyramid_pretrain_GLFA_split_occ_attn_highres_2shot_20210111' \
 --K=3\
 --gpu=3 \
 --phase 'test' \
 --align_corner  \
 --use_bone_RGB\
 --use_tps_sim\
 --use_pose_decoder\
 --use_attn_reg\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_15_batch_14000_G' 
#  --output_all \
#  --test_samples 400

CUDA_VISIBLE_DEVICES=3 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v2_pose_aware_pyramid_pretrain_GLFA_split_occ_attn_highres_2shot_20210111/epoch_15_batch_14000_G/3_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v2_highres_ep15_3shot
