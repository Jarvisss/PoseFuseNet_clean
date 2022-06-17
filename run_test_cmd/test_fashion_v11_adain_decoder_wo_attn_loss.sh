cd ..
# python run_fashion_pyramid_K+1attn_adain.py \
#  --test_id='fashion_v11_adain_decoder_k+1_no_attnreg_align_input_lsGAN_2shot_20210419' \
#  --K=2\
#  --gpu=0 \
#  --phase 'test' \
#  --align_corner  \
#  --align_input\
#  --use_tps_sim\
#  --use_pose_decoder\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_25_batch_10000_G' 
# #  --output_all \
# #  --test_samples 400


# CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v11_adain_decoder_k+1_no_attnreg_align_input_lsGAN_2shot_20210419/epoch_25_batch_10000_G/2_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v11_highres_align_input_adain_decoder_ep25_1w_2shot

# python run_fashion_pyramid_K+1attn_adain.py \
#  --test_id='fashion_v11_adain_decoder_k+1_no_attnreg_align_input_lsGAN_2shot_20210419' \
#  --K=2\
#  --gpu=0 \
#  --phase 'test' \
#  --align_corner  \
#  --align_input\
#  --use_tps_sim\
#  --use_pose_decoder\
#  --anno_size 256 176 \
#  --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
#  --test_ckpt_name 'epoch_25_batch_15000_G' 
# #  --output_all \
# #  --test_samples 400


# CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
# --gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
# --distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v11_adain_decoder_k+1_no_attnreg_align_input_lsGAN_2shot_20210419/epoch_25_batch_15000_G/2_shot_eval/ \
# --fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
# --name v11_highres_align_input_adain_decoder_ep25_1w5_2shot

python run_fashion_pyramid_K+1attn_adain.py \
 --test_id='fashion_v11_adain_decoder_k+1_no_attnreg_align_input_lsGAN_2shot_20210419' \
 --K=2\
 --gpu=0 \
 --phase 'test' \
 --align_corner  \
 --align_input\
 --use_tps_sim\
 --use_pose_decoder\
 --anno_size 256 176 \
 --path_to_dataset '/dataset/ljw/deepfashion/GLFA_split/fashion'\
 --test_ckpt_name 'epoch_27_batch_10000_G' 
#  --output_all \
#  --test_samples 400


CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/test_256  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/fashion_v11_adain_decoder_k+1_no_attnreg_align_input_lsGAN_2shot_20210419/epoch_27_batch_10000_G/2_shot_eval/ \
--fid_real_path /home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion/train_256  \
--name v11_highres_align_input_adain_decoder_ep27_2shot