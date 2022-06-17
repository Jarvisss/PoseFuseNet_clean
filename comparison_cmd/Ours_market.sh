
CUDA_VISIBLE_DEVICES=1 python -m script.metrics_market \
--gt_path /dataset/ljw/market/test  \
--distorated_path /home/ljw/playground/poseFuseNet/test_result/market_v2_pose_aware_pyramid_pretrain_occ_attn_align_input_2shot_20210506/epoch_2_batch_15000_G/2_shot_eval/ \
--fid_real_path /dataset/ljw/market/train  \
--name market_v2_highres_align_input_ep2_2shot
