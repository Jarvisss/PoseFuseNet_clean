cd ..
python run_fashion_pyramid_occ+attn.py \
 --id=fashion_v7-1_wsc_tps \
 --K=2 \
 --gpu=1\
 --phase 'train'\
 --align_corner\
 --batch_size 1 \
 --lr 1e-4 \
 --lr_D 1e-5 \
 --use_adv \
 --use_spectral_D \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --use_pose_decoder \
 --use_flow_reg\
 --lambda_flow_reg 0.0025\
 --lambda_flow_attn 5\
 --use_flow_attn_loss \
 --use_tps_sim\
 --tps_sim_beta1 2\
 --tps_sim_beta2 40\
 --model_save_freq 5000 \
 --img_save_freq 1 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_35_batch_4000_G'

 # --use_correctness    不用 correctness loss
 # --use_tps_sim        使用我们预计算的tps similarity
 #      tps_sim_beta1   tps 相似度计算的参数1(放大项)
 #      tps_sim_beta2   tps 相似度计算的参数2(最大距离)
 # --use_global_sim     使用全局的similarity
 #      joints_for_cos_sim 全局相似度计算使用几个点
