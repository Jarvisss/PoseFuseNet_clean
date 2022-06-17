cd ..
python run_mask_net.py \
 --id=fashion_pretrain_masknet_softmax_inputy \
 --K=2 \
 --gpu=1\
 --phase 'train'\
 --align_corner\
 --use_input_mask\
 --use_input_y\
 --batch_size 8 \
 --lr 1e-4 \
 --use_spectral_D \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --model_save_freq 5000 \
 --flow_exp_name 'fashion_pretrain_flow_v0_with_occ_1shot_20201201'\
 --which_flow_epoch 'epoch_35_batch_4000_G'

 #  --use_correctness \ 不用 correctness loss
