cd ..
python run_fashion_pyramid.py --id=1_pyramid_pretrain_GLFA_split --K=2 --gpu=0 \
 --phase 'train' --align_corner\
 --batch_size 5 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_correctness\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --anno_size 256 176 \
 --flow_exp_name 'fashion_v0_flow_pretrain_GLFA_lsGAN_1shot_20201123'