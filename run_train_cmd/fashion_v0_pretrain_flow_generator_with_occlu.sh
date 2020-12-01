cd ..
python run_fashion.py --id=0_with_occ --K=1 --gpu=1 \
 --phase 'train' --align_corner\
 --batch_size 10 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D\
 --anno_size 256 176 \
 --use_flow_reg \
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion' 
