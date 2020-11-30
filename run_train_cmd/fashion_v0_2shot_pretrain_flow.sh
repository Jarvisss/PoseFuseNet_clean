cd ..
python run_fashion.py --id=0_flow_pretrain --K=1 --gpu=3 \
 --phase 'train' --align_corner\
 --batch_size 5 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D
