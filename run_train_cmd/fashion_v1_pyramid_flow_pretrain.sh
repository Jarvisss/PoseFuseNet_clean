cd ..
python run_fashion_pyramid.py --id=1_pyramid_G_flow_pretrain --K=2 --gpu=0 \
 --phase 'train' --align_corner\
 --batch_size 5 --lr 1e-4 --lr_D 1e-5 --use_adv --use_spectral_D --use_correctness
