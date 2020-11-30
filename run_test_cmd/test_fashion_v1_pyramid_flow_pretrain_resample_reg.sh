cd ..
python run_fashion_pyramid.py --test_id='fashion_v1_pyramid_G_resample_flow_pretrain_lsGAN_2shot_20201122' --K=2 --gpu=3 \
 --phase 'test' --align_corner --G_use_resample --anno_size 256 256
