cd ..
python run.py --id='15_add_shoudersim' \
 --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023'\
 --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027'\
 --K=2 --gpu=1 --use_parsing --use_attnflow --use_rigid --use_tv --lambda_attnflow 10