cd ..
python run.py --id=14_large_flow \
 --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023'\
 --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_lr0.0001-20201027'\
 --K=2 --gpu=2 --use_parsing --use_attnflow --use_rigid --lambda_attnflow 1000