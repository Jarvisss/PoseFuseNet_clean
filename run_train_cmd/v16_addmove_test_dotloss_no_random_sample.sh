cd ..
python run.py --id='16_addmove_test_dotloss_no_random_sample' \
 --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023'\
 --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027'\
 --K=2 --gpu=0 --use_parsing --use_attnflow --use_rigid --use_tv --move_rigid --use_dot --lambda_attnflow 1000