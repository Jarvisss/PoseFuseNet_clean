cd ..
python run.py --id='18_moverigid_align_corner_addtv_clothregion' \
 --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023'\
 --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027'\
 --K=2 --gpu=0 --use_parsing --use_self_flow --use_rigid --use_tv --move_rigid --use_dot --gmm_pixel_wise --align_corner --lambda_selfflow 0 --lambda_tv 100
