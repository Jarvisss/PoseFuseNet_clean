cd ..
python run.py --id='18_self_attention_moverigid_shoulderdot_align_corner_withtv' \
 --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023'\
 --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027'\
 --K=2 --gpu=0 --use_parsing --use_self_flow --use_rigid --use_tv --move_rigid --use_dot --gmm_pixel_wise --align_corner --lambda_selfflow 10 --lambda_tv 10
