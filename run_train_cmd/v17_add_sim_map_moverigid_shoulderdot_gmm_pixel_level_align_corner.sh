cd ..
python run.py --id='17_add_sim_map_moverigid_shoulderdot_gmm_pixel_level_align_corner' \
 --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023'\
 --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027'\
 --K=2 --gpu=1 --use_parsing --use_attnflow --use_rigid --use_tv --move_rigid --use_dot --gmm_pixel_wise --lambda_attnflow 10 --align_corner --use_simmap