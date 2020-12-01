cd .. 
python run_gmm.py --test \
 --test_dataset 'iper'\
 --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
 --ref_id=0 --test_source '026_1_1' --test_target_motion '026_1_1'
 --align_corner
 --align_parsing

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=100 --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=200 --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=0 --test_source '91lel7JgImS' --test_target_motion '91lel7JgImS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=100 --test_source '91lel7JgImS' --test_target_motion '91lel7JgImS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=200 --test_source '91lel7JgImS' --test_target_motion '91lel7JgImS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=0 --test_source 'A1VAMY3TBhS' --test_target_motion 'A1VAMY3TBhS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=100 --test_source 'A1VAMY3TBhS' --test_target_motion 'A1VAMY3TBhS'

# python run_gmm.py --test \
# --rigid_ckpt_name 'Geo_v2_loss_parse_lr0.0001-20201023' \
# --tps_ckpt_name 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
# --ref_id=200 --test_source 'A1VAMY3TBhS' --test_target_motion 'A1VAMY3TBhS'
