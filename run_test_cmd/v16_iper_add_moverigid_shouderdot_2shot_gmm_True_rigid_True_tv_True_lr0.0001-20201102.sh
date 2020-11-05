cd ..

# python run.py --test --test_id 'v16_add_moverigid_shouderdot_2shot_gmm_True_rigid_True_tv_True_lr0.0001-20201102' \
#     --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' \
#     --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
#     --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
#     --use_attnflow --move_rigid --use_rigid \
#     --ref_ids=0,236 --test_source '919iQ+Yy6qS' --test_target '919iQ+Yy6qS' 

# python run.py --test --test_id 'v16_add_moverigid_shouderdot_2shot_gmm_True_rigid_True_tv_True_lr0.0001-20201102' \
#     --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' \
#     --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
#     --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
#     --use_attnflow --move_rigid --use_rigid \
#     --ref_ids=0,158 --test_source '919iQ+Yy6qS' --test_target '919iQ+Yy6qS' 

python run.py --test --test_id 'v16_add_moverigid_shouderdot_2shot_gmm_True_rigid_True_tv_True_lr0.0001-20201102' \
    --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' \
    --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
    --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
    --use_attnflow --move_rigid --use_rigid \
    --ref_ids=0,70,136,200,257 --test_source_dataset 'danceFashion' --test_target_dataset 'iper' --test_source '919iQ+Yy6qS' --test_target '010_3_1' --test_freq 5

python run.py --test --test_id 'v16_add_moverigid_shouderdot_2shot_gmm_True_rigid_True_tv_True_lr0.0001-20201102' \
    --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' \
    --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
    --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
    --use_attnflow --move_rigid --use_rigid \
    --ref_ids=0,47,91,179,257 --test_source_dataset 'danceFashion' --test_target_dataset 'iper' --test_source 'A15Ei5ve9BS' --test_target '010_3_1' --test_freq 5

python run.py --test --test_id 'v16_add_moverigid_shouderdot_2shot_gmm_True_rigid_True_tv_True_lr0.0001-20201102' \
    --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' \
    --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
    --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
    --use_attnflow --move_rigid --use_rigid \
    --ref_ids=0,47,107,179,236 --test_source_dataset 'danceFashion' --test_target_dataset 'iper' --test_source 'A1dLq8J8cjS' --test_target '010_3_1' --test_freq 5

python run.py --test --test_id 'v16_add_moverigid_shouderdot_2shot_gmm_True_rigid_True_tv_True_lr0.0001-20201102' \
    --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' \
    --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' \
    --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
    --use_attnflow --move_rigid --use_rigid \
    --ref_ids=91,179,257 --test_source_dataset 'danceFashion' --test_target_dataset 'iper' --test_source 'A15Ei5ve9BS' --test_target '010_3_1' --test_freq 5
