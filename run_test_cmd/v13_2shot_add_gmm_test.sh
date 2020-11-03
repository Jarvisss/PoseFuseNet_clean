cd ..
python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
--test_id_parse 'parsing_vattn_3_4shot_20201019' \
--ref_ids=0,150,200,220,230 --use_parsing --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS' 

python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
--test_id_parse 'parsing_vattn_3_4shot_20201019' \
--ref_ids=0,236 --use_parsing --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS' 

python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
--test_id_parse 'parsing_vattn_3_4shot_20201019' \
--ref_ids=0,158 --use_parsing --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS' 
