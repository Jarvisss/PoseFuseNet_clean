cd ..
# python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
# --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
# --ref_ids=0,70,136,200,257 --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS' 

# python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
# --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing --parse_use_attn \
# --ref_ids=0,236 --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS' 

# python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
# --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing --parse_use_attn \
# --ref_ids=0,158 --test_source '919iQ+Yy6qS' --test_target_motion '919iQ+Yy6qS' 

python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
--test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
--ref_ids=0,47,91,179,257 --test_source_dataset 'danceFashion' --test_target_dataset 'iper' --test_source 'A15Ei5ve9BS' --test_target '023_3_2' --test_freq 5

# python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
# --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
# --ref_ids=0,47,107,179,236 --test_source 'A1dLq8J8cjS' --test_target_motion 'A1dLq8J8cjS' 

# python run.py --test --test_id 'v13_2shot_mask_True_soft_True_maskNormtype_softmax_parsing_True_lr0.0001-20201025' \
# --test_id_parse 'parsing_vattn_3_4shot_20201019' --use_parsing  --parse_use_attn \
# --ref_ids=91,179,257 --test_source_dataset 'danceFashion' --test_target_dataset 'iper' --test_source 'A15Ei5ve9BS' --test_target '023_3_2' --test_freq 5