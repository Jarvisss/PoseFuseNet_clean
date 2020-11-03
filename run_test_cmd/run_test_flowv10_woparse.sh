cd ..

:`
# python run.py --test --test_id 'v11_2shot_mask_True_soft_True_maskNormtype_softmax_lr0.0001-20201014'\
#   --ref_ids=0,150,200,220,230 
`
python run.py --test --test_id 'v10_aug_2shot_mask_True_soft_True_maskNormtype_divsum_lr0.0001-20201015'\
  --test_source 'A15Ei5ve9BS' --test_target_motion 'A15Ei5ve9BS' --ref_ids=0,158  --mask_norm_type "divsum"

python run.py --test --test_id 'v10_aug_2shot_mask_True_soft_True_maskNormtype_divsum_lr0.0001-20201015'\
  --test_source 'A15Ei5ve9BS' --test_target_motion 'A15Ei5ve9BS' --ref_ids=0,236  --mask_norm_type "divsum"

python run.py --test --test_id 'v10_aug_2shot_mask_True_soft_True_maskNormtype_divsum_lr0.0001-20201015'\
  --test_source 'A1dLq8J8cjS' --test_target_motion 'A1dLq8J8cjS' --ref_ids=158,290 --mask_norm_type "divsum"