cd ..
python run_gmm.py --id=3_iper_align_input_tv100_align_corner --gpu=3 --use_tvloss --lambda_tv 40 --align_parsing --align_corner --batch_size 4 --dataset 'iper'