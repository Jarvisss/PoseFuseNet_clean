cd ..
python run_gmm.py --id=3_iper_align_input_tv10_align_corner --gpu=2 --use_tvloss --lambda_tv 10 --align_parsing --align_corner --batch_size 4 --dataset 'iper'