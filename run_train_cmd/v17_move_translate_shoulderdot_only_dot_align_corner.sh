cd ..
python run.py --id='17_pretranslate_shoulderdot_only_dot_align_corner' \
 --tps_ckpt 'Geo_v3_tps_only_use_tv_lr0.0001-20201106'\
 --K=2 \
 --gpu=2 \
 --use_parsing \
 --use_attnflow \
 --use_tv \
 --align_input \
 --use_dot \
 --only_pose \
 --lambda_attnflow 10 \
 --align_corner 