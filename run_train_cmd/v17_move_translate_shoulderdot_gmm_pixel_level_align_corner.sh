cd ..
python run.py --id='17_pretranslate_shoulderdot_gmm_pixel_level_align_corner' \
 --tps_ckpt 'Geo_v3_align_input_tv100_align_corner'\
 --K=2 \
 --gpu=3 \
 --use_parsing \
 --use_attnflow \
 --use_tv \
 --align_input \
 --use_dot \
 --gmm_pixel_wise \
 --lambda_attnflow 10 \
 --align_corner 