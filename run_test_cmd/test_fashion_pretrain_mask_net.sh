cd ..
python run_mask_net.py \
 --test_id='fashion_pretrain_masknet_2shot_20210107' \
 --K=2\
 --gpu=1\
 --phase 'test'\
 --align_corner\
 --use_input_mask\
 --anno_size 256 176\
 --path_to_dataset '/home/ljw/playground/Global-Flow-Local-Attention/dataset/fashion'\
 --test_ckpt_name 'epoch_3_batch_5000_G' 
#  --output_all
