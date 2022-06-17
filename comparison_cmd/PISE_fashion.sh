
CUDA_VISIBLE_DEVICES=0 python -m script.metrics \
--gt_path /home/ljw/playground/Pose-Transfer/fashion_data/test \
--distorated_path /home/ljw/playground/Pose-Transfer/results/fashion_PISE/eval_results/latest_176 \
--fid_real_path /home/ljw/playground/Pose-Transfer/fashion_data/train  \
--name PISE_176256