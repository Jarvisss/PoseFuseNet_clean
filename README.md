# PoseFuseNet
This is the official code released for paper: [IJCAI 2021]
![Pipeline](https://github.com/Jarvisss/PoseFuseNet/blob/master/Pipeline_v2.png)


## Training Procedure
# 1. Train the parser
`python run_parsing_net.py --K=2 --use_attn 
`
# 2. Train the gmm
### fisrt train the rigid gmm
`python run_gmm.py --id=rigid_gmm --rigid`
### then the tps gmm

* to use tvloss on tps grid: 
`python run_gmm.py --id=tps_tv_gmm --use_tvloss`

* not to use tvloss on tps grid: 
`python run_gmm.py --id=tps_gmm`

# 3. Train the Generator
`python run.py --rigid_ckpt 'Geo_v2_loss_parse_lr0.0001-20201023' --tps_ckpt 'Geo_v2_tps_after_rigidaffine_use_parsingl1_use_tv_lr0.0001-20201027' --K=2 --use_parsing --use_attnflow --use_rigid --use_tv --move_rigid --use_dot --lambda_attnflow 10
`
