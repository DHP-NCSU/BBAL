python run_ceal/cb_ceal_xe.py --lambda_weight 0.0 --gamma_weight 0.0 --training_epoch 15 > logs/gs_15e_0.0_0.0.log 2>&1 &
python run_ceal/cb_ceal_xe.py --lambda_weight 0.4 --gamma_weight 0.0 --training_epoch 15 > logs/gs_15e_0.4_0.0.log 2>&1 &
python run_ceal/cb_ceal_xe.py --lambda_weight 0.0 --gamma_weight 0.4 --training_epoch 15 > logs/gs_15e_0.0_0.4.log 2>&1 &
python run_ceal/cb_ceal_xe.py --lambda_weight 0.2 --gamma_weight 0.2 --training_epoch 15 > logs/gs_15e_0.2_0.2.log 2>&1 &

