for i in 0 1 2
do
    python run/bbal.py --lambda_weight 3.0 --gamma_weight 1.0 > logs/gs_3.0_1.0_${i}.log 2>&1 &
    python run/bbal.py --lambda_weight 1.0 --gamma_weight 5.0 > logs/gs_1.0_5.0_${i}.log 2>&1 &
done