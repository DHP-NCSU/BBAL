for l in 0.0 1.0 3.0
do
    for g in 0.0 1.0
    do
        python run_ceal/cb_ceal_exp.py --lambda_weight $l --gamma_weight $g > logs/gs_${l}_${g}.log 2>&1 &
    done
done
