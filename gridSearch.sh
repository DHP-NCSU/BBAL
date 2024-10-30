for l in 0.5 1.0
do
    for g in 3.0 5.0
    do
        python run_ceal/cb_ceal_exp.py --lambda_weight $l --gamma_weight $g > logs/gs_${l}_${g}.log 2>&1 &
    done
done
