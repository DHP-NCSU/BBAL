for l in 0.05 0.1 0.2 0.5
do
    for g in 0.1 0.2
    do
        python run_ceal/cb_ceal_exp.py --lambda_weight $l --gamma_weight $g > logs/gs_${l}_${g} 2>&1 &
    done
done
