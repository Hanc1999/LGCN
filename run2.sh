for lr in 0.01 0.005 0.001 0.0005
do
    for lamda in 0.1 0.05 0.02 0.01
    do
        for layer in 1 2
        do
            for batch in 10000 40000
            do
                python3 run_exp.py --model 6  --lr $lr --lamda $lamda --layer $layer --batch $batch
            done
        done
    done
done
