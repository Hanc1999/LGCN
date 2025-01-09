for lr in 0.02 0.01 0.005
do
    for lamda in 0.02 0.01 0.005
    do
        for layer in 1 2
        do
            for batch in 10000 40000
            do
                for alpha in 1e-4
                do
                    python3 run_exp.py --model 13 --lr $lr --lamda $lamda --layer $layer --batch $batch --epoch 300 --afd_alpha $alpha
                done
            done
        done
    done
done