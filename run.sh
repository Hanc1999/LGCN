for lr in 0.02 0.01 0.005
do
    for lamda in 0.02 0.01 0.005
    do
        for layer in 2
        do
          for batch in 10000 40000
            do
              python3 run_exp.py --model 13 --lr $lr --lamda $lamda --layer $layer 
            done
        done
    done
done