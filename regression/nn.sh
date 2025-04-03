#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT

declare -i mode_value
mode_value=8

nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 0 -s 0.5 -n 500 > test_nn.log 2>&1 &
nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 50 -s 0.5 -n 500 > test_nn2.log 2>&1 &
nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 100 -s 0.5 -n 500 > test_nn3.log 2>&1 &
nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 150 -s 0.5 -n 500 > test_nn4.log 2>&1 &

nohup python -u nn.py -c 2 -m $mode_value -w 8 -seed 0 -s 0.5 -n 1000 > test_nn5.log 2>&1 &
nohup python -u nn.py -c 2 -m $mode_value -w 8 -seed 50 -s 0.5 -n 1000 > test_nn6.log 2>&1 &
nohup python -u nn.py -c 5 -m $mode_value -w 8 -seed 100 -s 0.5 -n 1000 > test_nn7.log 2>&1 &
nohup python -u nn.py -c 5 -m $mode_value -w 8 -seed 150 -s 0.5 -n 1000 > test_nn8.log 2>&1 &

nohup python -u nn.py -c 7 -m $mode_value -w 8 -seed 0 -s 0.5 -n 2000 > test_nn9.log 2>&1 &
nohup python -u nn.py -c 8 -m $mode_value -w 8 -seed 50 -s 0.5 -n 2000 > test_nn10.log 2>&1 &
nohup python -u nn.py -c 9 -m $mode_value -w 8 -seed 100 -s 0.5 -n 2000 > test_nn11.log 2>&1 &
nohup python -u nn.py -c 10 -m $mode_value -w 8 -seed 150 -s 0.5 -n 2000 > test_nn12.log 2>&1 &

mode_value=9

nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 0 -s 0.5 -n 500 > test_nn13.log 2>&1 &
nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 50 -s 0.5 -n 500 > test_nn14.log 2>&1 &
nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 100 -s 0.5 -n 500 > test_nn15.log 2>&1 &
nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 150 -s 0.5 -n 500 > test_nn16.log 2>&1 &

nohup python -u nn.py -c 3 -m $mode_value -w 8 -seed 0 -s 0.5 -n 1000 > test_nn17.log 2>&1 &
nohup python -u nn.py -c 3 -m $mode_value -w 8 -seed 50 -s 0.5 -n 1000 > test_nn18.log 2>&1 &
nohup python -u nn.py -c 6 -m $mode_value -w 8 -seed 100 -s 0.5 -n 1000 > test_nn19.log 2>&1 &
nohup python -u nn.py -c 6 -m $mode_value -w 8 -seed 150 -s 0.5 -n 1000 > test_nn20.log 2>&1 &

nohup python -u nn.py -c 11 -m $mode_value -w 8 -seed 0 -s 0.5 -n 2000 > test_nn21.log 2>&1 &
nohup python -u nn.py -c 12 -m $mode_value -w 8 -seed 50 -s 0.5 -n 2000 > test_nn22.log 2>&1 &
nohup python -u nn.py -c 13 -m $mode_value -w 8 -seed 100 -s 0.5 -n 2000 > test_nn23.log 2>&1 &
nohup python -u nn.py -c 15 -m $mode_value -w 8 -seed 150 -s 0.5 -n 2000 > test_nn24.log 2>&1 &
# wait
# mode_value=2

# nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 0 -s 0.5 -n 500 > test_nn.log 2>&1 &
# nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 50 -s 0.5 -n 500 > test_nn2.log 2>&1 &
# nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 100 -s 0.5 -n 500 > test_nn3.log 2>&1 &
# nohup python -u nn.py -c 0 -m $mode_value -w 8 -seed 150 -s 0.5 -n 500 > test_nn4.log 2>&1 &

# nohup python -u nn.py -c 2 -m $mode_value -w 8 -seed 0 -s 0.5 -n 1000 > test_nn5.log 2>&1 &
# nohup python -u nn.py -c 2 -m $mode_value -w 8 -seed 50 -s 0.5 -n 1000 > test_nn6.log 2>&1 &
# nohup python -u nn.py -c 5 -m $mode_value -w 8 -seed 100 -s 0.5 -n 1000 > test_nn7.log 2>&1 &
# nohup python -u nn.py -c 5 -m $mode_value -w 8 -seed 150 -s 0.5 -n 1000 > test_nn8.log 2>&1 &

# nohup python -u nn.py -c 7 -m $mode_value -w 6 -seed 0 -s 0.5 -n 2000 > test_nn9.log 2>&1 &
# nohup python -u nn.py -c 8 -m $mode_value -w 8 -seed 50 -s 0.5 -n 2000 > test_nn10.log 2>&1 &
# nohup python -u nn.py -c 9 -m $mode_value -w 8 -seed 100 -s 0.5 -n 2000 > test_nn11.log 2>&1 &
# nohup python -u nn.py -c 10 -m $mode_value -w 8 -seed 150 -s 0.5 -n 2000 > test_nn12.log 2>&1 &

# mode_value=3

# nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 0 -s 0.5 -n 500 > test_nn13.log 2>&1 &
# nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 50 -s 0.5 -n 500 > test_nn14.log 2>&1 &
# nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 100 -s 0.5 -n 500 > test_nn15.log 2>&1 &
# nohup python -u nn.py -c 1 -m $mode_value -w 8 -seed 150 -s 0.5 -n 500 > test_nn16.log 2>&1 &

# nohup python -u nn.py -c 3 -m $mode_value -w 8 -seed 0 -s 0.5 -n 1000 > test_nn17.log 2>&1 &
# nohup python -u nn.py -c 3 -m $mode_value -w 8 -seed 50 -s 0.5 -n 1000 > test_nn18.log 2>&1 &
# nohup python -u nn.py -c 6 -m $mode_value -w 8 -seed 100 -s 0.5 -n 1000 > test_nn19.log 2>&1 &
# nohup python -u nn.py -c 6 -m $mode_value -w 8 -seed 150 -s 0.5 -n 1000 > test_nn20.log 2>&1 &

# nohup python -u nn.py -c 11 -m $mode_value -w 8 -seed 0 -s 0.5 -n 2000 > test_nn21.log 2>&1 &
# nohup python -u nn.py -c 12 -m $mode_value -w 8 -seed 50 -s 0.5 -n 2000 > test_nn22.log 2>&1 &
# nohup python -u nn.py -c 13 -m $mode_value -w 8 -seed 100 -s 0.5 -n 2000 > test_nn23.log 2>&1 &
# nohup python -u nn.py -c 15 -m $mode_value -w 8 -seed 150 -s 0.5 -n 2000 > test_nn24.log 2>&1 &

echo "waiting..."
wait

echo "All scripts have been run successfully."

