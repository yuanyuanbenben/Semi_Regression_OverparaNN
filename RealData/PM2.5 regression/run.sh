#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# estimation
nohup python -u main.py -c 3 -l 1 > test.log 2>&1 &
nohup python -u nn.py -c 5 > test2.log 2>&1 &
nohup python -u kernel.py > test3.log 2>&1 &
nohup python -u spline.py > test4.log 2>&1 &
nohup python -u local_linear.py > test5.log 2>&1 &
nohup python -u linear.py > test6.log 2>&1 &

echo "Waiting"

wait

echo "All scripts have been run successfully."