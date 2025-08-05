#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT

echo "waiting..."


nohup python -u kernel.py -m 8 -s 0.5 -n 500 > test_kernel.log 2>&1 &

nohup python -u kernel.py -m 8 -s 0.5 -n 1000 > test_kernel2.log 2>&1 &

nohup python -u kernel.py -m 8 -s 0.5 -n 2000 > test_kernel3.log 2>&1 &

nohup python -u kernel.py -m 9 -s 0.5 -n 500 > test_kernel4.log 2>&1 &

nohup python -u kernel.py -m 9 -s 0.5 -n 1000 > test_kernel5.log 2>&1 &

nohup python -u kernel.py -m 9 -s 0.5 -n 2000 > test_kernel6.log 2>&1 &

wait



echo "All scripts have been run successfully."

