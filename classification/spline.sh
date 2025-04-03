#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT

echo "waiting..."

# nohup python -u spline.py -m 0 -s 0.5 -n 500 > test_spline.log 2>&1 &
# nohup python -u spline.py -m 0 -s 0.5 -n 1000 > test_spline2.log 2>&1 &
# nohup python -u spline.py -m 0 -s 0.5 -n 2000 > test_spline3.log 2>&1 &

nohup python -u spline.py -m 8 -s 0.5 -n 500 > test_spline4.log 2>&1 &
nohup python -u spline.py -m 8 -s 0.5 -n 1000 > test_spline5.log 2>&1 &
nohup python -u spline.py -m 8 -s 0.5 -n 2000 > test_spline6.log 2>&1 &
# nohup python -u spline.py -m 0 -s 0.5 -n 1000 > test_spline2.log 2>&1 &
# nohup python -u spline.py -m 0 -s 0.5 -n 2000 > test_spline3.log 2>&1 &
wait

echo "All scripts have been run successfully."

