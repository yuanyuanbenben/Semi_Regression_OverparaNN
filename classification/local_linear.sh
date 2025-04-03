#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT

echo "waiting..."

nohup python -u local_linear.py -m 8 -s 0.5 -n 500 > local_linear.log 2>&1 &

nohup python -u local_linear.py -m 8 -s 0.5 -n 1000 > local_linear2.log 2>&1 &

nohup python -u local_linear.py -m 8 -s 0.5 -n 2000 > local_linear3.log 2>&1 &

nohup python -u local_linear.py -m 9 -s 0.5 -n 500 > local_linear4.log 2>&1 &

nohup python -u local_linear.py -m 9 -s 0.5 -n 1000 > local_linear5.log 2>&1 &

nohup python -u local_linear.py -m 9 -s 0.5 -n 2000 > local_linear6.log 2>&1 &
wait
echo "All scripts have been run successfully."

