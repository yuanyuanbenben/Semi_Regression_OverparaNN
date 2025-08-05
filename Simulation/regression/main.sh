#!/bin/bash

function terminate_scripts {
    echo "Terminating running scripts..."
    pkill -P $$  
    exit 1
}


trap terminate_scripts SIGINT


# run for our method
declare -i mode_value

mode_value=0
# estimation
nohup python -u main.py -c 0 -m $mode_value -seed 0 -s 0.5 -n 2000 -l 1 > test.log 2>&1 &
nohup python -u main.py -c 1 -m $mode_value -seed 50 -s 0.5 -n 2000 -l 1 > test2.log 2>&1 &
nohup python -u main.py -c 2 -m $mode_value -seed 100 -s 0.5 -n 2000 -l 1 > test3.log 2>&1 &
nohup python -u main.py -c 3 -m $mode_value -seed 150 -s 0.5 -n 2000 -l 1 > test4.log 2>&1 &

nohup python -u main.py -c 4 -m $mode_value -seed 0 -s 0.5 -n 1000 -l 1 > test5.log 2>&1 &
nohup python -u main.py -c 5 -m $mode_value -seed 50 -s 0.5 -n 1000 -l 1 > test6.log 2>&1 &
nohup python -u main.py -c 6 -m $mode_value -seed 100 -s 0.5 -n 1000 -l 1 > test7.log 2>&1 &
nohup python -u main.py -c 7 -m $mode_value -seed 150 -s 0.5 -n 1000 -l 1 > test8.log 2>&1 &

nohup python -u main.py -c 8 -m $mode_value -seed 0 -s 0.5 -n 500 -l 1 > test9.log 2>&1 &
nohup python -u main.py -c 9 -m $mode_value -seed 50 -s 0.5 -n 500 -l 1 > test10.log 2>&1 &
nohup python -u main.py -c 10 -m $mode_value -seed 100 -s 0.5 -n 500 -l 1 > test11.log 2>&1 &
nohup python -u main.py -c 11 -m $mode_value -seed 150 -s 0.5 -n 500 -l 1 > test12.log 2>&1 &

# size
# nohup python -u main.py -c 0 -m $mode_value -seed 200 -s 0.5 -n 500 -l 1 > test.log 2>&1 &
# nohup python -u main.py -c 0 -m $mode_value -seed 250 -s 0.5 -n 500 -l 1 > test2.log 2>&1 &
# nohup python -u main.py -c 1 -m $mode_value -seed 300 -s 0.5 -n 500 -l 1 > test3.log 2>&1 &
# nohup python -u main.py -c 1 -m $mode_value -seed 350 -s 0.5 -n 500 -l 1 > test4.log 2>&1 &
# nohup python -u main.py -c 2 -m $mode_value -seed 400 -s 0.5 -n 500 -l 1 > test5.log 2>&1 &
# nohup python -u main.py -c 2 -m $mode_value -seed 450 -s 0.5 -n 500 -l 1 > test6.log 2>&1 &
# nohup python -u main.py -c 3 -m $mode_value -seed 200 -s 0.5 -n 1000 -l 1 > test7.log 2>&1 &
# nohup python -u main.py -c 3 -m $mode_value -seed 250 -s 0.5 -n 1000 -l 1 > test8.log 2>&1 &
# nohup python -u main.py -c 4 -m $mode_value -seed 300 -s 0.5 -n 1000 -l 1 > test9.log 2>&1 &
# nohup python -u main.py -c 4 -m $mode_value -seed 350 -s 0.5 -n 1000 -l 1 > test10.log 2>&1 &
# nohup python -u main.py -c 5 -m $mode_value -seed 400 -s 0.5 -n 1000 -l 1 > test11.log 2>&1 &
# nohup python -u main.py -c 5 -m $mode_value -seed 450 -s 0.5 -n 1000 -l 1 > test12.log 2>&1 &
# nohup python -u main.py -c 6 -m $mode_value -seed 200 -s 0.5 -n 2000 -l 1 > test13.log 2>&1 &
# nohup python -u main.py -c 6 -m $mode_value -seed 250 -s 0.5 -n 2000 -l 1 > test14.log 2>&1 &
# nohup python -u main.py -c 7 -m $mode_value -seed 300 -s 0.5 -n 2000 -l 1 > test15.log 2>&1 &
# nohup python -u main.py -c 7 -m $mode_value -seed 350 -s 0.5 -n 2000 -l 1 > test16.log 2>&1 &
# nohup python -u main.py -c 8 -m $mode_value -seed 400 -s 0.5 -n 2000 -l 1 > test17.log 2>&1 &
# nohup python -u main.py -c 8 -m $mode_value -seed 450 -s 0.5 -n 2000 -l 1 > test18.log 2>&1 &

echo "All scripts have been run successfully."

