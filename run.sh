#!/bin/bash
process_name=$1
process_number=$2
pids=()

# check arguments
if [[ $# -ne 2 ]];
then
    echo "./run.sh <process_name> <process_number>"
    exit 1
fi

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT
function ctrl_c() {
    echo "** END PROCESS"
    for pid in ${pids[@]}
    do
        kill -SIGKILL $pid
    done
}

# run multi process
for ((i = 0; i < $process_number; i++))
do
    python $process_name &
    pids+=($!)
done

# wait while process is running
wait