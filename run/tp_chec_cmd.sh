#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
start=${1:-0}
step=${2:-25}
end=${3:-100}
PY_ARGS=${@:4}


# # 循环
for x in $(seq "$start" "$step" "$end"); do 
    ./run/scan_aux.sh 1 2 10 $[$x+285] aux_scan sim_main.py 3  --gen_benchmark --e2e_latency 0.09 $PY_ARGS

    ./run/scan_aux.sh 1 2 10 $[$x+305] aux_scan sim_main.py 3  --gen_benchmark --e2e_latency 0.08 $PY_ARGS

    ./run/scan_aux.sh 1 2 10 $[$x+250] aux_scan sim_main.py 3  --gen_benchmark --e2e_latency 0.1 $PY_ARGS

done

