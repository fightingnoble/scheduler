#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
core_offset_start=${1:-0}
core_offset_step=${2:-25}
core_offset_end=${3:-100}
tp_start=${4:-1}
tp_step=${5:-2}
tp_end=${6:-10}
PY_ARGS=${@:7}


# # 循环
for y in $(seq "$core_offset_start" "$core_offset_step" "$core_offset_end"); do 
    ./run/scan_aux.sh $tp_start $tp_step $tp_end $[$y+285] aux_scan sim_main.py 3  --gen_benchmark --e2e_latency 0.09 $PY_ARGS

    ./run/scan_aux.sh $tp_start $tp_step $tp_end $[$y+305] aux_scan sim_main.py 3  --gen_benchmark --e2e_latency 0.08 $PY_ARGS

    ./run/scan_aux.sh $tp_start $tp_step $tp_end $[$y+250] aux_scan sim_main.py 3  --gen_benchmark --e2e_latency 0.1 $PY_ARGS

done

