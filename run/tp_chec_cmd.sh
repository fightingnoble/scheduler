#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
core_offset_start=${1:-0}
core_offset_step=${2:-25}
core_offset_end=${3:-100}
tp_start=${4:-1}
tp_step=${5:-2}
tp_end=${6:-10}
pre_alloc=${7:-"False"}
static_sim=${8:-"False"}
glb_dyn=${9:-"True"}
dyn=${10:-"True"}
root_dir=${11:-"aux_scan"}
PY_ARGS=${@:12}


# # 循环
for y in $(seq "$core_offset_start" "$core_offset_step" "$core_offset_end"); do 
    {
        ./run/scan_aux.sh $tp_start $tp_step $tp_end $[$y+285] $root_dir sim_main.py 3 $pre_alloc $static_sim $glb_dyn $dyn --gen_benchmark --e2e_latency 0.09 $PY_ARGS 

        ./run/scan_aux.sh $tp_start $tp_step $tp_end $[$y+305] $root_dir sim_main.py 3 $pre_alloc $static_sim $glb_dyn $dyn --gen_benchmark --e2e_latency 0.08 $PY_ARGS 

        ./run/scan_aux.sh $tp_start $tp_step $tp_end $[$y+250] $root_dir sim_main.py 3 $pre_alloc $static_sim $glb_dyn $dyn --gen_benchmark --e2e_latency 0.1 $PY_ARGS 
    }&
done
wait