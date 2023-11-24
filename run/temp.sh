#!/bin/bash
# set -x

tp_start=${1:-1}
tp_step=${2:-1}
tp_end=${3:-10}
root_dir=${4:-"coalescing"}
scan_param=${5:-"False"}
scan_seed=${6:-"False"}
plot=${7:-'dis'}
wsc_slack_ratio=${8:-0.80}
seed_start=${9:-0}
seed_end=${10:-9}


# plot_args, "--plot ''", if plot is 'dis', else " --plot True "
if [ $plot == "dis" ]; then
    PY_ARGS=""
else
    PY_ARGS="--plot True"
fi

if [ $scan_param == "True" ]; then
    wsc_e=1.0
else
    wsc_e=$wsc_slack_ratio
fi

for wsc_slack_ratio in $(seq $wsc_slack_ratio 0.05 $wsc_e); do
    tem_s=`echo "scale=3;1.00 - $wsc_slack_ratio" | bc`
    for exec_t_comp_ratioA in $(seq $tem_s -0.05 0); do
        ./run/scan_coalescing.sh $tp_start $tp_step $tp_end $root_dir sim_main.py 3 True False 0 $PY_ARGS --gen_benchmark --e2e_latency 0.09 --bin_pack_cfg "Bp_coalescing.json" --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA
        ./run/scan_coalescing.sh $tp_start $tp_step $tp_end $root_dir sim_main.py 3 True False 0 $PY_ARGS --gen_benchmark --e2e_latency 0.08 --bin_pack_cfg "Bp_coalescing.json" --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA
        ./run/scan_coalescing.sh $tp_start $tp_step $tp_end $root_dir sim_main.py 3 True False 0 $PY_ARGS --gen_benchmark --e2e_latency 0.1  --bin_pack_cfg "Bp_coalescing.json" --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA

        if [ $scan_seed == "True" ]; then
            for ((seed=$seed_start; seed<=$seed_end; seed++)); do
                echo "./run/scan_coalescing.sh $tp_start $tp_step $tp_end $root_dir sim_main.py 3 False True $seed $PY_ARGS --gen_benchmark --e2e_latency 0.09 --bin_pack_cfg "Bp_coalescing.json" --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA" 
                echo "./run/scan_coalescing.sh $tp_start $tp_step $tp_end $root_dir sim_main.py 3 False True $seed $PY_ARGS --gen_benchmark --e2e_latency 0.08 --bin_pack_cfg "Bp_coalescing.json" --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA" 
                echo "./run/scan_coalescing.sh $tp_start $tp_step $tp_end $root_dir sim_main.py 3 False True $seed $PY_ARGS --gen_benchmark --e2e_latency 0.1  --bin_pack_cfg "Bp_coalescing.json" --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA" 
                wait
            done
        fi
    done
done

wait
