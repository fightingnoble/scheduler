#!/bin/bash
# set -x

tp_scan=${1:-"False"}
tp_start=${2:-0}  
tp_step=${3:-1}
tp_end=${4:-9}
scan_seed=${5:-"False"}
plot=${6:-"dis"}
seed_start=${7:-0}  
seed_end=${8:-9}
root_dir=${9:-"coalescing_scan"}

if [ $tp_scan != "True" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    list_n_bin_cfg=(24 22 20 18 16 14 12 10 8 6 4 2 1) 
else
    list_n_bin_cfg=(26) 
fi

# plot_args, "--plot ''", if plot is 'dis', else " --plot True "
if [ $plot == "dis" ]; then
    PY_ARGS=""
else
    PY_ARGS="--plot True"
fi
n_p=3
PY_ARGS="$PY_ARGS --profiling_filename profiling/profiling_light.csv --n_p $n_p --gen_benchmark --bin_pack_cfg "Bp_split.json""

echo "********************************************"

# exec_t_comp_ratioA, exec_t_comp_ratioB, wsc_slack_ratio exec_var_en
# 0.3, 0.3, 1, True
# 0, 0.05, 1, False
list_exec_t_comp_ratioA=(0.3 0)
list_exec_t_comp_ratioB=(0.3 0.05)
list_wsc_slack_ratio=(1 1)
list_exec_var_en=(True False)
list_suffix=("var_0.2_0.3" "var_0.2")
list_var_sim_cfg=("var_sim_cfg.json" "var_sim_cfg.json")
    
for i in {0..1}; do
  {
    exec_t_comp_ratioA=${list_exec_t_comp_ratioA[$i]}
    exec_t_comp_ratioB=${list_exec_t_comp_ratioB[$i]}
    wsc_slack_ratio=${list_wsc_slack_ratio[$i]}
    exec_var_en=${list_exec_var_en[$i]}
    suffix=${list_suffix[$i]}
    var_sim_cfg=${list_var_sim_cfg[$i]}

    for n_bin in ${list_n_bin_cfg[@]}; do
        echo "********************************************"
        echo "n_bin: $n_bin"
        echo "********************************************"

        RATE_CFG="--wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA --exec_t_comp_ratioB $exec_t_comp_ratioB --num_bins $n_bin --root_dir "$root_dir/n_bins_${n_bin}" "
        echo "./run/abla_scalablility.sh $tp_start $tp_step $tp_end "$root_dir/n_bins_${n_bin}" sim_main.py True False $VAR_ARGS "xx" $PY_ARGS $RATE_CFG"
        ./run/abla_scalablility.sh $tp_start $tp_step $tp_end "$root_dir/n_bins_${n_bin}" sim_main.py True False "xx" "xx" $PY_ARGS $RATE_CFG

        if [ $scan_seed == True ]; then
            VAR_ARGS="--var_sim_cfg $var_sim_cfg --file_suffix $suffix --jitter_sim_en"
            if [ $exec_var_en == "True" ]; then
                VAR_ARGS="$VAR_ARGS --exec_var_en"
            fi
            for ((seed=$seed_start; seed<=$seed_end; seed++)); do
                
                DYN_ARGS="$VAR_ARGS --seed $seed"
                # file_suffix="jitter_en_0.2_seed_$seed"
                file_suffix="jitter_en_${suffix[@]:4}_seed_$seed"
                echo "./run/abla_scalablility.sh $tp_start $tp_step $tp_end $root_dir sim_main.py False True $DYN_ARGS $file_suffix $PY_ARGS $RATE_CFG" 
                ./run/abla_scalablility.sh $tp_start $tp_step $tp_end $root_dir sim_main.py False True $DYN_ARGS $file_suffix $PY_ARGS $RATE_CFG 
                wait
            done
        fi
    done
  }
done
wait
echo "********************************************"
