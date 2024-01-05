#!/bin/bash
# set -x

scan_type=${1:-"tp"}
scan_seed=${2:-"False"}
plot=${3:-"dis"}
tp_start=${4:-0}  
tp_step=${5:-1}
tp_end=${6:-9}
seed_start=${7:-0}
seed_end=${8:-9}  
root_dir=${9:-"coalescing_scan"}

cfg_idx_max=1
list_n_bin_cfg=(8 4) 
ratioB_incr_list=(0)
if [ $scan_type == "bin" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    list_n_bin_cfg=(24 22 20 18 16 14 12 10 8 6 4 2 1) 
elif [ $scan_type == "cfg" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    cfg_idx_max=7
elif [ $scan_type == "ratioB" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    ratioB_incr_list=(-0.05 0. 0.05 0.1) 
fi

# plot_args, "--plot ''", if plot is 'dis', else " --plot True "
if [ $plot == "dis" ]; then
    Orin_PY_ARGS=""
else
    Orin_PY_ARGS="--plot True"
fi
n_p=3
Orin_PY_ARGS="$Orin_PY_ARGS --profiling_filename profiling/profiling_light.csv --n_p $n_p --gen_benchmark"

echo "********************************************"

# exec_t_comp_ratioA, exec_t_comp_ratioB, wsc_slack_ratio exec_var_en
list_exec_t_comp_ratioA=(0.3 0 0.25 0.2 0.15 0 0 0)
list_exec_t_comp_ratioB=(0.15 0.05 0.15 0.15 0.15 0.05 0.05 0.05)
list_wsc_slack_ratio=(1 1 1 1 1 1 1 1)
list_exec_var_en=(True False True True True False False False)
list_suffix=("var_0.2_0.3" "var_0.2" "var_0.25_0.15" "var_0.2_0.1" "var_0.15_0.05" "var_0.15" "var_0.1" "var_0.05")
list_var_sim_cfg=("var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json")
list_jitter_comp_cfg=(0.2 0.2 0.15 0.1 0.05 0.15 0.1 0.05)


for i in $(seq 0 1 $cfg_idx_max); do
  {
    exec_t_comp_ratioA=${list_exec_t_comp_ratioA[$i]}
    exec_t_comp_ratioB=${list_exec_t_comp_ratioB[$i]}
    wsc_slack_ratio=${list_wsc_slack_ratio[$i]}
    exec_var_en=${list_exec_var_en[$i]}
    suffix=${list_suffix[$i]}
    var_sim_cfg=${list_var_sim_cfg[$i]}
    jitter_comp_cfg=${list_jitter_comp_cfg[$i]}

    for n_bin in ${list_n_bin_cfg[@]}; do
        echo "********************************************"
        echo "n_bin: $n_bin"
        echo "********************************************"

        # initial bin packing generation
        bin_root="$root_dir/n_bins_${n_bin}"
        RATE_CFG="--jitter_t_comp_ratio $jitter_comp_cfg --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA --num_bins $n_bin --root_dir $bin_root"
        PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_split.json" --exec_t_comp_ratioB $exec_t_comp_ratioB"
        echo "./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py "xx" $PY_ARGS"
        ./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py  "xx" $PY_ARGS
        
        # repacking: replace the item in each bin 
        for ratioB in ${ratioB_incr_list[@]}; do
            new_ratioB=`echo "$exec_t_comp_ratioB + $ratioB"|bc`
            PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_repack.json" --exec_t_comp_ratioB $new_ratioB"
            echo "./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py "_ov_${exec_t_comp_ratioB}_repack" $PY_ARGS"
            ./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py "_ov_${exec_t_comp_ratioB}_repack" $PY_ARGS 
        done
        wait

        if [ $scan_seed == True ]; then
            VAR_ARGS="--var_sim_cfg $var_sim_cfg --jitter_sim_en"
            if [ $exec_var_en == "True" ]; then
                VAR_ARGS="$VAR_ARGS --exec_var_en"
            fi
            for ((seed=$seed_start; seed<=$seed_end; seed++)); do
                {
                    DYN_ARGS="$VAR_ARGS --seed $seed"

                    PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_split.json" --exec_t_comp_ratioB $exec_t_comp_ratioB --file_suffix $suffix"
                    # file_suffix="jitter_en_0.2_seed_$seed"
                    log_suffix="jitter_en_${suffix[@]:4}_seed_$seed"

                    # isolated-time within each bin
                    echo "./run/abla_scalablility.sh cyc $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS" 
                    ./run/abla_scalablility.sh cyc $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS 

                    # enable time sharing within each bin
                    echo "./run/abla_scalablility.sh pglb $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS" 
                    ./run/abla_scalablility.sh pglb $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS 

                    # constrained sharing among and along chains within each bin
                    for ratioB in ${ratioB_incr_list[@]}; do
                        {
                            new_ratioB=`echo $exec_t_comp_ratioB $ratioB | awk '{ printf "%0.2f\n", $1+$2}'`
                            file_suffix="$suffix_ov_${new_ratioB}_repack"
                            PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_repack.json" --i_file_suffix "_ov_${new_ratioB}_repack" --exec_t_comp_ratioB $new_ratioB --file_suffix $file_suffix"
                            log_suffix="jitter_en_${suffix[@]:4}_seed_${seed}_repack"
                            echo "./run/abla_scalablility.sh dyn $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS " 
                            ./run/abla_scalablility.sh dyn $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS 
                        }&
                    done
                }&
                wait
            done
        fi
    done
  }
done
wait
echo "********************************************"
