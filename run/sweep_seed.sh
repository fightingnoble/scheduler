#!/bin/bash
# set -x

# Default parameters
seed_end=${1:-9}
glb_dyn=${2:-"True"}
scan_seed=${3:-"False"}
plot=${4:-'dis'}

# plot_args, "--plot ''", if plot is 'dis', else " --plot True "
if [ $plot == 'dis' ]; then
    PY_ARGS="--bin_pack_cfg "Bp_reside.json""
else
    PY_ARGS="--bin_pack_cfg "Bp_reside.json" --plot True"
fi

core_scan_dir=core_scan
aux_scan_dir=aux_scan
lat_scan_dir=lat_scan
mkdir -p log/barycenter/$core_scan_dir
mkdir -p log/barycenter/$aux_scan_dir
mkdir -p log/barycenter/$lat_scan_dir

echo "start scan_seed!!!!"

if [ $scan_seed != "True" ]; then
    seed_end=0
fi

for ((seed=0; seed<=$seed_end; seed++)); do
    {
        echo $seed
        # sleep 2
        dyn="True"
        # if seed is 0
        if [ $seed == 0 ]; then
            pre_alloc="True"
            static_sim="True"
        else
            pre_alloc="False"
            static_sim="False"
        fi
        echo "nohup ./run/exp_cmd_min_core.sh barycenter/core_scan $pre_alloc $static_sim $glb_dyn $dyn $seed $PY_ARGS > log/barycenter/$core_scan_dir/cmd_seed_$seed.log.txt 2>&1&"
        nohup ./run/exp_cmd_min_core.sh barycenter/core_scan $pre_alloc $static_sim $glb_dyn $dyn $seed $PY_ARGS > log/barycenter/$core_scan_dir/cmd_seed_$seed.log.txt 2>&1&
        echo "nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 $pre_alloc $static_sim $glb_dyn $dyn barycenter/aux_scan $seed $PY_ARGS > log/barycenter/$aux_scan_dir/cmd_seed_$seed.log.txt 2>&1&"
        wait
    }
done

if [ $scan_seed == "True" ]; then
    echo "force worst case!!!!"
    echo "nohup ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True -1 $PY_ARGS > log/barycenter/$core_scan_dir/cmd_seed_wc.log.txt 2>&1&"
    nohup ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True -1 $PY_ARGS > log/barycenter/$core_scan_dir/cmd_seed_wc.log.txt 2>&1&
    echo "nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan -1 $PY_ARGS > log/barycenter/$aux_scan_dir/cmd_seed_wc.log.txt 2>&1&"
    nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan -1 $PY_ARGS > log/barycenter/$aux_scan_dir/cmd_seed_wc.log.txt 2>&1&
fi
wait
