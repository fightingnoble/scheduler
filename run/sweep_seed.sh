#!/bin/bash
# set -x

# Default parameters
seed_start=${1:-0}
seed_end=${2:-9}
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

# ./run/exp_cmd_min_core.sh barycenter/core_scan True True True True 0 $PY_ARGS 
nohup ./run/exp_cmd_min_core.sh barycenter/core_scan True True False False 0 $PY_ARGS > log/barycenter/core_scan/cmd.log.txt 2>&1
# ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9  True True True True barycenter/aux_scan 0 $PY_ARGS &
nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9  True True False False barycenter/aux_scan 0 $PY_ARGS > log/barycenter/aux_scan/cmd.log.txt 2>&1&
# nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9  True True True True barycenter/aux_scan 0 --lateness_mode all_soft $PY_ARGS > log/barycenter/aux_scan/all_soft_cmd.log.txt 2>&1&
# nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 True True True True barycenter/lat_scan 0 $PY_ARGS > log/barycenter/lat_scan/cmd.log.txt 2>&1&
# nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 True True True True barycenter/lat_scan 0 --lateness_mode all_soft $PY_ARGS > log/barycenter/lat_scan/all_soft_cmd.log.txt 2>&1&
wait

echo "start scan_seed!!!!"

if [ $scan_seed == "True" ]; then
    for ((seed=$seed_start; seed<=$seed_end; seed++)); do
        {
            echo $seed
            sleep 2
            # ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True $seed $PY_ARGS &
            nohup ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True $seed $PY_ARGS > log/barycenter/$core_scan_dir/cmd_seed_$seed.log.txt 2>&1&
            # ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed $PY_ARGS &
            nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed $PY_ARGS > log/barycenter/$aux_scan_dir/cmd_seed_$seed.log.txt 2>&1&
            # ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed $PY_ARGS
            # nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed --lateness_mode all_soft $PY_ARGS > log/barycenter/$aux_scan_dir/all_soft_cmd_seed_$seed.log.txt 2>&1&
            # nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 False False True True barycenter/lat_scan $seed $PY_ARGS > log/barycenter/$lat_scan_dir/cmd_seed_$seed.log.txt 2>&1&
            # nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 False False True True barycenter/lat_scan $seed --lateness_mode all_soft $PY_ARGS > log/barycenter/$lat_scan_dir/all_soft_cmd_seed_$seed.log.txt 2>&1&
            wait
        }
    done
fi
wait
echo "force worst case!!!!"

# ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True $seed $PY_ARGS &
nohup ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True -1 $PY_ARGS > log/barycenter/$core_scan_dir/cmd_seed_wc.log.txt 2>&1&
# ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed $PY_ARGS &
nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan -1 $PY_ARGS > log/barycenter/$aux_scan_dir/cmd_seed_wc.log.txt 2>&1&

wait