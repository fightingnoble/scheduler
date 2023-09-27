#!/bin/bash
# set -x

# Default parameters
seed_start=${1:-0}
seed_end=${2:-9}
plot=${3:-"False"}
check_max_core=${4:-"False"}
for ((seed=$seed_start; seed<=$seed_end; seed++)); do
    {
        echo $seed
        core_scan_dir=core_scan
        aux_scan_dir=aux_scan
        lat_scan_dir=lat_scan
        mkdir -p log/barycenter/$core_scan_dir
        mkdir -p log/barycenter/$aux_scan_dir
        mkdir -p log/barycenter/$lat_scan_dir

        nohup ./run/exp_cmd_min_core.sh barycenter/core_scan False False True True $seed --bin_sort barycenter --plot $plot > log/barycenter/$core_scan_dir/cmd_seed_$seed.log.txt 2>&1&
        nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed --bin_sort barycenter --plot $plot > log/barycenter/$aux_scan_dir/cmd_seed_$seed.log.txt 2>&1&
        nohup ./run/exp_cmd_max_tp.sh 0 25 200 0 1 9 False False True True barycenter/aux_scan $seed --lateness_mode all_soft --bin_sort barycenter --plot $plot > log/barycenter/$aux_scan_dir/all_soft_cmd_seed_$seed.log.txt 2>&1&
        # nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 False False True True barycenter/lat_scan $seed --bin_sort barycenter --plot $plot > log/barycenter/$lat_scan_dir/cmd_seed_$seed.log.txt 2>&1&
        # nohup ./run/exp_cmd_ctx_switch.sh 0 25 0 0 1 16 False False True True barycenter/lat_scan $seed --lateness_mode all_soft --bin_sort barycenter --plot $plot > log/barycenter/$lat_scan_dir/all_soft_cmd_seed_$seed.log.txt 2>&1&
    }
done

