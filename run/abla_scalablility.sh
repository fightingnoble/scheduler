#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
aux_start=${1:-1}
aux_step=${2:-1}
aux_end=${3:-10}
root_dir=${4:-"aux_scan"}
p_fn=${5:-"sim_main.py"}
pre_alloc=${6:-"False"}
dyn=${7:-"True"}
DYN_ARGS=${8:-"0"}
file_suffix=${9:-""}
PY_ARGS=${@:10}


echo "|target path log/$root_dir/." 
for lat in $(seq 0.1 -0.01 0.08); do
    for n_aux in $(seq "$aux_start" "$aux_step" "$aux_end"); do 
        {
            LOAD_VAR="--e2e_latency $lat --aux_scale_factor $n_aux"
            cfg=$(python run/cfg_parser.py $PY_ARGS $LOAD_VAR)
            folder_path="log/$root_dir/./$cfg"
            file_path="$folder_path/bin_pack_new.log.txt"   
            dir_path=$(dirname "$file_path")
            
            # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
            if [ ! -d "$dir_path" ]; then
                mkdir -p "$dir_path"
            fi
            echo "=start ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            {
                if [ $pre_alloc == "True" ]; then
                    echo "==start (bin_pack-coaleasing) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                    nohup python $p_fn --test_case bin_pack_new $PY_ARGS $LOAD_VAR > $dir_path/bin_pack_new.log.txt 2>&1 
                    echo "==finish (bin_pack-coaleasing) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                fi
                if [ $dyn == "True" ]; then
                    
                    log_name=$dir_path/cyclic_${file_suffix}.log.txt
                    if [ -f $log_name ]; then
                        rm $log_name
                    fi
                    echo "==start (cyclic) ./$cfg/cyclic_${file_suffix}" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                    nohup python $p_fn --test_case cyclic $PY_ARGS $LOAD_VAR $DYN_ARGS > $log_name 2>&1& 
                    echo "==finish (cyclic) ./$cfg/cyclic_${file_suffix}" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                fi
            }
            echo "finish ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            sleep 1.5

        } 
    done
    echo "-------------------"
    wait
done

wait

