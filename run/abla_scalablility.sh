#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
case=${1:-"bp"}
aux_start=${2:-1}  
aux_step=${3:-1}
aux_end=${4:-10}
root_dir=${5:-"aux_scan"}
p_fn=${6:-"sim_main.py"}
file_suffix=${7:-""} 
# DYN_ARGS=${7:-"0"}
PY_ARGS=${@:8}

if [ $file_suffix == "xx" ] || [ $file_suffix == "x" ]; then
    file_suffix=""
fi

case_input="bin_pack_new"
case_sign="bin_pack_new"
if [ $case == "dyn" ]; then
    case_input="dynamic"
    case_sign="dyn"
elif [ $case == "pglb" ]; then
    case_input="partitioned_glb_dynamic"
    case_sign="pglb"
elif [ $case == "cyc" ]; then
    case_input="cyclic"
    case_sign="cyclic"
fi


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

                if [ $case == "bp" ]; then
                    log_name=$dir_path/${case_sign}${file_suffix}.log.txt
                    if [ -f $log_name ]; then
                        rm $log_name
                    fi
                    echo "==start (bin_pack-coaleasing) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                    nohup python $p_fn --test_case bin_pack_new $PY_ARGS $LOAD_VAR > $log_name 2>&1 
                    echo "==finish (bin_pack-coaleasing) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                
                elif [ $case != "" ]; then
                    log_name=$dir_path/${case_sign}_${file_suffix}.log.txt
                    if [ -f $log_name ]; then
                        rm $log_name
                    fi
                    echo "==start (${case_input}) ./$cfg/${case_sign}_${file_suffix}" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                    nohup python $p_fn --test_case ${case_input} $PY_ARGS $LOAD_VAR > $log_name 2>&1& 
                    echo "==finish (${case_input}) ./$cfg/${case_sign}_${file_suffix}" `date "+%Y-%m-%d %H:%M:%S.%3N"`
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

