#!/bin/bash
# set -x

# Default parameters
start=${1:-1}
step=${2:-1}
end=${3:-10}
root_dir=${4:-"core_scan"}
p_fn=${5:-"sim_main.py"}
n_p=${6:-"3"}
pre_alloc=${7:-"False"}
static_sim=${8:-"False"}
glb_dyn=${9:-"True"}
dyn=${10:-"True"}
PY_ARGS=${@:11}

echo "target path log/$root_dir/$cfg"
cfg=$(python run/cfg_parser.py --n_p ${n_p} ${PY_ARGS})

for x in $(seq "$start" "$step" "$end"); do
    {
        file_path="log/$root_dir/$cfg/$x/bin_pack_new_$x.log.txt"
        dir_path=$(dirname "$file_path")

        # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
        if [ ! -d "$dir_path" ]; then
            mkdir -p "$dir_path"
        fi
        echo "start $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
        {
            if [ $glb_dyn == "True" ]; then
                echo "start (glb) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                if [ "$static_sim" == "True" ]; then
                    if [ -f "$dir_path/glb_dyn_${x}_ideal.log.txt" ]; then
                        rm "$dir_path/glb_dyn_${x}_ideal.log.txt"
                    fi
                    if [ -f "$dir_path/glb_dyn_${x}_jitter_dis.log.txt" ]; then
                        rm "$dir_path/glb_dyn_${x}_jitter_dis.log.txt"
                    fi

                    nohup python $p_fn --test_case glb_dynamic --barrier_dis --num_cores $x --root_dir ${root_dir} --n_p $n_p ${PY_ARGS} > $dir_path/glb_dyn_${x}_ideal.log.txt 2>&1 
                    nohup python $p_fn --test_case glb_dynamic --num_cores $x --root_dir ${root_dir} --n_p $n_p ${PY_ARGS} > $dir_path/glb_dyn_${x}_jitter_dis.log.txt 2>&1 
                fi
                if [ -f "$dir_path/glb_dyn_${x}_jitter_en.log.txt" ]; then
                    rm "$dir_path/glb_dyn_${x}_jitter_en.log.txt"
                fi
                nohup python $p_fn --test_case glb_dynamic --jitter_sim_en --file_suffix var_0.2 --num_cores $x --root_dir ${root_dir} --n_p $n_p ${PY_ARGS} > $dir_path/glb_dyn_${x}_jitter_en.log.txt 2>&1 
                echo "finish (glb) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`

            fi
        }
        {
            if [ $pre_alloc == "True" ]; then
                echo "start (bin_pack) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
                nohup python $p_fn  --test_case bin_pack_new --num_cores $x --root_dir ${root_dir} --n_p $n_p ${PY_ARGS} > $dir_path/bin_pack_new_$x.log.txt 2>&1 
                echo "finish (bin_pack) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
            if [ $dyn == "True" ]; then
                echo "start (dyn) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                sleep 0.1
                if [ $static_sim == "True" ]; then
                    if [ -f "$dir_path/dyn_${x}_jitter_dis.log.txt" ]; then
                        rm "$dir_path/dyn_${x}_jitter_dis.log.txt"
                    fi
                    nohup python $p_fn --test_case dynamic --num_cores $x --root_dir ${root_dir} --n_p $n_p ${PY_ARGS} > $dir_path/dyn_${x}_jitter_dis.log.txt 2>&1 
                fi
                if [ -f "$dir_path/dyn_${x}_jitter_en.log.txt" ]; then
                    rm "$dir_path/dyn_${x}_jitter_en.log.txt"
                fi
                nohup python $p_fn --test_case dynamic --jitter_sim_en --file_suffix var_0.2 --num_cores $x --root_dir ${root_dir} --n_p $n_p ${PY_ARGS} > $dir_path/dyn_${x}_jitter_en.log.txt 2>&1 
                echo "finish (dyn) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
        }
        echo "start (log_analyse) $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
        python -m analyze.log_analyse --folder ./$dir_path/ --output ./$dir_path/new_bin_pack$x.csv 
        echo "finish $cfg/$x" `date "+%Y-%m-%d %H:%M:%S.%3N"`
    }&
done
wait