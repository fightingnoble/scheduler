#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
start=${1:-200}
step=${2:-20}
end=${3:-360}
cfg=${4:-"heavy"}
p_fn=${5:-"allocator_agent.py"}
PY_ARGS=${@:6}

# 循环
for x in $(seq "$start" "$step" "$end"); do
    file_path="log/$cfg/$x/bin_pack_new_$x.log.txt"
    dir_path=$(dirname "$file_path")

    if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
    fi

    if [ -f "log/$cfg/$x/dyn_${x}_jitter_dis.log.txt" ]; then
        rm "log/$cfg/$x/dyn_${x}_jitter_dis.log.txt"
    fi
    if [ -f "log/$cfg/$x/dyn_${x}_jitter_en.log.txt" ]; then
        rm "log/$cfg/$x/dyn_${x}_jitter_en.log.txt"
    fi
    # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作

    nohup python $p_fn --test_case dynamic --num_cores $x --n_p 3 ${PY_ARGS} > log/$cfg/$x/dyn_${x}_jitter_dis.log.txt 2>&1 &
    nohup python $p_fn --test_case dynamic --jitter_sim_en --file_suffix var_0.2 --num_cores $x --n_p 3 ${PY_ARGS} > log/$cfg/$x/dyn_${x}_jitter_en.log.txt 2>&1 &
done
wait

# 循环
for x in $(seq "$start" "$step" "$end"); do
    python log_analyse.py --folder ./log/$cfg/$x/ --output ./log/$cfg/$x/new_bin_pack$x.csv 
done
wait
