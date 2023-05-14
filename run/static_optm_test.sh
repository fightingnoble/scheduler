#!/bin/bash

# 设置默认值为 196 10 300
start=${1:-196}
step=${2:-10}
end=${3:-300}

# 循环
for x in $(seq "$start" "$step" "$end"); do
    file_path="log/$x/bin_pack_new_$x.log.txt"
    dir_path=$(dirname "$file_path")

    if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
    fi

    # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
    nohup python allocator_agent.py  --test_case bin_pack_new --num_cores "$x"  --n_p 3 > log/$x/bin_pack_new_$x.log.txt 2>&1 &
done
wait

# 循环
for x in $(seq "$start" "$step" "$end"); do
    python log_analyse.py --folder ./log/$x/ --output ./log/$x/new_bin_pack$x.csv 
done
wait
