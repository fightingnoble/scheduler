#!/bin/bash

# 设置默认值为 200 20 360
start=${1:-285}
step=${2:-20}
end=${3:-360}
cfg=${4:-"heavy"}

# read -p "请输入内容（light/heavy）：" cfg

if [ "$cfg" = "light" ]; then
    fn="profiling_light.csv"
elif [ "$cfg" = "medium" ]; then
    fn="profiling_medium.csv"
elif [ "$cfg" = "heavy" ]; then
    fn="profiling.csv"
else
    echo "无效的输入"
    exit 1
fi

echo "文件名为：$fn"

# 循环
for x in $(seq "$start" "$step" "$end"); do
    file_path="log/$cfg/$x/bin_pack_new_$x.log.txt"
    dir_path=$(dirname "$file_path")

    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
    fi

    rm "log/$cfg/$x/bin_pack_new_$x.log.txt"
    rm "log/$cfg/$x/glb_dyn_${x}_ideal.log.txt"
    rm "log/$cfg/$x/glb_dyn_${x}_jitter_dis.log.txt"
    rm "log/$cfg/$x/glb_dyn_${x}_jitter_en.log.txt"
    rm "log/$cfg/$x/dyn_${x}_jitter_dis.log.txt"
    rm "log/$cfg/$x/dyn_${x}_jitter_en.log.txt"
done

sh run/compare_glb.sh $start $step $end $cfg
sh run/static_optm_test.sh $start $step $end $cfg
sh run/compare_dyn.sh $start $step $end $cfg