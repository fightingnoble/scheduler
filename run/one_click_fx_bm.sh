#!/bin/bash

# 设置默认值为 200 20 360
start=${1:-285}
step=${2:-20}
end=${3:-360}
cfg=${4:-"heavy"}
p_fn=${5:-"allocator_agent.py"}
PY_ARGS=${@:6}

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
cfg=$(python run/file_path_prepare.py --start $start --step $step --end $end --profiling_filename $fn ${PY_ARGS})
echo "cfg is $cfg"
./run/compare_glb.sh $start $step $end $cfg ${p_fn} --profiling_filename $fn ${PY_ARGS}
./run/static_optm_test.sh $start $step $end ${p_fn} $cfg --profiling_filename $fn ${PY_ARGS}
./run/compare_dyn.sh $start $step $end $cfg ${p_fn} --profiling_filename $fn ${PY_ARGS}