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
    e2e_t=0.1
elif [ "$cfg" = "medium" ]; then
    fn="profiling_medium.csv"
    e2e_t=0.09
elif [ "$cfg" = "heavy" ]; then
    fn="profiling.csv"
    e2e_t=0.09
else
    echo "无效的输入"
    exit 1
fi

echo "文件名为：$fn"
cfg=$(python run/cfg_parser.py --profiling_filename $fn --e2e_latency $e2e_t ${PY_ARGS})
echo "cfg is $cfg"
python run/file_path_prepare.py --start $start --step $step --end $end --cfg_n $cfg 
./run/compare_glb.sh $start $step $end $cfg ${p_fn} --profiling_filename $fn --e2e_latency $e2e_t ${PY_ARGS}
./run/static_optm_test.sh $start $step $end $cfg ${p_fn} --profiling_filename $fn --e2e_latency $e2e_t ${PY_ARGS}
./run/compare_dyn.sh $start $step $end $cfg ${p_fn} --profiling_filename $fn --e2e_latency $e2e_t ${PY_ARGS}