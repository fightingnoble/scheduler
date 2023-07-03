#!/bin/bash
# set -x 
# 设置默认值为 200 20 360
start=${1:-285}
step=${2:-20}
end=${3:-360}
p_fn=${4:-"sim_main.py"}
PY_ARGS=${@:5}

# read -p "请输入内容（light/heavy）：" cfg

cfg=$(python run/file_path_prepare.py --start $start --step $step --end $end ${PY_ARGS})
echo "cfg is $cfg"
./run/compare_glb.sh $start $step $end $cfg ${p_fn} ${PY_ARGS}
./run/static_optm_test.sh $start $step $end $cfg ${p_fn} ${PY_ARGS}
./run/compare_dyn.sh $start $step $end $cfg ${p_fn} ${PY_ARGS}