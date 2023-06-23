#!/bin/bash

# 设置默认值为 200 20 360
start=${1:-285}
step=${2:-20}
end=${3:-360}

sh run/compare_glb.sh $start $step $end
sh run/static_optm_test.sh $start $step $end
sh run/compare_dyn.sh $start $step $end