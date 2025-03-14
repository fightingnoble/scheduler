#!/bin/bash
set -x

# 1st stage: pre allocation
# ```
#                 "--test_case", "bin_pack_new",
#                 "--bin_pack_cfg", "Bp_split.json", 
# ```
# Basic setting:
# No task colocates in the same partition, 
# ```
#                 "--num_bins", "-1",
#                 "--root_dir", "coalescing_scan/n_bins_max",
#                  --jitter_t_comp_ratio 0.2 --wsc_slack_ratio 1 --exec_t_comp_ratioA 0.3 --exec_t_comp_ratioB 0.15
# ```

# 2nd stage: dynamic allocation
# variability setting: 
# ```
#                 "--seed", "-1",
#                 "--var_sim_cfg", "var_sim_cfg.json",
                # "--jitter_sim_en", "--exec_var_en",
# ```

# variable to be varied: 
# ```
#                 "--exec_var_para", "{\"scale\": 0.31}",
#                 "--jitter_sim_para", "{\"scale\": 0.1}",
#                 "--file_suffix", "var_0.1_0.31",
# ```

# Two cases:  
# 1. Cyc-RT-like (Robustness)
#   In each partition, task are multiplexed, and isolated Temporally, 
#   all tasks is preserved for a fixed processing window, 
#   regardless their runtime status. 
# ```
#                 "--test_case", "cyclic",
#                 "--bin_pack_cfg", "Bp_split.json", 
# ```

# Cyc-RT-S (Robustness improvement)
#   In each partition, task are multiplexed, and remove temporal isolation

                # "--test_case", "partitioned_glb_dynamic",
                # "--bin_pack_cfg", "Bp_split.json", 
                # "--barrier_dis",


scan_type=${1:-"tp"}
scan_seed=${2:-"False"}
plot=${3:-"dis"}
tp_start=${4:-0}  
tp_step=${5:-1}
tp_end=${6:-9}
seed_start=${7:-0}
seed_end=${8:-9}  
root_dir=${9:-"coalescing_scan"}

cfg_idx_min=1
list_n_bin_cfg=(8 4) 
bin_unroll_factor=0
cfg_unroll_factor=0
ratioB_incr_list=(0)

if [ $scan_type == "bin" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    list_n_bin_cfg=(24 20 18 16 12 10 2 1) 
    bin_unroll_factor=3
fi

# basic setting
if [ $plot == "dis" ]; then
    Orin_PY_ARGS=""
else
    Orin_PY_ARGS="--plot True"
fi 

n_p=3
Orin_PY_ARGS="$Orin_PY_ARGS --profiling_filename profiling/profiling_light.csv --n_p $n_p --gen_benchmark"

n_bin=-1

exec_t_comp_ratioA=0.3
exec_t_comp_ratioB=0.15
wsc_slack_ratio=1
var_sim_cfg="var_sim_cfg.json"
jitter_comp_cfg=0.2

printf "\nGenerating initial bin packing...\n" && sleep 1
bin_root="$root_dir/n_bins_max"
RATE_CFG="--jitter_t_comp_ratio $jitter_comp_cfg --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA --num_bins $n_bin --root_dir $bin_root"
PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_split.json" --exec_t_comp_ratioB $exec_t_comp_ratioB"

lat=0.1
n_aux=4
LOAD_VAR="--e2e_latency $lat --aux_scale_factor $n_aux"
cfg=$(python run/cfg_parser.py ${PY_ARGS} ${LOAD_VAR})
folder_path="log/$bin_root/./$cfg"
file_path="$folder_path/bin_pack_new.log.txt"   
dir_path=$(dirname "$file_path")
# 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi

log_name=$dir_path/bin_pack_new.log.txt
if [ -f $log_name ]; then
    rm $log_name
fi
case_para=("--test_case bin_pack_new") 
# echo "sim_main.py ${case_para} ${PY_ARGS} ${LOAD_VAR} > $log_name 2>&1"
# python sim_main.py ${case_para} ${PY_ARGS} ${LOAD_VAR} > $log_name 2>&1 
if [ $? -eq 0 ]; then
    echo "initial bin packing success."
else
    echo "initial bin packing fail."
    exit 1
fi


VAR_ARGS="--var_sim_cfg $var_sim_cfg --jitter_sim_en"
VAR_ARGS="$VAR_ARGS --exec_var_en"

seed=-1
DYN_ARGS="$VAR_ARGS --seed ${seed}"
DYN_ARGS="$DYN_ARGS --forbid_miss"
case_para=("--test_case partitioned_glb_dynamic --barrier_dis" "--test_case cyclic") 
case_input=("partitioned_glb_dynamic" "cyclic")
case_sign=("pglb" "cyclic")

start_time=`date +%s`              #定义脚本运行的开始时间
[ -e /tmp/fd1 ] || mkfifo /tmp/fd1 #创建有名管道
exec 3<>/tmp/fd1                   #创建文件描述符，以可读（<）可写（>）的方式关联管道文件，这时候文件描述符3就有了有名管道文件的所有特性
rm -rf /tmp/fd1                    #关联后的文件描述符拥有管道文件的所有特性,所以这时候管道文件可以删除，我们留下文件描述符来用就可以了
for ((i=1;i<=20;i++))
do
        echo >&3                   #&3代表引用文件描述符3，这条命令代表往管道里面放入了一个"令牌"
done
 
# 设置 var_sen 和 var_slowdown 的初始范围
var_sen_low=0.0
var_sen_high=0.31
var_slowdown_low=0.0
var_slowdown_high=0.4

# 设置最细网格粒度
grid_granularity=0.01

# 定义 Python 脚本路径
python_script="sim_main.py"

# 定义函数来执行 Python 脚本并返回成功/失败标志
execute_and_check() {
    local var_sen=$1
    local var_slowdown=$2
    local case_idx=$3
    local case_para=${case_para[$case_idx]}
    local case_sign=${case_sign[$case_idx]}
    local Jitter_sym="var_${var_sen}(J)"
    local suffix="${Jitter_sym}_${var_slowdown}(T)"
    local var_para="--jitter_sim_para "{\"scale\":$var_sen}" --exec_var_para "{\"scale\":$var_slowdown}" --file_suffix ${suffix}"
    local log_name=$dir_path/${case_sign}_jitter_en_${suffix[@]:4}_seed_${seed}.log.txt

    sleep 1 
    if [ -f $log_name ]; then
        rm $log_name
    fi
    # 执行 Python 脚本
    # echo "sim_main.py ${case_para} ${PY_ARGS} ${DYN_ARGS} ${var_para} ${LOAD_VAR} > $log_name 2>&1"
    python sim_main.py ${case_para} ${PY_ARGS} ${DYN_ARGS} ${var_para} ${LOAD_VAR} > $log_name 2>&1 
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "${case_sign} ${suffix} success."
        return 0
    else
        echo "${case_sign} ${suffix} fail."
        return 1
    fi
}

# 定义二分查找函数
binary_search() {
    local search_var_name=$1
    local var_low=$2
    local var_high=$3
    local fixed_var_name=$4
    local fixed_var_value=$5
    local case_idx=$6

    # 检查边界情况
    # 最大边界成功 -> 都成功
    execute_and_check $var_high $fixed_var_value $case_idx
    if [ $? -eq 0 ]; then 
        echo "Case:$((case_idx+1)) w/ $fixed_var_name $fixed_var_value, ${search_var_name} boundary $var_high - inf" 
        return 0
    fi

    # 最小边界失败 -> 都失败
    execute_and_check $var_low $fixed_var_value $case_idx
    if [ $? -eq 1 ]; then
        echo "Case:$((case_idx+1)) w/ $fixed_var_name $fixed_var_value, ${search_var_name} boundary -inf - $var_low" 
        return 0
    fi

    
    while (( $(echo "$var_high - $var_low > $grid_granularity" | bc -l) )); do
        local mid=$(echo "($var_low + $var_high) / 2" | bc -l)
        mid=$(printf "%.2f" $mid) # 四舍五入到两位小数
        execute_and_check $mid $fixed_var_value $case_idx
        if [ $? -eq 0 ]; then
            var_low=$mid
        else
            var_high=$mid
        fi
    done
    # 返回高低两个边界
    echo "Case:$((case_idx+1)) w/ $fixed_var_name $fixed_var_value, ${search_var_name} boundary $var_low - $var_high" 
    return 0
}

# 主逻辑
# scan over the variation variation
for case_idx in 0 1; do
    for var_slowdown in $(seq 0 0.01 0.4); do
    read -u3                           #代表从管道中读取一个令牌
    {
        var_sen_low=0.0
        var_sen_high=0.4
        binary_search "var_sen" $var_sen_low $var_sen_high "var_slowdown" $var_slowdown $case_idx
        echo >&3                   #代表我这一次命令执行到最后，把令牌放回管道
    }&
    done
done
wait 
# sim_main.py --num_bins -1 --root_dir coalescing_scan/n_bins_max --gen_benchmark --e2e_latency 0.1 --aux_scale_factor 4 
# --var_sim_cfg var_sim_cfg.json --jitter_t_comp_ratio 0.2 --wsc_slack_ratio 1 --exec_t_comp_ratioA 0.3 --exec_t_comp_ratioB 0.15 --test_case bin_pack_new --bin_pack_cfg Bp_split.json 

stop_time=`date +%s`  #定义脚本运行的结束时间
 
echo "TIME:`expr $stop_time - $start_time`"
exec 3<&-                       #关闭文件描述符的读
exec 3>&-                       #关闭文件描述符的写
