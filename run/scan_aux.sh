#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
start=${1:-1}
step=${2:-1}
end=${3:-10}
num_cores=${4:-"300"}
root_dir=${5:-"aux_scan"}
p_fn=${6:-"sim_main.py"}
n_p=${7:-"3"}
PY_ARGS=${@:8}


# 循环
for x in $(seq "$start" "$step" "$end"); do 
    cfg=$(python run/cfg_parser.py --aux_scale_factor $x --n_p ${n_p} ${PY_ARGS} )
    echo "cfg is $root_dir/$num_cores/$cfg"
    folder_path="log/$root_dir/$num_cores/$cfg"
    file_path="$folder_path/bin_pack_new.log.txt"   
    dir_path=$(dirname "$file_path")

    if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
    fi

    # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
    nohup python $p_fn --test_case glb_dynamic --aux_scale_factor $x --barrier_dis --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/glb_dyn_ideal.log.txt 2>&1 &
    nohup python $p_fn --test_case glb_dynamic --aux_scale_factor $x --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/glb_dyn_jitter_dis.log.txt 2>&1 &
    nohup python $p_fn --test_case glb_dynamic --aux_scale_factor $x --jitter_sim_en --file_suffix var_0.2 --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/glb_dyn_jitter_en.log.txt 2>&1 &
    nohup python $p_fn --test_case bin_pack_new --aux_scale_factor $x --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/bin_pack_new.log.txt 2>&1 &
    wait
    nohup python $p_fn --test_case dynamic --aux_scale_factor $x --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/dyn_jitter_dis.log.txt 2>&1 &
    nohup python $p_fn --test_case dynamic --aux_scale_factor $x --jitter_sim_en --file_suffix var_0.2 --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/dyn_jitter_en.log.txt 2>&1 &
    wait
    python log_analyse.py --folder $folder_path --output $folder_path/new_bin_pack.csv --n_p ${n_p} --aux_scale_factor $x --get_ref_num_exec
    wait
done

