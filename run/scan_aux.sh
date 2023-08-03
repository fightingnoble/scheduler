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
pre_alloc=${8:-"False"}
static_sim=${9:-"False"}
glb_dyn=${10:-"True"}
dyn=${11:-"True"}
PY_ARGS=${@:12}


echo "target path log/$root_dir/$num_cores" 

# 循环
for x in $(seq "$start" "$step" "$end"); do 
    {
        cfg=$(python run/cfg_parser.py --aux_scale_factor $x --n_p ${n_p} ${PY_ARGS} )
        folder_path="log/$root_dir/$num_cores/$cfg"
        file_path="$folder_path/bin_pack_new.log.txt"   
        dir_path=$(dirname "$file_path")

        # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
        if [ ! -d "$dir_path" ]; then
            mkdir -p "$dir_path"
        fi
        echo "start $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
        {
            if [ $glb_dyn == "True" ]; then
                echo "start (glb) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                if [ $static_sim == "True" ]; then
                    nohup python $p_fn --test_case glb_dynamic --aux_scale_factor $x --barrier_dis --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/glb_dyn_ideal.log.txt 2>&1 
                    nohup python $p_fn --test_case glb_dynamic --aux_scale_factor $x --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/glb_dyn_jitter_dis.log.txt 2>&1 
                fi 
                nohup python $p_fn --test_case glb_dynamic --aux_scale_factor $x --jitter_sim_en --file_suffix var_0.2 --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/glb_dyn_jitter_en.log.txt 2>&1 
                echo "finish (glb) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
        }
        {
            if [ $pre_alloc == "True" ]; then
                echo "start (bin_pack) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                nohup python $p_fn --test_case bin_pack_new --aux_scale_factor $x --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/bin_pack_new.log.txt 2>&1 
                echo "finish (bin_pack) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
            if [ $dyn == "True" ]; then
                echo "start (dyn) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                if [ $static_sim == "True" ]; then
                    nohup python $p_fn --test_case dynamic --aux_scale_factor $x --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/dyn_jitter_dis.log.txt 2>&1 
                fi
                nohup python $p_fn --test_case dynamic --aux_scale_factor $x --jitter_sim_en --file_suffix var_0.2 --num_cores ${num_cores} --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $folder_path/dyn_jitter_en.log.txt 2>&1 
                echo "finish (dyn) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
        }
        echo "start (log_analyse) $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
        python -m analyze.log_analyse --folder $folder_path --output $folder_path/new_bin_pack.csv --n_p ${n_p} --aux_scale_factor $x --get_ref_num_exec
        echo "finish $num_cores/$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
    }&
done
wait

