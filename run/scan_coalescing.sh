#!/bin/bash
# set -x 

# 设置默认值为 200 20 360
start=${1:-1}
step=${2:-1}
end=${3:-10}
root_dir=${4:-"aux_scan"}
p_fn=${5:-"sim_main.py"}
n_p=${6:-"3"}
pre_alloc=${7:-"False"}
dyn=${8:-"True"}
seed=${9:-"0"}
PY_ARGS=${@:10}


echo "target path log/$root_dir/." 

# 循环
for x in $(seq "$start" "$step" "$end"); do 
    sleep 1
    {
        cfg=$(python run/cfg_parser.py --aux_scale_factor $x --n_p ${n_p} ${PY_ARGS} )
        folder_path="log/$root_dir/./$cfg"
        file_path="$folder_path/bin_pack_new.log.txt"   
        dir_path=$(dirname "$file_path")

        # 目录已经存在或者已经创建成功，接下来就可以进行其他的操作
        if [ ! -d "$dir_path" ]; then
            mkdir -p "$dir_path"
        fi
        echo "start ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
        {
            if [ $pre_alloc == "True" ]; then
                echo "start (bin_pack) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                nohup python $p_fn --test_case bin_pack_new --aux_scale_factor $x  --root_dir ${root_dir} --n_p ${n_p} ${PY_ARGS}  > $dir_path/bin_pack_new.log.txt 2>&1 
                echo "finish (bin_pack) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
            if [ $dyn == "True" ]; then
                echo "start (dyn) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`

                if [ -f "$dir_path/glb_dyn_${x}_jitter_en.log.txt" ]; then
                    rm "$dir_path/glb_dyn_${x}_jitter_en.log.txt"
                fi
                echo "start (dyn seed_$seed) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                nohup python $p_fn --test_case dynamic --aux_scale_factor $x --root_dir ${root_dir} --n_p ${n_p} --jitter_sim_en --file_suffix var_0.2 --seed $seed ${PY_ARGS} > $dir_path/dyn_jitter_en_seed_$seed.log.txt 2>&1& 
                echo "finish (dyn seed_$seed) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
                echo "finish (dyn) ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
            fi
        }
        echo "finish ./$cfg" `date "+%Y-%m-%d %H:%M:%S.%3N"`
    }
done
wait

