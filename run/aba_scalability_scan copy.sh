#!/bin/bash
# set -x

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
elif [ $scan_type == "cfg" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    cfg_idx_max=7
    cfg_unroll_factor=2
elif [ $scan_type == "ratioB" ]; then
    tp_start=9
    tp_step=1
    tp_end=9
    # ratioB_incr_list=(-0.05 0.05 0.1) 
    ratioB_incr_list=(-0.05 0.05 0.1) 
fi

# plot_args, "--plot ''", if plot is 'dis', else " --plot True "
if [ $plot == "dis" ]; then
    Orin_PY_ARGS=""
else
    Orin_PY_ARGS="--plot True"
fi
n_p=3
Orin_PY_ARGS="$Orin_PY_ARGS --profiling_filename profiling/profiling_light.csv --n_p $n_p --gen_benchmark"


start_time=`date +%s`              #定义脚本运行的开始时间
[ -e /tmp/fd1 ] || mkfifo /tmp/fd1 #创建有名管道
exec 3<>/tmp/fd1                   #创建文件描述符，以可读（<）可写（>）的方式关联管道文件，这时候文件描述符3就有了有名管道文件的所有特性
rm -rf /tmp/fd1                    #关联后的文件描述符拥有管道文件的所有特性,所以这时候管道文件可以删除，我们留下文件描述符来用就可以了
for ((i=1;i<=10;i++))
do
        echo >&3                   #&3代表引用文件描述符3，这条命令代表往管道里面放入了一个"令牌"
done
 
for ((i=1;i<=1000;i++))
do
read -u3                           #代表从管道中读取一个令牌
{
        sleep 1  #sleep 1用来模仿执行一条命令需要花费的时间（可以用真实命令来代替）
        echo 'success'$i       
        echo >&3                   #代表我这一次命令执行到最后，把令牌放回管道
}&
done
wait
 
stop_time=`date +%s`  #定义脚本运行的结束时间
 
echo "TIME:`expr $stop_time - $start_time`"
exec 3<&-                       #关闭文件描述符的读
exec 3>&-                       #关闭文件描述符的写
printf "\n********************* Start ***********************\n"

# Strategy:
#   remain the sim_var as constant (0.2/0.3/3), only enable/disable one/two/three of them 
#   vary the Compensation rate
#   we use exec_t_comp_ratioB to scan the slowdown compemsation rate, 
#   and use exec_t_comp_ratioA as typical value

# ============ Compensation rate Args =============
# exec_t_comp_ratioA, exec_t_comp_ratioB, wsc_slack_ratio exec_var_en
list_jitter_comp_cfg=(0.2 0.2 0.15 0.1 0.05 0.15 0.1 0.05)
list_exec_t_comp_ratioA=(0.3 0 0.25 0.2 0.15 0 0 0)
list_exec_t_comp_ratioB=(0.15 0.05 0.15 0.15 0.15 0 0 0)
list_wsc_slack_ratio=(1 1 1 1 1 1 1 1)

# ============ Simulation Args =============
list_exec_var_en=(True False True True True False False False)
# list_suffix=("var_0.2_0.3" "var_0.2" "var_0.25_0.15" "var_0.2_0.1" "var_0.15_0.05" "var_0.15" "var_0.1" "var_0.05")
list_n_var=(2 1 2 2 2 1 1 1)
list_var_sim_cfg=("var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json" "var_sim_cfg.json")
var_slowdown=0.3
var_sen=0.2
var_load=3
# suffix is defined as "var_"+var_sen+"(J)_"+var_slowdown+"(T)_"+var_load+"(L)"
Jitter_sym="var_${var_sen}(J)"
w_slowdown_sym="${Jitter_sym}_${var_slowdown}(T)"
load_sym="${w_slowdown_sym}_${var_load}(L)"

# ============ Scan Args =============
# set cfg_idx_end as cfg_idx_max
if [ $scan_type == "cfg" ]; then
    cfg_idx_end=$cfg_idx_max
    cfg_idx_init=$((cfg_idx_min+1))
else 
    cfg_idx_end=$cfg_idx_min
    cfg_idx_init=0
fi
# cfg_idx_init=5
# cfg_idx_end=6

# add "--force_num_cores", "--file_suffix", "_force",
if [ $scan_type == "tp" ]; then
    Orin_PY_ARGS="$Orin_PY_ARGS --force_num_cores"
fi

for cfg_roll_iterB in $(seq 0 1 1); do
  for cfg_roll_iterA in $(seq 0 1 $cfg_unroll_factor); do
  {
    cfg_idx=$((cfg_idx_init+cfg_roll_iterA*2+cfg_roll_iterB))
    printf "\ncfg_idx: $cfg_idx\n"
    exec_t_comp_ratioA=${list_exec_t_comp_ratioA[$cfg_idx]}
    exec_t_comp_ratioB=${list_exec_t_comp_ratioB[$cfg_idx]}
    wsc_slack_ratio=${list_wsc_slack_ratio[$cfg_idx]}
    exec_var_en=${list_exec_var_en[$cfg_idx]}
    # suffix=${list_suffix[$cfg_idx]}
    var_sim_cfg=${list_var_sim_cfg[$cfg_idx]}
    jitter_comp_cfg=${list_jitter_comp_cfg[$cfg_idx]}
    # n_var=${list_n_var[$cfg_idx]}
    # if not exec_var_en, then w_slowdown_sym is the same as Jitter_sym
    # else, w_slowdown_sym is the sum of Jitter_sym and exec_t_comp_ratioA
    if [ $exec_var_en == "True" ]; then
        suffix=${w_slowdown_sym}
    else
        suffix=${Jitter_sym}
    fi

    for bin_loop_iter in $(seq 0 1 ${bin_unroll_factor}); do
    {
        # iterate from index [bin_loop_iter * 2: bin_loop_iter * 2 + 1]
        # for n_bin in ${list_n_bin_cfg[@]}; do
        idx_s=$((bin_loop_iter*2)); idx_e=$((idx_s+1))
        for n_bin_idx in $(seq $idx_s $idx_e); do
        {   
            n_bin=${list_n_bin_cfg[$n_bin_idx]}
            printf "\n******************* n_bin: $n_bin *******************\n" && sleep 1
            # initial bin packing generation
            {
                printf "\nGenerating initial bin packing...\n" && sleep 1
                bin_root="$root_dir/n_bins_${n_bin}"
                RATE_CFG="--jitter_t_comp_ratio $jitter_comp_cfg --wsc_slack_ratio $wsc_slack_ratio --exec_t_comp_ratioA $exec_t_comp_ratioA --num_bins $n_bin --root_dir $bin_root"
                PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_split.json" --exec_t_comp_ratioB $exec_t_comp_ratioB"
                echo "./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py ${log_suffix} $PY_ARGS"
                if [ $scan_type != "tp" ]; then
                    log_suffix="xx"
                    ./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py ${log_suffix} $PY_ARGS
                else
                    log_suffix="_force"
                    # run TP=9 first, then run TP=0-8
                    printf "\nTest TP=9 first...\n" && sleep 1
                    ./run/abla_scalablility.sh bp 9 1 9 $bin_root sim_main.py ${log_suffix} $PY_ARGS
                    printf "\nTest TP=0-8...\n" && sleep 1
                    ./run/abla_scalablility.sh bp $tp_start $tp_step 8 $bin_root sim_main.py ${log_suffix} $PY_ARGS
                fi
            } && {
                # if scan ratioB skip the following steps
                if [ $scan_type != "ratioB" ]; then
                    if [ $scan_seed == True ]; then
                        printf "\nTest cyc and pglb ...\n"
                        VAR_ARGS="--var_sim_cfg $var_sim_cfg --jitter_sim_en"
                        if [ $exec_var_en == "True" ]; then
                            VAR_ARGS="$VAR_ARGS --exec_var_en"
                        fi
                        for ((seed=$seed_start; seed<=$seed_end; seed++)); do
                            {
                                printf "\nTest cyc and pglb with seed: $seed...\n"
                                DYN_ARGS="$VAR_ARGS --seed $seed"

                                file_suffix=${suffix}
                                log_suffix="jitter_en_${suffix[@]:4}_seed_$seed"
                                if [ $scan_type == "tp" ]; then
                                    log_suffix="force_${log_suffix}"
                                fi
                                PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_split.json" --exec_t_comp_ratioB $exec_t_comp_ratioB --file_suffix ${file_suffix}"
                                # 1. isolated-time within each bin
                                echo "./run/abla_scalablility.sh cyc $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS" 
                                ./run/abla_scalablility.sh cyc $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS 

                                # 3. enable time sharing within each bin
                                echo "./run/abla_scalablility.sh pglb $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS" 
                                ./run/abla_scalablility.sh pglb $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS 
                            } 
                        done
                        wait
                    fi
                fi

                printf "\nScan over RatioB... $(date "+%Y-%m-%d %H:%M:%S.%3N")"
                # repacking: replace the item in each bin 
                for ratioB in ${ratioB_incr_list[@]}; do
                    {
                        {
                            printf "\nGenerating bin packing with ratioB: $ratioB...\n" && sleep 1
                            new_ratioB=`echo $exec_t_comp_ratioB $ratioB | awk '{ printf "%0.2f\n", $1+$2}'`
                            PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_repack.json" --exec_t_comp_ratioB $new_ratioB"
                            echo "./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py "_ov_${new_ratioB}_repack" $PY_ARGS" 
                            ./run/abla_scalablility.sh bp $tp_start $tp_step $tp_end $bin_root sim_main.py "_ov_${new_ratioB}_repack" $PY_ARGS 
                        } && {
                            if [ $scan_seed == True ]; then
                                printf "\nTest dyn with seed ..." && sleep 1
                                VAR_ARGS="--var_sim_cfg $var_sim_cfg --jitter_sim_en"
                                if [ $exec_var_en == "True" ]; then
                                    VAR_ARGS="$VAR_ARGS --exec_var_en"
                                fi
                                for ((seed=$seed_start; seed<=$seed_end; seed++)); do
                                    {
                                        printf "\nTest dyn with seed: $seed... $(date "+%Y-%m-%d %H:%M:%S.%3N")"
                                        DYN_ARGS="$VAR_ARGS --seed $seed"

                                        # 4.constrained sharing among and along chains within each bin
                                        file_suffix="${suffix}_ov_${new_ratioB}_repack"
                                        log_suffix="jitter_en_${suffix[@]:4}_seed_${seed}_ov_${new_ratioB}_repack"
                                        if [ $scan_type == "tp" ]; then
                                            log_suffix="force_${log_suffix}"
                                        fi
                                        # PY_ARGS="$Orin_PY_ARGS $RATE_CFG --bin_pack_cfg "Bp_repack.json" --exec_t_comp_ratioB $new_ratioB --i_file_suffix "_ov_${new_ratioB}_repack" --file_suffix $file_suffix"
                                        PY_ARGS="${PY_ARGS} --i_file_suffix "_ov_${new_ratioB}_repack" --file_suffix ${file_suffix}"
                                        echo "./run/abla_scalablility.sh dyn $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS " 
                                        ./run/abla_scalablility.sh dyn $tp_start $tp_step $tp_end $bin_root sim_main.py $log_suffix $PY_ARGS $DYN_ARGS 
                                    } 
                                done
                                wait
                            fi
                        }
                    } &
                done
                wait
                echo "Done. | n_bin: $n_bin | ratioB"
            }
        } 
        done
        wait
    } &
    done
    wait
    echo "Done. | n_bin: `echo ${list_n_bin_cfg[@]} | tr ' ' ','`"
  } &
  done
  wait
done
wait
echo "Done. | All"
