## Generate config files for pglb, seq-pglb, cyc, and reserv, respectively.

1. 1st stage: pre allocation
```
                "--test_case", "bin_pack_new",
                "--bin_pack_cfg", "Bp_split.json", 
```
Basic setting:
No task colocates in the same partition, 
```
                "--num_bins", "-1",
                "--root_dir", "new_scan/n_bins_max",
                 --jitter_t_comp_ratio 0.2 --wsc_slack_ratio 1 --exec_t_comp_ratioA 0.3 --exec_t_comp_ratioB 0.15
```

```shell
plot_flag=""                                        # 是否绘图，""或"--plot True"
profiling_filename="profiling/profiling_light.csv"  # 任务profile文件
n_p=3                                               # 并行度/处理器数量
gen_benchmark="--gen_benchmark"                     # 固定开启
jitter_t_comp_ratio=0.2                             # sensor端jitter比例
wsc_slack_ratio=1                                   # slack比例
exec_t_comp_ratioA=0.3                              # A类任务执行时间变异比例
exec_t_comp_ratioB=0.15                             # B类任务执行时间变异比例
e2e_latency=0.1                                     # 端到端延迟（秒）
aux_scale_factor=4                                  # 辅助缩放因子
test_case="bin_pack_new"                            # 固定
num_bins=-1                                         # bin数量，-1表示自动
root_dir="new_scan/n_bins_max"               # 结果输出根目录
bin_pack_cfg="Bp_split.json"                        # bin packing配置文件

python sim_main.py {plot_flag}\
    --profiling_filename {profiling_filename} \
    --n_p {n_p} \
    --gen_benchmark \
    --jitter_t_comp_ratio {jitter_t_comp_ratio} \
    --wsc_slack_ratio {wsc_slack_ratio} \
    --exec_t_comp_ratioA {exec_t_comp_ratioA} \
    --exec_t_comp_ratioB {exec_t_comp_ratioB} \
    --e2e_latency {e2e_latency} \
    --aux_scale_factor {aux_scale_factor} \ 
    --test_case bin_pack_new \
    --num_bins {num_bins} \
    --root_dir {root_dir} \
    --bin_pack_cfg {bin_pack_cfg} 
```

```shell
run_bin_pack() {
    local policy_var="$1"
    local load_var="$2"
    local setting_var="$3"

    python sim_main.py \
        $policy_var \
        $load_var \
        $setting_var
}

# case相关参数（主控参数）
setting_var="{plot_flag}\
    --profiling_filename {profiling_filename} \
    --n_p {n_p} \
    --gen_benchmark \
    --jitter_t_comp_ratio {jitter_t_comp_ratio} \
    --wsc_slack_ratio {wsc_slack_ratio} \
    --exec_t_comp_ratioA {exec_t_comp_ratioA} \
    --exec_t_comp_ratioB {exec_t_comp_ratioB}" 

# load相关参数（负载参数）
spec_var="\
    --e2e_latency {e2e_latency} \
    --aux_scale_factor {aux_scale_factor} "

# 1. cyc
policy_var="--test_case bin_pack_new --num_bins -1 --root_dir new_scan/n_bins_max --bin_pack_cfg Bp_split.json"
run_bin_pack "$policy_var" "$spec_var" $setting_var 

# 2. pglb
policy_var="--test_case bin_pack_new --num_bins 3 --root_dir new_scan/n_bins_3 --bin_pack_cfg Bp_split.json"
run_bin_pack "$policy_var" "$spec_var" $setting_var 

# 3. reserv
policy_var="---test_case bin_pack_new -num_bins 3 --root_dir new_scan/n_bins_3 --bin_pack_cfg Bp_repack.json"
run_bin_pack "$policy_var" "$spec_var" $setting_var 

```