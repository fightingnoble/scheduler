import math
FLOPS_PER_CORE = 0.5

overhead_pushpull_per_core = {0.6:83, 0.8:87,} # cycles
overhead_of_enqueuing_op = 1000 # cycles
overhead_of_dequeuing_op = 1000 # cycles

clock_period = 1e-9 # Seconds
SRAM_size_per_core = 1.25 # MB

trace_file = "trace.txt"
trace_list = []

numerical_error_tol_abs = 1e-12
numerical_tol_bit = 12
numerical_error_tol_rel = 0.01
elim_nume_error = lambda x: round(x, numerical_tol_bit)
def elim_error(x, n_dig, abs_err, mod='round'):
    if mod == 'up':
        return round(x + 0.5 * abs_err, n_dig)
    elif mod == 'down':
        return round(x - 0.5 * abs_err, n_dig)
    else:
        return round(x, n_dig)

flop_error_tol_abs = FLOPS_PER_CORE * 1e-6
flop_error_tol_bit = 7

fork_pid_base = 1000
max_fork_candi_num = 20
max_fork_pid = 100

GLB_BUFFER_SIZE = 40E6 # Bytes
GLB_BUFFER_SIZE_PER_CORE = 156.25E3 # Bytes
BW_DRAM = 100E9 # Bytes/s
LAT_PER_HOP = 10e-9 # Seconds
AVG_HOP_NUM = 10
MIN_CORE_NUM = 256

BROADCAST_SCALER = 1

W_perc = 1/3
A_perc = 1/3
O_perc = 1/3

cfg_dir = "./cfgs"
log_dir = "./log"
plot_dir = "./plot"
trace_dir = "./trace"
cache_dir = "./cache"

import os 
cfg_root_fmt = r"x{aux_scale_factor}_{e2e_latency}s_rda-{jitter_t_comp_ratio:.2%}(J)_{wsc_slack_ratio:.2%}(T)_{exec_t_comp_ratioA:.2%}(S)_ignore"
cache_root_fmt = os.path.join(cache_dir, r"{root_dir}", r"{cfg_n}")
plot_root_fmt = os.path.join(plot_dir, r"{root_dir}", r"{cfg_n}", r"{num_cores}")
trace_root_fmt = os.path.join(trace_dir, r"{root_dir}", r"{cfg_n}")

bin_fn_fmt = r"bin_list_{num_cores}{i_file_suffix}.pkl"
routing_table_fn_fmt = r"routing_table_{num_cores}{i_file_suffix}.pkl"
bin_save_fmt = os.path.join(cache_root_fmt, bin_fn_fmt)
routing_table_save_fmt = os.path.join(cache_root_fmt, routing_table_fn_fmt)

# case: cyclic, sta_dyn, glb_dyn

plt_fn_w_seed_fmt = r"{plot_root}/{case}_{plt_size}_{num_cores}_seed_{seed}{file_suffix}.pdf"
plt_fn_wo_seed_fmt = r"{plot_root}/{case}_{plt_size}_{num_cores}{file_suffix}.pdf"

trace_fn_wo_seed_fmt = r"{trace_root}/{case}_e2e_trace_{num_cores}.pkl" 
trace_fn_w_seed_fmt = r"{trace_root}/{case}_e2e_trace_{num_cores}_seed_{seed}{file_suffix}.pkl" 

core_stat_fn_fmt = r"{cache_root}/{case}_max_core_stat.pkl" # dyn_max_core_stat.pkl

case_name_bp = r"bin_pack_new"
case_name_glb = r"glb_dyn"
case_name_dyn = r"dyn"
case_name_cyc = r"cyclic"
case_name_pglb = r"pglb"

case_name_bp_input = r"bin_pack_new"
case_name_glb_input = r"glb_dynamic"
case_name_dyn_input = r"dynamic"
case_name_cyc_input = r"cyclic"
case_name_pglb_input = r"partitioned_glb_dynamic"