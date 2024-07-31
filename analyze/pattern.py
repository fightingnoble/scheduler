import re, os, copy
from typing import Callable, List
from collections import OrderedDict
# regular match expression for path
from global_var import case_name_cyc, case_name_dyn, case_name_glb, case_name_pglb, case_name_bp, case_suffix_N_barrier

bin_case = [case_name_bp]
bin_dep_case = [case_name_cyc, case_name_dyn]
bin_indep_case = [case_name_glb, case_name_pglb]
all_case = bin_case + bin_dep_case + bin_indep_case

# filter for the non-displayed keys
nondisplay_key_pattern = r'[\w_]*(str|sign)'
keys_filter = lambda x: re.match(nondisplay_key_pattern, x) is None
# remove_nondisplay_keys = lambda x: list(filter(keys_filter, x))
def remove_nondisplay_keys(x, extra_keys=[]):
    if isinstance(x, list):
        l = [k for k in x if keys_filter(k) and k not in extra_keys]
    elif isinstance(x, dict):
        l = OrderedDict([(k, v) for k, v in x.items() if keys_filter(k) and k not in extra_keys])
    else:
        raise ValueError("Unsupported type: {}".format(type(x)))
    return l

# ================== static parameters ================

# =========== path pattern ===========
folder_pattern = {
    "num_cores": r"(\d+)",
    "cfg_n": r"x(\d+)_(\d+\.\d+)s_rda-((\d+\.\d+)%\(J\)_)?(\d+\.\d+)%\(T\)_(\d+\.\d+)%\(S\)_(\w+)", 
    "cfg_option": r"(soft|heavy|medium)",
    "num_bins": r"n_bins_(\d+)",
}#x0_0.1s_rda-20.00%(J)_80.00%(T)_5.00%(S)_ignore
folder_pattern_keys = {
    "num_cores": ["num_cores",],
    "cfg_n": ["aux_scale_factor", "e2e_latency", "_sen_cp_str", "jitter_t_comp_ratio", "wsc_slack_ratio", "exec_t_comp_ratioA", "lateness_mode"], 
    "cfg_option": ["cfg_option",],
    "num_bins": ["num_bins",],
}
folder_type = {
    "num_cores": [int,],
    "cfg_n": [int, float, str, float, float, float, str],
    "cfg_option": [str],
    "num_bins": [int],
}

# =========== case file pattern ===========
# case_re = r"(?P<method>cyclic|sta_dyn|glb_dyn|dyn|dynamic)"
# r"(?P<method>cyclic|glb_dyn|dyn|pglb)"
case_re = rf"(?P<method>{case_name_bp}|{case_name_cyc}|{case_name_dyn}|{case_name_glb}|{case_name_pglb})(?P<barrier>{case_suffix_N_barrier})"
case_re_key = ['method',"barrier"]
case_re_type = [str,]

pure_core_pattern = r"(?P<num_cores>\d+)"
core_pattern = r"(?P<force_core>_force)?(?P<num_cores>\d+)?"
core_pattern_key = ["force_core", "num_cores"]
core_pattern_type = [bool, int]

# ================== jitter parameters ================

seed_re_withname = r"(_seed_(?P<seed>-?\d+))?"
seed_re_key = ["_seed_str", "seed"]
seed_re_type = [str, int]

# =========== file suffix pattern ===========
#   ```shell
#   Jitter_sym="var_${jitter_comp_cfg}(J)"
#   w_slowdown_sym="${Jitter_sym}_${exec_t_comp_ratioA}(T)"
#   w_ld1_sym="${w_slowdown_sym}_${load_bursty_ratio1}(LD1)"
#   ```
suffix_var_order = ["head", "sen", "slowdown", "ld", "repack"]
var_pattern = {
        "head": r"var", 
        "sen": r"([\d\.]+)(\(J\))?",
        "slowdown": r"([\d\.]+)(\(T\))?", 
        "ld": r"([\d\.]+)(\(LD\d*\))?", 
        "repack": r"ov_([\d\.]+)_repack",
    }
var_pattern_keys = {
    "head": ["jitter_en"],
    "sen": ["_sen_str","var_sen", "_sen_sign"], 
    "slowdown": ["_slowdown_str", "var_slowdown", "_slowdown_sign"], 
    "ld": ["_ld_str", "var_ld", "_ld_sign"], 
    "repack": ["_repack_str", "repack_ratio"] 
}
var_pattern_type = {
    "head": [bool,],
    "sen": [str, float, str], 
    "slowdown": [str, float, str], 
    "ld": [str, float, str], 
    "repack": [str, float]
}
jitter_keys = ['jitter_en', 'var_sen', 'var_slowdown', 'var_ld', 'seed'] 


# =========== trace file pattern ===========
# file_suffix="${suffix}_ov_${new_ratioB}_repack
# example: cyclic_e2e_trace_300_seed_3var_0.1(J)_0.2(T)_3(LD1).pkl
# glb_dyn_e2e_trace_250.pkl  glb_dyn_e2e_trace_325_seed_3var_0.2.pkl
# example: var_0.1(J)_0.2(T)_3(LD1), var_0.1(J)_0.2(T)_3(LD1)_ov_0.15_repack
def get_trace_regexp(suffix_var_order=["sen", "slowdown", "ld"]): 
    suffix_var_order = ["head", *suffix_var_order, "repack"]
    trace_surffix_pattern = rf"(?P<head>{var_pattern['head']})?"
    trace_surffix_pattern += r"".join(rf"(?P<{key}>_{var_pattern[key]})?" for key in suffix_var_order[1:])
    file_surffix_pattern_keys = []
    file_surffix_pattern_type = []
    for key in suffix_var_order: 
        file_surffix_pattern_keys.extend(var_pattern_keys[key])
        file_surffix_pattern_type.extend(var_pattern_type[key])    

    trace_pattern = case_re+r"_e2e_trace_"+core_pattern+seed_re_withname+trace_surffix_pattern+r"\.pkl"
    #  (dynamic|glb_dyn)_e2e_trace_(\d+)?(_seed_(\d+))?(var_[\d\.]+)?\.pkl
    trace_pattern_keys = case_re_key + core_pattern_key + seed_re_key + file_surffix_pattern_keys
    trace_pattern_type = case_re_type + core_pattern_type + seed_re_type + file_surffix_pattern_type
    return trace_pattern, trace_pattern_keys, trace_pattern_type

# =========== log file pattern ===========
# ```shell
# log_suffix="jitter_en_${suffix[@]:4}_seed_${seed}_ov_${new_ratioB}_repack"
# log_suffix="jitter_en_${suffix[@]:4}_seed_$seed"
# log_name=$dir_path/${case_sign}${file_suffix}.log.txt
# log_name=$dir_path/${case_sign}_${file_suffix}.log.txt
# ```
# example: 
# dyn_jitter_en_0.2(J)_0.3(T)_seed_0_ov_0.15_repack.log.txt
def get_log_regexp(suffix_var_order=["sen", "slowdown", "ld"]):
    suffix_var_order = ["head", *suffix_var_order, "repack"]
    log_infix_pattern = r"".join(rf"(?P<{key}>_{var_pattern[key]})?" for key in suffix_var_order[1:-1])
    file_infix_pattern_keys = []
    file_infix_pattern_type = []
    for key in suffix_var_order[1:-1]: 
        file_infix_pattern_keys.extend(var_pattern_keys[key])
        file_infix_pattern_type.extend(var_pattern_type[key]) 

    log_suffix_pattern = rf"(?P<repack>_{var_pattern['repack']})?"
    log_pattern = case_re+core_pattern+r"(_jitter_(dis|en))?" + log_infix_pattern +seed_re_withname+log_suffix_pattern+r"\.log\.txt" 
    log_pattern_keys = case_re_key + core_pattern_key + ["_jitter_en_str", "jitter_en"] + file_infix_pattern_keys + seed_re_key + var_pattern_keys['repack']
    log_pattern_type = case_re_type + core_pattern_type + [str, str] + file_infix_pattern_type + seed_re_type + var_pattern_type['repack']
    return log_pattern, log_pattern_keys, log_pattern_type


# Metrices list
metric_lat = ['confidence', 'ddl_percentile', 'rt_percentile', 'miss_rate']
metric_switch_tp = ['n_ctx_switch', 'cum_time', 'throughput']

# =========== helper functions ===========
# get_group_dict = lambda pattern_keys, match, pattern_type: {k: t(v) for k,v,t in zip(pattern_keys, match.groups(), pattern_type) if v is not None}
default_dict = {float: "", int: "", str: "", bool: False}
def get_group_dict(pattern_keys, match, pattern_type, set_default=False):
    if set_default:
        return {k: t(v) if v is not None else default_dict[t] for k,v,t in zip(pattern_keys, match.groups(), pattern_type)}
    else:
        return {k: t(v) for k,v,t in zip(pattern_keys, match.groups(), pattern_type) if v is not None}
    
def get_path_var_scaner(hook_list: List[Callable], match_pattern: dict, match_pattern_keys: dict, match_type, search_seq: list):
    def path_var_scaner(df, folder_path, target_depth, path_var_dict, current_depth=0):
        if current_depth == target_depth:
            for hook in hook_list:
                df = hook(df, folder_path, path_var_dict)
            return df

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                item_match = re.match(match_pattern[search_seq[current_depth]], item)
                if item_match:
                    path_var_dict_new = copy.deepcopy(path_var_dict)
                    for key, values, t in zip(match_pattern_keys[search_seq[current_depth]], item_match.groups(), match_type[search_seq[current_depth]]):
                        path_var_dict_new[key] = t(values)
                    df = path_var_scaner(df, item_path, target_depth, path_var_dict_new, current_depth + 1)
        return df
    return path_var_scaner
