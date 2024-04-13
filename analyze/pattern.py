import re, os, copy
from typing import Callable, List
from collections import OrderedDict
# regular match expression for path
from global_var import case_name_cyc, case_name_dyn, case_name_glb, case_name_pglb

# =========== path pattern ===========
folder_pattern = {
    "num_cores": r"(\d+)",
    "cfg_n": r"x(\d+)_(\d+\.\d+)s_rda-((\d+\.\d+)%\(J\)_)?(\d+\.\d+)%\(T\)_(\d+\.\d+)%\(S\)_(\w+)", 
    "cfg_option": r"(soft|heavy|medium)",
    "num_bins": r"n_bins_(\d+)",
}#x0_0.1s_rda-20.00%(J)_80.00%(T)_5.00%(S)_ignore
folder_pattern_keys = {
    "num_cores": ["num_cores",],
    "cfg_n": ["aux_scale_factor", "e2e_latency", "", "jitter_t_comp_ratio", "wsc_slack_ratio", "exec_t_comp_ratioA", "lateness_mode"], 
    "cfg_option": ["cfg_option",],
    "num_bins": ["num_bins",],
}
folder_type = {
    "num_cores": [int,],
    "cfg_n": [int, float, str, float, float, float, str],
    "cfg_option": [str],
    "num_bins": [int],
}

# =========== file suffix pattern ===========
#   ```shell
#   Jitter_sym="var_${jitter_comp_cfg}(J)"
#   w_slowdown_sym="${Jitter_sym}_${exec_t_comp_ratioA}(T)"
#   w_ld1_sym="${w_slowdown_sym}_${load_bursty_ratio1}(LD1)"
#   ```

suffix_var_order = ["head", "jitter", "slowdown", "ld"]
var_pattern = {
        "head": r"var", 
        "jitter": r"([\d\.]+)(\(J\))?",
        "slowdown": r"([\d\.]+)(\(T\))?", 
        "ld": r"([\d\.]+)(\(LD\d*\))?" 
    }
var_pattern_keys = {
    "head": ["var_en"],
    "jitter": ["jitter_str","var_jitter", "jitter_sign"], 
    "slowdown": ["slowdown_str", "var_slowdown", "slowdown_sign"], 
    "ld": ["ld_str", "var_ld", "ld_sign"] 
}
var_pattern_type = {
    "head": [str,],
    "jitter": [str, float, str], 
    "slowdown": [str, float, str], 
    "ld": [str, float, str] 
}
    
# example: var_0.1(J)_0.2(T)_3(LD1) 
file_surffix_pattern = r"_".join(rf"(?P<{key}>{var_pattern[key]})?" for key in suffix_var_order) 
file_surffix_pattern_keys = []
file_surffix_pattern_type = []
for key in suffix_var_order: 
    file_surffix_pattern_keys.extend(var_pattern_keys[key])
    file_surffix_pattern_type.extend(var_pattern_type[key])    

def get_var_dict(suffix_str):
    match = re.match(file_surffix_pattern, suffix_str)
    if match:
        return get_group_dict(file_surffix_pattern_keys, match, file_surffix_pattern_type)
    else:
        return None

# =========== case file pattern ===========
# case_re = r"(?P<method>cyclic|sta_dyn|glb_dyn|dyn|dynamic)"
case_re = r"(?P<method>cyclic|glb_dyn|dyn|pglb)"
seed_re = r"-?\d+"
seed_re_wn = r"(_seed_(?P<seed>-?\d+))?"

# =========== trace file pattern ===========
trace_pattern = case_re+r"_e2e_trace_(?P<num_cores>\d+)?"+seed_re_wn+r"(?P<jitter_en>var_[\d\.]+)?\.pkl"
#  (dynamic|glb_dyn)_e2e_trace_(\d+)?(_seed_(\d+))?(var_[\d\.]+)?\.pkl
trace_pattern_keys = ['method', "num_cores", "", "seed", "jitter_en"]
trace_pattern_type = [str, int, str, int, bool]
# dynamic_e2e_trace_250.pkl  dynamic_e2e_trace_300var_0.2.pkl
# glb_dyn_e2e_trace_250.pkl  glb_dyn_e2e_trace_300var_0.2.pkl
# dynamic_e2e_trace_250.pkl  dynamic_e2e_trace_275_seed_7var_0.2.pkl  
# glb_dyn_e2e_trace_250.pkl  glb_dyn_e2e_trace_325_seed_3var_0.2.pkl

# =========== log file pattern ===========
log_pattern = case_re+r"(_\d+)?_jitter_(dis|en)(_seed_("+seed_re+r"))?\.log\.txt"# file_pattern = r"(glb_dyn)(_\d+)?_jitter_(dis|en)\.log\.txt"
log_pattern_keys = ['method', "", "jitter_en", "", "seed"]
log_pattern_type = [str, str, str, str, int]
# bin_pack_new_{x}.log.txt
# glb_dyn_{x}_ideal.log.txt
# glb_dyn_{x}_jitter_dis.log.txt
# glb_dyn_{x}_jitter_en_seed_{y}.log.txt 
# dyn_{x}_jitter_dis.log.txt
# dyn_{x}_jitter_en_seed_{y}.log.txt
# bin_pack_new.log.txt          dyn_jitter_en_seed_1.log.txt  dyn_jitter_en_seed_5.log.txt  dyn_jitter_en_seed_9.log.txt  glb_dyn_jitter_en_seed_0.log.txt  glb_dyn_jitter_en_seed_4.log.txt  glb_dyn_jitter_en_seed_8.log.txt
# dyn_jitter_dis.log.txt        dyn_jitter_en_seed_2.log.txt  dyn_jitter_en_seed_6.log.txt  glb_dyn_ideal.log.txt         glb_dyn_jitter_en_seed_1.log.txt  glb_dyn_jitter_en_seed_5.log.txt  glb_dyn_jitter_en_seed_9.log.txt
# dyn_jitter_en.log.txt         dyn_jitter_en_seed_3.log.txt  dyn_jitter_en_seed_7.log.txt  glb_dyn_jitter_dis.log.txt    glb_dyn_jitter_en_seed_2.log.txt  glb_dyn_jitter_en_seed_6.log.txt  new_bin_pack.csv
# dyn_jitter_en_seed_0.log.txt  dyn_jitter_en_seed_4.log.txt  dyn_jitter_en_seed_8.log.txt  glb_dyn_jitter_en.log.txt     glb_dyn_jitter_en_seed_3.log.txt  glb_dyn_jitter_en_seed_7.log.txt


# =========== helper functions ===========
get_group_dict = lambda pattern_keys, match, pattern_type: {k: t(v) for k,v,t in zip(pattern_keys, match.groups(), pattern_type) if v is not None}

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
