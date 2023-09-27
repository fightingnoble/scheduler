import re, os, copy
from typing import Callable, List

folder_pattern = {
    "num_cores": r"(\d+)",
    "cfg_n": r"x(\d+)_(\d+\.\d+)s_rda-(\d+\.\d+)%\(T\)_(\d+\.\d+)%\(S\)_(\w+)", 
    "cfg_option": r"(soft|heavy|medium)"
}
folder_pattern_keys = {
    "num_cores": ["num_cores",],
    "cfg_n": ["aux_scale_factor", "e2e_latency", "wsc_slack_ratio", "temporal_rda_ratio", "lateness_mode"], 
    "cfg_option": ["cfg_option",]
}
folder_type = {
    "num_cores": [int,],
    "cfg_n": [int, float, float, float, str],
    "cfg_option": str
}

trace_pattern = r"(?P<method>dynamic|glb_dyn)_e2e_trace_(?P<num_cores>\d+)?(_seed_(?P<seed>\d+))?(?P<jitter_en>var_[\d\.]+)?\.pkl"
#  (dynamic|glb_dyn)_e2e_trace_(\d+)?(_seed_(\d+))?(var_[\d\.]+)?\.pkl
trace_pattern_keys = ["method", "num_cores", "", "seed", "jitter_en"]
trace_pattern_type = [str, int, str, int, bool]
# dynamic_e2e_trace_250.pkl  dynamic_e2e_trace_300var_0.2.pkl
# glb_dyn_e2e_trace_250.pkl  glb_dyn_e2e_trace_300var_0.2.pkl
# dynamic_e2e_trace_250.pkl  dynamic_e2e_trace_275_seed_7var_0.2.pkl  
# glb_dyn_e2e_trace_250.pkl  glb_dyn_e2e_trace_325_seed_3var_0.2.pkl

log_pattern = r"(dyn|glb_dyn)(_\d+)?_jitter_(dis|en)(_seed_(\d+))?\.log\.txt"
# file_pattern = r"(glb_dyn)(_\d+)?_jitter_(dis|en)\.log\.txt"
log_pattern_keys = ["method", "", "jitter_en", "", "seed"]
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
