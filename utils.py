import os
import pickle, argparse, time
from functools import wraps, reduce
from global_var import cfg_dir
from typing import Dict, Callable
import pandas as pd

def dump_and_check(save_path, obj2save):
    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(save_path, "wb") as f:
        pickle.dump(obj2save, f)
    try:
            # load the bin_list and the init_p_list
        with open(save_path, "rb") as f:
            obj2save = pickle.load(f)
        print(f"{save_path} saved and loaded successfully")
    except:
        print(f"{save_path} not found")
        exit()

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except:
        raise FileNotFoundError(f"{path} not found")
    return obj

class Found(Exception):
    pass

def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="all", help="task name")
    parser.add_argument("--plot", type=bool, default=False, help="plot")
    parser.add_argument("--test_all", default=False, help="test all the task")
    parser.add_argument("--num_cores", default=266, type=int, help="number of cores")
    parser.add_argument("--num_bins", default=-1, type=int, help="number of bins")
    parser.add_argument("--BinExtendRule", default="list", type=str, help="Rule for when and how to extend the bin")
    parser.add_argument("--preemptable", default=False, action="store_true", help="enable preemption")
    parser.add_argument("--quantum_check_en", default=False, action="store_true", help="enable quantum check")
    parser.add_argument("--quantumSize", default=2, type=int, help="quantum size, # of simulation steps")
    # parser.add_argument("--hyper_p", default=None, type=float, help="hyper period")
    parser.add_argument("--warmup_dis", default=False, action="store_true", help="warmup disable")
    # parser.add_argument("--drain", default=False, action="store_true", help="drain")
    # parser.add_argument("--sim_step", default=None, type=float, help="simulation step")
    parser.add_argument("--n_p", default=1, type=int, help="number of periods")
    
    parser.add_argument("--jitter_sim_en", default=False, action="store_true", help="enable jitter simulation")
    parser.add_argument("--jitter_sim_para", default={"loc":0, "scale":0.2, "force_wc":False}, type=dict, help="jitter simulation parameters")
    
    parser.add_argument("--var_sim_cfg", default="var_sim_cfg.json", type=str, help="variation simulation config file")

    parser.add_argument("--load_var_sim_en", default=False, action="store_true", help="enable dynamic object simulation")
    parser.add_argument("--load_var_sim_para", default=dict(), type=dict, help="dynamic object simulation parameters")

    parser.add_argument("--e2e_var_sim_en", default=False, action="store_true", help="enable e2e latency variation simulation")
    parser.add_argument("--e2e_var_sim_para", default=dict(), type=dict, help="e2e latency variation simulation parameters")
    
    parser.add_argument("--file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--i_file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--barrier_dis", default=False, action="store_true", help="disable barrier")
    parser.add_argument("--data_lifetime_mode", default="static", type=str, help="lifetime mode: most_recent, ref_count, timeout, watermark") 
    parser.add_argument("--spatial_rda_ratio", default=0.2, type=float, help="spatial ratio")
    parser.add_argument("--temporal_rda_ratio", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling_light.csv", help="profiling filename")
    parser.add_argument("--lateness_mode", type=str, default="ignore", help="lateness mode")
    # parser.add_argument("--lateness_threshold", type=float, default=0.0, help="lateness threshold")
    # parser.add_argument("--cbs_en", default=False, action="store_true", help="enable cbs")
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    # parser.add_argument("--freq", type=float, default=10, help="frequency")
    parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
    parser.add_argument("--slack_threshold", default=5e-4, type=float, help="slack threshold")
    parser.add_argument("--aux_scale_factor", default=1, type=int, help="aux scale factor")
    parser.add_argument("--gen_benchmark", default=False, action="store_true", help="generate benchmark")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")

    parser.add_argument("--bin_pack_cfg", default="bin_pack_cfg.json", type=str, help="bin pack config file")
    parser.add_argument("--bin_pack_para", default=dict(), type=dict, help="bin pack algorithm parameters")
    # parser.add_argument("--bin_sort", default="EAT", type=str, help="bin sort: EAT, barycenter")
    # parser.add_argument("--bin_sort_reverse", default=True, type=bool, help="bin sort reverse")
    
    parser.add_argument("--max_core_stat", default=False, type=bool, help="max core stat")
    args = parser.parse_args()
    return args

# define a wrapper for displaying current function, start time, end time, and execution time
def time_cnt(description:str):
    def time_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("="*10+description+"="*10)
            start_time = time.time()
            print("Start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
            t_s = time.monotonic()
            result = func(*args, **kwargs)
            t_e = time.monotonic()
            s, ms = divmod((t_e - t_s) * 1000, 1000)
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            print("%d:%02d:%02d:%03d" % (h, m, s, ms))
            end_time = time.time()
            print("End time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
            return result
        return wrapper
    return time_decorator

def update_df(df, index_dict:Dict, info_dict:Dict, update_fn:Callable=lambda x,y:y):
    criteria = [df[col] == val for col, val in index_dict.items()]
    data_idx = reduce(lambda x, y: x&y, criteria)

    if df.loc[data_idx].size:
        for key in info_dict.keys():
            origin = df.loc[data_idx, key]
            df.loc[data_idx, key] = update_fn(origin, info_dict[key])
    else:
        index_dict.update(info_dict)
        df = pd.concat([df, pd.DataFrame(index_dict, index=[0])], ignore_index=True)
    return df
