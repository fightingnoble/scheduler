import os
import pickle, argparse, time
from functools import wraps, reduce
from global_var import cfg_dir
from typing import Dict, Callable
import pandas as pd
import numpy as np
import json
from global_var import *
import ast

import h5py
import numpy as np

# 定义分块大小
CHUNK_SIZE = 25
def get_next_chunk_id(f):
    try:
        chunk_ids = [int(name.split('_')[1]) for name in f.keys()]
        if chunk_ids:
            return max(chunk_ids) + 1
        else:
            return 0
    except (OSError, KeyError):
        return 0

def save_chunk(file_name, data: list, force:bool=False):
    check_parents_path(file_name)
    if len(data) < CHUNK_SIZE and not force:
        return
    if len(data) > 0:
        with h5py.File(file_name, 'a') as f:
            chunk_id = get_next_chunk_id(f)
            dataset_name = f'chunk_{chunk_id}'
            data_as_json = np.array([json.dumps(item) for item in data], dtype=h5py.special_dtype(vlen=str))
            f.create_dataset(dataset_name, data=data_as_json, compression="gzip")
        data.clear()
    if force:
        try:
            load_h5_file(file_name)
            print(f"{file_name} saved and loaded successfully")
        except:
            print(f"{file_name} bad")
            exit()

def load_h5_file(file_name):
    data = []
    with h5py.File(file_name, 'r') as f:
        for name in f.keys():
            # 只有使用 [:]，你才能获得 h5py.Dataset 对象中的实际数据内容
            data_as_json = f[name][:]
            dataset = [json.loads(item) for item in data_as_json]
            data.extend(dataset)
    return data


def dump_and_check(save_path, obj2save):
    check_parents_path(save_path)

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

def check_parents_path(save_path):
    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_pickle(path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except:
        raise FileNotFoundError(f"{path} not found")
    return obj

class Found(Exception):
    pass

def dict_type(string):
    try:
        return ast.literal_eval(string)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid dictionary format")
    
def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="all", help="task name")
    parser.add_argument("--plot", type=bool, default=False, help="plot")
    parser.add_argument("--plot_fmt", type=str, default="svg,pdf", help="plot format")
    parser.add_argument("--test_all", default=False, help="test all the task")
    parser.add_argument("--num_cores", default=266, type=int, help="number of cores")
    parser.add_argument("--force_num_cores", default=False, action="store_true", help="force to use the number of cores")
    
    parser.add_argument("--num_bins", default=-1, type=int, help="number of bins")
    parser.add_argument("--timestepxus", default=10, type=int, help="timestep in us")
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
    parser.add_argument("--jitter_sim_para", default={}, type=dict_type, help="jitter simulation parameters")

    parser.add_argument("--exec_var_en", default=False, action="store_true", help="enable exec jitter simulation")
    parser.add_argument("--exec_var_para", default={}, type=dict_type, help="exec jitter simulation parameters")

    parser.add_argument("--var_sim_cfg", default="var_sim_cfg.json", type=str, help="variation simulation config file")

    parser.add_argument("--load_var_sim_en", default=False, action="store_true", help="enable dynamic object simulation")
    parser.add_argument("--load_var_sim_para", default=dict(), type=dict_type, help="dynamic object simulation parameters")

    parser.add_argument("--e2e_var_sim_en", default=False, action="store_true", help="enable e2e latency variation simulation")
    parser.add_argument("--e2e_var_sim_para", default=dict(), type=dict_type, help="e2e latency variation simulation parameters")
    
    parser.add_argument("--file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--i_file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--barrier_dis", default=False, action="store_true", help="disable barrier")
    parser.add_argument("--data_lifetime_mode", default="static", type=str, help="lifetime mode: most_recent, ref_count, timeout, watermark") 
    
    parser.add_argument("--jitter_t_comp_ratio", default=0.2, type=float, help="spatial ratio")
    parser.add_argument("--exec_t_comp_ratioA", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--exec_t_comp_ratioB", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--load_t_comp_ratio", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--var_estimation", default={}, type=dict_type, help="estimation parameters")
    
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
    parser.add_argument("--bin_pack_para", default=dict(), type=dict_type, help="bin pack algorithm parameters")
    # parser.add_argument("--bin_sort", default="EAT", type=str, help="bin sort: EAT, barycenter")
    # parser.add_argument("--bin_sort_reverse", default=True, type=bool, help="bin sort reverse")
    
    parser.add_argument("--max_core_stat", default=False, type=bool, help="max core stat")
    parser.add_argument("--forbid_miss", default=False, action="store_true", help="forbid miss")
    parser.add_argument("--progress_aware", default=False, action="store_true", help="consider the execution porgress")
    parser.add_argument("--allow_realloc", default=False, action="store_true", help="allow reallocation of resources amount")
    args = parser.parse_args()

    jitter_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['jitter'] 
    # print(args.jitter_sim_para)
    jitter_sim_para.update(args.jitter_sim_para)
    args.jitter_sim_para = jitter_sim_para
    exec_var_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['exec']
    exec_var_para.update(args.exec_var_para)
    args.exec_var_para = exec_var_para
    load_var_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['load_var']
    load_var_sim_para.update(args.load_var_sim_para)
    args.load_var_sim_para = load_var_sim_para
    e2e_var_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['e2e_var']
    e2e_var_sim_para.update(args.e2e_var_sim_para)
    args.e2e_var_sim_para = e2e_var_sim_para
    binpack_cfg = json.load(open(os.path.join(cfg_dir, args.bin_pack_cfg), "r"))
    binpack_cfg.update(args.bin_pack_para)
    args.binpack_cfg = binpack_cfg
    # copy from jitter_t_comp_ratio, exec_t_comp_ratioA, load_t_comp_ratio
    var_estimation = {
        "jitter": args.jitter_t_comp_ratio,
        "exec": args.exec_t_comp_ratioA,
        "load_var": args.load_t_comp_ratio,
    }
    var_estimation.update(args.var_estimation)
    args.var_estimation = var_estimation
    
    if args.e2e_var_sim_en:
        assert args.gen_benchmark == True

    args.plt_fmt = args.plot_fmt.split(",")
    return args

def args_postprocess(args):
    root_dir = args.root_dir
    args.binpack_cfg.update({"exec_t_comp_ratioB": args.exec_t_comp_ratioB}) 
    para_scan_group2 = {"num_cores": args.num_cores}

    if args.force_num_cores and args.aux_scale_factor!= 9:
        args.force_suffix="force_"
    else:
        args.force_suffix=""

    cfg_para_dict, para_scan_group1, cfg_n = get_cfg_n(args)
    path_para_dict = {"root_dir": root_dir, "cfg_n": cfg_n, "i_file_suffix": args.i_file_suffix, "force_suffix": args.force_suffix}
    # remain parameters in group1 unfilled
    cfg_n_format = cfg_root_fmt.format(**cfg_para_dict, **{}.fromkeys(para_scan_group1, r"{}"))

    plot_root = plot_root_fmt.format(**path_para_dict, **para_scan_group2)
    trace_root = trace_root_fmt.format(**path_para_dict, **para_scan_group2)
    bin_path_format = os.path.join('cache', root_dir, cfg_n_format, r"bin_list_{}"+f"{args.i_file_suffix}.pkl")
    # remaining parameters in group2 unfilled
    # bin_save_fmt.format(**{**path_para_dict, 'cfg_n': cfg_n_format, "num_cores": r"{}"})

    trace_path_para = {
        "trace_root": trace_root, "num_cores": args.num_cores, 
        "seed": args.seed, "file_suffix": args.file_suffix, 
        "force_suffix": args.force_suffix
        }
    plot_path_para = {"plot_root": plot_root, "num_cores": args.num_cores, 
                      "seed": args.seed, "file_suffix": args.file_suffix,
                      "force_suffix": args.force_suffix
                      }
    
    # worst case: state seed = -1, seed value is not used but set to 0, print as -1

    if args.seed == -1:
        args.seed = 0
        # set enforce_wc
        args.jitter_sim_para.update({"enforce_wc": True})
        args.exec_var_para.update({"enforce_wc": True})
        
    if args.jitter_sim_en: 
        enforce_wc = args.jitter_sim_para.get("enforce_wc", False) 
    else:
        enforce_wc = False

    if enforce_wc:
        assert args.jitter_sim_en
        plot_path_para.update({"seed": "-1"})
        trace_path_para.update({"seed": "-1"})

    csv_xlxs_root = os.path.join(log_dir, root_dir)
    return cfg_para_dict,para_scan_group1,para_scan_group2,path_para_dict,bin_path_format,trace_path_para,plot_path_para,csv_xlxs_root

def get_cfg_n(args):
    cfg_para_dict = {
        "wsc_slack_ratio": args.wsc_slack_ratio, "exec_t_comp_ratioA": args.exec_t_comp_ratioA, 
        "lateness_mode": args.lateness_mode, "jitter_t_comp_ratio": args.jitter_t_comp_ratio,
        }
    para_scan_group1 = {"aux_scale_factor": args.aux_scale_factor, "e2e_latency": args.e2e_latency}
    if not args.gen_benchmark:
        if args.profiling_filename == "profiling/profiling.csv":
            cfg_n = "heavy"
        else:
            cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1] 
        cfg_n += f"_{args.lateness_mode}"
    else:
        cfg_n = cfg_root_fmt.format(**cfg_para_dict, **para_scan_group1)
    return cfg_para_dict, para_scan_group1, cfg_n

def get_case_path_str(args):
    if args.test_case == case_name_pglb_input:
        case_pth = case_name_pglb
    elif args.test_case == case_name_cyc_input:
        case_pth = case_name_cyc
    elif args.test_case == case_name_fifo_input:
        case_pth = case_name_fifo
    elif args.test_case == case_name_dyn_input:
        case_pth = case_name_dyn
    elif args.test_case == case_name_glb_input:
        case_pth = case_name_glb
    elif args.test_case == case_name_bp_input:
        case_pth = case_name_bp
    
    if args.barrier_dis:
        case_pth += case_suffix_N_barrier
    return case_pth


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
            df.loc[data_idx, key] = update_fn(origin.values, info_dict[key])
    else:
        index_dict.update(info_dict)
        df = pd.concat([df, pd.DataFrame(index_dict, index=[0])], ignore_index=True)
    return df

from pyinstrument import Profiler

# profiler = Profiler()
# profiler.start()

# # code you want to profile

# profiler.stop()

# profiler.print()
def pyinstr_profiler(description:str):
    def pyinstr_profiler_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("="*10+description+"="*10)
            profiler = Profiler(interval=0.0001)
            profiler.start()
            result = func(*args, **kwargs)
            profiler.stop()
            profiler.print()
            return result
        return wrapper
    return pyinstr_profiler_decorator


def csv_fmt_check(filename, cols):
    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=cols).to_csv(filename, index=False)
    else:
        # check the column, if not exist, add it, and clear the content
        df = pd.read_csv(filename)
        if set(df.columns) != set(cols):
            pd.DataFrame(columns=cols).to_csv(filename, index=False)

def core_distr(rsc_map, score_dict, curr_aval_rsc):
    cum_score_reverse = np.cumsum(list(reversed(score_dict.values())))
    cum_size = [curr_aval_rsc * s / cum_score_reverse[-1] for s in cum_score_reverse]
    for i, pid in enumerate(reversed(score_dict.keys())):
        if i == 0:
            size = int(cum_size[0])
            rsc_map[pid] += size
            cum_size[0] = size
        elif i == len(score_dict) - 1:
            size = int(curr_aval_rsc - cum_size[i - 1])
            rsc_map[pid] += size
        else:
            size = int(cum_size[i] - cum_size[i - 1])
            rsc_map[pid] += size
            cum_size[i] = size + cum_size[i - 1]
