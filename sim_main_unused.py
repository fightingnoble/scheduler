"""Unused CSV and cache helpers, preserved without behavior changes."""

import os
import re

from paths import PathContext
from global_var import cache_root_fmt, bin_fn_fmt
from utils import load_pickle
from sim_main import compare_paths, generate_bin_paths


def ensure_csv(csv_path_and_fn, cfg_para_dict, para_scan_group1):
    import pandas as pd
    if not os.path.exists(csv_path_and_fn):
        cols = list(cfg_para_dict.keys()) + list(para_scan_group1.keys()) + ["num_cores"]
        pd.DataFrame(columns=cols).to_csv(csv_path_and_fn, index=False)


def prepare_induced_env_if_needed(path_para_dict, path_ctx: PathContext,
                                  trace_path_para, plot_path_para
                                  ):
    """
    当 args.binpack_cfg["core_size"] == "induced" 时：
    - 从已存在的 bin_list 文件名解析 num_cores
    - 更新 plot/trace 的 num_cores
    - 生成 bin_list/routing_table 路径
    - 加载 bin_list

    返回：(num_cores, bin_list)
    """
    folder, files, match = get_core_num_from_trace_name(path_para_dict, path_ctx)
    if not match:
        return None
    matched_num_cores = int(match.group(1))

    plot_path_para['num_cores'] = matched_num_cores
    trace_path_para['num_cores'] = matched_num_cores
    bin_list_save_path, routing_table_save_path = generate_bin_paths(path_para_dict, path_ctx, matched_num_cores, "initial load")
    bin_list = load_pickle(bin_list_save_path)
    return matched_num_cores, bin_list


def get_core_num_from_trace_name(path_para_dict, path_ctx: PathContext):
    # 使用旧方法生成路径
    old_folder = cache_root_fmt.format(**path_para_dict)

    # 使用新方法生成路径并比较
    new_folder = path_ctx.cache_root
    compare_paths(old_folder, new_folder, "cache_root")
    folder = new_folder

    # 使用旧方法生成正则表达式
    old_bin_fn_regex = bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"})
    
    # 使用新方法生成正则表达式
    new_bin_fn_regex = path_ctx.get_bin_fn_regex()
    # 比较正则表达式
    compare_paths(old_bin_fn_regex, new_bin_fn_regex, "bin_fn_regex")
    bin_fn_regex = new_bin_fn_regex
    
    root,dirs,files = os.walk(folder).__next__()
    assert len(dirs) == 0
    for fn in files:
        if match := re.match(bin_fn_regex, fn):
            break
    if not match: 
        print(f"!!! Warning: no bin_list file "+ bin_fn_regex +f" in {folder}:{files} !!!")
    return folder,files,match


def check_max_bin_num(args, num_cores, bin_path_format, path_ctx: PathContext):
    max_num_bins = 0
    for e2e_latency, aux_scale_factor in args.e2e_var_sim_para['event_list']:            
        # 使用旧方法生成路径
        old_bin_list_save_path = bin_path_format.format(aux_scale_factor, e2e_latency, num_cores)
        
        # 使用新方法生成路径（直接更新 ctx 并刷新）
        path_ctx.aux_scale_factor = aux_scale_factor
        path_ctx.e2e_latency = e2e_latency
        path_ctx.refresh_config()
        path_ctx.num_cores = num_cores
        new_bin_list_save_path = path_ctx.get_bin_list_path()
        
        # 比较路径
        compare_paths(old_bin_list_save_path, new_bin_list_save_path, f"bin_list_save_path (e2e_var: {aux_scale_factor}, {e2e_latency})")
        
        # 使用新路径
        bin_list = load_pickle(new_bin_list_save_path)
        max_num_bins = max(max_num_bins, len(bin_list))
    num_bins = max_num_bins
    return num_bins
