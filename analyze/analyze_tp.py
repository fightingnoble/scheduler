import re
from analyze.stat_num_exec import extract_num_exec
import pandas as pd
import numpy as np
import os
from functools import reduce

from analyze.pattern import *
from utils import update_df

folder_search_seq = ["num_cores", "cfg_n"]

def set_tp(df, info_dict):
    # Create a dictionary with the values
    data = remove_nondisplay_keys({**info_dict})
    data.pop('aux_scale_factor')

    # Append a row to the dataframe with the values
    update_fn = lambda x,y: max([x[0],y])
    df = update_df(df, data, {'throughput': info_dict['aux_scale_factor']}, update_fn)
    return df

def check_status(stat_df, num_exec, group):
    # check if there is any missed count
    missed = stat_df.loc[:, group].xs("Missed Count", level=1, axis=1).loc['sum'].values.any()
    # read number of exec 
    num_comp = stat_df.loc[:, group].xs("Completed Count", level=1, axis=1).loc['sum'].values
    # check whether the number of exec is the same
    success = np.all(num_comp == num_exec) and not missed
    return success

def get_throughput_extracter(stat_csv_filename, profiling_filename, n_p, warmup_dis, sim_param_seq):
    log_pattern, log_pattern_keys, log_pattern_type = get_log_regexp(sim_param_seq) 
    def extract_throughput(df, folder, info_dict):
        root,dirs,files = os.walk(folder).__next__()
        assert len(dirs) == 0
        for fn in files:
            match = re.match(stat_csv_filename, fn)
            if match:
                break
        if not match:
            print(f"!!! Warning: no stat file in {folder} !!!")
            return df
        file_path = f"{folder}/{fn}"
        print(file_path)
        stat_df = pd.read_csv(file_path, index_col=0, header=[0, 1])

        # find row named "num_exec"
        if "num_exec" in stat_df.index:
            num_exec = stat_df.loc["num_exec"].values[0]
        else:
            aux_scale_factor = info_dict['aux_scale_factor']
            num_exec = extract_num_exec(profiling_filename, aux_scale_factor, n_p, warmup_dis)
        item_name = stat_df.columns.get_level_values(0).unique().values
        case_group_dict=dict()
        for i in item_name:
            if match:= re.match(log_pattern, i):
                info_sim = get_group_dict(log_pattern_keys, match, log_pattern_type, True)
                data = {**info_dict, **info_sim}
                data = remove_nondisplay_keys(data)
                # cyclic|glb_dyn|dyn|pglb|bin_pack_new
                # classify the data by 3 levels: other, jitter_param, seed 
                # filter out the unused jitter_keys such as ld_var by data keys
                idx1 = tuple(info_sim[k] for k in data if k not in jitter_keys and k in info_sim) 
                if info_sim['jitter_en']:
                    idx2 = tuple(info_sim[k] for k in data if k in jitter_keys and k !='seed' and k in info_sim) 
                else:
                    idx2 = (False,)
                if idx1 not in case_group_dict:
                    case_group_dict[idx1] = dict()
                if idx2 not in case_group_dict[idx1]:
                    case_group_dict[idx1][idx2] = []
                case_group_dict[idx1][idx2].append(i)

        # record the failed cases
        bin_fail_case = []
        for L1_case, L1_dict in case_group_dict.items():
            if L1_case[0] in bin_case:
                if not check_status(stat_df, num_exec, L1_dict[(False,)]):
                    bin_fail_case.append(L1_case[1:])
        
        for L1_case, L1_dict in case_group_dict.items():
            # check bin_case status
            if (L1_case[0] in bin_dep_case and L1_case[1:] in bin_fail_case) or (L1_case[0] in bin_case):
                # skip the bin case, and the bin dependent but the dependent bin case failed
                continue
                
            # check static case
            if (False,) in L1_dict:
                if check_status(stat_df, num_exec, L1_dict[(False,)]):
                    # set static throughput
                    df = set_tp(df, info_dict)
                else:
                    # skip all dynamic cases
                    continue
            for case_L2 in L1_dict: 
                if case_L2 == (False,):
                    continue
                if check_status(stat_df, num_exec, L1_dict[case_L2]):
                    # set dynamic throughput
                    df = set_tp(df, info_dict) 
        return df
    
    return extract_throughput

if __name__ == "__main__":
    import argparse
    from utils import csv_fmt_check
    
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling_light.csv", help="profiling filename")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="throughput", help="filename")
    parser.add_argument("--stat_csv_filename", type=str, default="new_bin_pack.csv", help="csv filename")
    parser.add_argument("--folder_search_seq", type=str, default="num_cores,cfg_n", help="core list")
    parser.add_argument("--n_p", type=int, default=3, help="number of processors")
    parser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")

    args = parser.parse_args()
    root_dir = args.root_dir

    filename = args.filename
    filename = f"{filename}.csv"
    filename = os.path.join(args.output_dir, filename)
    folder_search_seq = args.folder_search_seq.split(",")
    folder_search_seq = args.folder_search_seq.split(",")

    cfg_keys = [key for search_key in folder_search_seq for key in folder_pattern_keys[search_key] if keys_filter(key)]
    cfg_keys = remove_nondisplay_keys(cfg_keys, ["aux_scale_factor"])
    trace_pattern, trace_pattern_keys, trace_pattern_type = get_log_regexp(args.sim_seq.split(","))
    sim_keys = remove_nondisplay_keys(trace_pattern_keys, ['seed'])
    cols =  cfg_keys + sim_keys + ['throughput']

    csv_fmt_check(filename, cols)

    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('log', root_dir)
    tp_extracter = get_throughput_extracter(
        args.stat_csv_filename, args.profiling_filename, 
        args.n_p, args.warmup_dis, args.sim_param_seq.split(","),
        sim_keys
    ) 
    scanner = get_path_var_scaner([tp_extracter, ], folder_pattern, folder_pattern_keys, folder_type, folder_search_seq)
    df = scanner(df, root_path, len(folder_search_seq), dict(), 0)
    df.to_csv(filename, index=False)
