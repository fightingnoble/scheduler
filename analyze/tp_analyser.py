import re
from typing import Callable, List
from analyze.log_analyse import extract_num_exec
import pandas as pd
import numpy as np
import os, copy
from functools import reduce

folder_search_seq = {
    "num_cores": r"(\d+)",
    "cfg_n": r"x(\d+)_(\d+\.\d+)s_rda-(\d+\.\d+)%\(T\)_(\d+\.\d+)%\(S\)_(\w+)"
}
folder_pattern_keys = {
    "num_cores": ["num_cores",],
    "cfg_n": ["aux_scale_factor", "e2e_latency", "wsc_slack_ratio", "temporal_rda_ratio", "lateness_mode"]
}
folder_type = {
    "num_cores": [int,],
    "cfg_n": [int, float, float, float, str]
}
search_seq = ["num_cores", "cfg_n"]

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
    

def get_throughput_extracter(stat_csv_filename, profiling_filename, n_p, warmup_dis):
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
        group_glb = [i for i in item_name if i.startswith("glb")]
        group_dyn = [i for i in item_name if i.startswith("dyn") or i.startswith("bin")]
        assert len(group_glb) in [0, 3]
        assert len(group_dyn) in [0, 3]
        if len(group_glb) == 3:
            # check if there is any missed count
            glb_missed = stat_df.loc[:, group_glb].xs("Missed Count", level=1, axis=1).loc['sum'].values.any()
            # read number of exec 
            glb_num_exec = stat_df.loc[:, group_glb].xs("Completed Count", level=1, axis=1).loc['sum'].values
            # check whether the number of exec is the same
            glb_success = np.all(glb_num_exec == num_exec) and not glb_missed
            if glb_success:
                # Create a dictionary with the values
                data = {**info_dict, **{'method': 'glb_dyn'}}
                data.pop('aux_scale_factor')
        
                # Append a row to the dataframe with the values
                criteria = [df[col] == val for col, val in data.items()]
                data_idx = reduce(lambda x, y: x&y, criteria)
                data.update({'throughput': info_dict['aux_scale_factor']})

                if df.loc[data_idx].size:
                    origin = df.loc[data_idx, 'throughput'].values[0]
                    df.loc[data_idx, 'throughput'] = max([origin, data['throughput']])
                else:
                    df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)

        if len(group_dyn) == 3:
            dyn_missed = stat_df.loc[:, group_dyn].xs("Missed Count", level=1, axis=1).loc['sum'].values.any()
            dyn_num_exec = stat_df.loc[:, group_dyn].xs("Completed Count", level=1, axis=1).loc['sum'].values
            dyn_success = np.all(dyn_num_exec == num_exec) and not dyn_missed
            if dyn_success:
                # Create a dictionary with the values
                data = {**info_dict, **{'method': 'dyn'}}
                data.pop('aux_scale_factor')
        
                # Append a row to the dataframe with the values
                criteria = [df[col] == val for col, val in data.items()]
                data_idx = reduce(lambda x, y: x&y, criteria)
                data.update({'throughput': info_dict['aux_scale_factor']})

                if df.loc[data_idx].size:
                    origin = df.loc[data_idx, 'throughput'].values[0]
                    df.loc[data_idx, 'throughput'] = max([origin, data['throughput']])
                else:
                    df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
        return df
    return extract_throughput

if __name__ == "__main__":
    import argparse
    
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
    search_seq = args.folder_search_seq.split(",")

    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=[
            'e2e_latency', 'wsc_slack_ratio', 'temporal_rda_ratio', 'lateness_mode', 
            'num_cores', 'method', 'throughput', 
        ]).to_csv(filename, index=False)
    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('log', root_dir)
    tp_extracter = get_throughput_extracter(args.stat_csv_filename, args.profiling_filename, args.n_p, args.warmup_dis) 
    scanner = get_path_var_scaner([tp_extracter, ], folder_search_seq, folder_pattern_keys, folder_type, search_seq)
    df = scanner(df, root_path, len(search_seq), dict(), 0)
    df.to_csv(filename, index=False)
