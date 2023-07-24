import re
from typing import Callable
from log_analyse import extract_num_exec
import pandas as pd
import numpy as np
import os, copy

match_pattern = {
    "num_cores": r"(\d+)",
    "cfg_n": r"x(\d+)_(\d+\.\d+)s_rda-(\d+\.\d+)%\(T\)_(\d+\.\d+)%\(S\)_(\w+)"
}
match_pattern_keys = {
    "num_cores": ["num_cores",],
    "cfg_n": ["aux_scale_factor", "e2e_latency", "wsc_slack_ratio", "temporal_rda_ratio", "lateness_mode"]
}
search_seq = ["num_cores", "cfg_n"]

def get_path_var_scaner(hook: Callable, match_pattern: dict, match_pattern_keys: dict, search_seq: list):
    def path_var_scaner(df, folder_path, target_depth, path_var_dict, current_depth=0):
        if current_depth == target_depth:
            df = hook(df, folder_path, path_var_dict)
            return df

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                item_match = re.match(match_pattern[search_seq[current_depth]], item)
                if item_match:
                    path_var_dict_new = copy.deepcopy(path_var_dict)
                    path_var_dict_new.update(dict(zip(match_pattern_keys[search_seq[current_depth]], item_match.groups())))
                    df = path_var_scaner(df, item_path, target_depth, path_var_dict_new, current_depth + 1)
        return df
    return path_var_scaner
    

def get_throughput_extracter(stat_csv_filename, profiling_filename, n_p, warmup_dis):
    def extract_throughput(df, folder, info_dict):
        root,dirs,files = os.walk(folder).__next__()
        assert len(dirs) == 0
        assert stat_csv_filename in files
        file_path = f"{folder}/{stat_csv_filename}"
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
        assert len(group_glb) == 3
        assert len(group_dyn) == 3
            # check if there is any missed count
        glb_missed = stat_df.loc[:, group_glb].xs("Missed Count", level=1, axis=1).loc['sum'].values.any()
        dyn_missed = stat_df.loc[:, group_dyn].xs("Missed Count", level=1, axis=1).loc['sum'].values.any()
            # read number of exec 
        glb_num_exec = stat_df.loc[:, group_glb].xs("Completed Count", level=1, axis=1).loc['sum'].values
        dyn_num_exec = stat_df.loc[:, group_dyn].xs("Completed Count", level=1, axis=1).loc['sum'].values
            # check whether the number of exec is the same
        glb_success = np.all(glb_num_exec == num_exec) and not glb_missed
        dyn_success = np.all(dyn_num_exec == num_exec) and not dyn_missed
        if glb_success:
            data = {
                    'e2e_latency': info_dict['e2e_latency'],
                    'wsc_slack_ratio': info_dict['wsc_slack_ratio'],
                    'temporal_rda_ratio': info_dict['temporal_rda_ratio'],
                    'lateness_mode': info_dict['lateness_mode'],
                    'num_cores': info_dict['num_cores'],
                    'fn': 'glb',
                    'throughput': info_dict['aux_scale_factor']
                }
            data_idx = (df['e2e_latency'] == data['e2e_latency']) & \
                        (df['wsc_slack_ratio'] == data['wsc_slack_ratio']) & \
                        (df['temporal_rda_ratio'] == data['temporal_rda_ratio']) & \
                        (df['lateness_mode'] == data['lateness_mode']) & \
                        (df['num_cores'] == data['num_cores']) & \
                            (df['fn'] == data['fn']) 
            if df.loc[data_idx].size:
                origin = df.loc[data_idx, 'throughput'].values[0]
                df.loc[data_idx, 'throughput'] = max([origin, data['throughput']])
            else:
                df = pd.concat([df, pd.DataFrame(data, index=[0])], ignore_index=True)
        if dyn_success:
            data = {
                    'e2e_latency': info_dict['e2e_latency'],
                    'wsc_slack_ratio': info_dict['wsc_slack_ratio'],
                    'temporal_rda_ratio': info_dict['temporal_rda_ratio'],
                    'lateness_mode': info_dict['lateness_mode'],
                    'num_cores': info_dict['num_cores'],
                    'fn': 'dyn',
                    'throughput': info_dict['aux_scale_factor']
                }
            data_idx = (df['e2e_latency'] == data['e2e_latency']) & \
                        (df['wsc_slack_ratio'] == data['wsc_slack_ratio']) & \
                        (df['temporal_rda_ratio'] == data['temporal_rda_ratio']) & \
                        (df['lateness_mode'] == data['lateness_mode']) & \
                        (df['num_cores'] == data['num_cores']) & \
                            (df['fn'] == data['fn'])
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
    parser.add_argument("--profiling_filename", type=str, default="profiling.csv", help="profiling filename")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="throughput", help="filename")
    parser.add_argument("--stat_csv_filename", type=str, default="new_bin_pack.csv", help="csv filename")
    parser.add_argument("--search_seq", type=str, default="num_cores,cfg_n", help="core list")
    parser.add_argument("--n_p", type=int, default=1, help="number of processors")
    parser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")

    args = parser.parse_args()
    root_dir = args.root_dir

    filename = args.filename
    filename = f"{filename}.csv"

    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=[
            'e2e_latency', 'wsc_slack_ratio', 'temporal_rda_ratio', 'lateness_mode', 
            'num_cores', 'fn', 'throughput', 
        ]).to_csv(filename, index=False)
    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('log', root_dir)
    search_seq = args.search_seq.split(",")
    tp_extracter = get_throughput_extracter(args.stat_csv_filename, args.profiling_filename, args.n_p, args.warmup_dis) 
    scanner = get_path_var_scaner(tp_extracter, match_pattern, match_pattern_keys, search_seq)
    df = scanner(df, root_path, len(search_seq), dict(), 0)
    df.to_csv(filename, index=False)
