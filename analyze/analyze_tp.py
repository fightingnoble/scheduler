import re
from analyze.stat_num_exec import extract_num_exec
import pandas as pd
import numpy as np
import os
from functools import reduce

from analyze.pattern import folder_pattern, folder_pattern_keys, folder_type, get_path_var_scaner
from analyze.pattern import log_pattern, log_pattern_keys, log_pattern_type, seed_re
from utils import update_df

search_seq = ["num_cores", "cfg_n"]

def read_tp(df, info_dict, stat_df, num_exec, group_dyn, group_stat, jitter_en='en', method='glb_dyn'):
    # check if there is any missed count
    missed = stat_df.loc[:, group_dyn+group_stat].xs("Missed Count", level=1, axis=1).loc['sum'].values.any()
    # read number of exec 
    num_comp = stat_df.loc[:, group_dyn+group_stat].xs("Completed Count", level=1, axis=1).loc['sum'].values
    # check whether the number of exec is the same
    success = np.all(num_comp == num_exec) and not missed
    if success:
        # Create a dictionary with the values
        data = {**info_dict, **{'method': method, 'jitter_en': jitter_en}}
        data.pop('aux_scale_factor')
    
        # Append a row to the dataframe with the values
        update_fn = lambda x,y: max([x[0],y])
        df = update_df(df, data, {'throughput': info_dict['aux_scale_factor']}, update_fn)
    return df

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
        # TODO: use pattern in analyze.pattern 
        group_glb = [i for i in item_name if re.match(r"glb_dyn(_\d+)?_jitter_en(_seed_("+seed_re+r"))\.log\.txt", i)]
        group_dyn = [i for i in item_name if re.match(r"dyn(_\d+)?_jitter_en(_seed_("+seed_re+r"))\.log\.txt", i)]
        group_dyn_stat = [i for i in item_name if re.match(r"dyn(_\d+)?_jitter_dis\.log\.txt", i) or i.startswith("bin")]
        group_glb_stat = [i for i in item_name if re.match(r"glb_dyn(_\d+)?(_ideal|_jitter_dis)\.log\.txt", i)]
        
        assert len(group_dyn_stat) in [0, 1, 2]
        assert len(group_glb_stat) in [0, 1, 2]

        if len(group_glb_stat):
            df = read_tp(df, info_dict, stat_df, num_exec, [], group_glb_stat, 'dis', 'glb_dyn')
        if len(group_dyn_stat):
            df = read_tp(df, info_dict, stat_df, num_exec, [], group_dyn_stat, 'dis', 'dyn')
        if len(group_glb):
            df = read_tp(df, info_dict, stat_df, num_exec, group_glb, group_glb_stat, 'en', 'glb_dyn')
        if len(group_dyn):
            df = read_tp(df, info_dict, stat_df, num_exec, group_dyn, group_dyn_stat, 'en', 'dyn')
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
            'e2e_latency', 'wsc_slack_ratio', 'exec_t_comp_ratioA', 'lateness_mode', 
            'num_cores', 'method', 'jitter_en', 'throughput', 
        ]).to_csv(filename, index=False)
    else:
        # check the column, if not exist, add it, and clear the content
        df = pd.read_csv(filename)
        if set(df.columns) != set([
            'e2e_latency', 'wsc_slack_ratio', 'exec_t_comp_ratioA', 'lateness_mode', 
            'num_cores', 'method', 'jitter_en', 'throughput', 
        ]):
            pd.DataFrame(columns=[
                'e2e_latency', 'wsc_slack_ratio', 'exec_t_comp_ratioA', 'lateness_mode', 
                'num_cores', 'method', 'jitter_en', 'throughput', 
            ]).to_csv(filename, index=False)

    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('log', root_dir)
    tp_extracter = get_throughput_extracter(args.stat_csv_filename, args.profiling_filename, args.n_p, args.warmup_dis) 
    scanner = get_path_var_scaner([tp_extracter, ], folder_pattern, folder_pattern_keys, folder_type, search_seq)
    df = scanner(df, root_path, len(search_seq), dict(), 0)
    df.to_csv(filename, index=False)
