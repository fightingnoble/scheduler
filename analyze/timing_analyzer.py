import re
from functools import reduce
import pandas as pd
import os
import numpy as np
from analyze.log_analyse import extract_num_exec
from model.Context_message import trace_analyser

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
folder_search_seq = ["cfg_n"]
file_pattern = r"(?P<method>dynamic|glb_dyn)_e2e_trace_(?P<num_cores>\d+)?(?P<jitter_en>var_[\d\.]+)?\.pkl"
file_pattern_keys = ["method", "num_cores", "jitter_en"]
file_pattern_type = [str, int, bool]
# dynamic_e2e_trace_250.pkl  dynamic_e2e_trace_300var_0.2.pkl
# glb_dyn_e2e_trace_250.pkl  glb_dyn_e2e_trace_300var_0.2.pkl

def get_e2e_checker(file_pattern, timing_flag_dict, profiling_filename=None, n_p=None, warmup_dis=None, stat_csv_filename=None):
    def e2e_checker(df, folder, info_dict):
        aux_scale_factor = info_dict['aux_scale_factor']
        num_exec, sink_pred = extract_num_exec(profiling_filename, aux_scale_factor, n_p, warmup_dis, "e2e")

        root,dirs,files = os.walk(folder).__next__()
        assert len(dirs) == 0
        for fn in sorted(os.listdir(folder)):
            fn_match = re.match(file_pattern, fn)
            if fn_match:
                file_path = os.path.join(folder, fn)
                print(file_path)
                # Create a dictionary with the values
                data = {**info_dict, **dict({key: t(fn_match.group(key)) for key, t in zip(fn_match.groupdict().keys(), file_pattern_type)})}
                data.pop("", None)
        
                # Append a row to the dataframe with the values
                criteria = [df[col] == val for col, val in data.items()]
                data_idx = reduce(lambda x, y: x&y, criteria)

                if data["lateness_mode"] == "all_soft" and data['method'] == 'glb_dyn':
                    e2e_latency_list, n_violation = trace_analyser(timing_flag_dict, file_path, data["e2e_latency"], data["lateness_mode"], True)
                    assert num_exec == len(e2e_latency_list[0]) + len(e2e_latency_list[1])
                    miss_rate = n_violation / num_exec
                else:
                    e2e_latency_list = trace_analyser(timing_flag_dict, file_path, data["e2e_latency"], data["lateness_mode"])

                    if stat_csv_filename is not None:
                        log_folder = os.path.join(folder.replace('trace', 'log'), str(data['num_cores']))
                        root,dirs,files = os.walk(log_folder).__next__()
                        assert len(dirs) == 0

                        for log_fn in files:
                            match = re.match(stat_csv_filename, log_fn)
                            if match:
                                break
                        if not match:
                            print(f"!!! Warning: no stat file in {log_folder} !!!")
                            return df
                        log_path = os.path.join(log_folder, log_fn)
                        print(file_path)
                        stat_df = pd.read_csv(log_path, index_col=0, header=[0, 1])
                        # r"(dyn|glb_dyn)(_\d+)?_jitter_(dis|en)\.log\.txt"
                        index_name = 'dyn' if data['method'] == 'dynamic' else 'glb_dyn'
                        index_name += f"_{data['num_cores']}"  
                        index_name += '_jitter_dis' if not data['jitter_en'] else '_jitter_en'
                        index_name += '.log.txt'
                        num_comp = 0 
                        for idx in stat_df.loc[:, index_name]["Completed Count"].index:
                            if "_".join(idx.split('_')[0:-2]) in sink_pred:
                                num_comp += stat_df.loc[:, index_name]["Completed Count"][idx]
                        assert num_comp == len(e2e_latency_list[0]) + len(e2e_latency_list[1])
                    miss_rate = 1- (len(e2e_latency_list[0]) + len(e2e_latency_list[1])) / num_exec

                # aplly histogram analysis
                rt_e2e_latency_list = np.array(e2e_latency_list[0])
                ddl_e2e_latency_list = np.array(e2e_latency_list[1])
                print(f"mean: {np.mean(rt_e2e_latency_list):.6f}, std: {np.std(rt_e2e_latency_list):.6f}, max: {np.max(rt_e2e_latency_list):.6f}, min: {np.min(rt_e2e_latency_list):.6f}")
                print(f"mean: {np.mean(ddl_e2e_latency_list):.6f}, std: {np.std(ddl_e2e_latency_list):.6f}, max: {np.max(ddl_e2e_latency_list):.6f}, min: {np.min(ddl_e2e_latency_list):.6f}")
                # calculate the percentile
                rt_percentiles = np.percentile(rt_e2e_latency_list, [90, 95, 99, 99.9, 99.99])
                ddl_percentiles = np.percentile(ddl_e2e_latency_list, [90, 95, 99, 99.9, 99.99])
                print(f"rt_percentile: {rt_percentiles}")
                print(f"ddl_percentile: {ddl_percentiles}")

                if df.loc[data_idx].size:
                    df.loc[data_idx, 'ddl_percentile'] = ddl_percentiles
                    df.loc[data_idx, 'rt_percentile'] = rt_percentiles
                    df.loc[data_idx, 'confidence'] = [90, 95, 99, 99.9, 99.99]
                    df.loc[data_idx, ['miss_rate']] = miss_rate
                else:
                    data.update({'ddl_percentile': ddl_percentiles,
                                    'rt_percentile': rt_percentiles,
                                    'confidence': [90, 95, 99, 99.9, 99.99],
                                    'miss_rate': miss_rate})
                    df = pd.concat([df, pd.DataFrame(data)])
        return df
    return e2e_checker

if __name__ == "__main__":
    import argparse
    from analyze.tp_analyser import get_path_var_scaner
    
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="timing", help="filename")
    parser.add_argument("--folder_search_seq", type=str, default="cfg_n,num_cores", help="core list")
    parser.add_argument("--profiling_filename", type=str, default="profiling_light.csv", help="profiling filename")
    parser.add_argument("--stat_csv_filename", type=str, default=r"new_bin_pack(\d+)?.csv", help="csv filename")
    parser.add_argument("--n_p", type=int, default=3, help="number of processors")
    parser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")

    args = parser.parse_args()
    root_dir = args.root_dir
    
    # load the criticality 
    from task.task_cfg import load_taskattrib
    glb_n_task_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=False) 
    timing_flag_dict = {}
    for task_name in glb_n_task_dict:
        timing_flag_dict[task_name] = glb_n_task_dict[task_name].timing_flag

    filename = args.filename
    filename = f"{filename}.csv"
    filename = os.path.join(args.output_dir, filename)
    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=
            [key for search_key in folder_search_seq for key in folder_pattern_keys[search_key]]
            + [key for key in file_pattern_keys if key != ""]
            +['confidence', 'ddl_percentile', 'rt_percentile', 'miss_rate']).to_csv(filename, index=False)
    
    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('trace', root_dir)
    ctx_extracter = get_e2e_checker(file_pattern, timing_flag_dict, args.profiling_filename, args.n_p, args.warmup_dis, args.stat_csv_filename) 
    scanner = get_path_var_scaner([ctx_extracter, ], folder_pattern, folder_pattern_keys, folder_type, folder_search_seq)
    df = scanner(df, root_path, len(folder_search_seq), dict(), 0)
    df.to_csv(filename, index=False)
