import re
from functools import reduce
import pandas as pd
import os

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
folder_search_seq = ["cfg_n", "num_cores"]
file_pattern = r"(dyn|glb_dyn)(_\d+)?_jitter_(dis|en)\.log\.txt"
# file_pattern = r"(glb_dyn)(_\d+)?_jitter_(dis|en)\.log\.txt"
file_pattern_keys = ["method", "", "jitter_en"]
# bin_pack_new_{x}.log.txt
# glb_dyn_{x}_ideal.log.txt
# glb_dyn_{x}_jitter_dis.log.txt
# glb_dyn_{x}_jitter_en.log.txt
# dyn_{x}_jitter_dis.log.txt
# dyn_{x}_jitter_en.log.txt

# 读取文件内容
def get_ctx_switch_info(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # 定义正则表达式模式
    pattern = r"number of context switch (\d+)\ncumulative context switch ([\d.]+e{0,1}[-\d]*)"

    # 匹配模式并提取数值
    matches = re.findall(pattern, content)

    # 输出提取到的数值
    if not matches:
        print(f"Warning: no context switch info in {filename}")
        return 0, 0
    else:
        match = matches[-1]
        number_of_context_switch = int(match[0])
        cumulative_context_switch_time = float(match[1])
        print(f"Number of context switch: {number_of_context_switch}")
        print(f"Cumulative context switch: {cumulative_context_switch_time}")
        return number_of_context_switch, cumulative_context_switch_time

def get_ctx_extracter(file_pattern):
    def extract_ctx_num(df, folder, info_dict):
        root,dirs,files = os.walk(folder).__next__()
        assert len(dirs) == 0
        for fn in sorted(os.listdir(folder)):
            fn_match = re.match(file_pattern, fn)
            if fn_match:
                file_path = os.path.join(folder, fn)
                print(file_path)
                # Create a dictionary with the values
                data = {**info_dict, **dict(zip(file_pattern_keys, fn_match.groups()))}
                data.pop("")
        
                # Append a row to the dataframe with the values
                criteria = [df[col] == val for col, val in data.items()]
                data_idx = reduce(lambda x, y: x&y, criteria)

                number_of_context_switch, cumulative_context_switch_time = get_ctx_switch_info(file_path)
                if df.loc[data_idx].size:
                    df.loc[data_idx, 'n_ctx_switch'] = number_of_context_switch
                    df.loc[data_idx, 'cum_time'] = cumulative_context_switch_time
                    df.loc[data_idx, 'throughput'] = -1
                else:
                    data.update({'n_ctx_switch': number_of_context_switch, 
                                 'cum_time': cumulative_context_switch_time,
                                 'throughput': -1})
                    df = pd.concat([df, pd.DataFrame.from_dict(data, orient='index').T], ignore_index=True)
        return df
    return extract_ctx_num

if __name__ == "__main__":
    import argparse
    from analyze.tp_analyser import get_path_var_scaner, get_throughput_extracter
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="ctx_switch", help="filename")
    parser.add_argument("--folder_search_seq", type=str, default=','.join(folder_search_seq), help="core list")

    parser.add_argument("--profiling_filename", type=str, default="profiling_light.csv", help="profiling filename")
    parser.add_argument("--stat_csv_filename", type=str, default=r"new_bin_pack(\d+)?.csv", help="csv filename")
    parser.add_argument("--n_p", type=int, default=3, help="number of processors")
    parser.add_argument("--warmup_dis", type=bool, default=False, help="whether to warm up the system")
    parser.add_argument("--output_dir", type=str, default=".", help="output directory")

    args = parser.parse_args()
    root_dir = args.root_dir

    filename = args.filename
    filename = f"{filename}.csv"
    filename = os.path.join(args.output_dir, filename)
    folder_search_seq = args.folder_search_seq.split(",")

    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=
            [key for search_key in folder_search_seq for key in folder_pattern_keys[search_key]]
            + [key for key in file_pattern_keys if key != ""]
        + ['n_ctx_switch', 'cum_time', 'throughput']).to_csv(filename, index=False)

    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('log', root_dir)
    ctx_extracter = get_ctx_extracter(file_pattern) 
    tp_extracter = get_throughput_extracter(args.stat_csv_filename, args.profiling_filename, args.n_p, args.warmup_dis) 
    scanner = get_path_var_scaner([ctx_extracter, tp_extracter], folder_pattern, folder_pattern_keys, folder_type, folder_search_seq)
    df = scanner(df, root_path, len(folder_search_seq), dict(), 0)
    df.to_csv(filename, index=False)
