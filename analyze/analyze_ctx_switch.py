import re
from functools import reduce
import pandas as pd
import os

from analyze.pattern import folder_pattern, folder_pattern_keys, folder_type, get_path_var_scaner, get_group_dict
from analyze.pattern import log_pattern, log_pattern_keys, log_pattern_type
from utils import update_df

folder_search_seq = ["cfg_n", "num_cores"]
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

def get_ctx_extracter(file_pattern, file_pattern_keys, file_pattern_type):
    def extract_ctx_num(df, folder, info_dict):
        root,dirs,files = os.walk(folder).__next__()
        assert len(dirs) == 0
        for fn in sorted(os.listdir(folder)):
            fn_match = re.match(file_pattern, fn)
            if fn_match:
                file_path = os.path.join(folder, fn)
                print(file_path)
                # Create a dictionary with the values
                data = {**info_dict, **get_group_dict(file_pattern_keys, fn_match, file_pattern_type)}
                data.pop("")
                if 'seed' not in data:
                    data['seed'] = ""

                # NOTE: Jitter_en flag is defined differently in 
                # trace and log name, "var_[\d\.]+" and "dis|en", respectively.
                if data['jitter_en'] == "en" and data['seed'] == "":
                    print(f"(Passed!) Warning: no seed in {file_path.split('/')[-1]}")
                    continue
                        
                number_of_context_switch, cumulative_context_switch_time = get_ctx_switch_info(file_path)
                df = update_df(df, data, {'n_ctx_switch': number_of_context_switch,
                                            'cum_time': cumulative_context_switch_time,
                                            'throughput': -1})
                1+1
        return df
    return extract_ctx_num

if __name__ == "__main__":
    import argparse
    from analyze.analyze_tp import get_throughput_extracter
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--filename", type=str, default="ctx_switch", help="filename")
    parser.add_argument("--folder_search_seq", type=str, default=','.join(folder_search_seq), help="core list")

    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling_light.csv", help="profiling filename")
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
            + [key for key in log_pattern_keys if key != ""]
        + ['n_ctx_switch', 'cum_time', 'throughput']).to_csv(filename, index=False)

    # Load the dataframe
    df = pd.read_csv(filename)

    root_path = os.path.join('log', root_dir)
    ctx_extracter = get_ctx_extracter(log_pattern, log_pattern_keys, log_pattern_type) 
    # for find minimum required cores
    tp_extracter = get_throughput_extracter(args.stat_csv_filename, args.profiling_filename, args.n_p, args.warmup_dis) 
    scanner = get_path_var_scaner([ctx_extracter, tp_extracter], folder_pattern, folder_pattern_keys, folder_type, folder_search_seq)
    df = scanner(df, root_path, len(folder_search_seq), dict(), 0)
    df.to_csv(filename, index=False)
