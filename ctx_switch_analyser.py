import re

# 读取文件内容
def get_ctx_switch_info(filename):
    with open(filename, 'r') as file:
        content = file.read()

    # 定义正则表达式模式
    pattern = r"number of context switch (\d+)\ncumulative context switch ([\d.]+)"

    # 匹配模式并提取数值
    match = re.findall(pattern, content)[0]

    # 输出提取到的数值
    number_of_context_switch = int(match[0])
    cumulative_context_switch_time = float(match[1])
    print(f"Number of context switch: {number_of_context_switch}")
    print(f"Cumulative context switch: {cumulative_context_switch_time}")
    return number_of_context_switch, cumulative_context_switch_time


if __name__ == "__main__":
    import numpy as np
    import argparse
    # load the trace list from the file
    import pickle
    import pandas as pd
    import copy
    import os 
    import torch
    import plotly.graph_objects as go
    import plotly.io as pio   
    pio.kaleido.scope.mathjax = None
    
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--core_list", type=str, default="300", help="core list")
    parser.add_argument("--profiling_filename", type=str, default="profiling.csv", help="profiling filename")
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    # parser.add_argument("--freq", type=float, default=10, help="frequency")
    parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
    parser.add_argument("--slack_threshold", default=5e-4, type=float, help="slack threshold")
    parser.add_argument("--aux_scale_factor", default=1, type=int, help="aux scale factor")
    parser.add_argument("--gen_benchmark", default=False, action="store_true", help="generate benchmark")
    parser.add_argument("--root_dir", default=".", type=str, help="root directory")
    parser.add_argument("--lateness_mode", type=str, default="ignore", help="lateness mode")
    parser.add_argument("--filename", type=str, default="ctx_switch", help="filename")
    parser.add_argument("--temporal_rda_ratio", default=0.05, type=float, help="temporal ratio")

    args = parser.parse_args()
    e2e_latency = args.e2e_latency

    core_list = [int(i) for i in args.core_list.split(",")]
    if not args.gen_benchmark:
        if args.profiling_filename == "profiling.csv":
            cfg_n = "heavy"
        else:
            cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
        if cfg_n == "light":
            args.aux_scale_factor = 1
        elif cfg_n == "heavy":
            args.aux_scale_factor = 6
        elif cfg_n == "medium":
            args.aux_scale_factor = 4

        if cfg_n == "light":
            args.e2e_latency = 1
        else:
            args.e2e_latency = 0.09

    else:
        cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio-args.temporal_rda_ratio):.2%}(T)_{args.temporal_rda_ratio:.2%}(S)"
    if args.lateness_mode:
        cfg_n += f"_{args.lateness_mode}"
    root_dir = args.root_dir

    filename = args.filename
    if args.lateness_mode:
        filename = args.lateness_mode + filename
    filename = f"{filename}.csv"
    if not os.path.exists(filename):
        # Create a dataframe with the values
        pd.DataFrame(columns=[
            'aux_scale_factor', 'e2e_latency', 'wsc_slack_ratio', 'temporal_rda_ratio', 'lateness_mode', 'filename',
            'num_cores', 'n_ctx_switch', 'cum_time'
        ]).to_csv(filename, index=False)
    # Load the dataframe
    df = pd.read_csv(filename)

    result_dict = {}
    for num_cores in core_list:
        folder = f"log/{root_dir}/{cfg_n}/{num_cores}"
        for fn in sorted(os.listdir(folder)):
            fn_match = re.match(r"glb_dyn_\d+_jitter_(en|dis).log.txt", fn)
            if fn_match:
                file_path = os.path.join(folder, fn)
                print(file_path)
                number_of_context_switch, cumulative_context_switch_time = get_ctx_switch_info(file_path)
                # Create a dictionary with the values
                data = {
                    'aux_scale_factor': args.aux_scale_factor,
                    'e2e_latency': args.e2e_latency,
                    'wsc_slack_ratio': args.wsc_slack_ratio - args.temporal_rda_ratio,
                    'temporal_rda_ratio': args.temporal_rda_ratio,
                    'lateness_mode': args.lateness_mode,
                    'filename': f"{fn_match[1]}",
                    'num_cores': num_cores,
                    'n_ctx_switch': number_of_context_switch,
                    'cum_time': cumulative_context_switch_time,
                }
        
                # Append a row to the dataframe with the values
                data_idx = (df['aux_scale_factor'] == args.aux_scale_factor) & \
                            (df['e2e_latency'] == args.e2e_latency) & \
                                (df['wsc_slack_ratio'] == args.wsc_slack_ratio - args.temporal_rda_ratio) & \
                                    (df['temporal_rda_ratio'] == args.temporal_rda_ratio) & \
                                        (df['lateness_mode'] == args.lateness_mode) & \
                                            (df['filename'] == f"{fn_match[1]}") & \
                                                (df['num_cores'] == num_cores) 

                if df.loc[data_idx].size:
                    df.loc[data_idx, 'n_ctx_switch'] = number_of_context_switch
                    df.loc[data_idx, 'cum_time'] = cumulative_context_switch_time
                else:
                    df = pd.concat([df, pd.DataFrame.from_dict(data, orient='index').T], ignore_index=True)

    df.to_csv(filename, index=False)
