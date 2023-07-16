import argparse
def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="all", help="task name")
    parser.add_argument("--plot", action="store_true", help="plot the task timeline")
    parser.add_argument("--test_all", default=False, help="test all the task")
    parser.add_argument("--num_cores", default=266, type=int, help="number of cores")
    parser.add_argument("--BinExtendRule", default="list", type=str, help="Rule for when and how to extend the bin")
    parser.add_argument("--preemptable", default=False, action="store_true", help="enable preemption")
    parser.add_argument("--quantum_check_en", default=False, action="store_true", help="enable quantum check")
    parser.add_argument("--quantumSize", default=2, type=int, help="quantum size, # of simulation steps")
    # parser.add_argument("--hyper_p", default=None, type=float, help="hyper period")
    parser.add_argument("--warmup", default=False, action="store_true", help="warmup")
    # parser.add_argument("--drain", default=False, action="store_true", help="drain")
    # parser.add_argument("--sim_step", default=None, type=float, help="simulation step")
    parser.add_argument("--n_p", default=1, type=int, help="number of periods")
    
    parser.add_argument("--jitter_sim_en", default=False, action="store_true", help="enable jitter simulation")
    parser.add_argument("--jitter_sim_para", default={"loc":0, "scale":0.2}, type=dict, help="jitter simulation parameters")
    
    parser.add_argument("--load_var_sim_en", default=False, action="store_true", help="enable dynamic object simulation")
    parser.add_argument("--load_var_sim_para", default={}, type=dict, help="dynamic object simulation parameters")
    parser.add_argument("--load_var_para_file", default="load_var_para.json", type=str, help="dynamic object simulation parameters file")

    parser.add_argument("--e2e_var_sim_en", default=False, action="store_true", help="enable e2e latency variation simulation")
    parser.add_argument("--e2e_var_sim_para", default={"loc":0, "scale":0.2, "period":0.1}, type=dict, help="e2e latency variation simulation parameters")
    
    parser.add_argument("--file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--i_file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--barrier_dis", default=False, action="store_true", help="disable barrier")
    parser.add_argument("--data_lifetime_mode", default="static", type=str, help="lifetime mode: most_recent, ref_count, timeout, watermark") 
    parser.add_argument("--spatial_rda_ratio", default=0.2, type=float, help="spatial ratio")
    parser.add_argument("--temporal_rda_ratio", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--profiling_filename", type=str, default="profiling_light.csv", help="profiling filename")
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

    args = parser.parse_args()
    return args


args = input_parser() 
# print(args)
if not args.gen_benchmark:
    if args.profiling_filename == "profiling.csv":
        cfg_n = "heavy"
    else:
        cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
else:
    cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio-args.temporal_rda_ratio):.2%}(T)_{args.temporal_rda_ratio:.2%}(S)"
cfg_n += f"_{args.lateness_mode}"

print(f"{cfg_n}")