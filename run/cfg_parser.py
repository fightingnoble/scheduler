import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(parent_dir)
sys.path.append('.')

from utils import input_parser
args = input_parser() 
# print(args)
if not args.gen_benchmark:
    if args.profiling_filename == "profiling.csv":
        cfg_n = "heavy"
    else:
        cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
else:
    cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio):.2%}(T)_{args.temporal_rda_ratio:.2%}(S)"
cfg_n += f"_{args.lateness_mode}"

print(f"{cfg_n}")