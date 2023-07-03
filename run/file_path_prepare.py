"""
if [ "$cfg" = "light" ]; then
    fn="profiling_light.csv"
elif [ "$cfg" = "medium" ]; then
    fn="profiling_medium.csv"
elif [ "$cfg" = "heavy" ]; then
    fn="profiling.csv"
else
    echo "无效的输入"
    exit 1
fi

echo "文件名为：$fn"

# 循环
for x in $(seq "$start" "$step" "$end"); do
    file_path="log/$cfg/$x/bin_pack_new_$x.log.txt"
    dir_path=$(dirname "$file_path")

    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
    fi

    rm "log/$cfg/$x/bin_pack_new_$x.log.txt"
    rm "log/$cfg/$x/glb_dyn_${x}_ideal.log.txt"
    rm "log/$cfg/$x/glb_dyn_${x}_jitter_dis.log.txt"
    rm "log/$cfg/$x/glb_dyn_${x}_jitter_en.log.txt"
    rm "log/$cfg/$x/dyn_${x}_jitter_dis.log.txt"
    rm "log/$cfg/$x/dyn_${x}_jitter_en.log.txt"
done

"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", action="store_true", help="verbose")
parser.add_argument("--temporal_rda_ratio", default=0.05, type=float, help="temporal ratio")
parser.add_argument("--profiling_filename", type=str, default="profiling_light.csv", help="profiling filename")
parser.add_argument("--lateness_mode", type=str, default="ignore", help="lateness mode")
parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
parser.add_argument("--slack_threshold", default=5e-4, type=float, help="slack threshold")
parser.add_argument("--aux_scale_factor", default=1, type=int, help="aux scale factor")
parser.add_argument("--gen_benchmark", default=False, action="store_true", help="generate benchmark")
parser.add_argument("--start", default=200, type=int, help="start")
parser.add_argument("--step", default=20, type=int, help="step")
parser.add_argument("--end", default=360, type=int, help="end")
args = parser.parse_args() 
# print(args)
if not args.gen_benchmark:
    if args.profiling_filename == "profiling.csv":
        cfg_n = "heavy"
    else:
        cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
else:
    cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio-args.temporal_rda_ratio):.2%}(T)_{args.temporal_rda_ratio:.2%}(S)"


fn = args.profiling_filename
start = args.start
step = args.step
end = args.end

# print("文件名为：", fn)

# 循环
for x in range(start, end + 1, step):
    file_path = f"log/{cfg_n}/{x}/bin_pack_new_{x}.log.txt"
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for log_name in [
        f"log/{cfg_n}/{x}/bin_pack_new_{x}.log.txt",
        f"log/{cfg_n}/{x}/glb_dyn_{x}_ideal.log.txt",
        f"log/{cfg_n}/{x}/glb_dyn_{x}_jitter_dis.log.txt",
        f"log/{cfg_n}/{x}/glb_dyn_{x}_jitter_en.log.txt",
        f"log/{cfg_n}/{x}/dyn_{x}_jitter_dis.log.txt",
        f"log/{cfg_n}/{x}/dyn_{x}_jitter_en.log.txt",
    ]:
        if os.path.exists(log_name):
            os.remove(log_name)

print(f"{cfg_n}")