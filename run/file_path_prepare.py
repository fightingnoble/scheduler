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
parser.add_argument("--cfg_n", default="light", type=str, help="cfg_n")
parser.add_argument("--start", default=200, type=int, help="start")
parser.add_argument("--step", default=20, type=int, help="step")
parser.add_argument("--end", default=360, type=int, help="end")
args = parser.parse_args() 
start = args.start
step = args.step
end = args.end
cfg_n = args.cfg_n

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