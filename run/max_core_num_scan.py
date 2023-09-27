# #!/bin/bash

# # root_dir=${5:-"aux_scan"}
# # p_fn=${6:-"sim_main.py"}
# # n_p=${7:-"3"}
# # pre_alloc=${8:-"False"}
# # static_sim=${9:-"False"}
# # glb_dyn=${10:-"True"}
# # dyn=${11:-"True"}
# # seed=${12:-"0"}
# # PY_ARGS=${@:13}

# root_dir=${1:-"lat_scan"}
# p_fn=${2:-"sim_main.py"}
# n_p=${3:-"3"}
# PY_ARGS=${@:4}


# num_cores=400

# for seed in {0..9}; do
#   # Dyn tests
#   for aux_lat in [[0.08,6],[0.09,11],[0.1,11]]; do  
#     aux=${aux_lat[0]}
#     lat=${aux_lat[1]}
    
#      python $p_fn --test_case dynamic \
#       --jitter_sim_en --file_suffix var_0.2  \
#       --num_cores $num_cores --root_dir $root_dir --n_p $n_p \
#       --aux_scale_factor $aux --e2e_latency $lat \
#       --seed $seed ${PY_ARGS} 
#   done

#   # GLB tests
#   for aux_lat in [[0.08,5],[0.09,5],[0.1,6]]; do
#     aux=${aux_lat[0]}
#     lat=${aux_lat[1]}

#     python $p_fn --test_case glb_dynamic \
#       --jitter_sim_en --file_suffix var_0.2 \  
#       --num_cores $num_cores --root_dir $root_dir --n_p $n_p \
#       --aux_scale_factor $aux --e2e_latency $lat \
#       --seed $seed ${PY_ARGS} 
#   done
  
# done


import subprocess, time
import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--type", type=str, default="all", help="type")
args = argparser.parse_args()

root_dir = "barycenter/lat_scan" 
p_fn = "sim_main.py"
n_p = 3
PY_ARGS = []

num_cores = 400

p_list = []

if args.type == "dyn":
  # Dyn tests
  for lat, aux in [[0.08,6], [0.09,11], [0.1,11]]:

    command = [
      "python", p_fn,
      "--test_case", "dynamic",
      "--num_cores", str(num_cores), "--root_dir", root_dir, "--n_p", str(n_p),
      "--aux_scale_factor", str(aux), "--e2e_latency", str(lat),
      "--max_core_stat", "True", "--gen_benchmark", 
    ] + PY_ARGS

    with open(os.devnull, 'w') as devnull:
      subprocess.run(command, stdout=devnull)

    for seed in range(10):

      command = [
        "python", p_fn,
        "--test_case", "dynamic",
        "--jitter_sim_en", "--file_suffix", "var_0.2",
        "--num_cores", str(num_cores), "--root_dir", root_dir, "--n_p", str(n_p),
        "--aux_scale_factor", str(aux), "--e2e_latency", str(lat),
        "--seed", str(seed), "--max_core_stat", "True", "--gen_benchmark", 
      ] + PY_ARGS
        
      with open(os.devnull, 'w') as devnull:
        subprocess.run(command, stdout=devnull)
else:
  for seed in range(10):
    # GLB tests    
    for lat, aux in [[0.08,5], [0.09,5], [0.1,6]]:  
      command = [
        "python", p_fn,
        "--test_case", "glb_dynamic",
        "--jitter_sim_en", "--file_suffix", "var_0.2",
        "--num_cores", str(num_cores), "--root_dir", root_dir, "--n_p", str(n_p), 
        "--aux_scale_factor", str(aux), "--e2e_latency", str(lat), 
        "--seed", str(seed), "--max_core_stat", "True", "--gen_benchmark", 
      ] + PY_ARGS
      with open(os.devnull, 'w') as devnull:
        subprocess.run(command, stdout=devnull)
