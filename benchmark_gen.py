import os
from task.task_cfg import create_init_p_list, gen_workloads
from task.task_cfg import affinity_cfg
from task.task_cfg import init_affinity
from task.task_agent import TaskInt 
from task.task_agent import TaskInt
from task.spec import Spec
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from sched.scheduler_agent import Scheduler
from sched.monitor_agent import Monitor
from allocator_agent import glb_sched, cyclic_sched
from model.event_gen.e2e_latency import jitter_gen_biside, get_intger_gen, discrete_event_sim, exp_jitter
import numpy as np 
import math
import matplotlib.pyplot as plt
from global_var import *

def main():
    import argparse
    import pickle
    from global_var import trace_list
    import json

    from utils import input_parser
    args = input_parser() 
    if not args.gen_benchmark:
        if args.profiling_filename == "profiling/profiling.csv":
            cfg_n = "heavy"
        else:
            cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
    else:
        cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio):.2%}(T)_{args.exec_t_comp_ratioA:.2%}(S)"
    num_cores = args.num_cores
    num_periods = args.n_p
    
    if args.jitter_sim_en: 
        if args.jitter_sim_para == {}:
            jitter_sim_para = args.jitter_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['jitter']
        else:
            jitter_sim_para = args.jitter_sim_para


    if args.load_var_sim_en:
        if args.load_var_sim_para == {}:
            args.load_var_sim_para = load_var_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['load_var']
        else:
            load_var_sim_para = args.load_var_sim_para
    
    if args.e2e_var_sim_en:
        if args.e2e_var_sim_para == {}:
            args.e2e_var_sim_para = e2e_var_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['e2e_var']
        else:
            e2e_var_sim_para = args.e2e_var_sim_para

    if args.bin_pack_para == {}:
        args.binpack_cfg = binpack_cfg = json.load(open(os.path.join(cfg_dir, args.bin_pack_cfg), "r"))
    else:
        binpack_cfg = args.bin_pack_para

    hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args)

    # generate the process list
    glb_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
    init_affinity(glb_p_list, mode='job', job_graph_nx=physical_graph_nx, verbose=args.verbose)

    # simlation settings
    np.random.seed(args.seed)
    if True:
        plot_distr_hist(args, glb_n_task_dict)


def plot_distr_hist(args, glb_n_task_dict):
    sensor_dict = {}
    for key, task in glb_n_task_dict.items():
        thread_n = key.split('_')[-2]
        troughput_n = key.split('_')[-1]
        task_n = key.replace("_"+thread_n, "").replace("_"+troughput_n, "")
        if task_n in sensor_dict:
            continue
        if task.trigger_mode == "event":
            sensor_dict[task_n] = task
    
    n_plot = len(sensor_dict)+2
    cols = math.ceil((n_plot)**0.5)
    rows = math.ceil((n_plot)/cols)
    fig, ax = plt.subplots(rows, cols, figsize=(cols*5, rows*5))

    axis_idx = 0 
    plot_sample_num = 1000
    rng = np.random.default_rng(args.seed)
    for key, task in sensor_dict.items():
        seed = rng.integers(0,plot_sample_num)
        print(seed)
        sensor_jitter = jitter_gen_biside(1/task.freq, jitter_sim_para=args.jitter_sim_para, size=plot_sample_num, seed=seed)
        ax[axis_idx//cols, axis_idx%cols].hist(sensor_jitter(), density=True, 
                                            bins='auto', histtype='stepfilled', alpha=0.2,
                                            label=f"{task.name}")
        ax[axis_idx//cols, axis_idx%cols].set_title(f"{task.name}-{task.freq}Hz")
        # set axis as scientific notation
        ax[axis_idx//cols, axis_idx%cols].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axis_idx += 1 

    if args.e2e_var_sim_en:
        e2e_jitter_gen = jitter_gen_biside(args.e2e_latency, jitter_sim_para=args.e2e_var_sim_para, 
                                size=plot_sample_num, seed=rng.integers(0,plot_sample_num))
        e2e_jitter = e2e_jitter_gen()
        ax[axis_idx//cols, axis_idx%cols].hist(e2e_jitter, density=True, 
                                            bins='auto', histtype='stepfilled', alpha=0.2,
                                            label=f"e2e")
        ax[axis_idx//cols, axis_idx%cols].set_title(f"e2e-{args.e2e_latency}ms")
        axis_idx += 1

    switch_size= (jitter_gen_biside(1, {'scale': 0.2, }, size=plot_sample_num, seed=0)()+0.8)*GLB_BUFFER_SIZE_PER_CORE/3
    # switch_size= jitter_gen_biside(GLB_BUFFER_SIZE_PER_CORE/3, {'scale': 0.2, }, size=plot_sample_num, seed=0)()+GLB_BUFFER_SIZE_PER_CORE/3*0.8

    ax[axis_idx//cols, axis_idx%cols].hist(switch_size, density=True,
                                        bins='auto', histtype='stepfilled', alpha=0.2,
                                        label=f"switch")
    ax[axis_idx//cols, axis_idx%cols].set_title(f"tile_size-{GLB_BUFFER_SIZE_PER_CORE*0.8/3/1e3:.2f}K")
    # set axis as scientific notation
    ax[axis_idx//cols, axis_idx%cols].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axis_idx += 1

    base_lat = GLB_BUFFER_SIZE_PER_CORE*0.8/3/BW_DRAM + AVG_HOP_NUM * LAT_PER_HOP
    trans_jitter = (exp_jitter(1, {'scale': 0.2, }, size=plot_sample_num, seed=0)()+1)*base_lat
    ax[axis_idx//cols, axis_idx%cols].hist(trans_jitter, density=True,
                                        bins='auto', histtype='stepfilled', alpha=0.2,
                                        label=f"switch")
    ax[axis_idx//cols, axis_idx%cols].set_title(f"tile_trans_jitter-[{base_lat*1e6:.3f}:{base_lat*1.2*1e6}]us")
    # set axis as scientific notation
    ax[axis_idx//cols, axis_idx%cols].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axis_idx += 1

    trans_jitter = (1-exp_jitter(1, {'scale': 0.2, }, size=plot_sample_num, seed=0)())
    ax[axis_idx//cols, axis_idx%cols].hist(trans_jitter, density=True,
                                        bins='auto', histtype='stepfilled', alpha=0.2,
                                        label=f"switch")
    ax[axis_idx//cols, axis_idx%cols].set_title(f"actual ops-[0:100]%")
    # set axis as scientific notation
    ax[axis_idx//cols, axis_idx%cols].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axis_idx += 1


    if args.load_var_sim_en:
        ax[axis_idx//cols, axis_idx%cols].hist(get_intger_gen(30,plot_sample_num,args.seed)(), density=True,
                bins='auto', histtype='stepfilled', alpha=0.2,
                label=f"n_dyn_obj")
        axis_idx += 1
    
    plt.savefig(f"doc/verification/jitter_sim.pdf")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()