import os
from task.task_cfg import create_init_p_list, gen_workloads
from task.task_cfg import affinity_cfg
from task.task_cfg import init_affinity
from global_sched import push_task_into_bins, push_task_into_bins_new
from task.task_agent import TaskInt 
from task.task_agent import TaskInt
from task.spec import Spec
from model.msg_dispatcher import MsgDispatcher
from model.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from scheduler_agent import Scheduler
from sched.monitor_agent import Monitor
from allocator_agent import glb_sched, cyclic_sched
from model.event_gen.e2e_latency import jitter_gen, num_dyn_obj_gen, dyn_obj_sim, e2e_var_sim
import numpy as np 
import math
import matplotlib.pyplot as plt
from model.event_gen.e2e_latency import jitter_gen

def main():
    import argparse
    import pickle
    from global_var import trace_list
    import json

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
    parser.add_argument("--load_var_para_file", default=None, type=str, help="dynamic object simulation parameters file")

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

    args = parser.parse_args() 
    if not args.gen_benchmark:
        if args.profiling_filename == "profiling.csv":
            cfg_n = "heavy"
        else:
            cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
    else:
        cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio-args.temporal_rda_ratio):.2%}(T)_{args.temporal_rda_ratio:.2%}(S)"
    num_cores = args.num_cores
    num_periods = args.n_p
    slack_threshold = args.slack_threshold
    warmup = args.warmup
    if args.load_var_sim_para == {}:
        load_var_sim_para = json.load(open(args.load_var_para_file, "r"))
    else:
        load_var_sim_para = args.load_var_sim_para

    hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args, slack_threshold)

    # generate the process list
    glb_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
    init_affinity(glb_p_list, mode='job', job_graph_nx=physical_graph_nx, verbose=args.verbose)

    # assert all the process has hard deadline
    if args.lateness_mode == "all_hard":
        for _p in glb_p_list:
            _p.task.criticality = "hard"
    elif args.lateness_mode == "all_soft":
        for _p in glb_p_list:
            _p.task.criticality = "soft"
    elif args.lateness_mode == "ignore":
        pass

    # simlation settings
    sim_step = min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32
    quantumSize = sim_step*args.quantumSize
    save_path = f"cache/{cfg_n}/bin_list_{num_cores}{args.i_file_suffix}.pkl"
    np.random.seed(args.seed)
    # from model.message_handler import gen_sensor_event
    # event_iter_dict = gen_sensor_event(glb_p_list, hyper_p, num_periods, True, args.jitter_sim_en, args.jitter_sim_para, args.seed)
    jitter_para_dict = dict(jitter_sim_en=args.jitter_sim_en, jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    event_iter_dict = TaskInt.get_event_generator(glb_n_task_dict, hyper_p, num_periods, True, **jitter_para_dict)
    # convert to list then convert to iterator
    # event_iter_dict = {task: [list(event_iter_dict[task][0]), list(event_iter_dict[task][1])] for task in event_iter_dict}
    # pickle.dump(event_iter_dict, open(f"cache/{cfg_n}/event_iter_dict_{args.test_case}_{args.jitter_sim_en}.pkl", "wb"))
    # event_iter_dict = {task: [iter(event_iter_dict[task][0]), iter(event_iter_dict[task][1])] for task in event_iter_dict}
    if True:
        plot_distr_hist(args, glb_n_task_dict)
    
    event_range = hyper_p * (num_periods+warmup)
    # integrated in to virtual sensor related source operator
    # if the handler find the var scaling factor is greater than 1, it spawns(wake up) x(factor-1) of new threads, 
    # which is marked as "spawned"
    # the "spawned" thread will be terminated as soon as they finish their job
    dyn_obj_sim(args.load_var_sim_para, 1, args.e2e_var_sim_para["period"], event_range, args.seed)
    # integrated in to every message from every sensor source operator
    e2e_var_sim(args.e2e_latency, args.e2e_var_sim_para, 1, args.e2e_var_sim_para["period"], event_range, args.seed)
    


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
        sensor_jitter = jitter_gen(1/task.freq, jitter_sim_para=args.jitter_sim_para, size=plot_sample_num, seed=seed)
        ax[axis_idx//cols, axis_idx%cols].hist(sensor_jitter(), density=True, 
                                            bins='auto', histtype='stepfilled', alpha=0.2,
                                            label=f"{task.name}")
        ax[axis_idx//cols, axis_idx%cols].set_title(f"{task.name}-{task.freq}Hz")
        # set axis as scientific notation
        ax[axis_idx//cols, axis_idx%cols].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axis_idx += 1 

    e2e_jitter_gen = jitter_gen(args.e2e_latency, jitter_sim_para=args.e2e_var_sim_para, 
                            size=plot_sample_num, seed=rng.integers(0,plot_sample_num))
    e2e_jitter = e2e_jitter_gen()
    ax[axis_idx//cols, axis_idx%cols].hist(e2e_jitter, density=True, 
                                        bins='auto', histtype='stepfilled', alpha=0.2,
                                        label=f"e2e")
    ax[axis_idx//cols, axis_idx%cols].set_title(f"e2e-{args.e2e_latency}ms")
    # set axis as scientific notation
    ax[axis_idx//cols, axis_idx%cols].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axis_idx += 1
    ax[axis_idx//cols, axis_idx%cols].hist(num_dyn_obj_gen(30,plot_sample_num,args.seed)(), density=True,
            bins='auto', histtype='stepfilled', alpha=0.2,
            label=f"n_dyn_obj")
    plt.savefig(f"doc/verification/jitter_sim.pdf")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()