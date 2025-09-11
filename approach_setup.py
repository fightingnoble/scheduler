
import os
import re
import pandas as pd
import numpy as np
from collections import namedtuple

from task.task_cfg import gen_workloads, affinity_cfg
from sched.global_sched import push_task_into_bins_new, coleasing_alloc_1bin
from sched.global_sched import coleasing_alloc_cluster
from task.task_agent import TaskInt
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from sched.scheduler_agent import Scheduler
from sched.monitor_agent import Monitor
from model.event_gen.e2e_latency import discrete_event_sim
from model.task_queue_agent import TaskQueue
from utils import (dump_and_check, load_pickle, build_path_old, 
                   check_parents_path, core_distr)
from sim_main import (build_paths_and_ctx, build_workload_and_criticality, 
                     build_scheduler_elements, build_simulation_env)
from global_var import *
from approach_initiator import instantiate_mygraph_from_json
from utils import get_case_path_str

# =================================================================================================
# Helper functions migrated from sim_main.py
# =================================================================================================

def _get_core_num_from_trace_name(path_para_dict):
    """
    Finds the number of cores from a saved bin_list filename in a given directory.
    """
    folder = cache_root_fmt.format(**path_para_dict)
    if not os.path.exists(folder):
        return folder, [], None
        
    root, dirs, files = os.walk(folder).__next__()
    assert len(dirs) == 0
    
    for fn in files:
        if match := re.match(bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"}), fn):
            return folder, files, match
            
    return folder, files, None


def _create_common_sim_elements(args, trace_path_para, glb_p_list, hyper_p, sim_step, bin_list):
    """
    Creates a collection of common simulation objects needed for bin-packing algorithms.
    """
    case_pth = f"policy_{args.policy}" # Use policy for path instead of test_case
    
    if args.jitter_sim_en:
        trace_path = trace_fn_w_seed_fmt.format(**trace_path_para, **{"case": case_pth})
    else:
        trace_path = trace_fn_wo_seed_fmt.format(**trace_path_para, **{"case": case_pth})

    scheduler_args = {
        "exec_t_comp_ratioB": args.exec_t_comp_ratioB,
        "barrier_en": not args.barrier_dis, 
        "forbid_miss": args.forbid_miss,
        "progress_aware": True if args.test_case in two_stage_case_coll else False,
        "allow_realloc": args.allow_realloc,
        "trace_path": trace_path
    }
    
    exec_para_dict = dict(exec_var_en=args.exec_var_en, exec_var_para=args.exec_var_para, seed=args.seed)
    rsc_list = [Resource_model_int(size=sched_tab.num_resources, **exec_para_dict) for sched_tab in bin_list]
    msg_dispatcher = MsgDispatcher(len(bin_list))
    a_data_pipe = DataPipe("activation", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    w_data_pipe = DataPipe("weight", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    scheduler_list = [Scheduler(bin_list[idx], args.e2e_latency, hyper_p, glb_p_list, 
                                        res_cfg=rsc_list[idx], **scheduler_args) for idx in range(len(bin_list))]
    monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]
    
    return msg_dispatcher, a_data_pipe, w_data_pipe, scheduler_list, monitor_list

# =================================================================================================
# Bin list generators, wrapping the logic from sim_main.py
# =================================================================================================

def _generate_coalescing_bins(args, common_params, path_params):
    """ Corresponds to 'cyc' policy """
    print_title("Generating Bins using Coalescing method")
    # Unpack all necessary parameters from common_params
    (bin_list, glb_p_list, hyper_p, _, 
     num_periods, sim_step, quantumSize, event_iter_dict,
     msg_dispatcher, a_data_pipe, w_data_pipe, scheduler_list, monitor_list) = common_params
    
    path_para_dict, csv_xlxs_root, bin_save_fmt, plot_path_para, cfg_para_dict, para_scan_group1 = path_params

    max_core_layout = coleasing_alloc_1bin(
        bin_list, glb_p_list, affinity_cfg, event_iter_dict,
        args.num_cores, args.quantum_check_en, quantumSize, 
        sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,
        scheduler_list, monitor_list, msg_dispatcher, a_data_pipe, w_data_pipe,
        num_periods, binpack_cfg=args.binpack_cfg,
        verbose=True, DEBUG_FG=False, warmup=True, drain=True, 
    )
    
    bin_list[0].to_sparse_dict()
    num_cores = sum(max_core_layout[1].values())

    if args.force_num_cores and args.aux_scale_factor != 9:
        _cfg_n_t = cfg_root_fmt.format(**cfg_para_dict, **{**para_scan_group1, "aux_scale_factor": 9})
        _path_para_dict = {"root_dir": args.root_dir, "cfg_n": _cfg_n_t, "i_file_suffix": args.i_file_suffix, "force_suffix": ""}
        folder, files, match = _get_core_num_from_trace_name(_path_para_dict)
        if not match:
            print(f"!!! Warning: no bin_list file for core forcing found in {folder}:{files} !!!")
            # Fallback or error
        else:
            forced_cores = int(match.group(1))
            if forced_cores > num_cores:
                bin_list[0].num_resources = forced_cores
                print(f"Force the num of Core {num_cores} -> {forced_cores}")
                num_cores = forced_cores
            elif forced_cores < num_cores:
                print(f"Error: Forced specified num of Core {forced_cores} should be larger than the estimated num of cores {num_cores}")
                import sys; sys.exit(1)

    bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
    dump_and_check(bin_list_save_path, bin_list)
    print(f"Saved coalescing bin_list to: {bin_list_save_path}")

    return bin_list

def _generate_split_bins(args, common_params, path_params):
    """ Corresponds to 'pglb' policy """
    print_title("Generating Bins using Bin Split method")
    # Unpack all necessary parameters from common_params
    (bin_list, glb_p_list, hyper_p, physical_graph_nx, 
     num_periods, sim_step, quantumSize, event_iter_dict,
     msg_dispatcher, a_data_pipe, w_data_pipe, scheduler_list, monitor_list) = common_params

    path_para_dict, csv_xlxs_root, bin_save_fmt, plot_path_para, cfg_para_dict, para_scan_group1 = path_params

    pid2_bin_id, bin_size_list = coleasing_alloc_cluster(
        bin_list, glb_p_list, affinity_cfg, event_iter_dict,
        args.num_cores, args.quantum_check_en, quantumSize,
        sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,
        scheduler_list, monitor_list, msg_dispatcher, a_data_pipe, w_data_pipe,
        num_periods, binpack_cfg=args.binpack_cfg,
        job_graph=physical_graph_nx, 
        n_partition=args.num_bins if args.num_bins != -1 else 9999,
        verbose=True, DEBUG_FG=False, warmup=True, drain=True, 
    )
    
    num_cores = sum(bin_size_list.values())

    if args.force_num_cores and args.aux_scale_factor != 9:
        _cfg_n_t = cfg_root_fmt.format(**cfg_para_dict, **{**para_scan_group1, "aux_scale_factor": 9})
        _path_para_dict = {"root_dir": args.root_dir, "cfg_n": _cfg_n_t, "i_file_suffix": args.i_file_suffix, "force_suffix": ""}
        folder, files, match = _get_core_num_from_trace_name(_path_para_dict)
        if not match:
             print(f"!!! Warning: no bin_list file for core forcing found in {folder}:{files} !!!")
        else:
            forced_cores = int(match.group(1))
            if forced_cores > num_cores:
                over_sub_ratio = forced_cores / num_cores
                score_dict = {_bin.id:_bin.num_resources for _bin in bin_list} 
                rsc_map = {**score_dict}
                curr_aval_rsc = forced_cores - num_cores
                core_distr(rsc_map, score_dict, curr_aval_rsc)
                assert sum(rsc_map.values()) == forced_cores
                for _bin in bin_list:
                    _bin.num_resources = rsc_map[_bin.id]
                print(f"Force the num of Core {num_cores} -> {forced_cores}, over-subscription ratio is {over_sub_ratio:.2f}")
                num_cores = forced_cores
            elif forced_cores < num_cores:
                print(f"Error: Forced specified num of Core {forced_cores} should be larger than the estimated num of cores {num_cores}")
                import sys; sys.exit(1)
    
    bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
    dump_and_check(bin_list_save_path, bin_list)
    print(f"Saved split bin_list to: {bin_list_save_path}")

    return bin_list

def _generate_repack_bins(args, common_params, path_params):
    """ Corresponds to 'reserv' policy """
    print_title("Generating Bins using Repack method")
    # Unpack relevant parameters
    (initial_bin_list, _, _, _, 
     num_periods, sim_step, quantumSize, event_iter_dict,
     _, _, _, _, _) = common_params

    path_para_dict, csv_xlxs_root, bin_save_fmt, plot_path_para, cfg_para_dict, para_scan_group1 = path_params

    # 1. Load the original bin configuration
    folder, files, match = _get_core_num_from_trace_name(path_para_dict)
    if not match:
        print(f"!!! ERROR: No base bin_list file found for repacking in {folder}:{files} !!!")
        import sys; sys.exit(1)
    
    num_cores = int(match.group(1))
    bin_list_load_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
    bin_list = load_pickle(bin_list_load_path)
    print(f"Loaded base bin_list from: {bin_list_load_path}")

    # 2. Modify args for repacking step as in sim_main
    args.exec_t_comp_ratioA = args.exec_t_comp_ratioB 
    args.binpack_cfg["slack_sharing"] = True
    
    # 3. Regenerate workloads with new slack distribution
    print("="* 20 + "Redistribute slack for repack" + "="* 20)
    hyper_p, _, _, glb_p_list = gen_workloads(args) # New glb_p_list

    # 4. Prepare for repacking
    print("="* 20 + "Extracting bin-assignment" + "="* 20)
    pid2_bin_id = {}
    for _bin in bin_list:
        pid_list = _bin.index_occupy_by_id().keys()
        for pid in pid_list:
            pid2_bin_id[pid] = _bin.id
        _bin.clear() # Clear the bin for refilling

    args.binpack_cfg["mapping"] = pid2_bin_id
    args.binpack_cfg["bin_sel_mod"] = "pre_defined"
    args.binpack_cfg["affinity_en"] = False

    # 5. Create new sim elements with new glb_p_list
    # Note: We create new elements here because glb_p_list is regenerated for repack
    msg_dispatcher, a_data_pipe, w_data_pipe, scheduler_list, monitor_list = _create_common_sim_elements(
        args, path_para_dict, glb_p_list, hyper_p, sim_step, bin_list)
    
    # 6. Run the repacking (push)
    repacked_bin_list = push_task_into_bins_new(
        bin_list, glb_p_list, affinity_cfg, event_iter_dict,
        num_cores, args.quantum_check_en, quantumSize, 
        sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,
        scheduler_list, monitor_list, msg_dispatcher, a_data_pipe, w_data_pipe,
        num_periods, binpack_cfg=args.binpack_cfg,
        verbose=True, DEBUG_FG=False, warmup=True, drain=True, 
    )

    # 7. Save the new bin_list
    path_para_dict['i_file_suffix'] += f"_ov_{args.exec_t_comp_ratioB:.2f}_repack"
    bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
    dump_and_check(bin_list_save_path, repacked_bin_list)
    print(f"Saved repacked bin_list to: {bin_list_save_path}")

    return repacked_bin_list

# =================================================================================================
# Main factory and setup function
# =================================================================================================

def get_bin_list_generator(policy: str):
    """
    Factory function to select the bin list generation algorithm based on the policy.
    """
    if policy == "cyc":
        return _generate_coalescing_bins
    elif policy == "pglb":
        return _generate_split_bins
    elif policy == "reserv":
        return _generate_repack_bins
    else:
        raise ValueError(f"Unknown policy for bin list generation: '{policy}'")


def setup_benchmark(args, time_norm_factor):
    """
    Main function to setup the benchmark. It generates workloads, creates or loads
    the scheduling information (bin_list), and prepares the logical graph.

    Returns:
        tuple: (G, pid2name, bin_list, policy, log_path_root, hyper_p)
    """
    print_title("Benchmark Setup Started")
    
    # ======================== 使用 sim_main API ========================
    # call api from sim_main.py, 
    # build the workload graph, setup the path, 
    # generate the bin_list
    # 1. 构建路径和上下文
    path_params, path_ctx = build_paths_and_ctx(args)
    
    # 2. 生成 workload 并设置 criticality
    workload = build_workload_and_criticality(args)
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list = workload
    
    # 3. 构建调度器元素
    scheduler_result = build_scheduler_elements(args, path_params, path_ctx, workload)
    if scheduler_result is None:
        print("Error: Failed to build scheduler elements")
        return None
        
    (task_spec, rsc_list, msg_dispatcher, a_data_pipe, w_data_pipe, 
     scheduler_list, monitor_list, trace_path, num_cores, bin_list, sim_step) = scheduler_result
    
    # 4. 构建仿真环境参数
    num_periods, warmup, quantumSize, event_range, event_iter_dict = build_simulation_env(
        args, workload, sim_step
    )

    # 5. generate schedule parameters
    G, pid2name = instantiate_mygraph_from_json(
        'cache/graph_w_ert_ddl.json',
        time_norm_factor=time_norm_factor
        )
    
    # 如果指定了 bin_list 路径则加载，否则使用生成的 bin_list
    if hasattr(args, 'load_bin_list_path') and args.load_bin_list_path:
        bin_list = load_pickle(args.load_bin_list_path)
        # 重新计算 hyper_p
        f_gcd = np.gcd.reduce([freq for task, freq in G.logical_graph.nodes(data="freq")])
        hyper_p = 1/f_gcd

    print_title("Benchmark Setup Finished")
    
    return G, pid2name, bin_list, args.policy, path_ctx.get_log_path(), hyper_p

def print_title(title):
    print("\n" + "="*25)
    print(f" {title}")
    print("="*25 + "\n")
