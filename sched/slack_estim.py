from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np
import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    from task.task_agent import TaskBase, ProcessBase, TaskIntAttr, ProcessInt
    from networkx import DiGraph
from typing import List, Any, Dict, Tuple, Union
from global_var import *
from task.graph_breakdown import decompose_dag_into_chains

def EstimCoreNums4Process(_p:ProcessInt, flops, expected_slack, 
                          round_mode="round", curr_aval_rsc:int=None):
    if round_mode == "ceil":
        round_func = math.ceil
    elif round_mode == "floor":
        round_func = math.floor
    else:
        round_func = round
    req_rsc_size = flops / expected_slack / FLOPS_PER_CORE

    constr = None
    if _p.parallel_mode in ["upb","range"]:
        req_rsc_size = min(round_func(req_rsc_size), _p.core_max)
        if req_rsc_size==_p.core_max: 
            constr = "upb"
    elif _p.parallel_mode in ["lwb", "range"]:
        if curr_aval_rsc is not None:
            if _p.core_min > curr_aval_rsc:
                # no available solution
                return 0, "N/A"
        req_rsc_size = max(round_func(req_rsc_size), _p.core_min)
        if req_rsc_size==_p.core_min:
            constr = "lwb"
    elif _p.parallel_mode == "list":
        # select the nearest one
        # filter the core_list by the current available resource
        if curr_aval_rsc is not None:
            core_list = [x for x in _p.core_list if 0 < x <= curr_aval_rsc] 
            if len(core_list) == 0:
                # no available solution
                return 0, "N/A"
        else:
            core_list = _p.core_list
        req_rsc_size = min(_p.core_list, key=lambda x:abs(x-req_rsc_size))
        if req_rsc_size==max(_p.core_list):
            constr = "upb"
        elif req_rsc_size==min(_p.core_list):
            constr = "lwb"
    else:
        req_rsc_size = max(round_func(req_rsc_size), 1)
    if curr_aval_rsc is not None:
        req_rsc_size = min(req_rsc_size, curr_aval_rsc)
    got_latency = flops / req_rsc_size / FLOPS_PER_CORE
    return req_rsc_size, got_latency, constr

def EstimCoreNums4Task(task_dict:Dict[str, TaskBase], flops_dict, node, expected_slack, round_mode="round"):
    if round_mode == "ceil":
        round_func = math.ceil
    elif round_mode == "floor":
        round_func = math.floor
    else:
        round_func = round
    req_rsc_size = flops_dict[node] / expected_slack / FLOPS_PER_CORE

    constr = None
    _task = task_dict[node]
    if _task.parallel_mode in ["upb","range"]:
        req_rsc_size = min(round_func(req_rsc_size), _task.core_max_compile)
        if req_rsc_size==_task.core_max_compile: 
            constr = "upb"
    elif _task.parallel_mode in ["lwb", "range"]:
        req_rsc_size = max(round_func(req_rsc_size), _task.core_min_compile)
        if req_rsc_size==_task.core_min_compile:
            constr = "lwb"
    elif _task.parallel_mode == "list":
        # select the nearest one
        # filter the core_list by the current available resource
        req_rsc_size = min(_task.core_list_compile, key=lambda x:abs(x-req_rsc_size))
        if req_rsc_size==max(_task.core_list_compile):
            constr = "upb"
        elif req_rsc_size==min(_task.core_list_compile):
            constr = "lwb"
    else:
        req_rsc_size = max(round_func(req_rsc_size), 1)
    got_latency = flops_dict[node] / req_rsc_size / FLOPS_PER_CORE
    return req_rsc_size, got_latency, constr


def alloc_func(rsc_map_w:Dict[str, Tuple[int, float]], 
               task_dict:Dict[str, TaskBase], 
               flops_dict:Dict[str, float], 
               ops_rem, slcak_rem, threshold):
    state = True
    for node in flops_dict:
        # estimate the slack
        slack_estm = flops_dict[node] / ops_rem * slcak_rem
        # estimate the resource
        req_rsc_size, got_latency, constr = EstimCoreNums4Task(task_dict, flops_dict, node, slack_estm, 'ceil')
        rsc_map_w[node] = (req_rsc_size, got_latency if constr else slack_estm, constr)
    
    # check the constraint
    constr_dict = {node:constr for node, (_, _, constr) in rsc_map_w.items() if node in flops_dict and constr is not None}
    if len(constr_dict) == 0:
        return state, 0, slcak_rem
    else:
        state = False
        curr_slack_rem = sum([slack_estm for node, (_, slack_estm, _) in rsc_map_w.items() if node in flops_dict]) -slcak_rem

        # exactly match the slack: 
        #  1. no process is reassigned due to the constraint
        #  2. the reassigned processes compensate each other
        if 0>=curr_slack_rem>=-threshold:
            return state, 0, curr_slack_rem
        elif curr_slack_rem > 0:
            # remove the process which has been reached the core_max
            for node,_constr in constr_dict.items():
                if _constr == "upb":
                    ops_rem -= flops_dict[node]
                    slcak_rem -= rsc_map_w[node][1]
                    flops_dict.pop(node)
        else:
            # remove the process which has been reached the core_min
            for node,_constr in constr_dict.items():
                if _constr == "lwb":
                    ops_rem -= flops_dict[node]
                    slcak_rem -= rsc_map_w[node][1]
                    flops_dict.pop(node)
    return state, ops_rem, slcak_rem


def build_score_dict_ref_flops(task_dict:Dict[str, TaskBase], nodes:Any, score_dict):
    for node_n in nodes: 
        assert node_n in task_dict
        _task = task_dict[node_n]
        score_dict[node_n] = _task.flops


def DistributeSlack(task_dict:Dict[str, TaskBase], e2e_latency, chains:List[List[Any]], temporal_rda_ratio, threshold):
    chains_info = []
    for chain in chains:
        flops_dict = {}
        build_score_dict_ref_flops(task_dict, chain, flops_dict)
        # TODO: select the e2e_latency, by the last node of the chain
        tail_task = task_dict[chain[-1]]
        if tail_task.timing_flag == "deadline":
            slcak_rem = e2e_latency
            is_ddl_constr = True
        else:
            slcak_rem = tail_task.freq_division_factor / tail_task.freq
            is_ddl_constr = False
        slcak_rem = (1-temporal_rda_ratio)*1e3*slcak_rem/1e3
        ops_rem = sum(flops_dict.values())
        chains_info.append((chain, flops_dict, slcak_rem, ops_rem, is_ddl_constr))
    
    sort_idx = np.array([(slcak_rem, not is_ddl_constr, -ops_rem) for _, _, slcak_rem, ops_rem, is_ddl_constr in chains_info], dtype=np.dtype('f8, ?, f8')).argsort()
    chains_info = [chains_info[i] for i in sort_idx]
    
    rsc_map_w:Dict[str, Tuple[int, float]] = {}
    for chain, flops_dict, slcak_rem, ops_rem, is_ddl_constr in chains_info:
        for node in chain:
            if node in rsc_map_w:
                ops_rem -= flops_dict[node]
                slcak_rem -= rsc_map_w[node][1]
                flops_dict.pop(node)
        state = False
        while not state and len(flops_dict) > 0:
            state, ops_rem, slcak_rem = alloc_func(rsc_map_w, task_dict, flops_dict, ops_rem, slcak_rem, threshold)
    return rsc_map_w

def rsc_slack_estim(taskJobs:Union[Dict[str, Union[TaskBase,ProcessBase]], List[Union[TaskBase,ProcessBase]]], 
             task_graph:DiGraph, start_nodes, end_nodes, e2e_latency, 
             temporal_rda_ratio, 
             threshold):
    """
    input:
        system specification:
            1. temporal_rda_ratio
            2. threshold
            3. e2e_latency
        task_graph: the task graph
        required properties of each task:
            1. freq
            2. freq_division_factor
            3. timing_flag
            4. name
            5. flops
            6. parallel_cfg_compile: parallel_mode, 1 of (core_min_compile, core_max_compile, core_list_compile)
    """
    if isinstance(taskJobs, list):
        if taskJobs[0].__class__.__name__ == "ProcessInt":
            task_dict = {i.task.name:i for i in taskJobs.task}
        elif taskJobs[0].__class__.__name__ == "TaskInt":
            task_dict = {i.name:i for i in taskJobs}
    elif isinstance(taskJobs, dict):
        # get a random element
        if next(iter(taskJobs.values())).__class__.__name__ == "ProcessInt":
            task_dict = {k:v.task for k,v in taskJobs.items()}
        else:
            task_dict = taskJobs
    else:
        raise TypeError("taskJobs should be a list or dict")
    
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    return DistributeSlack(task_dict, e2e_latency, chains, temporal_rda_ratio, threshold)

def estim_release_dll_time(task_graph_nx:DiGraph, 
                            comp_time: Dict[str, float]={},
                            io_time: Dict[str, float]={}, 
                            task_type: Dict[str, str]={}, 
                           temporal_rda_ratio=0, sched_step_comp=0, 
                           comm_compen_en=False, 
                           profiling_filename:str="profiling/profiling.csv",
                           verbose=False):
    """
    set the ERT and ddl property of each task: 
    traverse the job graph, 
    for each task, 
    ERT = max(ddl of all pred tasks) + io_time; 
    ddl = ERT + exp_comp_t; 
    """
    ert: Dict[str, float] = {}
    ddl: Dict[str, float] = {}

    if not comp_time:
        df:pd.DataFrame = pd.read_csv(profiling_filename, sep=",", index_col=0) 
        comp_time: Dict[str, float] = {task_n:df.loc[task_n, "Expected Latency (ms)"]/1000 for task_n in df.T}
        io_time: Dict[str, float] = {task_n:1e-6 for task_n in df.T}
        # task_type: Dict[str, str] = {task_n:df.loc[task_n, "Timing_flag"] for task_n in df.T}

    for node in nx.topological_sort(task_graph_nx):  # 拓扑排序遍历节点
        preds = task_graph_nx.pred[node]  # 获取当前节点的前驱节点
        if node not in comp_time:
            ert[node] = 0 if len(preds) == 0 else max([ddl[pred] for pred in preds])
            ddl[node] = ert[node]
        else:
            if len(preds) > 0:
                if comm_compen_en:
                    ert[node] = max([ddl[pred] + io_time[node] for pred in preds])
                else:
                    ert[node] = max([ddl[pred] for pred in preds])
            else:
                ert[node] = 0
            # compute the ddl
            # if task_type[node] == "RT": 
            #     ddl[node] = ert[node] + comp_time[node] + sched_step_comp 
            # else:
            ddl[node] = ert[node] + comp_time[node] *1e7 / (1 - temporal_rda_ratio)/ 1e7
    return ert, ddl

deduce_RDA = lambda size, temporal_rda_ratio, wsc_slack_ratio: math.ceil(size * (1-temporal_rda_ratio) / wsc_slack_ratio) - size
deduce_num_exec = lambda freq, f_gcd, thread_scaling_factor: math.ceil(freq / f_gcd) * thread_scaling_factor
deduce_no_stall_latency = lambda size, flops: flops / size / FLOPS_PER_CORE 
deduce_min_tot_rsc = lambda req_rsc, thread_scaling_factor, freq_division_factor: req_rsc * thread_scaling_factor * freq_division_factor
deduce_max_tot_rsc = lambda rda_size, size, thread_scaling_factor, freq_division_factor, var_factor: (rda_size + size) * thread_scaling_factor * freq_division_factor * var_factor
deduce_flops_typical = lambda flops, thread_scaling_factor, freq, f_gcd : flops * thread_scaling_factor * freq / f_gcd
deduce_flops_max = lambda flops_typical, var_factor: flops_typical * var_factor
deduce_equiv_core = lambda ops, hyper_p: ops / hyper_p / FLOPS_PER_CORE
deduce_util = lambda equiv_core, min_tot_rsc: equiv_core / min_tot_rsc

def deduce_task_attrib(taskattr: TaskIntAttr,
                        f_gcd: float,
                        hyper_p: int,
                        req_rsc_size: int,
                        temporal_rda_ratio: float,
                        wsc_slack_ratio: float,):

    rda_size = min(deduce_RDA(req_rsc_size, temporal_rda_ratio, wsc_slack_ratio), taskattr.core_max-req_rsc_size)
    taskattr.rda_size = rda_size
    taskattr.main_size = req_rsc_size
    taskattr.num_exec = deduce_num_exec(taskattr.freq, f_gcd, taskattr.thread_scaling_factor)
    taskattr.no_stall_latency = deduce_no_stall_latency(req_rsc_size, taskattr.flops)
    min_tot_rsc = deduce_min_tot_rsc(req_rsc_size, taskattr.thread_scaling_factor, taskattr.freq_division_factor)
    taskattr.min_tot_rsc = min_tot_rsc
    taskattr.max_tot_rsc = deduce_max_tot_rsc(rda_size, req_rsc_size, taskattr.thread_scaling_factor, taskattr.freq_division_factor, taskattr.var_factor)
    flops_typical = deduce_flops_typical(taskattr.flops, taskattr.thread_scaling_factor, taskattr.freq, f_gcd)
    taskattr.flops_typical = flops_typical
    taskattr.flops_max = deduce_flops_max(flops_typical, taskattr.var_factor)
    equiv_core = deduce_equiv_core(flops_typical, hyper_p)
    taskattr.equiv_core = equiv_core
    taskattr.util = deduce_util(equiv_core, min_tot_rsc)

def duduce_cfg(taskattr_dict, f_gcd, hyper_p, 
               logical_graph_nx, task_graph_srcs, task_graph_sinks, 
               slack_threshold, e2e_latency, temporal_rda_ratio, wsc_slack_ratio, verbose=False):
    rsc_map_w = rsc_slack_estim(taskattr_dict, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, e2e_latency, temporal_rda_ratio, slack_threshold) 
    if verbose:
        print(rsc_map_w)
    ert, ddl = estim_release_dll_time(logical_graph_nx, 
                            comp_time={node:slack_estm for node, (_, slack_estm, _) in rsc_map_w.items()},
                            io_time={node:1e-6 for node in taskattr_dict},
                            task_type={node:taskattr_dict[node].timing_flag for node in taskattr_dict},
                            temporal_rda_ratio=temporal_rda_ratio,)
    if verbose:
        print(ert, ddl) 
    for node, (req_rsc_size, slack_estm, constr) in rsc_map_w.items():
        taskattr:TaskIntAttr = taskattr_dict[node]
        taskattr.ERT = ert[node]
        taskattr.ddl = ddl[node] - ert[node]
        taskattr.exp_comp_t = slack_estm
        deduce_task_attrib(taskattr, f_gcd, hyper_p, req_rsc_size, temporal_rda_ratio, wsc_slack_ratio)
    if verbose:
        for node, taskattr in taskattr_dict.items():
            print(node, taskattr)
            print()

def test():
    import argparse
    import numpy as np 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling.csv", help="profiling filename") 
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    # parser.add_argument("--freq", type=float, default=10, help="frequency")
    parser.add_argument("--temporal_rda_ratio", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
    parser.add_argument("--slack_threshold", default=5e-4, type=float, help="slack threshold")
    parser.add_argument("--aux_scale_factor", default=1, type=int, help="aux scale factor")
    args = parser.parse_args() 

    from task.task_cfg import task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops
    from task.task_cfg import load_taskattrib, gen_taskint_from_cfg
    taskattr_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=args.verbose) 
    hyper_p = 1/f_gcd
    if args.aux_scale_factor > 1:
        for node, taskattr in taskattr_dict.items():
            # scale up the thread scaling factor
            if taskattr.timing_flag == "realtime":
                taskattr.thread_scaling_factor *= args.aux_scale_factor
    logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
    slack_threshold = args.slack_threshold
    duduce_cfg(taskattr_dict, f_gcd, hyper_p, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, slack_threshold, 
                         args.e2e_latency, args.temporal_rda_ratio, args.wsc_slack_ratio, 
                         True)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)

    for node, taskint in glb_n_task_dict.items():
        print(node, taskint)
        print()

if __name__ == "__main__":
    test()