from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np
if TYPE_CHECKING:
    from task.task_agent import TaskBase, ProcessBase
    from networkx import DiGraph
from task.graph_breakdown import decompose_dag_into_chains, sort_chains_by_ddl_flops
from typing import List, Any, Dict, Tuple, Union
from global_var import *

def EstimCoreNums(task_dict:Dict[str, TaskBase], flops_dict, node, expected_slack, round_mode="round"):
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
        req_rsc_size, got_latency, constr = EstimCoreNums(task_dict, flops_dict, node, slack_estm, 'ceil')
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

deduce_RDA = lambda size, temporal_rda_ratio, wsc_slack_ratio: math.ceil(size * (1-temporal_rda_ratio) / wsc_slack_ratio) - size


def test():
    from task.task_cfg import load_taskint, task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops
    import argparse
    import numpy as np 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--profiling_filename", type=str, default="profiling.csv", help="profiling filename") 
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    parser.add_argument("--freq", type=float, default=10, help="frequency")
    parser.add_argument("--temporal_rda_ratio", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
    args = parser.parse_args() 

    glb_n_task_dict, f_gcd = load_taskint(args.profiling_filename, False, False, verbose=args.verbose) 
    logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
    threshold = 5e-4
    rsc_map_w = rsc_slack_estim(glb_n_task_dict, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, args.e2e_latency, args.temporal_rda_ratio, threshold) 
    print(rsc_map_w)

if __name__ == "__main__":
    test()