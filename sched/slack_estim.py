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
    got_latency = elim_nume_error(flops_dict[node] / req_rsc_size / FLOPS_PER_CORE)
    return req_rsc_size, got_latency, constr


def alloc_func(rsc_map_w:Dict[str, Tuple[int, float]], 
               task_dict:Dict[str, TaskBase], 
               flops_dict:Dict[str, float], 
               ops_rem, slack_rem, threshold):
    state = True
    for node in flops_dict:
        # estimate the slack
        slack_estm = flops_dict[node] / ops_rem * slack_rem
        # estimate the resource
        req_rsc_size, got_latency, constr = EstimCoreNums4Task(task_dict, flops_dict, node, slack_estm, 'ceil')
        rsc_map_w[node] = (req_rsc_size, got_latency if constr else slack_estm, constr)
    
    # check the constraint
    constr_dict = {node:constr for node, (_, _, constr) in rsc_map_w.items() if node in flops_dict and constr is not None}
    if len(constr_dict) == 0:
        slack_rem -= sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in flops_dict])
        return state, 0, slack_rem
    else:
        state = False
        curr_slack_rem = sum([slack_estm for node, (_, slack_estm, _) in rsc_map_w.items() if node in flops_dict]) -slack_rem

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
                    slack_rem -= rsc_map_w[node][1]
                    flops_dict.pop(node)
        else:
            # remove the process which has been reached the core_min
            for node,_constr in constr_dict.items():
                if _constr == "lwb":
                    ops_rem -= flops_dict[node]
                    slack_rem -= rsc_map_w[node][1]
                    flops_dict.pop(node)
    return state, ops_rem, slack_rem


def build_score_dict_ref_flops(task_dict:Dict[str, TaskBase], nodes:Any, score_dict):
    for node_n in nodes: 
        assert node_n in task_dict
        _task = task_dict[node_n]
        score_dict[node_n] = _task.flops


def DistributeSlack(task_dict:Dict[str, TaskBase], e2e_latency, chains:List[List[Any]], temporal_rel, temporal_abs_en, threshold):
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
        # temporal_abs = max([1/task_dict[node].freq*temporal_rel for node in chain if task_dict[node].trigger_mode!='N']) if temporal_abs_en else 0
        if temporal_abs_en and task_dict[chain[0]].trigger_mode!='N':
            temporal_abs = elim_nume_error(1/task_dict[chain[0]].freq*temporal_rel)
            print(chain[0], temporal_abs)
        else:
            temporal_abs = 0
        # temporal_abs = 0
        slcak_rem = (1-temporal_rel)*1e3*(slcak_rem-temporal_abs)/1e3
        ops_rem = sum(flops_dict.values())
        chains_info.append((chain, flops_dict, slcak_rem, ops_rem, is_ddl_constr))
    
    sort_idx = np.array([(not is_ddl_constr, -ops_rem/slcak_rem) for _, _, slcak_rem, ops_rem, is_ddl_constr in chains_info], dtype=np.dtype('?, f8')).argsort()
    chains_info = [chains_info[i] for i in sort_idx]
    
    rsc_map_w:Dict[int, float, str] = {}
    # e2e_lat_info = []
    for chain, flops_dict, slcak_rem, ops_rem, is_ddl_constr in chains_info:
        for node in chain:
            if node in rsc_map_w:
                ops_rem -= flops_dict[node]
                slcak_rem -= rsc_map_w[node][1]
                flops_dict.pop(node)
        state = False
        while not state and len(flops_dict) > 0:
            state, ops_rem, slcak_rem = alloc_func(rsc_map_w, task_dict, flops_dict, ops_rem, slcak_rem, threshold)
        # e2e_lat_info.append(sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in chain]))
    return rsc_map_w

def rsc_slack_estim(taskJobs:Union[Dict[str, Union[TaskBase,ProcessBase]], List[Union[TaskBase,ProcessBase]]], 
             task_graph:DiGraph, start_nodes, end_nodes, e2e_latency, 
             temporal_rda_ratio, 
             threshold, abs_reserve_en=False):
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
    return DistributeSlack(task_dict, e2e_latency, chains, temporal_rda_ratio, abs_reserve_en, threshold)

def estim_release_dll_time(task_graph_nx:DiGraph, 
                            comp_time: Dict[str, float]={},
                            io_time: Dict[str, float]={}, 
                            task_type: Dict[str, str]={}, 
                           temporal_rel=0, temporal_abs_en=False, 
                           temporal_abs={},
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
            # for the start node and the end node respectively
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
            slack = comp_time[node] *1e7 / (1 - temporal_rel)/ 1e7
            # judge whether have start pred
            if temporal_abs_en:
                for pred in preds:
                    if pred not in comp_time:
                        # judge whether the pred is on the critical path
                        if ddl[pred] + temporal_abs[node] > ert[node]:
                            slack += temporal_abs[node]
                            print("Compensation:", node, temporal_abs[node])
                        break
            ddl[node] = ert[node] + slack
    # eliminate the numerical error
    for node in ddl:
        ddl[node] = elim_nume_error(ddl[node])
        ert[node] = elim_nume_error(ert[node])
    return ert, ddl

deduce_RDA = lambda size, temporal_rda_ratio, wsc_slack_ratio: math.ceil(size * (1-temporal_rda_ratio) / wsc_slack_ratio) - size if (1-temporal_rda_ratio) != wsc_slack_ratio else 0 
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

    rda_size = min(deduce_RDA(req_rsc_size, temporal_rda_ratio, wsc_slack_ratio), taskattr.core_max_compile-req_rsc_size)
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
               logical_graph_nx, task_graph_srcs, task_graph_sinks, sink_attr,
               slack_threshold, e2e_latency, temporal_rda_ratio, wsc_slack_ratio,
                 abs_reserve_en=False, verbose=False, plot=False):
    rsc_map_w = rsc_slack_estim(taskattr_dict, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, e2e_latency, temporal_rda_ratio, slack_threshold, abs_reserve_en) 
    if verbose:
        print(rsc_map_w)
    ert, ddl = estim_release_dll_time(logical_graph_nx, 
                            comp_time={node:slack_estm for node, (_, slack_estm, _) in rsc_map_w.items()},
                            io_time={node:1e-6 for node in taskattr_dict},
                            task_type={node:taskattr_dict[node].timing_flag for node in taskattr_dict},
                            temporal_rel=temporal_rda_ratio, temporal_abs_en=abs_reserve_en,
                            temporal_abs={
                                node: elim_nume_error(1/taskattr_dict[node].freq*temporal_rda_ratio) 
                                    for node in taskattr_dict if taskattr_dict[node].trigger_mode!='N'
                                    } if abs_reserve_en else {},
                            )
    # legality check: all sink node enforce the deadline constraint
    for node in task_graph_sinks:
        if sink_attr[node]=="deadline":
            assert ddl[node] <= e2e_latency
        else:
            assert ddl[node] <= hyper_p
    if verbose:
        print(ert, ddl) 
    # update ert, ddl, exp_comp_t to graph as well as the taskattr_dict
    for node, (req_rsc_size, slack_estm, constr) in rsc_map_w.items():
        taskattr:TaskIntAttr = taskattr_dict[node]
        taskattr.ERT = ert[node]
        taskattr.ddl = ddl[node] - ert[node]
        taskattr.exp_comp_t = slack_estm
        deduce_task_attrib(taskattr, f_gcd, hyper_p, req_rsc_size, temporal_rda_ratio, wsc_slack_ratio)
    for node in logical_graph_nx:
        logical_graph_nx.nodes[node]["ert"] = ert[node]
        logical_graph_nx.nodes[node]["ddl"] = ddl[node]
        logical_graph_nx.nodes[node]["exp_comp_t"] = rsc_map_w[node][1] if node in rsc_map_w else 0

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(111)

        node_color_map = {"op": "red", "sink": "blue", "src": "green"}
        edge_color_map = {"data": "red", "control": "blue"}
        
        node_colors = [node_color_map[d] for n, d in logical_graph_nx.nodes(data="type")]
        edge_colors = [edge_color_map[d] for u,v,d in logical_graph_nx.edges(data="type")]
        for layer, nodes in enumerate(nx.topological_generations(logical_graph_nx)):
            for node in nodes:
                logical_graph_nx.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(logical_graph_nx, subset_key="layer")
        # text with 45 degree rotation
        nx.draw(logical_graph_nx, pos, with_labels=False, node_size=100, node_color=node_colors, edge_color=edge_colors, font_size=10, ax=ax1)
        # Draw node labels "name" "ddl," "ert," and "exp_comp_t" attributes
        node_labels = {}
        for node, data in logical_graph_nx.nodes(data=True):
            node_labels[node] = f"{node}\n[{data['ddl']:.5f}:{data['ert']:.5f}]\nexp_comp_t:{data['exp_comp_t']:.5f}"
        text = nx.draw_networkx_labels(logical_graph_nx, pos, labels=node_labels, font_size=10, ax=ax1)

        for _, t in text.items():
            t.set_rotation(60)
        fig.tight_layout()
        plt.savefig("plot/{cfg_n}/jobTask_graph_dbg.pdf", format="pdf")

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

    from task.task_cfg import task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops, sink_attr
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
                         task_graph_sinks, sink_attr, slack_threshold, 
                         args.e2e_latency, args.temporal_rda_ratio, args.wsc_slack_ratio, 
                         verbose=True)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)

    for node, taskint in glb_n_task_dict.items():
        print(node, taskint)
        print()

if __name__ == "__main__":
    test()