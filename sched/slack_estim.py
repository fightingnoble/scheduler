from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from task.task_agent import TaskBase, ProcessBase, TaskIntAttr, ProcessInt
    from networkx import DiGraph
from typing import List, Any, Dict, Tuple, Union
from global_var import *
from task.graph_breakdown import decompose_dag_into_chains
from collections import OrderedDict
from sched.packing_solver.gurobi_MP_chain_assign import GurobiRscSlackEstim

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
        slack_estm = elim_nume_error(flops_dict[node] / ops_rem * slack_rem)
        # estimate the resource
        req_rsc_size, got_latency, constr = EstimCoreNums4Task(task_dict, flops_dict, node, slack_estm, 'ceil')
        rsc_map_w[node] = (req_rsc_size, got_latency, constr)
    
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

def GurobiDistributeSlack(task_dict:Dict[str, TaskBase], chains:List[Tuple[List[Any], float]], 
                          temporal_rel:Dict, temporal_abs:Dict):
    assert isinstance(temporal_rel, dict) and isinstance(temporal_abs, dict)
    chains_info = []
    for chain, slack_rem in chains:
        flops_dict = OrderedDict()
        build_score_dict_ref_flops(task_dict, chain, flops_dict)
        # TODO: select the e2e_latency, by the last node of the chain
        tail_task = task_dict[chain[-1]]
        if tail_task.timing_flag == "deadline":
            is_ddl_constr = True
        else:
            is_ddl_constr = False
        margin = {node:(temporal_abs[node], temporal_rel[node]) for node in chain}
        ops_rem = sum(flops_dict.values())
        chains_info.append((chain, flops_dict, margin, slack_rem, ops_rem, is_ddl_constr))
    
    sort_idx = np.array([(not is_ddl_constr, -ops_rem/slack_rem) for _, _, _, slack_rem, ops_rem, is_ddl_constr in chains_info], dtype=np.dtype('?, f8')).argsort()
    chains_info = [chains_info[i] for i in sort_idx]
    
    rsc_map_w:Dict[int, float, str] = {}
    # e2e_lat_info = []
    for chain, flops_dict, margin, slack_rem, ops_rem, is_ddl_constr in chains_info:
        for node in chain:
            if node in rsc_map_w:
                ops_rem -= flops_dict[node]
                slack_rem -= rsc_map_w[node][1] / (1 - margin[node][1]) + margin[node][0]
                flops_dict.pop(node)
                margin.pop(node)
        
        # while not state and len(flops_dict) > 0:
        #     state, ops_rem, slack_rem = alloc_func(rsc_map_w, task_dict, flops_dict, ops_rem, slack_rem, threshold)
        if (K:=len(flops_dict)) == 0:
            continue
        tot_cores = 150
        flops = list(flops_dict.values())
        node_list = list(flops_dict.keys())
        constr_core = [{"mode":task_dict[node].parallel_mode, "max":task_dict[node].core_max_compile, 
                        "min":task_dict[node].core_min_compile, "list":task_dict[node].core_list_compile
                        } for node in node_list]
        solver = GurobiRscSlackEstim(K, flops, slack_rem, [margin[node] for node in node_list], constr_core, tot_cores) # , verbose=True
        solver.create_variables()
        solver.define_constraints()
        sol = solver.solve()
        rsc_map_w.update(dict(zip(node_list, sol.values())))
        
        # collelct the slack of each components
        lt = [margin[node_list[0]][0]+sol[0][1]]
        for node in node_list[1:]:
            lt.append(rsc_map_w[node][1]/(1 - margin[node][1]) + margin[node][0])
        print(lt, chain, sum(lt))

        # e2e_lat_info.append(sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in chain]))
    return rsc_map_w


def DistributeSlack(task_dict:Dict[str, TaskBase], chains:List[Tuple[List[Any], float]], 
                    temporal_rel:float, temporal_abs:float, threshold):
    assert isinstance(temporal_rel, float) and isinstance(temporal_abs, float)
    chains_info = []
    for chain, slack_rem in chains:
        flops_dict = {}
        build_score_dict_ref_flops(task_dict, chain, flops_dict)
        # TODO: select the e2e_latency, by the last node of the chain
        tail_task = task_dict[chain[-1]]
        if tail_task.timing_flag == "deadline":
            is_ddl_constr = True
        else:
            is_ddl_constr = False
        slack_rem -= len(chain) * temporal_abs
        slack_rem = (1-temporal_rel)*1e3*slack_rem/1e3
        ops_rem = sum(flops_dict.values())
        chains_info.append((chain, flops_dict, slack_rem, ops_rem, is_ddl_constr))
    
    sort_idx = np.array([(not is_ddl_constr, -ops_rem/slack_rem) for _, _, slack_rem, ops_rem, is_ddl_constr in chains_info], dtype=np.dtype('?, f8')).argsort()
    chains_info = [chains_info[i] for i in sort_idx]
    
    rsc_map_w:Dict[int, float, str] = {}
    # e2e_lat_info = []
    for chain, flops_dict, slack_rem, ops_rem, is_ddl_constr in chains_info:
        for node in chain:
            if node in rsc_map_w:
                ops_rem -= flops_dict[node]
                slack_rem -= rsc_map_w[node][1]
                flops_dict.pop(node)
        state = False
        while not state and len(flops_dict) > 0:
            state, ops_rem, slack_rem = alloc_func(rsc_map_w, task_dict, flops_dict, ops_rem, slack_rem, threshold)
        # e2e_lat_info.append(sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in chain]))
    return rsc_map_w

def rsc_slack_estim(taskJobs:Union[Dict[str, Union[TaskBase,ProcessBase]], List[Union[TaskBase,ProcessBase]]], 
             task_graph:DiGraph, start_nodes, end_nodes, 
             temporal_rel:Union[float, Dict[str, float]], 
             threshold, temporal_abs:Dict={}, 
             algorithm="avg"):
    """
    input:
        system specification:
            1. exec_t_comp_ratioA
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
    # zip the chains with its e2e latency
    chains = [(chain[1:-1], task_graph.nodes[chain[-1]]['ert'] - task_graph.nodes[chain[0]]['ddl']) for chain in chains]
    assert algorithm in ["avg", 'gurobi']
    if algorithm == "avg":
        return DistributeSlack(task_dict, chains, temporal_rel, temporal_abs, threshold)
    else:
        return GurobiDistributeSlack(task_dict, chains, temporal_rel, temporal_abs)

def get_chains(task_graph:DiGraph, start_nodes, end_nodes, flops_dict):
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    
    # classify the chains by its sink types
    ddl_chains = []
    rt_chains = []
    for chain in chains:
        # task_graph_nx.nodes[node]['jitter']
        slack = task_graph.nodes[chain[-1]]['ddl'] - task_graph.nodes[chain[0]]['jitter']
        tot_ops = sum([flops_dict[node] for node in chain[1:-1]])
        if task_graph.nodes[chain[-1]]["chain_criticality"]:
            ddl_chains.append((chain[1:-1], tot_ops, slack))
        else:
            rt_chains.append((chain[1:-1], tot_ops, slack))
    rt_chains.sort(key=lambda x: (-x[1]/x[2], *x[0][-1].split("_")[-2:]))
    ddl_chains.sort(key=lambda x: (-x[1]/x[2], *x[0][-1].split("_")[-2:]))
    return rt_chains, ddl_chains

def estim_release_dll_time(task_graph_nx:DiGraph, 
                            comp_time: Dict[str, float]={},
                            io_time: Dict[str, float]={}, 
                            task_type: Dict[str, str]={}, 
                           temporal_rel=0, algorithm="avg", 
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

            if algorithm == 'avg':
                slack = comp_time[node] *1e7 / (1 - temporal_rel)/ 1e7
            else:
                slack = comp_time[node] *1e7 / (1 - temporal_rel[node])/ 1e7 
            # judge whether have start pred
            for pred in preds:
                if pred not in comp_time:
                    # judge whether the pred is on the critical path
                    if ddl[pred] + temporal_abs[node] > ert[node]:
                        slack += temporal_abs[node]
                        print("Compensation:", node, temporal_abs[node])
                    break
                # if ert[node] == 0:
                #     slack = temporal_abs[node] + comp_time[node]
                # else:
                #     for pred in preds:
                #         if pred not in comp_time:
                #             # judge whether the pred is on the critical path
                #             if ddl[pred] + temporal_abs[node] > ert[node]:
                #                 src_comp = ddl[pred] + temporal_abs[node] - ert[node]
                #                 inter_comp = slack - comp_time[node]
                #                 if src_comp > inter_comp:
                #                     slack = src_comp + comp_time[node]
                #                     print("Compensation:", node, src_comp-inter_comp)
                #             # suppose each node has only one sensor pred
                #             break
            ddl[node] = ert[node] + slack
    # eliminate the numerical error
    for node in ddl:
        ddl[node] = elim_nume_error(ddl[node])
        ert[node] = elim_nume_error(ert[node])
    return ert, ddl

def init_graph_time_attr(task_graph_nx:DiGraph, 
                            comp_time: Dict[str, float]={},
                           temporal_rel=0, 
                           temporal_abs={},
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

    for node in nx.topological_sort(task_graph_nx):  # 拓扑排序遍历节点
        preds = task_graph_nx.pred[node] 
        succs = task_graph_nx.succ[node] 
        # filter the sink and and the src nodes
        if len(succs) == 0 or len(preds) == 0:
            ert[node] = task_graph_nx.nodes[node]['ert']
            ddl[node] = task_graph_nx.nodes[node]['ddl']
            continue
        assert node in comp_time
        ert[node] = 0 if len(preds) == 0 else max([ddl[pred] for pred in preds]) 
        if isinstance(temporal_rel, float):
            slack = comp_time[node] *1e7 / (1 - temporal_rel)/ 1e7
        else:
            slack = comp_time[node] *1e7 / (1 - temporal_rel[node])/ 1e7 + temporal_abs[node] 
        ddl[node] = ert[node] + slack
    
    # For each src node if its ddl=exp_comp_t>0, then transfer the slack to its succ nodes,
    # the slack extent advance the ert of the succ nodes, 
    # but not earlier than the latest ert of the succ node's other preds,
    #   and set its ddl=exp_comp_t=0
    for node in task_graph_nx:
        if task_graph_nx.in_degree[node] == 0:
            if ddl[node] > 0: 
                for succ in task_graph_nx.succ[node]:
                    wifes = [ddl[pred] for pred in task_graph_nx.pred[succ] if pred != node]
                    wifes_ddl = max(wifes) if len(wifes) > 0 else 0
                    ert[succ] -= min(task_graph_nx.nodes[node]['exp_comp_t'], ert[succ]-wifes_ddl)
                ddl[node] = 0
                task_graph_nx.nodes[node]['jitter'] = elim_nume_error(task_graph_nx.nodes[node]['exp_comp_t'])
                task_graph_nx.nodes[node]['exp_comp_t'] = 0
                task_graph_nx.nodes[node]['ddl'] = 0

    # eliminate the numerical error
    for node in ddl:
        ddl[node] = elim_nume_error(ddl[node])
        ert[node] = elim_nume_error(ert[node])
    return ert, ddl


deduce_RDA = lambda size, exec_t_comp_ratioA, wsc_slack_ratio: math.ceil(size * (1-exec_t_comp_ratioA) / wsc_slack_ratio) - size if (1-exec_t_comp_ratioA) != wsc_slack_ratio else 0 
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
                        exec_t_comp_ratioA: float,
                        wsc_slack_ratio: float,):

    rda_size = min(deduce_RDA(req_rsc_size, exec_t_comp_ratioA, wsc_slack_ratio), taskattr.core_max_compile-req_rsc_size)
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
               slack_threshold, e2e_latency, exec_t_comp_ratioA, jitter_t_comp_ratio, wsc_slack_ratio,
                 algorithm='avg', timestep_size=10,
                 verbose=False, plot=False):
    temporal_abs = {
        node: elim_nume_error(1/taskattr_dict[node].freq*jitter_t_comp_ratio) 
        for node in taskattr_dict if taskattr_dict[node].trigger_mode!='N'
        } if algorithm=='gurobi' else {}
    temporal_rel = exec_t_comp_ratioA if algorithm == 'avg' else {
        node: exec_t_comp_ratioA if taskattr_dict[node].trigger_mode=='N' else 0 for node in taskattr_dict}
    
    rsc_map_w = rsc_slack_estim(taskattr_dict, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, e2e_latency, temporal_rel, slack_threshold, 
                         temporal_abs=temporal_abs, algorithm=algorithm) 
    if verbose:
        print(rsc_map_w)
    ert, ddl = estim_release_dll_time(logical_graph_nx, 
                            comp_time={node:slack_estm for node, (_, slack_estm, _) in rsc_map_w.items()},
                            io_time={node:1e-6 for node in taskattr_dict},
                            task_type={node:taskattr_dict[node].timing_flag for node in taskattr_dict},
                            temporal_rel=temporal_rel, algorithm=algorithm,
                            temporal_abs=temporal_abs)

    # legality check: all sink node enforce the deadline constraint
    # correct the ert and ddl of the sink nodes, and ddl of its predecessors
    for sink in task_graph_sinks:
        if sink_attr[sink] == "deadline":
            assert ddl[sink] <= e2e_latency
            ddl[sink] = e2e_latency
        else:
            assert ddl[sink] <= hyper_p
            ddl[sink] = hyper_p
        for pred in logical_graph_nx.pred[sink]:
            ddl[pred] = ddl[sink]

    if verbose:
        print(ert, ddl) 
    # update ert, ddl, exp_comp_t to graph as well as the taskattr_dict
    for node, (req_rsc_size, slack_estm, constr) in rsc_map_w.items():
        taskattr:TaskIntAttr = taskattr_dict[node]
        taskattr.ERT = ert[node]
        taskattr.ddl = ddl[node] - ert[node]
        taskattr.exp_comp_t = slack_estm
        deduce_task_attrib(taskattr, f_gcd, hyper_p, req_rsc_size, exec_t_comp_ratioA, wsc_slack_ratio)
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
            node_labels[node] = f"{node}\n{data['ddl']-data['ert']:.5f}[{data['ert']:.5f}:{data['ddl']:.5f}]\nexp_comp_t:{data['exp_comp_t']:.5f}"
        text = nx.draw_networkx_labels(logical_graph_nx, pos, labels=node_labels, font_size=10, ax=ax1)

        for _, t in text.items():
            t.set_rotation(60)
        fig.tight_layout()
        plt.savefig(f"plot/jobTask_graph_dbg.pdf", format="pdf")
        plt.close()

    if verbose:
        for node, taskattr in taskattr_dict.items():
            print(node, taskattr)
            print()

def deduce_cfg2(taskattr_dict, f_gcd, hyper_p, 
               logical_graph_nx, task_graph_srcs, task_graph_sinks, sink_attr, src_attr,
               slack_threshold, e2e_latency, exec_t_comp_ratioA, jitter_t_comp_ratio, wsc_slack_ratio,
                 algorithm='avg', timestep_size=10,
                 verbose=False, plot=False):

    # set the the ert and ddl of the sink nodes and the src nodes
    for sink in task_graph_sinks:
        e2e_constr = e2e_latency if sink_attr[sink] == "deadline" else hyper_p
        logical_graph_nx.nodes[sink]["ert"] = e2e_constr
        logical_graph_nx.nodes[sink]["ddl"] = e2e_constr
        logical_graph_nx.nodes[sink]["exp_comp_t"] = 0
    for src in task_graph_srcs: 
        jitter_t_comp = 1/src_attr[src]*jitter_t_comp_ratio if algorithm == 'gurobi' else 0
        logical_graph_nx.nodes[src]["ert"] = 0
        logical_graph_nx.nodes[src]["ddl"] = jitter_t_comp
        logical_graph_nx.nodes[src]["exp_comp_t"] = jitter_t_comp

    # set abs compensation and rel compensation
    # use abs comp for coalecing and use rel comp for ours
    exec_t_comp_abs = {node: 5* timestep_size * 1e-6 for node in taskattr_dict} if algorithm=='gurobi' else 0.
    exec_t_comp_rel = exec_t_comp_ratioA if algorithm == 'avg' else {
        node: exec_t_comp_ratioA for node in taskattr_dict}
    rsc_map_w = rsc_slack_estim(taskattr_dict, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, exec_t_comp_rel, slack_threshold, 
                         temporal_abs=exec_t_comp_abs, algorithm=algorithm) 
    if verbose:
        print(rsc_map_w)
    ert, ddl = init_graph_time_attr(
        logical_graph_nx, 
        comp_time={node:slack_estm for node, (_, slack_estm, _) in rsc_map_w.items()},
        temporal_rel=exec_t_comp_rel, temporal_abs=exec_t_comp_abs)

    # legality check: all sink node and their preds enforce the deadline constraint
    for sink in task_graph_sinks:
        e2e_constr = logical_graph_nx.nodes[sink]['ddl']
        for pred in logical_graph_nx.pred[sink]:
            assert ddl[pred] <= e2e_constr
            # if the sink is it unique succ, then set the ddl of the pred to the sink's ddl
            if len(logical_graph_nx.succ[pred]) == 1:
                ddl[pred] = e2e_constr
    if verbose:
        print(ert, ddl) 

    # update ert, ddl, exp_comp_t to graph as well as the taskattr_dict
    for node, (req_rsc_size, slack_estm, constr) in rsc_map_w.items():
        taskattr:TaskIntAttr = taskattr_dict[node]
        taskattr.ERT = ert[node]
        taskattr.ddl = ddl[node] - ert[node]
        taskattr.exp_comp_t = slack_estm
        deduce_task_attrib(taskattr, f_gcd, hyper_p, req_rsc_size, exec_t_comp_ratioA, wsc_slack_ratio)
        logical_graph_nx.nodes[node]["ert"] = ert[node]
        logical_graph_nx.nodes[node]["ddl"] = ddl[node]
        logical_graph_nx.nodes[node]["exp_comp_t"] = slack_estm
    
    # propagate the chain_criticality to all nodes from the sink nodes
    # init all node attr chain_criticality as True
    for node in logical_graph_nx:
        logical_graph_nx.nodes[node]["chain_criticality"] = True
        # TODO: double check
        # if node in taskattr_dict:
        #     taskattr_dict[node].chain_criticality = 'hard'
    for sink in task_graph_sinks:
        # sort if sink_attr[sink] != "deadline"
        if sink_attr[sink] != "deadline":
            logical_graph_nx.nodes[sink]["chain_criticality"] = False
            for node in nx.ancestors(logical_graph_nx, sink):
                logical_graph_nx.nodes[node]["chain_criticality"] = False
                if node in taskattr_dict:
                    taskattr_dict[node].chain_criticality = 'soft'

    if plot:
        plot_timeline_graph(logical_graph_nx)

    if verbose:
        for node, taskattr in taskattr_dict.items():
            print(node, taskattr)
            print()

def deduce_eq_wsc(task_graph, start_nodes, end_nodes, src_attr,jitter_t_comp_ratio):
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    wcs = []
    for chain in chains:
        src = chain[0]
        sink = chain[-1]
        jitter_t_comp = 1/src_attr[src]*jitter_t_comp_ratio 
        e2e_constr = task_graph.nodes[sink]['ddl']
        wcs.append(jitter_t_comp/e2e_constr)
    return 1-max(wcs)

def plot_timeline_graph(logical_graph_nx, path=f"plot/jobTask_graph_dbg.pdf"):
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
        node_labels[node] = f"{node}\n{data['ddl']-data['ert']:.5f}[{data['ert']:.5f}:{data['ddl']:.5f}]\nexp_comp_t:{data['exp_comp_t']:.5f}"
    text = nx.draw_networkx_labels(logical_graph_nx, pos, labels=node_labels, font_size=10, ax=ax1)

    for _, t in text.items():
        t.set_rotation(60)
    fig.tight_layout()
    plt.savefig(path, format="pdf")
    plt.close()  


def test():
    import argparse
    import numpy as np 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling.csv", help="profiling filename") 
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    # parser.add_argument("--freq", type=float, default=10, help="frequency")
    parser.add_argument("--exec_t_comp_ratioA", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
    parser.add_argument("--slack_threshold", default=5e-4, type=float, help="slack threshold")
    parser.add_argument("--aux_scale_factor", default=1, type=int, help="aux scale factor")
    args = parser.parse_args() 

    from task.task_cfg import task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops, sink_attr
    from task.task_cfg import load_taskattrib, gen_taskint_from_cfg
    taskattr_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=args.verbose) 
    hyper_p = 1/f_gcd
    if args.aux_scale_factor != 1:
        for node, taskattr in taskattr_dict.items():
            # scale up the thread scaling factor
            if taskattr.timing_flag == "realtime":
                taskattr.thread_scaling_factor *= args.aux_scale_factor
    logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
    slack_threshold = args.slack_threshold
    duduce_cfg(taskattr_dict, f_gcd, hyper_p, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, sink_attr, slack_threshold, 
                         args.e2e_latency, args.exec_t_comp_ratioA, args.wsc_slack_ratio, 
                         verbose=True)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)

    for node, taskint in glb_n_task_dict.items():
        print(node, taskint)
        print()

if __name__ == "__main__":
    test()