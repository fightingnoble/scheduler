from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from task.task_agent import TaskBase, ProcessBase, TaskIntAttr, ProcessInt, TaskInt
    from networkx import DiGraph
from typing import List, Any, Dict, Tuple, Union
from global_var import *
from task.graph_breakdown import decompose_dag_into_chains
from collections import OrderedDict
from sched.packing_solver.gurobi_MP_chain_assign import GurobiRscSlackEstim
from model.performance import cal_lat, slack_comp

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
        slack_rem -= sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in flops_dict]) + time1n_error_tol_abs
        # redistributed the slack to the remaining processes
        if slack_rem > 0:
            for node in flops_dict:
                slack_redis = flops_dict[node] / ops_rem * slack_rem
                rsc_map_w[node] = (rsc_map_w[node][0], elim_nume_error(rsc_map_w[node][1] + slack_redis), rsc_map_w[node][2])
        return state, 0, slack_rem
    else:
        state = False
        curr_slack_rem = sum([slack_estm for node, (_, slack_estm, _) in rsc_map_w.items() if node in flops_dict]) -slack_rem

        # exactly match the slack: 
        #  1. no process is reassigned due to the constraint
        #  2. the reassigned processes compensate each other
        
        # if no constraint, all task is assigned slack propotional to its flops, 
        # and all task requires same core numbers, e.g., core_max_ideal
        # w/ constraints: 
        # core_max_actual > core_max_ideal
        # A. if a task is resource upper bounded, 
            # it use less core and more slack than the ideal, other task use more core than the ideal
            # the upper bounded items is upper bounded by the core_max_ideal, 
            # it's no meaning to assign more slack to them and it's not legal to assign more core to them
            # so we move them. 
        # B. core_min_actual come from maxumum core of the lower bounded items in current iterations, 
            # but not must be the final result. Because there may be other tasks become upper bounded, and lower bounded items may be eliminated.
            # Consequently, we leave them unprocessed until there is no upper bounded item. 
        # C. The remaining slack (<= total_slack * percent of flops of remaining tasks), 
        # and is distributed remaining processes
        # 
        # the item that not upper bounded items are maximumly 
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
                # slack_rem -= rsc_map_w[node][1]/(1 - margin[node][1]) + margin[node][0]
                slack_rem -= cal_lat(rsc_map_w[node][1], margin[node][0], margin[node][1])
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
        solver = GurobiRscSlackEstim(K, flops, slack_rem, [margin[node] for node in node_list], constr_core, tot_cores, verbose=False) # , verbose=True
        solver.create_variables()
        solver.define_constraints()
        sol = solver.solve()
        rsc_map_w.update(dict(zip(node_list, sol.values())))
        
        # collelct the slack of each components
        lt = [margin[node_list[0]][0]+sol[0][1]]
        for node in node_list[1:]:
            # lt.append(rsc_map_w[node][1]/(1 - margin[node][1]) + margin[node][0])
            lt.append(cal_lat(rsc_map_w[node][1], margin[node][0], margin[node][1]))
        print("Alloted slack: {} {} sum: {:.3e}".format(lt, chain, sum(lt)))

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
        # slack_rem -= len(chain) * temporal_abs
        # slack_rem = (1-temporal_rel)*1e3*slack_rem/1e3
        slack_rem = slack_comp(slack_rem, len(chain) * temporal_abs, temporal_rel)
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
        # lt = [lat/(1-temporal_rel) for node, (_, lat, _) in rsc_map_w.items() if node in chain]
        # lt = [elim_nume_error(rsc_map_w[node][1]/(1-temporal_rel)+temporal_abs) for node in chain]
        lt = [elim_nume_error(cal_lat(rsc_map_w[node][1], temporal_abs, temporal_rel)) for node in chain]
        print(list(zip(chain, lt, np.cumsum(lt))))
    assert check_sol(rsc_map_w, chains, temporal_abs, temporal_rel, task_dict)
    return rsc_map_w

def check_sol(sol, chains, temporal_abs, temporal_rel, task_dict):
    # sol: (n_core, lat, constr)
    for chain, e2e_constr in chains:
        for node in chain:
            _task:TaskInt = task_dict[node]
            # Check core constraints
            if _task.parallel_mode == "list":
                if sol[node][0] not in _task.core_list_compile:
                    return False
            elif _task.parallel_mode == "lwb":
                if sol[node][0] < _task.core_min_compile:
                    return False
            elif _task.parallel_mode == "upb":
                if sol[node][0] > _task.core_max_compile:
                    return False
            elif _task.parallel_mode == "range":
                if sol[node][0] < _task.core_min_compile or sol[node][0] > _task.core_max_compile:
                    return False
            # check the flops constraint
            if elim_nume_error(_task.flops - sol[node][0] * sol[node][1] * FLOPS_PER_CORE)>0:
                return False
        # check the e2e_latency constraint
        # e2e_lat = sum([sol[node][1] / (1 - temporal_rel) + temporal_abs for node in chain])
        e2e_lat = sum([cal_lat(sol[node][1], temporal_abs, temporal_rel) for node in chain])
        if elim_nume_error(e2e_lat - e2e_constr)>0:
            return False

    return True

def rsc_slack_estim(taskJobs:Union[Dict[str, Union[TaskBase,ProcessBase]], List[Union[TaskBase,ProcessBase]]], 
             chains:List[Tuple[List[Any], float]],
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
            ddl_chains.append((chain[1:-1], tot_ops, elim_nume_error(slack)))
        else:
            rt_chains.append((chain[1:-1], tot_ops, elim_nume_error(slack)))
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
                # slack = comp_time[node] *1e7 / (1 - temporal_rel)/ 1e7
                slack = cal_lat(comp_time[node], 0, temporal_rel)
            else:
                # slack = comp_time[node] *1e7 / (1 - temporal_rel[node])/ 1e7 
                slack = cal_lat(comp_time[node], 0, temporal_rel[node])
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
            # slack = comp_time[node] *1e7 / (1 - temporal_rel)/ 1e7 + temporal_abs
            slack = cal_lat(comp_time[node], temporal_abs, temporal_rel)
        else:
            # slack = comp_time[node] *1e7 / (1 - temporal_rel[node])/ 1e7 + temporal_abs[node] 
            slack = cal_lat(comp_time[node], temporal_abs[node], temporal_rel[node])
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
                task_graph_nx.nodes[node]['exp_comp_t'] = 0
                task_graph_nx.nodes[node]['ddl'] = 0

    # eliminate the numerical error
    for node in ddl:
        ddl[node] = elim_nume_error(ddl[node])
        ert[node] = elim_nume_error(ert[node])
    return ert, ddl


# deduce_RDA = lambda size, exec_t_comp_ratioA, wsc_slack_ratio: math.ceil(size * (1-exec_t_comp_ratioA) / wsc_slack_ratio) - size if (1-exec_t_comp_ratioA) != wsc_slack_ratio else 0 
deduce_RDA = lambda size, exec_t_comp_ratioA, wsc_slack_ratio: math.ceil(elim_nume_error(size * (wsc_slack_ratio -1))) if wsc_slack_ratio > 1 else 0 
deduce_num_exec = lambda freq, f_gcd, thread_scaling_factor: math.ceil(freq / f_gcd) * thread_scaling_factor
deduce_no_stall_latency = lambda size, flops: flops / size / FLOPS_PER_CORE 
deduce_min_tot_rsc = lambda req_rsc, thread_scaling_factor, freq_division_factor: req_rsc * thread_scaling_factor * freq_division_factor
deduce_max_tot_rsc = lambda rda_size, size, thread_scaling_factor, freq_division_factor, var_factor: (rda_size + size) * thread_scaling_factor * freq_division_factor * var_factor
deduce_flops_ModelSum = lambda flops, thread_scaling_factor, freq, f_gcd : flops * thread_scaling_factor * freq / f_gcd
deduce_flops_ModelSumMax = lambda flops_ModelSum, var_factor: flops_ModelSum * var_factor
deduce_equiv_core = lambda ops, hyper_p: ops / hyper_p / FLOPS_PER_CORE
deduce_util = lambda equiv_core, min_tot_rsc: equiv_core / min_tot_rsc

def deduce_task_attrib(taskattr: TaskIntAttr,
                        f_gcd: float,
                        hyper_p: int,
                        req_rsc_size: int,
                        exec_t_comp_ratioA: float,
                        wsc_slack_ratio: float,):

    equiv_core = taskattr.equiv_core
    rda_size = min(deduce_RDA(req_rsc_size, exec_t_comp_ratioA, wsc_slack_ratio), taskattr.core_max_compile-req_rsc_size)
    taskattr.num_exec = deduce_num_exec(taskattr.freq, f_gcd, taskattr.thread_scaling_factor)
    taskattr.rda_size = rda_size
    taskattr.main_size = req_rsc_size
    taskattr.no_stall_latency = deduce_no_stall_latency(req_rsc_size, taskattr.flops)
    min_tot_rsc = deduce_min_tot_rsc(req_rsc_size, taskattr.thread_scaling_factor, taskattr.freq_division_factor)
    taskattr.min_tot_rsc = min_tot_rsc
    taskattr.max_tot_rsc = deduce_max_tot_rsc(rda_size, req_rsc_size, taskattr.thread_scaling_factor, taskattr.freq_division_factor, taskattr.var_factor)
    taskattr.util = deduce_util(equiv_core, min_tot_rsc)

def deduce_cfg2(taskattr_dict, f_gcd, hyper_p, 
               logical_graph_nx, task_graph_srcs, task_graph_sinks, sink_attr, src_attr,
               slack_threshold, e2e_latency, exec_t_comp_ratioA, jitter_t_comp_ratio, wsc_slack_ratio,
                 algorithm='avg', timestep_size=10, 
                 var_estimation={},
                 verbose=False, plot=False):

    # set abs compensation and rel compensation
    # use abs comp for coalecing and use rel comp for ours
    # exec_t_comp_abs: the estimation for slot grid displacement
    # exec_t_comp_rel: the estimation for rate of exec. slowdown
    # TODO: add transfer delay compensation
    if algorithm == 'avg':
        # using the same slowdown ratio for all nodes
        exec_t_comp_abs = 5* timestep_size * 1e-6 
        exec_t_comp_rel = exec_t_comp_ratioA
    else:
        # support different slowdown ratio for different nodes
        exec_t_comp_abs = {node: 5* timestep_size * 1e-6 for node in taskattr_dict} 
        exec_t_comp_rel = {node: exec_t_comp_ratioA for node in taskattr_dict} 
    

    chains = get_chains_info(logical_graph_nx, task_graph_srcs, task_graph_sinks)
    rsc_map_w = rsc_slack_estim(taskattr_dict, chains, 
                                exec_t_comp_rel, slack_threshold, 
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

    if algorithm == 'avg':
        wsc_by_name = deduce_eq_wsc(
            wsc_slack_ratio,
            logical_graph_nx, task_graph_srcs, task_graph_sinks, 
            src_attr, 
            exec_t_comp_rel, 
            exec_t_comp_abs, 
            var_estimation,
            verbose
        )
    else:
        wsc_by_name = None

    # update ert, ddl, exp_comp_t to graph as well as the taskattr_dict
    for node, (req_rsc_size, slack_estm, constr) in rsc_map_w.items():
        taskattr:TaskIntAttr = taskattr_dict[node]
        taskattr.ERT = ert[node]
        taskattr.ddl = ddl[node] - ert[node]
        taskattr.exp_comp_t = slack_estm
        # calculate the wsc_slack_ratio for each node
        spatial_ratio = wsc_by_name[node] if wsc_by_name is not None else 1 
        deduce_task_attrib(taskattr, f_gcd, hyper_p, req_rsc_size, exec_t_comp_ratioA, spatial_ratio)
        logical_graph_nx.nodes[node]["ert"] = ert[node]
        logical_graph_nx.nodes[node]["ddl"] = ddl[node]
        logical_graph_nx.nodes[node]["exp_comp_t"] = slack_estm
    

    if plot:
        plot_timeline_graph(logical_graph_nx)

    if verbose:
        for node, taskattr in taskattr_dict.items():
            print(node, taskattr)
            print()

def get_chains_info(task_graph, start_nodes, end_nodes):
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    # zip the chains with its e2e latency
    chains = [(chain[1:-1], task_graph.nodes[chain[-1]]['ert'] - task_graph.nodes[chain[0]]['ddl']) for chain in chains]
    return chains

def deduce_eq_wsc(
    wsc_slack_ratio:float,
    task_graph, start_nodes, end_nodes, src_attr, 
    temporal_rel:Union[float, Dict[str, float]], 
    temporal_abs:Dict={}, 
    var_estimation={},
    verbose=False,
    debug = False,
    ) -> List[Tuple[List[str], float]]:
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    # wsc is determined by timing jitter, exe_slowdown, displacemnt, and e2e latency
    # wsc_estm = (e2e_latency - jitter - temporal_abs of all nodes) * (1-slowdown ratio)
    # wsc_comp = (e2e_latency - jitter_t_comp - temporal_abs of all nodes) * (1-temporal_rel)
    # wsc_ratio = input_wcs_ratio * (wsc_estm/wsc_comp)
    wcs_by_chain = []
    wcs_by_name = {}
    for chain in chains:
        src = chain[0]
        sink = chain[-1]
        e2e_constr = task_graph.nodes[sink]['ddl']
        jitter_var = elim_nume_error(1/src_attr[src]*var_estimation.get('jitter', 0))
        exe_slowdown_var = var_estimation.get('exec', 0)
        jitter_t_comp = task_graph.nodes[src]["jitter"]
        if isinstance(temporal_rel, float):
            displacement = len(chain[1:-1]) * temporal_abs
        else:
            displacement = sum([temporal_abs[node] for node in chain[1:-1]])
        if isinstance(temporal_rel, float):
            # wsc_comp = (e2e_constr - jitter_t_comp - displacement) * (1-temporal_rel)
            wsc_comp = slack_comp(e2e_constr, jitter_t_comp+displacement, temporal_rel)
        else:
            # wsc_comp = (e2e_constr - jitter_t_comp - displacement) * (1-temporal_rel.values()[0])
            wsc_comp = slack_comp(e2e_constr, jitter_t_comp+displacement, temporal_rel.values()[0])
        # wsc_estm = (e2e_constr - jitter_var - displacement) * (1-exe_slowdown_var)
        wsc_estm = slack_comp(e2e_constr, jitter_var+displacement, exe_slowdown_var)
        wsc_ratio = wsc_comp/wsc_estm * wsc_slack_ratio
        if debug:
            wcs_by_chain.append((chain[1:-1], wsc_ratio))
            print(f"chain: {chain}, wsc_comp: {wsc_comp}, wsc_estm: {wsc_estm}")
        for node in chain[1:-1]:
            wcs_by_name[node] = max(wcs_by_name.get(node, 1), wsc_ratio) 
    return wcs_by_name

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

    from task.task_cfg import task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops, task_sink_attr
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
    deduce_cfg2(taskattr_dict, f_gcd, hyper_p, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, task_sink_attr, slack_threshold, 
                         args.e2e_latency, args.exec_t_comp_ratioA, args.wsc_slack_ratio, 
                         verbose=True)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)

    for node, taskint in glb_n_task_dict.items():
        print(node, taskint)
        print()

if __name__ == "__main__":
    test()