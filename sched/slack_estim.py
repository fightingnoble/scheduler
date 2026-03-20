from __future__ import annotations
from typing import TYPE_CHECKING
import math
import numpy as np
import networkx as nx
import re 
import matplotlib.pyplot as plt
from approach_Eq import Variation

if TYPE_CHECKING:
    from task.task_agent import TaskBase, ProcessBase, TaskIntAttr, ProcessInt, TaskInt
    from networkx import DiGraph
from typing import List, Any, Dict, Tuple, Union
from global_var import *
from task.graph_breakdown import decompose_dag_into_chains
from collections import OrderedDict
from sched.packing_solver.chain_slack_assign import GurobiRscSlackEstim, HeuriRscSlackEstim
from approach_Eq import AccVarDist, SenVarDist
from sched.ref_alloc_search import TaskConstraints

## Old defines

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


def build_score_dict_ref_flops(task_dict:Dict[str, TaskBase], nodes:Any, score_dict):
    for node_n in nodes: 
        assert node_n in task_dict
        _task = task_dict[node_n]
        score_dict[node_n] = _task.flops

def DistributeSlack(task_graph: DiGraph, task_dict:Dict[str, TaskBase], chains_info:List[Dict[str, Any]],
                    quantile: float, threshold: float = None, algorithm: str = "gurobi", total_cores: int = 300) -> Dict[str, Tuple[int, float, str]]:
    """
    统一的松弛时间分配函数，支持多种求解算法。
    
    参数:
        task_graph: 任务图
        task_dict: 任务字典
        chains_info: 链信息列表，每个元素是包含 "chain_nodes", "total_ops", "slack", "is_ddl", "name_idx" 的字典。
        quantile: 分位数
        threshold: 阈值（仅启发式算法需要）
        algorithm: 求解算法，可选 "heuristic" 或 "gurobi"
    按链的优先级顺序，为任务分配核心数，以满足端到端延迟分位数约束。

    核心逻辑:
    1. 按关键性(硬/软约束)和紧急性(计算量/延迟预算)对所有链进行排序。
    2. 遍历排序后的链，为每条链迭代求解资源分配：
        a. 计算当前链上已分配节点的延迟总和。
        b. 计算未分配节点的剩余延迟预算 (remaining_deadline)。
        c. 调用求解器，在 remaining_deadline 约束下，为未分配节点找到最小化核心数的方案。
        d. 更新全局资源分配表。
    返回:
        资源分配映射，格式为 {节点: (核心数, 延迟, 约束标签)}
    """
    rsc_map_w:Dict[str, Tuple[int, float, Union[str, None]]] = {}
    
    for info in chains_info:
        chain = info["chain_nodes"]
        e2e_deadline = info["slack"]
        # 1. 识别未分配的 op 节点
        unassigned = [node for node in chain if node not in rsc_map_w and task_graph.nodes[node]['type'] != 'sink']
        if not unassigned:
            continue

        # 2. 计算剩余延迟预算
        # (core, latency, constr_tag) in rsc_map_w
        remaining_deadline = e2e_deadline - sum(rsc_map_w[node][1] for node in chain if node in rsc_map_w) 
        if remaining_deadline <= 0:
            raise ValueError(f"No remaining deadline for chain {chain}, terminating.")

        # 3. 准备求解器输入
        node_var_dists = [task_graph.nodes[node]['var_dist'] for node in unassigned]
        
        # 根据节点类型创建约束：只有非AccVarDist类型的节点才需要TaskConstraints
        constr_core = []
        for i, node in enumerate(unassigned):
            var_dist = node_var_dists[i]
            if isinstance(var_dist, SenVarDist):
                # SenVarDist类型的节点（通常是传感器任务）不需要复杂的核心约束
                # 使用默认约束：核心数固定为1
                constraints = TaskConstraints(
                    parallel_mode="range",
                    core_min=1,
                    core_max=1,
                    core_list=[]
                )
            else:
                # 其他类型的节点需要从任务配置中获取约束
                task = task_dict[node]
                constraints = TaskConstraints(
                    parallel_mode=task.parallel_mode,
                    core_min=task.core_min_compile,
                    core_max=task.core_max_compile,
                    core_list=task.core_list_compile 
                )
            constr_core.append(constraints)
        
        tot_cores = total_cores  # 使用函数参数，可配置
        
        # 4. 根据算法选择求解器
        if algorithm == "gurobi":
            solver = GurobiRscSlackEstim(node_var_dists, remaining_deadline, quantile, constr_core, tot_cores, verbose=False)
            solver.create_variables()
            solver.define_constraints()
            solution = solver.solve()
        elif algorithm == "avg":
            if threshold is None:
                raise ValueError("threshold parameter is required for heuristic algorithm")
            solution = HeuriRscSlackEstim(node_var_dists, remaining_deadline, quantile, constr_core, tot_cores, threshold)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # 5. 更新资源分配
        # 将整数索引映射回节点名称
        solution_with_names = {unassigned[i]: solution[i] for i in range(len(unassigned))}
        rsc_map_w.update(solution_with_names)
        # collect the slack of each components
        lt = [rsc_map_w[node][1] for node in chain[0:-1]]
        print("Alloted slack: {} {} sum: {:.3e}".format(lt, chain, sum(lt)))
    return rsc_map_w

def rsc_slack_estim(task_graph: DiGraph,
             taskJobs:Union[Dict[str, Union[TaskBase,ProcessBase]], List[Union[TaskBase,ProcessBase]]],
             chains_info:List[Dict[str, Any]],
             threshold,
             algorithm="avg",
             quantile:float=0.95,
             total_cores:int=300):
    """
    资源和松弛时间估计的顶层函数。
    input:
        system specification:
            1. quantile
            2. threshold
            3. e2e_latency
        task_graph: the task graph
        required properties of each task:
            1. freq
            2. freq_division_factor
            3. timing_flag
            4. name
            5. var_dist
            6. parallel_cfg_compile: parallel_mode, 1 of (core_min_compile, core_max_compile, core_list_compile)
    """
    if isinstance(taskJobs, list):
        if taskJobs[0].__class__.__name__ == "ProcessInt":
            task_dict = {i.task.name:i for i in taskJobs.task}
        elif taskJobs[0].__class__.__name__ == "TaskInt":
            task_dict = {i.name:i for i in taskJobs}
    elif isinstance(taskJobs, dict):
        if next(iter(taskJobs.values())).__class__.__name__ == "ProcessInt":
            task_dict = {k:v.task for k,v in taskJobs.items()}
        else:
            task_dict = taskJobs
    else:
        raise TypeError("taskJobs should be a list or dict")
    
    assert algorithm in ["avg", 'gurobi']
    return DistributeSlack(task_graph, task_dict, chains_info, quantile, threshold, algorithm, total_cores)

def get_chains(task_graph:DiGraph, start_nodes, end_nodes, 
                task_dict:Dict[str, 'TaskBase'], quantile:float=0.95,
                node_var_dists: List[Variation]=None, 
                remove_src_sink: bool=False
                ):
    """
    将任务图分解为链，对其进行分类，并按关键性对其进行排序。
    这是一个重构版本，与DistributeSlack中的逻辑保持一致。
    """
    raw_chains = []
    for start_node in start_nodes:
        raw_chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)

    chains_info = []
    for chain in raw_chains:
        # get nodes, slack，get flops, get timing_flag
        op_nodes = [node for node in chain if task_graph.nodes[node]['type'] == 'op']
        if not op_nodes:
            continue
        
        e2e_lat = task_graph.nodes[chain[-1]]['ddl']
        if node_var_dists is None:
            slack = e2e_lat
            total_ops = sum(task_dict[node].flops for node in op_nodes)
        else:
            total_ops = 0
            total_time_fixed = 0
            for i, dist in enumerate(node_var_dists):
                if isinstance(dist, SenVarDist):
                    total_time_fixed += dist.quantile(quantile)
                else: 
                    total_ops += dist.load_dist.quantile(quantile)
                    total_time_fixed += dist.exec_dist.quantile(quantile)

            slack = elim_nume_error(e2e_lat - total_time_fixed)

        is_ddl_chain = (task_dict[chain[-2]].timing_flag == "deadline")
        match = re.search(r"((_\d+)+)$", chain[-2])
        if match:
            # Steering_speed_0_0
            # match.group(1) 提取整个匹配到的子串，即 "_0_0"
            numbers_str = match.group(1)
            # 使用split进行分割，得到一个 ['', '0', '0'] 的列表
            split_list = numbers_str.split('_')
            # 使用列表切片去除第一个空字符串，并转换为整数列表
            name_idx = [int(n) for n in split_list[1:]]
        else:
            name_idx = [0]

        chains_info.append({
            "chain_nodes": chain[1:-1] if remove_src_sink else chain, # 像旧函数一样排除src/sink
            "total_ops": total_ops,
            "slack": slack,
            "is_ddl": is_ddl_chain,
            "name_idx": name_idx
        })

    # 排序: 1. 硬约束优先; 2. (计算量/延迟预算)比值高的优先 (即更紧急的)
    chains_info.sort(key=lambda info: (
        not info["is_ddl"],
        -info["total_ops"] / info["slack"] if info["slack"] > 0 else -np.inf,
        *info["name_idx"]
    ))
    return chains_info



def init_topo_time_attr(
        task_graph_nx:nx.DiGraph, 
        rsc_map_w: Dict[str, Tuple[int, float, Union[str, None]]] = None,
    ):
    """
    set the ERT and ddl property of each task: 
    traverse the job graph, 
    for each task, 
    ERT = max(ddl of all pred tasks) + io_time; 
    ddl = ERT + exp_comp_t; 

    1. 汇点的特性：截止时间和前面的slack分配结果无关，执行时间为0（需要可配置，还没添加），ert需要特殊处理，ert等于截止时间减去执行时间（标记特殊处理，以及通用化修正的TODO。）
    2. 其他节点：函数按照拓扑顺序遍历图中的所有节点。ERT等于前驱节点中最晚的ddl，ddl等于自身ert加上分配结果中的延迟，对于没有前驱的节点（源节点），ert 被设置为 0。

    3. 汇点 ddl 合法性检查与传播：
    对于所有汇点，函数会进行合法性检查，确保其前驱节点的 ddl 小于或等于汇点的 ddl 约束。
    如果汇点是其某个前驱节点的唯一后继，则该前驱节点的 ddl 会被更新为汇点的 ddl。
    4. 数值误差消除：
    所有的 ddl 和 ert 值都会通过 elim_nume_error 函数进行处理，以消除潜在的浮点数计算误差。
    """
    ert: Dict[str, float] = {}
    ddl: Dict[str, float] = {}

    for node in nx.topological_sort(task_graph_nx):  # 拓扑排序遍历节点
        preds = task_graph_nx.pred[node] 
        # the ddl of the sink node, and the ops that only go to the sink node are not 
        # affected by allocation result
        # 1. the ddls of the sink nodes are inputs 
        if task_graph_nx.nodes[node]['type'] == "sink":
            ddl[node] = task_graph_nx.nodes[node]['ddl']
            ert[node] = task_graph_nx.nodes[node]['ert']
        else:
            ert[node] = 0 if len(preds) == 0 else max([ddl[pred] for pred in preds]) 
            slack = rsc_map_w[node][1]
            ddl[node] = ert[node] + slack

    # eliminate the numerical error
    for node in ddl:
        ddl[node] = elim_nume_error(ddl[node])
        ert[node] = elim_nume_error(ert[node])

    # update ert, ddl, exp_comp_t to graph based on solver outputs
    for node, (_, slack_estm, _) in rsc_map_w.items():
        task_graph_nx.nodes[node]["ert"] = ert[node]
        task_graph_nx.nodes[node]["ddl"] = ddl[node]
        task_graph_nx.nodes[node]["exp_comp_t"] = slack_estm
    
    # legality check: all sink node and their preds enforce the deadline constraint
    for sink_, type_ in task_graph_nx.nodes(data="type"):
        if type_ == "sink":
            for pred in task_graph_nx.pred[sink_]:
                assert ddl[pred] <= task_graph_nx.nodes[sink_]['ert']

                 # 2. the ddls of the node that only go to the sink node should align with the sink's ert
                succ_ = list(task_graph_nx.succ[pred])
                if len(succ_) == 1: # off course, the sink node is its unique succ
                    ddl[pred] = task_graph_nx.nodes[succ_[0]]['ert']
                    continue

    return ert, ddl


deduce_RDA = lambda size, wsc_slack_ratio: math.ceil(elim_nume_error(size * (wsc_slack_ratio -1))) if wsc_slack_ratio > 1 else 0 
deduce_num_exec = lambda freq, f_gcd, thread_scaling_factor: math.ceil(freq / f_gcd) * thread_scaling_factor
deduce_no_stall_latency = lambda size, flops: flops / size / FLOPS_PER_CORE 
deduce_min_tot_rsc = lambda req_rsc, thread_scaling_factor, freq_division_factor: req_rsc * thread_scaling_factor * freq_division_factor
deduce_max_tot_rsc = lambda rda_size, size, thread_scaling_factor, freq_division_factor, max_var_factor: (rda_size + size) * thread_scaling_factor * freq_division_factor * max_var_factor
deduce_flops_ModelSum = lambda flops, thread_scaling_factor, freq, f_gcd : flops * thread_scaling_factor * freq / f_gcd
deduce_flops_ModelSumMax = lambda flops_ModelSum, max_var_factor: flops_ModelSum * max_var_factor
deduce_equiv_core = lambda ops, hyper_p: ops / hyper_p / FLOPS_PER_CORE
deduce_util = lambda equiv_core, min_tot_rsc: equiv_core / min_tot_rsc

def deduce_task_attrib(taskattr: TaskIntAttr,
                        f_gcd: float,
                        hyper_p: int,
                        req_rsc_size: int,
                        wsc_slack_ratio: float,):

    equiv_core = taskattr.equiv_core
    rda_size = min(deduce_RDA(req_rsc_size, wsc_slack_ratio), taskattr.core_max_compile-req_rsc_size)
    taskattr.num_exec = deduce_num_exec(taskattr.freq, f_gcd, taskattr.thread_scaling_factor)
    taskattr.rda_size = rda_size
    taskattr.main_size = req_rsc_size
    taskattr.no_stall_latency = deduce_no_stall_latency(req_rsc_size, taskattr.flops)
    min_tot_rsc = deduce_min_tot_rsc(req_rsc_size, taskattr.thread_scaling_factor, taskattr.freq_division_factor)
    taskattr.min_tot_rsc = min_tot_rsc
    taskattr.max_tot_rsc = deduce_max_tot_rsc(rda_size, req_rsc_size, taskattr.thread_scaling_factor, taskattr.freq_division_factor, max(taskattr.var_factor))
    taskattr.util = deduce_util(equiv_core, min_tot_rsc)

def _fixcore_slack_estim(
    logical_graph_nx,
    fix_core_map: Dict[str, int],
    quantile: float,
    scale_factor: float = None,
) -> Dict[str, Tuple[int, float, Union[str, None]]]:
    """
    Repack mode: recompute task latency using ratioB quantile while keeping
    Phase 1 core counts fixed. Returns rsc_map_w compatible with init_topo_time_attr.

    Args:
        scale_factor: If provided, scale latency by this factor instead of using quantile.
                      This creates the "tail margin" effect when scale_factor < 1.
    """
    rsc_map_w: Dict[str, Tuple[int, float, Union[str, None]]] = {}
    for node, data in logical_graph_nx.nodes(data=True):
        if data.get('type') == 'sink':
            continue
        var_dist = data.get('var_dist')
        if var_dist is None:
            continue
        if isinstance(var_dist, SenVarDist):
            latency = float(var_dist.quantile(quantile))
            cores = 1
        else:
            cores = fix_core_map.get(node, 1)
            load_q = float(var_dist.load_dist.quantile(quantile))
            io_q   = float(var_dist.exec_dist.quantile(quantile))
            latency = load_q / max(cores, 1) / FLOPS_PER_CORE + io_q
            # Apply scaling factor if provided
            if scale_factor is not None:
                latency = latency * scale_factor
        rsc_map_w[node] = (cores, elim_nume_error(latency), None)
    return rsc_map_w

def deduce_cfg2(taskattr_dict,
               logical_graph_nx, task_graph_srcs, task_graph_sinks,
               quantile, slack_threshold,
               verbose=False, plot=False,
               fix_core_map: Dict[str, int] = None,
               scale_factor: float = None):
    """
    配置推导函数

    Args:
        quantile: 用于分布分位数计算的分位数
        fix_core_map: Phase 1 的核心分配，用于 repack
        scale_factor: repack 时的 latency 缩放因子，用于创建尾部余量
    """

    if fix_core_map is not None:
        # Repack mode: fixed cores, recompute latency at quantile (ratioB)
        rsc_map_w = _fixcore_slack_estim(logical_graph_nx, fix_core_map, quantile, scale_factor)
    else:
        # Normal Phase 1: solve for optimal core count
        chains_info = get_chains(logical_graph_nx, task_graph_srcs, task_graph_sinks, taskattr_dict,
                                   quantile=quantile, remove_src_sink=False)
        rsc_map_w = rsc_slack_estim(logical_graph_nx, taskattr_dict, chains_info,
                                    slack_threshold,
                                    algorithm="avg",
                                    quantile=quantile)
    if verbose:
        print(rsc_map_w)
    
    ert, ddl = init_topo_time_attr(
        logical_graph_nx, 
        rsc_map_w
    )

    if verbose:
        print(ert, ddl) 

    if plot:
        plot_timeline_graph(logical_graph_nx)
    
    return ert, ddl, rsc_map_w

def update_taskattr_dict(ert, ddl, rsc_map_w, taskattr_dict, f_gcd, hyper_p, 
               logical_graph_nx, verbose=False):
    for node, (req_rsc_size, slack_estm, constr) in rsc_map_w.items():
        if logical_graph_nx.nodes[node]['type'] != "op":
            continue
        taskattr:TaskIntAttr = taskattr_dict[node]
        taskattr.ERT = ert[node]
        taskattr.ddl = ddl[node] - ert[node]
        taskattr.exp_comp_t = slack_estm
        # calculate the wsc_slack_ratio for each node
        spatial_ratio = 1
        deduce_task_attrib(taskattr, f_gcd, hyper_p, req_rsc_size, spatial_ratio)

    if verbose:
        for node, taskattr in taskattr_dict.items():
            print(node, taskattr)
            print()

def get_chains_info(task_graph, start_nodes, end_nodes):
    chains = []
    for start_node in start_nodes:
        chains += decompose_dag_into_chains(task_graph, start_node, end_nodes)
    # zip the chains with its e2e latency
    chains = [(chain, task_graph.nodes[chain[-1]]['ddl']) for chain in chains]
    return chains

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

    from task.task_cfg import task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops
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
    ert, ddl, rsc_map_w = deduce_cfg2(taskattr_dict, 
               logical_graph_nx, task_graph_srcs, task_graph_sinks, 0.99,
               slack_threshold, verbose=True)
    
    update_taskattr_dict(ert, ddl, rsc_map_w, taskattr_dict, f_gcd, hyper_p, 
               logical_graph_nx, verbose=False)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)

    for node, taskint in glb_n_task_dict.items():
        print(node, taskint)
        print()

if __name__ == "__main__":
    test()