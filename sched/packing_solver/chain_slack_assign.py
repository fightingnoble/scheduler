from typing import List, Dict, Tuple, Union, Any
import gurobipy as gp
from gurobipy import GRB, quicksum
from utils import time_cnt
from global_var import FLOPS_PER_CORE, elim_nume_error, flop1n_error_tol_abs, time1n_error_tol_abs
# Import necessary distribution classes
from approach.approach_Eq import SenVarDist, AccVarDist, Variation
from sched.ref_alloc_search import TaskConstraints

from approach.approach_Eq import find_legal
from collections import OrderedDict
import math


def check_solution_validity(
    sol: Dict[int, Tuple[int, float, Union[str, None]]], 
    node_var_dists: List[Variation], 
    e2e_lat: float, 
    quantile: float, 
    constr_core: List[TaskConstraints], 
    tot_cores: int
) -> bool:
    """
    独立的解决方案验证函数，用于校验求解结果是否满足约束条件。
    
    参数:
        sol: 解决方案字典，格式为 {i: (核心数, 延迟, 约束标签)}
        node_var_dists: 节点变量分布列表
        e2e_lat: 端到端延迟约束
        quantile: 分位数
        constr_core: 核心约束列表
        tot_cores: 总核心数上限
    
    返回:
        bool: 解决方案是否有效
    """
    K = len(node_var_dists)
    
    # 端到端约束: sum(lat) ≤ e2e_lat
    tot_lat = sum(sol[i][1] for i in range(K))
    if elim_nume_error(tot_lat - e2e_lat) > 0:
        return False

    # 任务级约束
    for i in range(K):
        n_core, lat, _ = sol[i]
        dist = node_var_dists[i]
        
        if isinstance(dist, SenVarDist):
            lat_q = float(dist.quantile(quantile))
            if n_core != 1 or lat < lat_q:
                return False
        elif isinstance(dist, AccVarDist):
            io_q = dist.exec_dist.quantile(quantile)
            load_q = dist.load_dist.quantile(quantile)
            if n_core * (lat-io_q) * FLOPS_PER_CORE < load_q:
                return False

        constr = constr_core[i]
        mode = constr.parallel_mode
        if mode == 'list':
            if n_core not in constr.core_list:
                return False
        elif mode == 'lwb':
            if n_core < constr.core_min:
                return False
        elif mode == 'upb':
            if n_core > constr.core_max:
                return False
        elif mode == 'range':
            if n_core < constr.core_min or n_core > constr.core_max:
                return False
        if lat < 0:
            return False

    # 总核心数上限检查
    max_used_cores = max(sol[i][0] for i in range(K))
    if max_used_cores > tot_cores:
        return False
        
    return True

class GurobiRscSlackEstim:
    """
    基于分位数的串行链路资源分配（Gurobi）

    常量参数:
      - node_var_dists: List[Variation]
          每个任务的延迟分布对象（src: SenVarDist, op: AccVarDist）。
      - e2e_lat: float
          端到端延迟约束（绝对值）。因链路为串行，聚合为 max(lat_i)。
      - quantile: float
          目标分位数（例如 0.95）。
      - constr_core: List[TaskConstraints]
          每个任务的核心数约束配置，支持 'upb'/'lwb'/'range'/'list' 四种模式。
      - tot_cores: int
          系统总核心数上限（用于限制 max_core）。

    变量:
      - core[i]: int ≥ 1  每个任务分配的核心数
      - lat[i]:  float ≥ 0 每个任务的延迟（分位数近似）
      - max_core: int  所有任务所用核心数的最大值

    约束:
      1) 端到端延迟（串行场景）: max_i(lat[i]) ≤ e2e_lat
      2) 分布与核心关系:
         - src: lat[i] = SenVarDist.quantile(q), core[i] = 1
         - op:  lat[i] ≥ load_q/(core[i]*FLOPS_PER_CORE) + io_q （q 为目标分位数）
      3) 每任务核心数约束（上/下界/列表/区间）
      4) 总核心数上限: max_core ≤ tot_cores

    目标:
      - 最小化 max_core
    """
    def __init__(self, node_var_dists: List[Variation], e2e_lat: float, quantile: float, constr_core: List[TaskConstraints], tot_cores: int, verbose: bool = False):
        self.node_var_dists = node_var_dists
        self.K = len(node_var_dists)
        self.e2e_lat = float(e2e_lat)
        self.quantile = float(quantile)
        self.constr_core: List[TaskConstraints] = constr_core
        self.tot_cores = int(tot_cores)
        self.model = gp.Model("rsc_slack_estim")
        self.model.setParam('NonConvex', 2)
        self.model.setParam('OutputFlag', int(verbose))

        # will be filled in create_variables
        self.core = []
        self.lat = []
        self.core_sel: Dict[int, Any] = {}

    def create_variables(self):
        # 每个任务的核心数与分位数延迟
        for i in range(self.K):
            self.core.append(self.model.addVar(lb=1, vtype=GRB.INTEGER, name=f'core[{i}]'))
            self.lat.append(self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f'lat[{i}]'))
            # 如果是列表模式，增加one-hot选择变量
            if self.constr_core[i].parallel_mode == 'list':
                num_options = len(self.constr_core[i].core_list)
                self.core_sel[i] = self.model.addVars(num_options, vtype=GRB.BINARY, name=f"sel_{i}")
        # 最大核心数
        self.max_core = self.model.addVar(lb=0, ub=self.tot_cores, vtype=GRB.INTEGER, name="max_core")

    def define_constraints(self):
        # 1. 端到端延迟约束: sum(L_i) <= e2e_deadline
        self.model.addConstr(quicksum(self.lat[i] for i in range(self.K)) + time1n_error_tol_abs <= self.e2e_lat, "e2e_latency_constr")

        # 2. 任务延迟与核心数的关系约束
        for i in range(self.K):
            dist = self.node_var_dists[i]
            
            if isinstance(dist, SenVarDist):
                lat_q = float(dist.quantile(self.quantile))
                self.model.addConstr(self.lat[i] >= lat_q + time1n_error_tol_abs, name=f"lat_src[{i}]")
                self.model.addConstr(self.core[i] == 1, name=f"core_src[{i}]")
            elif isinstance(dist, AccVarDist):
                load_q = dist.load_dist.quantile(self.quantile) + flop1n_error_tol_abs
                io_q = dist.exec_dist.quantile(self.quantile)
                
                # Gurobi 支持二次约束: (L_i - io_q) * C_i >= load_q / FLOPS_PER_CORE
                self.model.addQConstr((self.lat[i] - io_q - time1n_error_tol_abs) * self.core[i] >= load_q / FLOPS_PER_CORE, f"lat_core_rel_{i}")

        # 3. 核心数分配约束
        for i in range(self.K):
            constr = self.constr_core[i]
            mode = constr.parallel_mode
            if mode == 'upb':
                self.model.addConstr(self.core[i] <= constr.core_max, name=f"core_upb[{i}]")
            elif mode == 'lwb':
                self.model.addConstr(self.core[i] >= constr.core_min, name=f"core_lwb[{i}]")
            elif mode == 'range':
                self.model.addConstr(self.core[i] >= constr.core_min, name=f"core_range_min[{i}]")
                self.model.addConstr(self.core[i] <= constr.core_max, name=f"core_range_max[{i}]")
            elif mode == 'list':
                opts = constr.core_list
                self.model.addConstr(quicksum(self.core_sel[i][j] for j in range(len(opts))) == 1, name=f"core_list_sum[{i}]")
                self.model.addConstr(self.core[i] == quicksum(opts[j] * self.core_sel[i][j] for j in range(len(opts))), name=f"core_list_link[{i}]")

        # 4) 总核心上限 + 定义 max_core
        self.model.addGenConstrMax(self.max_core, self.core, name="max_core_def")
        self.model.addConstr(self.max_core <= self.tot_cores, name="tot_core_cap")


    @time_cnt("solve")
    def solve(self) -> Union[Dict[int, Tuple[int, float, Union[str, None]]], None]:
        print("Gurobi-RscSlackEstim!!\n")
        # 目标：最小化最大核心数
        self.model.setObjective(self.max_core, GRB.MINIMIZE)
        # try:
        self.model.optimize()
        print("status:", self.model.status)
        print("obj:", self.model.objVal)
        # build solution dic of (core, lat)
        # print(self.model.display())
        sol: Dict[int, Tuple[int, float, Union[str, None]]] = {}
        for i in range(self.K):
            n_core = int(round(self.core[i].X))
            lat = float(self.lat[i].X)
            constr_tag = None
            if constr:=self.constr_core[i]:
                constr_tag = self.check_constr(i, constr)
            sol[i] = (n_core, elim_nume_error(lat), constr_tag)
        assert check_solution_validity(sol, self.node_var_dists, self.e2e_lat, self.quantile, self.constr_core, self.tot_cores), "Gurobi solution failed post-check"
        return sol
        # except gp.GurobiError as e:
        #     print('GurobiError', e.errno, str(e))
        #     return None
        # except Exception as e:
        #     print('Unexpected error:', str(e))
        #     return None

    def check_constr(self, i, constr):
        constr_type = constr.parallel_mode
        n_core = int(self.core[i].x)
        constr_t = None
        if constr_type in ['lwb', 'upb', 'range']: 
            if n_core == constr.core_min:
                constr_t = "lwb"
            elif n_core == constr.core_max:
                constr_t = "upb"
        elif constr_type == 'list':
            if n_core == min(constr.core_list):
                constr_t = "lwb"
            elif n_core == max(constr.core_list):
                constr_t = "upb"
        return constr_t



def HeuriRscSlackEstim(
    node_var_dists: List[Variation], e2e_lat: float, quantile: float, constr_core: List[TaskConstraints], tot_cores: int, 
    threshold: float
) -> Union[Dict[int, Tuple[int, float, Union[str, None]]], None]: # 返回值可能为 None
    """
    "约束驱动的理想均分"启发式求解器。
    为单条链上的一组未分配任务，在给定的延迟预算内，找到最优核心数分配。

    exactly match the slack: 
        1. no process is reassigned due to the constraint
        2. the reassigned processes compensate each other
    
    if w/o constraint, all task is assigned slack propotional to its flops, and all task requires same core numbers, e.g., core_max_ideal
    if w/ constraints: 
    core_max_actual > core_max_ideal
    A. if a task is resource upper bounded, 
        it use less core and more slack than the ideal, other task use more core than the ideal
        the upper bounded items is upper bounded by the core_max_ideal, 
        it's no meaning to assign more slack to them and it's not legal to assign more core to them
        so we move them. 
    B. core_min_actual come from maxumum core of the lower bounded items in current iterations, 
        but not must be the final result. Because there may be other tasks become upper bounded, and lower bounded items may be eliminated.
        Consequently, we leave them unprocessed until there is no upper bounded item. 
    C. The remaining slack (<= total_slack * percent of flops of remaining tasks), 
    and is distributed remaining processes

    """
    print("Heuri-RscSlackEstim!!\n")
    
    # 1. 前置可行性检查
    min_possible_latency = 0
    for i, dist in enumerate(node_var_dists):
        # 假设 find_legal 能找到约束下的最大核心数
        # 注意: 这里的 'ideal_cores' 设为一个极大值来探测上界
        max_cores, _ = find_legal(constr_core[i], tot_cores, tot_cores) 
        min_possible_latency += dist.quantile(quantile, max_cores)

    if min_possible_latency > e2e_lat:
        raise ValueError(f"Heuristic pre-check failed. Min possible latency ({min_possible_latency:.2e}) > budget ({e2e_lat:.2e}).")
    
    # 1a. 分离计算负载和IO时间
    flops_rem = 0
    total_time_fixed = 0
    flops_dict = OrderedDict()
    for i, dist in enumerate(node_var_dists):
        if isinstance(dist, SenVarDist):
            total_time_fixed += dist.quantile(quantile)
        else: 
            flops_rem += dist.load_dist.quantile(quantile)
            total_time_fixed += dist.exec_dist.quantile(quantile)
            flops_dict[i] = dist.load_dist.quantile(quantile)

    slack_rem = elim_nume_error(e2e_lat - total_time_fixed - time1n_error_tol_abs * (len(node_var_dists)+1))
    rsc_map_w:Dict[int, Tuple[int, float, Union[str, None]]] = {}

    state = False
    while not state and len(flops_dict) > 0:                
        # 1b. 核心公式：计算理想核心数
        ideal_cores = int(math.ceil(flops_rem / (slack_rem * FLOPS_PER_CORE)))

        # 2. 检查约束冲突，将冲突节点“固定”
        newly_fixed_nodes = []
        for node in flops_dict:
            req_rsc_size, constr = find_legal(constr_core[node], tot_cores, ideal_cores)
            got_latency = elim_nume_error(flops_dict[node] / req_rsc_size / FLOPS_PER_CORE)
            rsc_map_w[node] = (req_rsc_size, got_latency, constr)
            if constr is not None:
                newly_fixed_nodes.append(node)
        
        # check the constraint
        constr_dict = {node:constr for node, (_, _, constr) in rsc_map_w.items() if node in flops_dict and constr != 'none'}
        lat_remm_ops = sum([lat for node, (_, lat, _) in rsc_map_w.items() if node in flops_dict])
        curr_slack_gap = lat_remm_ops - slack_rem

        # achieved latency fall within the e2e latency, by wasting some slack, due to the quantization effects,
        if len(constr_dict) == 0:
            # we redistribute the remaining slack to the remaining processes, since there is no constraint
            for node in flops_dict:
                slack_redis = flops_dict[node] / flops_rem * (-curr_slack_gap)
                rsc_map_w[node] = (rsc_map_w[node][0], elim_nume_error(rsc_map_w[node][1] + slack_redis), rsc_map_w[node][2])        
            # no need to iterate again
            state, flops_rem, slack_rem = True, 0, 0
            break
        elif 0>=curr_slack_gap>=-threshold:            
            # no need to iterate again
            state, flops_rem, slack_rem = True, 0, -curr_slack_gap
            break
        else:
            # the item that not upper bounded items are maximumly 
            if curr_slack_gap > 0:
                # remove the process which has been reached the core_max
                for node,_constr in constr_dict.items():
                    if _constr == "upb":
                        flops_rem -= flops_dict[node]
                        slack_rem -= rsc_map_w[node][1]
                        flops_dict.pop(node)
            else:
                # remove the process which has been reached the core_min
                for node,_constr in constr_dict.items():
                    if _constr == "lwb":
                        flops_rem -= flops_dict[node]
                        slack_rem -= rsc_map_w[node][1]
                        flops_dict.pop(node)

    # add io time to the lat 
    for i, dist in enumerate(node_var_dists):
        if isinstance(dist, AccVarDist):
            fixed_io_time = dist.exec_dist.quantile(quantile)
            rsc_map_w[i] = (int(rsc_map_w[i][0]), elim_nume_error(float(rsc_map_w[i][1] + fixed_io_time + time1n_error_tol_abs)), rsc_map_w[i][2])

    # add the src nodes
    for i, dist in enumerate(node_var_dists):
        if isinstance(dist, SenVarDist):
            rsc_map_w[i] = (1, elim_nume_error(float(dist.quantile(quantile)+time1n_error_tol_abs)), None)

    # 验证解决方案
    if not check_solution_validity(rsc_map_w, node_var_dists, e2e_lat, quantile, constr_core, tot_cores):
        raise ValueError("Heuristic solution failed post-check")
    
    return rsc_map_w

