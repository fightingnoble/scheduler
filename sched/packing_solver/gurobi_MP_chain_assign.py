from typing import List, Dict, Tuple, Union, Any
import gurobipy as gp
from gurobipy import GRB
import numpy as np 
from utils import time_cnt
from global_var import FLOPS_PER_CORE, elim_nume_error

class GurobiRscSlackEstim():
    """
    problem define
    constant: 
        K: number of items
        flops, (k,): number of operations
        e2e: end-to-end latency
        margin, (k,): margin of each item
        constr_core: number constraints of each cores
            format: dict {
                key: "max", "min", "list"
                value: int, int, list
            }
        tot_cores: total number of cores
    variable:
        core, (k,): number of cores
        lat, (k,): latency of each item
    constraints:
        flops_k = core_k * lat_k, \forall k \in K
        e2e <= \sum_{k=1}^{K} lat_k * (1 + margin_k)
    objective:
        minimize the maximum of core
    """

    def __init__(self, K, flops, e2e, margin:List[Tuple[float, float]], constr_core:Dict[str, Union[str, int, List[int]]], tot_cores, verbose:bool=False):
        # constants
        self.K = K
        self.flops = flops
        self.e2e = e2e
        self.margin:List[Tuple[float, float]] = margin
        self.model = gp.Model("rsc_slack_estim")
        self.constr_core:Dict[str, Union[str, int, List[int]]] = constr_core
        self.tot_cores = tot_cores
        # variables
        self.core = []
        self.core_sel = {}
        self.lat = []
        self.max_core = 0
        self.model.setParam('NonConvex', 2)
        self.model.setParam('OutputFlag', int(verbose))

    def create_variables(self):
        for i in range(self.K):
            self.lat.append(self.model.addVar(lb=0, ub=self.e2e, vtype=GRB.CONTINUOUS, name=f'lat[{i}]'))
            if constr:=self.constr_core[i]:
                constr_type = constr['mode']
                if constr_type == 'list':
                    # binary type for list
                    # self.core.append(self.model.addVars(len(constr["list"]), vtype=GRB.BINARY, name=f'core[{i}]'))
                    self.core.append(self.model.addVar(lb=min(constr["list"]), ub=max(constr["list"]), vtype=GRB.INTEGER, name=f'core[{i}]'))
                    self.core_sel[i] = self.model.addVars(len(constr["list"]), vtype=GRB.BINARY, name=f'core_sel[{i}]')
                    raise NotImplementedError
                elif constr_type == 'lwb':
                    # min type
                    self.core.append(self.model.addVar(lb=constr["min"], ub=self.tot_cores, vtype=GRB.INTEGER, name=f'core[{i}]'))
                elif constr_type == 'upb':
                    # max type
                    self.core.append(self.model.addVar(lb=1, ub=constr["max"], vtype=GRB.INTEGER, name=f'core[{i}]'))
                elif constr_type == "range":
                    # range type
                    self.core.append(self.model.addVar(lb=constr["min"], ub=constr["max"], vtype=GRB.INTEGER, name=f'core[{i}]'))
                elif constr_type is None:
                    self.core.append(self.model.addVar(lb=1, ub=self.tot_cores, vtype=GRB.INTEGER, name=f'core[{i}]'))
                else:
                    raise NotImplementedError
        self.max_core = self.model.addVar(lb=0, ub=self.tot_cores, vtype=GRB.INTEGER, name="max_core")
    
    def define_constraints(self):
        # Constraints
        for i in range(self.K):
            # add list constraints for core
            if constr:=self.constr_core[i]:
                constr_type = constr['mode']
                if constr_type == 'list':
                    # binary type for list
                    self.model.addConstr(self.core[i] == sum([constr["list"][j] * self.core_sel[i][j] for j in range(len(constr["list"]))]), name=f'core[{i}]')
                    self.model.addConstr(sum([self.core_sel[i][j] for j in range(len(constr["list"]))]) == 1, name=f'core_sel[{i}]')
            self.model.addConstr(self.flops[i] == self.core[i] * self.lat[i] * FLOPS_PER_CORE, name=f'flops[{i}]')
        self.model.addConstr(sum([self.lat[i] / (1 - self.margin[i][1]) + self.margin[i][0] for i in range(self.K)]) <= self.e2e, name="e2e")
        self.model.addGenConstrMax(self.max_core, self.core, name="max_core")
        
    @time_cnt("solve")
    def solve(self):
        # Objective: minimize the maximum point of the packing
        self.model.setObjective(self.max_core, GRB.MINIMIZE)
        try:
            self.model.optimize()
            print("status:", self.model.status)
            print("obj:", self.model.objVal)
            # build solution dic of (core, lat)
            sol = {}
            for i in range(self.K):
                if constr:=self.constr_core[i]:
                    constr = self.check_constr(i, constr)
                n_core = round(self.core[i].x)
                lat = elim_nume_error(self.flops[i] / (n_core * FLOPS_PER_CORE))
                sol[i] = (int(self.core[i].x), lat, constr)
            return sol
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))
            return None
        except AttributeError:
            print('Encountered an attribute error')
            return None

    def check_constr(self, i, constr):
        constr_type = constr['mode']
        n_core = int(self.core[i].x)
        constr_t = None
        if constr_type in ['lwb', 'upb', 'range']: 
            if n_core == constr["min"]:
                constr_t = "lwb"
            elif n_core == constr["max"]:
                constr_t = "upb"
        elif constr_type == 'list':
            if n_core == min(constr["list"]):
                constr_t = "lwb"
            elif n_core == max(constr["list"]):
                constr_t = "upb"
        return constr_t
        
# if __name__ == "__main__":
#     # test
#     import numpy as np 
#     from task.task_cfg import load_taskattrib, creat_logical_graph, task_graph_srcs, task_graph_ops, task_graph_sinks
#     from task.graph_breakdown import decompose_dag_into_chains
#     from sched.slack_estim import build_score_dict_ref_flops, alloc_func
#     import copy
#     from utils import input_parser

#     args = input_parser() 
#     slack_threshold = args.slack_threshold
#     taskattr_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=args.verbose) 
#     hyper_p = 1/f_gcd
#     if args.aux_scale_factor > 1:
#         for node, taskattr in taskattr_dict.items():
#             # scale up the thread scaling factor
#             if taskattr.timing_flag == "realtime":
#                 taskattr.thread_scaling_factor *= args.aux_scale_factor

#     logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)

#     e2e = args.e2e_latency
#     chains = []
#     for start_node in task_graph_srcs:
#         chains += decompose_dag_into_chains(logical_graph_nx, start_node, task_graph_sinks)

#     task_dict = taskattr_dict
#     e2e_latency = args.e2e_latency
#     exec_t_comp_ratioA = args.exec_t_comp_ratioA
#     threshold = args.slack_threshold
#     chains_info = []
#     chains_info2 = []
#     for chain in chains:
#         flops_dict = {}
#         build_score_dict_ref_flops(task_dict, chain, flops_dict)
#         # TODO: select the e2e_latency, by the last node of the chain
#         tail_task = task_dict[chain[-1]]
#         if tail_task.timing_flag == "deadline":
#             slack_rem = e2e_latency
#             is_ddl_constr = True
#         else:
#             slack_rem = tail_task.freq_division_factor / tail_task.freq
#             is_ddl_constr = False
#         ops_rem = sum(flops_dict.values())
#         chains_info2.append((chain, copy.deepcopy(flops_dict), slack_rem, ops_rem, is_ddl_constr))
#         slack_rem = (1-exec_t_comp_ratioA)*1e3*slack_rem/1e3
#         chains_info.append((chain, flops_dict, slack_rem, ops_rem, is_ddl_constr))
    
#     sort_idx = np.array([(slack_rem, not is_ddl_constr, -ops_rem) for _, _, slack_rem, ops_rem, is_ddl_constr in chains_info], dtype=np.dtype('f8, ?, f8')).argsort()
#     chains_info = [chains_info[i] for i in sort_idx]
#     chains_info2 = [chains_info2[i] for i in sort_idx]
    
#     rsc_map_w:Dict[str, Tuple[int, float]] = {}
#     sol = {}
#     # for chain, flops_dict, slack_rem, ops_rem, is_ddl_constr in chains_info:
#     for (chain, flops_dict, slack_rem, ops_rem, is_ddl_constr), (chain2, flops_dict2, slack_rem2, ops_rem2, is_ddl_constr2) in zip(chains_info, chains_info2):
#         for node in chain:
#             if node in rsc_map_w:
#                 ops_rem -= flops_dict[node]
#                 slack_rem -= rsc_map_w[node][1]
#                 flops_dict.pop(node)
#         for node in chain2:
#             if node in rsc_map_w:
#                 ops_rem2 -= flops_dict2[node]
#                 slack_rem2 -= rsc_map_w[node][1]
#                 flops_dict2.pop(node)

#         state = False
#         while not state and len(flops_dict) > 0:
#             state, ops_rem, slack_rem = alloc_func(rsc_map_w, task_dict, flops_dict, ops_rem, slack_rem, threshold)

#         # node_list = list(taskattr_dict.keys())
#         # constr_core = [{"max":task_attr.core_max, "min":task_attr.core_min, "list":task_attr.core_list, "mode":task_attr.parallel_mode} for node, task_attr in taskattr_dict.items()]
#         # K = len(taskattr_dict)
#         # flops = [task_attr.flops for task_attr in taskattr_dict.values()]
#         # margin = [slack_threshold for _ in range(K)]

#         tot_cores = 120
#         K = len(flops_dict2)
#         flops = list(flops_dict2.values())
#         margin = [exec_t_comp_ratioA for _ in range(K)]
#         node_list = list(flops_dict2.keys())
#         constr_core = [{"mode":taskattr_dict[node].parallel_mode, "max":taskattr_dict[node].core_max_compile, 
#                         "min":taskattr_dict[node].core_min_compile, "list":taskattr_dict[node].core_list_compile
#                         } for node in node_list]
#         # {"max":task_attr.core_max_compile, "min":task_attr.core_min_compile, "list":task_attr.core_list_compile}
#         solver = GurobiRscSlackEstim(K, flops, slack_rem2, margin, constr_core, tot_cores)
#         solver.create_variables()
#         solver.define_constraints()
#         sol = solver.solve()
#         print(dict(zip(node_list, sol.values())))
