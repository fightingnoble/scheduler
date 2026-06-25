from gurobipy import Model, GRB, quicksum
from typing import List, Tuple, Dict, Union
import gurobipy as gp


class GurobiSemi2DClstMapping:
    """
    Problem Description: 
    I have a task graph (directed graph) with N tasks, and I need to deploy these tasks on M computing resources. The objective is to satisfy latency and data dependencies while balancing the utilization in spatial and temporal.

    Input: 
    Number of tasks N, number of resources M, number of partitions S.
    Connection relationships, source node and start time list, sink node and end time list.
    A list of workload for each task.
    A 2D table A, where A_ij's index: i is the task index, j is the number of resources (not an identifier!). A_ij represents the delay required to assign j resources to task i.

    Definition: 
    The total resources for each partition are calculated as: the maximum resource demand in the partition at all moments on the timeline.
    The resources assigned to each task are defined by its start time, duration, and the number of resources.
    The computational workload is defined as the assigned number of computational resources × duration.

    Constraints: Each source node in the task graph has a start time, and each sink node has a deadline. The start time of a source node must not be earlier than its own start time. The start time of each node must not be earlier than the end time of its predecessor node. The sink node must complete no later than its deadline.
    Each task i has a lower bound for the computational workload, and when the resource count equals the index j of the selected A_ij, the duration must be greater than A_ij.

    Objective: 
    The goal is to ensure that all deadlines are met, while minimizing the total computational resources required by all partitions.
    

    Formula Description: 

    Latex Algorithm Description: 
    """

    # N:int, M:int, S:int, NT:int, T:float, A:Dict[int, int], dependencies:List[Tuple[int, int]], compute_lower_bounds:List[int], start_constraints:List[int], end_constraints:List[int], time_format="float", timestep_size:float=None
    def __init__(self, N:int, M:Union[int, float], S:int, T:float, A:Dict[int, int], dependencies:List[Tuple[int, int]],  
                 compute_lower_bounds:List[int], start_constraints:List[int], end_constraints:List[int], path_info:List[Tuple[Union[List[int], float]]],
                 time_format="float", timestep_size:float=None, spt_fmt="float", 
                 fit_params:list=None, size_scale=1, 
                 verbose=True
                 ):
        self.N:int = N # Number of tasks
        self.M:int = M # Maximum number of resources
        self.S:int = S # Number of partitions
        self.NT:int = 2*N # Number of discrete time steps
        self.T:float = T # Period length
        self.A:Dict[int, Dict[int, int]] = A # 2D table A, where A_ij's index: i is the task index, j is the number of resources (not an identifier!). A_ij represents the delay required to assign j resources to task i.
        self.dependencies:List[Tuple[int, int]] = dependencies # Dependency relationships in the task graph
        self.compute_lower_bounds:List[int] = compute_lower_bounds # Lower bound for each task's workload
        self.start_constraints:List[int] = start_constraints # Earliest start time for each task (None means no constraint)
        self.end_constraints:List[int] = end_constraints # Latest completion time for each task
        self.path_info:List[Tuple[Union[List[int], float]]] = path_info # All edges along the path from source to sink 
        # self.critical_tick:List[int] = time_steps # Discrete time steps, used to calculate the maximum resource demand for the partition
        self.spt_fmt = GRB.CONTINUOUS if spt_fmt == "float" else GRB.INTEGER # Space variable type
        self.temp_fmt = GRB.CONTINUOUS if time_format == "float" else GRB.INTEGER # Time variable type
        if time_format != "float":
            assert (self.T - int(self.T/timestep_size)*timestep_size) < 1e-9, "timestep_size should be a divisor of T"
        self.t_upb = self.T if time_format == "float" else int(self.T/timestep_size) # Upper bound for time variable
        self.t_attr = {"lb": 0, "ub": self.t_upb, "vtype": self.temp_fmt} # Time variable attributes
        self.spt_attr_1_to_M = {"lb": 1, "ub": M, "vtype": self.spt_fmt} # Space variable attributes
        self.spt_attr_0_to_M = {"lb": 0, "ub": M, "vtype": self.spt_fmt} # Space variable attributes
        self.slackR = timestep_size if time_format == "float" else 1 # Slack interval right side, resource window is one time step ahead. So if the current moment is finished, the resource demand for the next moment is 0
        self.fit_params = fit_params
        self.size_scale = size_scale
        
        # Create model
        self.model = Model("Partitioned_Task_Scheduling")
        self._initialize_variables()
        self._add_constraints()
        self.scatter_bin_usage()
        self.model.setParam('NonConvex', 2)
        self.model.setParam('OutputFlag', int(verbose))
        self.verbose = verbose 

    def _initialize_variables(self):
        """初始化决策变量"""
        # ======================================================================
        # partition size: allow searching number of bins, using spt_attr_0_to_M, the bin with size of 0 is through unused
        # self.partition_size = self.model.addVars(self.S, name="Partition_Size", **self.spt_attr_0_to_M) # Partition capacity

        # Task variables: temporal placements
        self.start_times = self.model.addVars(self.N, name="Start_Time", **self.t_attr) # Task start times
        self.end_times = self.model.addVars(self.N, name="End_Time", **self.t_attr) # Task end times
        self.critical_tick = [self.start_times[i] for i in range(self.N)] + [self.end_times[i] for i in range(self.N)]
        # critical_tick = [model.cbGetSolution(model.getVarByName(f"Start_Time[{i}]")) for i in range(N)] + [model.cbGetSolution(model.getVarByName(f"End_Time[{i}]")) for i in range(N)]
        self.durations = self.model.addVars(self.N, name="Duration", **self.t_attr) # Task durations
        
        # Task variables: resource allocation
        self.resources = self.model.addVars(self.N, name="Resources", **self.spt_attr_1_to_M) # Number of resources assigned to each task

        # =================================================================================
        # Decision variables
        # =================================================================================
        self.is_resource_j = self.model.addVars(
            [(i, j) for i in range(self.N) for j in self.A[i].keys()],
            name="Is_Resource_j",
            vtype=GRB.BINARY,
        ) # whether jth implementation is selected for task i
        self.is_task_in_bin = self.model.addVars(
            [(i, s) for i in range(self.N) for s in range(self.S)],
            name="Task_In_Partition",
            vtype=GRB.BINARY,
        ) # whether task i is assigned to partition s
        

        # =================================================================================
        # Auxiliary variables
        # active at the critical ticks
        # The the peak usage of each partition        
        # =================================================================================
        self.active_at_t = {}
        self.Rsatisfied = {}
        self.Lsatisfied = {}
        self.active_at_t_cross_period = {}
        self.active_at_t_not_cross_period = {}
        # Lsatisfied = {(i,t): model.cbGetSolution(model.getVarByName(f"L_Satisfied[{i},{t}]")) for i in range(N) for t in range(2*N) if t%.N != i}
        # Rsatisfied = {(i,t): model.cbGetSolution(model.getVarByName(f"R_Satisfied[{i},{t}]")) for i in range(N) for t in range(2*N) if t%.N != i}
        # active_at_t_cross_period = {(i,t): model.cbGetSolution(model.getVarByName(f"Active_At_T_Cross_Period[{i},{t}]")) for i in range(N) for t in range(2*N) if t%.N != i}
        # active_at_t_not_cross_period = {(i,t): model.cbGetSolution(model.getVarByName(f"Active_At_T_Not_Cross_Period[{i},{t}]")) for i in range(N) for t in range(2*N) if t%.N != i}
        for i in range(self.N):
            for t in range(self.NT):
                if t%self.N == i: 
                    if t > self.N-1: 
                        # assert task i is not allocated to its end time
                        self.active_at_t[(i, t)] = 0
                    else: 
                        # task i is allocated to its start time
                        self.active_at_t[(i, t)] = 1
                else: 
                    # use binary variable to indicate if task i is active at t
                    self.active_at_t[(i, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"Active_At_T[{i},{t}]")
                    # 1 -> t>=start_times[i]
                    self.Lsatisfied[(i, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"L_Satisfied[{i},{t}]")
                    # 1 -> t<=end_times[i]-self.slackR
                    self.Rsatisfied[(i, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"R_Satisfied[{i},{t}]")
                    self.active_at_t_cross_period[(i, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"Active_At_T_Cross_Period[{i},{t}]")
                    self.active_at_t_not_cross_period[(i, t)] = self.model.addVar(vtype=GRB.BINARY, name=f"Active_At_T_Not_Cross_Period[{i},{t}]")


        # partition size: allow searching number of bins, using spt_attr_0_to_M, the bin with size of 0 is through unused
        self.peak_bin_usage = self.model.addVars(self.S, name="Max_Usage", **self.spt_attr_0_to_M) # Maximum usage of each partition


        # =================================================================================
        # Define the cyclic mapping constraints
        # =================================================================================
        self.is_cross_period = self.model.addVars(self.N, vtype=GRB.BINARY, name="Is_Cross_Period")
        self.is_edge_cross_period = self.model.addVars(
            [(i, j) for i, j in self.dependencies], vtype=GRB.BINARY, name="Is_Edge_Cross_Period"
        )
        # also add a variable to indicate if a source data is crossing a period
        for i in range(self.N):
            if self.start_constraints[i] is not None:
                self.is_edge_cross_period[-1, i] = self.model.addVar(vtype=GRB.BINARY, name=f"Is_Edge_Cross_Period_{-1}_{i}")
        # we not regulate the sink node time, so we need not add a variable to indicate if a sink data is crossing a period

        # =================================================================================
        # Target variables
        # =================================================================================        
        # Aux 1: speedup rate, chain by chain
        # self.speedup_rate = {}
        # for i in range(self.N):
        #     if self.end_constraints[i] is not None:
        #         self.speedup_rate[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"Speedup_Rate_{i}")
        
        # Aux 2: affinity score
        self.affinity_score = self.model.addVar(vtype=GRB.CONTINUOUS, name="affinity_score")
        self.spatial_affinity = self.model.addVars([(i,j) for i,j in self.dependencies], vtype=GRB.CONTINUOUS, name="spatial_affinity")
        self.temporal_affinity = self.model.addVars([(i,j) for i,j in self.dependencies], vtype=GRB.CONTINUOUS, name="temporal_affinity")
        self.temporal_affinity_bool = self.model.addVars([(i,j) for i,j in self.dependencies], vtype=GRB.BINARY, name="temporal_affinity_bool")
        self.st_affinity_bool = self.model.addVars([(i,j) for i,j in self.dependencies], vtype=GRB.CONTINUOUS, name="st_affinity_bool")
        # spatial_affinity = {(i,j): model.cbGetSolution(model.getVarByName(f"spatial_affinity[{i},{j}]")) for i,j in dependencies}
        # temporal_affinity = {(i,j): model.cbGetSolution(model.getVarByName(f"temporal_affinity[{i},{j}]")) for i,j in dependencies}
        # temporal_affinity_bool = {(i,j): model.cbGetSolution(model.getVarByName(f"temporal_affinity_bool[{i},{j}]")) for i,j in dependencies}
        # st_affinity_bool = {(i,j): model.cbGetSolution(model.getVarByName(f"st_affinity_bool[{i},{j}]")) for i,j in dependencies}
                
        # Aux 3: ratio of peak usage and partition size
        # self.spatial_rda_ratio = self.model.addVars(self.S, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="Spatial_RDA_Ratio")

        self.static_wasted_resources = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            name="Static_Wasted_Resources",
        )
        self.realloc_overhead_bin = self.model.addVars(
            self.S,
            vtype=GRB.CONTINUOUS,
            name="",
        )
        self.affinity_overhead = self.model.addVars(
            [(i,j) for i,j in self.dependencies],
            vtype=GRB.CONTINUOUS,
            name="Affinity_Overhead",
        )
        self.affinity_overhead_sum = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            name="Affinity_Overhead_Sum",
        )
        # affinity_overhead = {(i,j): model.cbGetSolution(model.getVarByName(f"Affinity_Overhead[{i},{j}]")) for i,j in dependencies}
        # affinity_overhead_sum = model.cbGetSolution(model.getVarByName(f"Affinity_Overhead_Sum"))
        
        self.dynamic_wasted_resources = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            name="Dynamic_Wasted_Resources",
        )
    
    def _add_constraints(self):
        """添加约束条件"""

        # =================================================================================
        # Define graph-related timing constraints
        # =================================================================================
        # All data dependencies must be satisfied. 
        # For a dependency from i to j, 
        # if crossing the hyperperiod, t[j] <= t[i] + d[i]
        # else, t[j] + T <= t[i] + d[i]. 
        # At most one time a path should cross the hyperperiod
        # =================================================================================
        self.add_e2e_graph_constrs()

        # =================================================================================
        # Constraint: amount and shape of Resource allocation 
        # =================================================================================
            # resources[i] == sum(is_resource_j * j)
            # duration[i] >= sum(A_ij * is_resource_j)
            # resources[i] * durations[i] >= compute_lower_bounds[i]
        # =================================================================================
        self.add_amount_and_shape_constrs()

        # =================================================================================
        # Constraint: Exactly one selection
        # =================================================================================
        self.add_exactly_one_sel_constrs()

        # =================================================================================
        # Constraint: Partition capacity
        # 6-1. the summation of the resources used by each partition equals to the M 
        # 6-2. at any time, the maximum resource usage should never exceed the partition size
        # =================================================================================
        self.add_bin_constrs()
        
        self.define_obj()

    def add_exactly_one_sel_constrs(self):
        for i in range(self.N):
            # only one partition is selected
            self.model.addConstr(
                quicksum(self.is_task_in_bin[i, s] for s in range(self.S)) == 1,
                name=f"Partition_Assignment_{i}",
            )  
            # only one implementation is selected
            self.model.addConstr(
                quicksum(self.is_resource_j[i, j] for j in self.A[i].keys()) == 1,
                name=f"Resource_Selection_{i}",
            ) 

    def add_amount_and_shape_constrs(self):
        for i in range(self.N):
            # Shape compatibility of allocated region and selected implementation
            # duration[i] >= sum(A_ij * is_resource_j)
            # self.model.addConstr(
            #     self.durations[i] >= quicksum(self.is_resource_j[i, j] * self.A[i][j] for j in self.A[i].keys()),
            #     name=f"Duration_Constraint_{i}",
            # ) 
            # resources[i] == sum(is_resource_j * j)
            self.model.addConstr(
                self.resources[i] == quicksum(self.is_resource_j[i, j] * j for j in self.A[i].keys()),
                name=f"Resource_Assignment_{i}",
            ) 

            # relation of arrival, execution and finish time
            self.model.addConstr(
                self.durations[i] == self.end_times[i] + self.T * self.is_cross_period[i] - self.start_times[i],
                name=f"Duration_Definition_{i}"
            )
        # Total amount (Budget) constraint
        # self.resources[i] * self.durations[i] >= compute_lower_bounds[i]
        for i in range(self.N):
            self.model.addConstr(
                self.resources[i] * self.durations[i] == self.compute_lower_bounds[i],
                name=f"Compute_Lower_Bound_{i}",
            )
            # self.model.addConstr(
            #     self.resources[i] * self.durations[i] <= self.compute_lower_bounds[i] + self.M,
            #     name=f"Compute_Lower_Bound_{i}",
            # )

    def add_e2e_graph_constrs(self):
        # =================================================================================
        # Define graph-related timing constraints
        # =================================================================================
        # All data dependencies must be satisfied. 
        # Specifically, if task \( i \) depends on task \( j \), then \( t_j \leq t_i + d_i \). 
        # For source nodes, a arrival time is specified.
        # For sink nodes, an deadline is specified.
        # =================================================================================

        # If crossing the hyperperiod, t[j] <= t[i] + d[i]
        # else, t[j] + T <= t[i] + d[i] 
        for i, j in self.dependencies:
            self.model.addConstr(
                self.start_times[j] - self.end_times[i] + self.T * self.is_edge_cross_period[i, j] == self.temporal_affinity[i,j],
                name=f"define_temporal_affinity[{i},{j}]"
            )
            self.model.addConstr(self.temporal_affinity[i,j] >=0, name=f"temporal_affinity_non_negative[{i},{j}]")
            self.model.addGenConstrIndicator(self.temporal_affinity_bool[i,j], True, self.temporal_affinity[i,j] == 0, name=f"temporal_affinity_bool_false[{i},{j}]")
            self.model.addGenConstrIndicator(self.temporal_affinity_bool[i,j], False, self.temporal_affinity[i,j] >= self.slackR, name=f"temporal_affinity_bool_true[{i},{j}]")

        M = self.T # Big M, uppper bound of the time
        for path, slack in self.path_info:
            # start time constraint connected to the source node
            first_node = path[0]
            assert self.start_constraints[first_node] is not None, "First node should have a start constraint"
            self.model.addConstr(self.start_times[first_node] >= self.start_constraints[first_node] - M * self.is_edge_cross_period[-1, first_node], name=f"Source_No_Cross_{first_node}")
            self.model.addConstr(self.start_times[first_node] + self.T >= self.start_constraints[first_node] - M * (1 - self.is_edge_cross_period[-1, first_node]), name=f"Source_Cross_{first_node}")


        # (1-2) source and sink node constraints
        # List[Tuple[Union[List[int], float]]]
        # At most one time a path should cross the hyperperiod
        for path, slack in self.path_info:
            # end time constraint connected to the sink node
            n_cross = self.is_cross_period[path[0]] + self.is_edge_cross_period[-1, path[0]]
            for i in range(1, len(path)):
                n_cross += self.is_edge_cross_period[path[i-1], path[i]] + self.is_cross_period[path[i]] 
            final_node = path[-1]
            assert self.end_constraints[final_node] is not None, "Final node should have a end constraint"
            self.model.addConstr(n_cross * self.T + self.end_times[path[-1]] <= self.end_constraints[final_node], name=f"Path_Cross_Period_{path[0]}_{path[-1]}")
            # speedup rate constraint
            # self.model.addConstr(self.speedup_rate[path[-1]] * (n_cross * self.T + self.end_times[path[-1]]) == self.end_constraints[final_node], 
            #                      name=f"Speedup_Rate_Constraint_{path[0]}_{path[-1]}")
            

    def add_bin_constrs(self):
        # =================================================================================
        # (6) Partition related constraints:
        # """
        # 6-1. the summation of the resources used by each partition equals to the M 
        # 6-2. at any time, the maximum resource usage should never exceed the partition size
        # """
        # =================================================================================
        
        # 6-1. The summation of the resources used by each partition equals to the M 
        # self.model.addConstr(
        #     quicksum(self.partition_size[s] for s in range(self.S)) == self.M,
        #     name=f"Partition_Size_Constraint"
        # )
        
        # ======================================================================
        # 6-2. at any time, the peak resource usage should never exceed the partition size
        # for each partition, we need to calculate sum along the spatial dim and calculate the max along the time dim
        # for s in range(self.S):
        #   max[sum(resources[i] * active_at_t[i, t] * is_task_in_bin[i, s] for i in range(self.N)) for t in self.critical_tick] <= partition_size[s]        # ======================================================================

        # step 1: combine resources[i] * active_at_t[i, t]
        self.resource_active_at_t = {}
        for i in range(self.N):
            for t in range(self.NT):
                # alternative: continuous variable or not? 
                self.resource_active_at_t[i, t] = self.model.addVar(name=f"r_{i}_{t}", **self.spt_attr_0_to_M)
                self.model.addConstr(self.resource_active_at_t[i, t] == self.resources[i] * self.active_at_t[i, t], name=f"Bilinear_{i}_resource_at_{t}")
                            
        # step 2: list allocated tasks at all time points
        # active_at_t: Binds to cross-cycle and non-cross-cycle logic via slack conditions.
                    
        for t in range(self.NT):
            for i in range(self.N):
                if t%self.N == i:
                    continue
                tick = self.critical_tick[t]
                # if start_times[i] <= t, then Lsatisfied[i, t] = 1
                # else, Lsatisfied[i, t] = 0
                self.model.addGenConstrIndicator(self.Lsatisfied[i, t], True, self.start_times[i] <= tick, name=f"Start_Active_Constraint_{i}_{t}")
                self.model.addGenConstrIndicator(self.Lsatisfied[i, t], False, self.start_times[i] - self.slackR >= tick, name=f"Start_Inactive_Constraint_{i}_{t}")
                # if end_times[i] < t, then Rsatisfied[i, t] = 1
                # else, Rsatisfied[i, t] = 0
                self.model.addGenConstrIndicator(self.Rsatisfied[i, t], True, tick <= self.end_times[i] - self.slackR, name=f"End_Active_Constraint_{i}_{t}")
                self.model.addGenConstrIndicator(self.Rsatisfied[i, t], False, tick >= self.end_times[i], name=f"End_Inactive_Constraint_{i}_{t}")
        
        # active_at_t_not_cross_period (non-cross-hyperperiod)：
        # active_at_t[i, t] = (start_times[i] <= t < start_times[i] + durations[i]) 
        for t in range(self.NT):
            for i in range(self.N):
                if t%self.N == i:
                    continue
                self.model.addGenConstrAnd(
                    self.active_at_t_not_cross_period[i, t], [self.Lsatisfied[i, t], self.Rsatisfied[i, t]],
                    name=f"Active_At_T_Constraint_{i}_{t}",
                )
                self.model.addGenConstrIndicator(
                    self.is_cross_period[i], False, self.active_at_t[i, t] == self.active_at_t_not_cross_period[i, t],
                    name=f"Active_At_T_Indicator_{i}_{t}",
                )

        # active_at_t_cross_period (cross-hyperperiod)
        # active_at_t[i, t] = (start_times[i] <= t or t< end_time[i]         
        for t in range(self.NT):
            for i in range(self.N):
                if t%self.N == i:
                    continue
                self.model.addGenConstrOr(
                    self.active_at_t_cross_period[i, t], [self.Lsatisfied[i, t], self.Rsatisfied[i, t]],
                    name=f"Active_At_T_Cross_Period_Constraint_{i}_{t}",
                )
                self.model.addGenConstrIndicator(
                    self.is_cross_period[i], True, self.active_at_t[i, t] == self.active_at_t_cross_period[i, t],
                    name=f"Active_At_T_Cross_Period_Indicator_{i}_{t}",
                )
                
        
        # Define resource usage along critical ticks
        self.usage_along_ticks = self.model.addVars(
            [(s, t) for s in range(self.S) for t in range(self.NT)],
            name="Usage_Along_Ticks",
            **self.spt_attr_0_to_M,
        )

        # Bringing resource_active_at_t[i, t] into 
        # max[sum(resources[i] * active_at_t[i, t] * is_task_in_bin[i, s] for i in range(self.N)) for t in self.critical_tick]
        # we can get:
        # max[sum(resource_active_at_t[i, t] * is_task_in_bin[i, s] for i in range(self.N)) for t in self.critical_tick] 
        
        # define usage_along_ticks[s, t] == sum(resource_active_at_t[i, t] * is_task_in_bin[i, s] for i in range(self.N)) for t in self.critical_tick
        # define peak_bin_usage[s] == max(usage_along_ticks[s, t] for t in self.critical_tick)
        for s in range(self.S):
            for t in range(self.NT):
                # 在时间 t 上分区 s 的资源需求
                self.model.addConstr(
                    self.usage_along_ticks[s, t] == quicksum(self.resource_active_at_t[i, t] * self.is_task_in_bin[i, s] for i in range(self.N)),
                    name=f"Usage_Along_Ticks_{s}_{t}",
                )
            # Max usage in each partition
            self.model.addGenConstrMax(
                self.peak_bin_usage[s], [self.usage_along_ticks[s, t] for t in range(self.NT)],
                name=f"Max_Usage_{s}",                    
            )

        # Bring peak_bin_usage[s] into the objective function, we can get:
        # peak_bin_usage[s] <= partition_size[s]
        # self.model.addConstrs((self.peak_bin_usage[s] <= self.partition_size[s] for s in range(self.S)), name="Peak_Bin_Usage_Constraint")
        self.model.addConstr(quicksum(self.peak_bin_usage[s] for s in range(self.S)) <= self.M, name="Totals_Usage_Constraint")

        # increased partition size constraint
        for s in range(self.S-1):
            self.model.addConstr(
                self.peak_bin_usage[s] >= self.peak_bin_usage[s+1],
                name=f"Partition_Size_Incr_{s}"
            )
        

    def define_obj(self):
        # Define the static wasted resources
        # $$\sum_{s}^{S} (T_{hp}*PeakUsage_{s} - \sum_{i=1}^{N} z_{i,s} * w_i)$$
        self.model.addConstr(
            self.static_wasted_resources == self.T * quicksum(self.peak_bin_usage[s] for s in range(self.S)) - sum(self.compute_lower_bounds),
            name="Static_Wasted_Resources_Constraint"
        )

        # Define the dynamic wasted resources
        # $$ \sum_{s}^{S} [PeakUsage_{s}**2 \cdot (\alpa + \beta * \sum_{i=1}^{N} z_{i,s}) ]
        self.scaled_peak_bin_usage = self.model.addVars(
            self.S,
            vtype=GRB.CONTINUOUS,
            name="scaled_peak_bin_usage",
        )
        self.model.addConstrs((self.scaled_peak_bin_usage[s] == self.peak_bin_usage[s] * self.size_scale for s in range(self.S)), name="Scaled_Peak_Bin_Usage_Constraint")
        for s in range(self.S):                    
            self.model.addGenConstrPoly(
                self.scaled_peak_bin_usage[s], self.realloc_overhead_bin[s], self.fit_params, 
                name = f"Dynamic_Wasted_Resources_Constraint_{s}"                
            )
        self.model.addConstr(
            self.dynamic_wasted_resources == self.T * quicksum(self.realloc_overhead_bin[s] for s in range(self.S)), 
            name="Dynamic_Wasted_Resources_Constraint"
        ) 
        
        for i in range(self.N): 
            self.model.addConstrs(
                (((self.resources[j]+self.resources[i])* self.size_scale / 2)**2 *40/256 /100 == self.affinity_overhead[i,j] for i,j in self.dependencies), 
                name=f"Affinity_Overhead_Constraint"
            )
        self.model.addConstrs(((self.spatial_affinity[i,j]) *self.temporal_affinity_bool[i,j]  == self.st_affinity_bool[i,j] for i,j in self.dependencies), name="def_ST_Affinity_Bool")
        self.model.addConstr(self.affinity_overhead_sum == quicksum((1-self.st_affinity_bool[i,j])  * self.affinity_overhead[i,j] 
                                                                    for i,j in self.dependencies), name="def_Affinity_Overhead_Sum")         
        self.model.setObjectiveN(self.static_wasted_resources + self.dynamic_wasted_resources+self.affinity_overhead_sum, index=0, priority=1, name="total_wasted_resources")

        # Define affinity score, to be maximized
        # spatial affinity: Ture if tasks are assigned to the same partition, False otherwise
        self.model.addConstrs((self.spatial_affinity[i, j] == quicksum(self.is_task_in_bin[i, s] * self.is_task_in_bin[j, s] for s in range(self.S)) for i, j in self.dependencies), name="def_spatial_affinity")
        

        # self.model.addConstr(
        #     self.affinity_score == quicksum(self.spatial_affinity[i, j]*(1-(self.temporal_affinity[i, j])/self.T) for i, j in self.dependencies)/len(self.dependencies),
        #     name="def_affinity_score"
        # )
        # self.model.setObjectiveN(-self.affinity_score, index=0, priority=2, name="max_affinity_score")
        
        # # define speedup rate, to be minimized
        # # self.model.setObjectiveN(quicksum(self.speedup_rate.values())/len(self.path_info), index=3, priority=0, name="speedup_rate")
        
        # # define spatial redundancy rate, to be minimized
        # self.model.addConstrs((self.spatial_rda_ratio[s] * self.peak_bin_usage[s] == self.partition_size[s] for s in range(self.S)), name="spatial_redundancy_rate")
        # # self.model.setObjectiveN(quicksum(self.spatial_rda_ratio.values())/self.S, index=4, priority=0, name="spatial_redundancy_rate")
        
    def scatter_bin_usage(self):
        """
        evenly distribute paths to bins
        """
        # .start, .VarHintVal
        allocated = set()
        bin_id = 0
        for path, slack in self.path_info:
            for i in path:
                if i not in allocated:
                    for s in range(self.S):
                        if s != bin_id:
                            # self.is_task_in_bin[i, s].start = 0
                            self.is_task_in_bin[i, s].VarHintVal = 0
                            self.is_task_in_bin[i, s].VarHintPri = 50
                        else:
                            # self.is_task_in_bin[i, s].start = 1
                            self.is_task_in_bin[i, s].VarHintVal = 1
                            self.is_task_in_bin[i, s].VarHintPri = 50
                    allocated.add(i)
            bin_id = (bin_id + 1) % self.S
            

    def solve(self):
        """求解模型"""
        try:
            self.model.printStats()
            # self.model.Params.Aggregate = 2 
            # self.model.Params.Presolve = 2  
            # self.model.Params.MIPFocus = 1  # 1: balanced, 2: feasibility, 3: optimality
            # self.model.setParam('MIPGap', 0.80)
            p = self.model.presolve()
            p.printStats()

            # Optimize model
            def mycallback(model, where):
                if where == gp.GRB.Callback.MIPSOL:
                    print("\n" + "=" * 30 + "\n")
                    obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)  # 获取目标函数值
                    partition_size = [model.cbGetSolution(model.getVarByName(f"Max_Usage[{s}]")) for s in range(S)]
                    print(f"当前最优解的目标函数值: {obj}")
                    print(f"Partition Size: {partition_size}")
                    static_wasted_resources = model.cbGetSolution(model.getVarByName("Static_Wasted_Resources"))
                    dynamic_wasted_resources = model.cbGetSolution(model.getVarByName("Dynamic_Wasted_Resources"))
                    affinity_overhead_sum = model.cbGetSolution(model.getVarByName("Affinity_Overhead_Sum"))
                    print(f"Static Wasted Resources: {static_wasted_resources:.2f}")
                    print(f"Dynamic Wasted Resources: {dynamic_wasted_resources:.2f}")
                    print(f"Affinity Overhead Sum: {affinity_overhead_sum:.2f}")
                    spatial_affinity = {(i,j): model.cbGetSolution(model.getVarByName(f"spatial_affinity[{i},{j}]")) for i,j in dependencies}
                    temporal_affinity = {(i,j): model.cbGetSolution(model.getVarByName(f"temporal_affinity[{i},{j}]")) for i,j in dependencies}
                    temporal_affinity_bool = {(i,j): model.cbGetSolution(model.getVarByName(f"temporal_affinity_bool[{i},{j}]")) for i,j in dependencies}
                    st_affinity_bool = {(i,j): model.cbGetSolution(model.getVarByName(f"st_affinity_bool[{i},{j}]")) for i,j in dependencies}
                    affinity_overhead = {(i,j): model.cbGetSolution(model.getVarByName(f"Affinity_Overhead[{i},{j}]")) for i,j in dependencies}
                    affinity_overhead_sum = model.cbGetSolution(model.getVarByName(f"Affinity_Overhead_Sum"))

                    r_s_d_l = {}
                    for i in range(N):
                        assigned_partition = [s for s in range(S) if model.cbGetSolution(model.getVarByName(f"Task_In_Partition[{i},{s}]")) > 0.5][0]
                        res = model.cbGetSolution(model.getVarByName(f"Resources[{i}]"))
                        start = model.cbGetSolution(model.getVarByName(f"Start_Time[{i}]")) 
                        duration = model.cbGetSolution(model.getVarByName(f"Duration[{i}]")) 
                        end = model.cbGetSolution(model.getVarByName(f"End_Time[{i}]")) 
                        # exp_comp_time = [model.cbGetSolution(model.getVarByName(f"Is_Resource_j[{i},{j}]")) * A[i][j] for j in self.A[i].keys() if model.cbGetSolution(model.getVarByName(f"Is_Resource_j[{i},{j}]")) > 0.5][0]
                        r_s_d_l[i] = (assigned_partition, start, end, duration, i, res)
                    for assigned_partition, start, end, duration, i, res in sorted(r_s_d_l.values()):
                        print(f"Task {i}: Start={start:.2f}, End={end:.2f}, Duration={duration:.2f}, Resources={res:.2f}, Partition={assigned_partition}")
                    print("\n" + "=" * 30 + "\n") 

            self.model.optimize(mycallback)

            # build solution dic of (core, lat)
            # print(self.model.display())
            return self.get_results()
        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError:
            print('Encountered an attribute error')

    def get_results(self) -> Union[Tuple[List[float], Dict[int, Tuple[int, int, float, float]]], None]:
        """获取结果"""
        status = self.model.Status
        print('Status: %g' % status)
        if status == GRB.OPTIMAL:
            print('Obj: %g' % self.model.ObjVal)
            sel = {}
            r_s_d_l = {}
            for i in range(self.N):
                assigned_partition = [s for s in range(self.S) if self.is_task_in_bin[i, s].x > 0.5][0]
                res = self.resources[i].x
                start = self.start_times[i].x
                duration = self.durations[i].x
                exp_comp_time = [self.is_resource_j[i, j].x * self.A[i][j] for j in self.A[i].keys() if self.is_resource_j[i, j].x > 0.5][0]
                sel[i] = assigned_partition
                r_s_d_l[i] = (res, start, duration, exp_comp_time)
                if self.verbose:
                    print(f"Task {i}: Start={start:.2f}, Duration={duration:.2f}, Resources={res}, Partition={assigned_partition}")
            partition_size = [self.peak_bin_usage[s].x for s in range(self.S)]
            
            if self.verbose:
                print(f"Partition Size: {partition_size}")
                print(f"Static Wasted Resources: {self.static_wasted_resources.x:.2f}")
                print(f"Dynamic Wasted Resources: {self.dynamic_wasted_resources.x:.2f}")
                print(f"Affinity Score: {self.affinity_score.x}")
                print(f"Affinity 1: {sum(sum(self.is_task_in_bin[i, s].X * self.is_task_in_bin[j, s].X for s in range(self.S))/len(self.dependencies) for i, j in self.dependencies)}")
                print(f"Affinity 2 {sum(1-(self.start_times[j].X + self.T*self.is_edge_cross_period[i, j].X - self.end_times[i].X)/self.T for i, j in self.dependencies)/len(self.dependencies)}")
                # print(f"Utilization: {[self.utilization[s].x for s in range(self.S)]}")
                # print(f"Max/Min Utilization: {self.max_util.x:.2f}/{self.min_util.x:.2f}")
                # print(f"Speedup Rate: {sum(map(lambda x: self.speedup_rate[x].x, self.speedup_rate.keys()))/len(self.path_info):.2f}")
                # print(f"Spatial Redundancy Rate: {sum(map(lambda x: self.spatial_rda_ratio[x].x, self.spatial_rda_ratio.keys()))/self.S:.2f}")

            return partition_size, sel, r_s_d_l

        elif status == GRB.INF_OR_UNBD or status == GRB.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)

            # do IIS
            print('The model is infeasible; computing IIS')
            removed = []

            # Loop until we reduce to a model that can be solved
            self.model.computeIIS()
            self.model.write("conflict.ilp")  # save the conflict constraints to a file
            while True:
                self.model.computeIIS()
                print('\nThe following constraint cannot be satisfied:')
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print('%s' % c.ConstrName)
                        # Remove a single constraint from the model
                        removed.append(str(c.ConstrName))
                        self.model.remove(c)
                        break
                print('')

        else:
            print("No optimal solution found.")

def debug(model):
    is_task_in_bin = {(i,s): model.cbGetSolution(model.getVarByName(f"Task_In_Partition[{i},{s}]")) for i in range(N) for s in range(S)}
    usage_along_ticks = {(s,t): model.cbGetSolution(model.getVarByName(f"Usage_Along_Ticks[{s},{t}]")) for s in range(S) for t in range(2*N)}
    is_cross_period = {i: model.cbGetSolution(model.getVarByName(f"Is_Cross_Period[{i}]")) for i in range(N)}

    start_times = {i: model.cbGetSolution(model.getVarByName(f"Start_Time[{i}]")) for i in range(N)}
    durations = {i: model.cbGetSolution(model.getVarByName(f"Duration[{i}]")) for i in range(N)}
    end_times = {i: model.cbGetSolution(model.getVarByName(f"End_Time[{i}]")) for i in range(N)}
    resources = {i: model.cbGetSolution(model.getVarByName(f"Resources[{i}]")) for i in range(N)}

    resource_active_at_t = {(i,t): model.cbGetSolution(model.getVarByName(f"r_{i}_{t}")) for i in range(N) for t in range(2*N)}
    active_at_t = {(i,t): model.cbGetSolution(model.getVarByName(f"Active_At_T[{i},{t}]")) if t%N != i else 1-int(t//N) for i in range(N) for t in range(2*N) }
    critical_tick = [model.cbGetSolution(model.getVarByName(f"Start_Time[{i}]")) for i in range(N)] + [model.cbGetSolution(model.getVarByName(f"End_Time[{i}]")) for i in range(N)]
    Lsatisfied = {(i,t): model.cbGetSolution(model.getVarByName(f"L_Satisfied[{i},{t}]")) for i in range(N) for t in range(2*N) if t%N != i}
    Rsatisfied = {(i,t): model.cbGetSolution(model.getVarByName(f"R_Satisfied[{i},{t}]")) for i in range(N) for t in range(2*N) if t%N != i}
    active_at_t_cross_period = {(i,t): model.cbGetSolution(model.getVarByName(f"Active_At_T_Cross_Period[{i},{t}]")) for i in range(N) for t in range(2*N) if t%N != i}
    active_at_t_not_cross_period = {(i,t): model.cbGetSolution(model.getVarByName(f"Active_At_T_Not_Cross_Period[{i},{t}]")) for i in range(N) for t in range(2*N) if t%N != i}


if __name__ == '__main__':
    import math
    # fitting alpha and beta
    y= [0.27088, 0.122085, 0.0342525, 0.0159225, 0.0162775, 0.0123425, 0.01288, 0.0066175, 0.003925, 0.0017825,] 
    x= [1, 2, 4, 8, 10, 12, 16, 18, 20, 24] 
    # fit the curve of 590/x and y    
    alpha=-2.1530989409936505e-09 
    beta=2.048110498544728e-06

    # # Input data
    # T = 30  # Period
    # N = 5  # Number of tasks
    # M = 4  # Maximum number of resources
    # S = 2  # Number of partitions

    # A = {
    #     0: {1: 3, 2: 2, 3: 1},
    #     1: {1: 5, 2: 3, 3: 2},
    #     2: {1: 4, 2: 2, 3: 1},
    #     3: {1: 6, 2: 4, 3: 2},
    #     4: {1: 3, 2: 2, 3: 1},
    # }
    # # List of paths from source to sink
    # path_info = [
    #     ([0, 1, 2], 25), 
    #     ((3, 4), 25)
    # ]
    # dependencies = [(0, 1), (1, 2), (3, 4)]  # (i, j): Task i must complete before task j
    # compute_lower_bounds = [10, 15, 8, 12, 10]  # Lower bound of computational workload for each task
    # start_constraints = [0, None, None, 25, None]  # Earliest start time for each task (None means no constraint)
    # end_constraints = [None, None, 25, None, 50]  # Latest completion time for each task

    size_scaler = 10
    T = 100  # Period
    N = 4  # Number of tasks
    M = 100/size_scaler  # Maximum number of resources
    S = 7  # Number of partitions

    # tag to id
    # D,C,A,B -> 0,1,2,3
    typical_lat = {
        2: 20/size_scaler,
        3: 25/size_scaler,
        1: 15/size_scaler,
        0: 45/size_scaler,
    }
    bursty_lat = dict(zip(range(N), [55, 25, 30, 40]))
        
    typical_size = 25/size_scaler
    assumed_efficiency = 0.9
    size_list = {
        2: [10/size_scaler, 20/size_scaler, 30/size_scaler, 40/size_scaler, 50/size_scaler],
        3: [10/size_scaler, 20/size_scaler, 30/size_scaler, 40/size_scaler, 50/size_scaler],
        1: [10/size_scaler, 20/size_scaler, 30/size_scaler, 40/size_scaler, 50/size_scaler],
        0: [10/size_scaler, 20/size_scaler, 30/size_scaler, 40/size_scaler, 50/size_scaler],
        # 2: [10, 20, 30, 40],
        # 3: [10, 20, 30, 40],
        # 1: [5, 10, 20, 30],
        # 0: [5, 10, 20, 30],
    }
    
    # load = typical_size * typical_lat
    ld = {i: typical_size * typical_lat[i] for i in range(N)}
    # load[i] / j/(assumed_efficiency*int((j-typical_size)/5))
    # polar: -typical_size/ln(assumed_efficiency)
    # in this case: ~50
    A = {i: {j: math.ceil(ld[i] / j/(assumed_efficiency**int(j/typical_size - 1))*size_scaler)/size_scaler for j in size_list[i]} for i in range(N)}    
    ld_sample_var = 2.5
    ld_ppf_value = {i: bursty_lat[i]/typical_lat[i] for i in range(N)}
    # ld_ppf_value = dict(zip(range(N), (i * typical_size for i in [55, 25, 30, 40, ])))
    jitter_sample_var = 0
    jitter_ppf_value = {i: jitter_sample_var for i in range(N)}

    # Max(ppf(ld, x%), max(A[i][j] for j in A[i]))
    start_constraints = [jitter_ppf_value[0], jitter_ppf_value[1], jitter_ppf_value[2], None]  # Earliest start time for each task (None means no constraint)
    end_constraints = [90, None, None, 90]  # Latest completion time for each task
    compute_lower_bounds = [ld_ppf_value[i]* max([k*v for k,v in A[i].items()]) for i in range(N)]
    # compute_lower_bounds = [max(ld_ppf_value[i], max([k*v for k,v in A[i].items()])) for i in range(N)]
    
    # from logical graph to physical graph, scale the number of nodes
    scaling_factor= [3, 1, 2, 2]
    from functools import reduce
    start_constraints = reduce(lambda x,y: x+y, [[start_constraints[i]]*scaling_factor[i] for i in range(N)])
    end_constraints = reduce(lambda x,y: x+y, [([end_constraints[i]]*scaling_factor[i]) for i in range(N)])
    compute_lower_bounds = reduce(lambda x,y: x+y, [[compute_lower_bounds[i]]*scaling_factor[i] for i in range(N)])
    idx = sum(scaling_factor) - 1
    for i in range(N-1, -1, -1):
        for scale_idx in range(scaling_factor[i]-1, -1, -1):
            size_list[idx] = size_list[i]
            A[idx] = A[i]
            ld[idx] = ld[i]
            ld_ppf_value[idx] = ld_ppf_value[i]
            jitter_ppf_value[idx] = jitter_ppf_value[i]
            if start_constraints[idx] is not None:
                start_constraints[idx] += T/scaling_factor[i] * scale_idx
            idx -= 1
    N = sum(scaling_factor)

        
    # List of paths from source to sink
    path_info = [
        ([0], 90), # 1st frame
        ([1], 90), # 2nd frame
        ([2], 90), # 3rd frame
        ([3, 0], 90), # 1st frame
        ([4, 3, 0], 90), # 1st frame
        ([4, 6], 90), # 1st frame
        ([5, 7], 90), # 2nd frame
    ]
    for path in path_info:
        end_constraints[path[0][-1]] = start_constraints[path[0][0]] + path[1]
    dependencies = [
        (3, 0),
        (4, 3),
        (4, 6),
        (5, 7)
        ]  # (i, j): Task i must complete before task j


    # input of the test 
    test_input = {
        "N": N,
        "M": M,
        "S": S,
        # "NT": 2 * N + 2,
        "T": T,
        "A": A,
        "dependencies": dependencies,
        "compute_lower_bounds": compute_lower_bounds,
        "start_constraints": start_constraints,
        "end_constraints": end_constraints,
        "path_info": path_info,
        "spt_fmt": "float",
        "time_format": "float",
        "timestep_size": 1e-3,
        "fit_params": [alpha, beta, 0, 0, 0],
        "size_scale": 590/M,
    }

    print("Test Input:")
    print(test_input)
    
    # Initialize scheduler and solve
    scheduler = GurobiSemi2DClstMapping(**test_input)
    scheduler.solve()
