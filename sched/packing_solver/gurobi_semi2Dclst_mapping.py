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
    def __init__(self, N:int, M:Union[int, float], S:int, NT:int, T:float, A:Dict[int, int], dependencies:List[Tuple[int, int]],  
                 compute_lower_bounds:List[int], start_constraints:List[int], end_constraints:List[int], path_info:List[Tuple[Union[List[int], float]]],
                 time_format="float", timestep_size:float=None, spt_fmt="int", 
                 verbose=True
                 ):
        self.N:int = N # Number of tasks
        self.M:int = M # Maximum number of resources
        self.S:int = S # Number of partitions
        self.NT:int = NT # Number of discrete time steps
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
        self.spt_attr = {"lb": 1, "ub": M, "vtype": self.spt_fmt} # Space variable attributes
        self.slackR = 1e-9 if time_format == "float" else 1 # Slack interval right side, resource window is one time step ahead. So if the current moment is finished, the resource demand for the next moment is 0
        
        # Create model
        self.model = Model("Partitioned_Task_Scheduling")
        self._initialize_variables()
        self._add_constraints()
        self.model.setParam('NonConvex', 2)
        self.model.setParam('OutputFlag', int(verbose))
        self.verbose = verbose 

    def _initialize_variables(self):
        """初始化决策变量"""
        # ======================================================================
        # temporal grids
        self.critical_tick = self.model.addVars(self.NT, name="Critical_Tick", **self.t_attr) # Discrete time steps, used to calculate the maximum resource demand for the partition
        # partition size
        self.partition_size = self.model.addVars(self.S, name="Partition_Size", **self.spt_attr) # Partition capacity

        # Task variables: temporal placements
        self.start_times = self.model.addVars(self.N, name="Start_Time", **self.t_attr) # Task start times
        self.end_times = self.model.addVars(self.N, name="End_Time", **self.t_attr) # Task end times
        self.durations = self.model.addVars(self.N, name="Duration", **self.t_attr) # Task durations
        
        # Task variables: resource allocation
        self.resources = self.model.addVars(self.N, name="Resources", **self.spt_attr) # Number of resources assigned to each task

        # =================================================================================
        # Decision variables
        # whether the ith task starts at the tth start critical tick
        # =================================================================================
        self.is_start_tick = self.model.addVars(self.N, self.NT, vtype=GRB.BINARY, name="Is_Start_Node") 
        # whether the ith task finishes at the tth end critical tick
        self.is_end_tick = self.model.addVars(self.N, self.NT, vtype=GRB.BINARY, name="Is_End_Node")  
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
        self.active_at_t = self.model.addVars(
            # [(i, t) for i in range(self.N) for t in self.critical_tick],
            [(i, t) for i in range(self.N) for t in range(self.NT)],
            vtype=GRB.BINARY,
            name="Active_At_T",
        )
        self.peak_bin_usage = self.model.addVars(self.S, name="Max_Usage", vtype=GRB.INTEGER, lb=0) # Maximum usage of each partition
        
        # =================================================================================
        # Target variables
        # =================================================================================
        # Main: utilization of each partition
        self.utilization = self.model.addVars(self.S, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="Utilization")
        self.min_util = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'min_util')
        self.max_util = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'max_util')
        
        # Aux 1: speedup rate, chain by chain
        self.speedup_rate = {}
        for i in range(self.N):
            if self.end_constraints[i] is not None:
                self.speedup_rate[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"Speedup_Rate_{i}")
        
        # Aux 2: affinity score
        self.affinity_score = self.model.addVar(lb=-len(self.dependencies), ub=len(self.dependencies), vtype=GRB.CONTINUOUS, name="affinity_score")
                
        # Aux 3: ratio of peak usage and partition size
        self.spatial_rda_ratio = self.model.addVars(self.S, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="Spatial_RDA_Ratio")

        
    
    def _add_constraints(self):
        """添加约束条件"""

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
        # Define the allocation constraints in temporal grids
        # =================================================================================
        # self.critical_tick[t] >= self.critical_tick[t-1]+self.slackR
        # self.start_times[i] == [self.is_start_tick[i, t] * critical_tick[t] for t in range(self.NT)]
        # self.end_times[i] == [self.is_end_tick[i, t] * critical_tick[t] for t in range(self.NT)]
        # sum(self.is_start_tick[i, t] for t in range(self.NT)) == 1
        # sum(self.is_end_tick[i, t] for t in range(self.NT)) == 1
        self.add_tgrid_constrs()
        # =================================================================================

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
            self.model.addConstr(
                quicksum(self.is_task_in_bin[i, s] for s in range(self.S)) == 1,
                name=f"Partition_Assignment_{i}",
            )  # 每个任务只能分配到一个分区
        for i in range(self.N):
            self.model.addConstr(
                quicksum(self.is_start_tick[i, k] for k in range(self.NT)) == 1,
                name=f"Start_Node_Selection_{i}",
            )
            self.model.addConstr(
                quicksum(self.is_end_tick[i, k] for k in range(self.NT)) == 1,
                name=f"End_Node_Selection_{i}",
            )

    def add_amount_and_shape_constrs(self):
        for i in range(self.N):
            # only one implementation is selected
            self.model.addConstr(
                quicksum(self.is_resource_j[i, j] for j in self.A[i].keys()) == 1,
                name=f"Resource_Selection_{i}",
            ) 
            # Shape compatibility of allocated region and selected implementation
            # duration[i] >= sum(A_ij * is_resource_j)
            self.model.addConstr(
                self.durations[i] >= quicksum(self.is_resource_j[i, j] * self.A[i][j] for j in self.A[i].keys()),
                name=f"Duration_Constraint_{i}",
            ) 
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
                self.resources[i] * self.durations[i] >= self.compute_lower_bounds[i],
                name=f"Compute_Lower_Bound_{i}",
            )

    def add_tgrid_constrs(self):

        # ======================================================================
        # Grid constraints: increasing critical ticks
        # self.critical_tick[t] >= self.critical_tick[t-1]+self.slackR
        # ======================================================================
        for t in range(1, self.NT):
            self.model.addConstr(self.critical_tick[t] >= self.critical_tick[t-1]+self.slackR , name=f"Critical_Tick_Incr_{t}")
        # ======================================================================

        # =================================================================================
        # Define the constraints of critical ticks and the start and end time of each task
        # self.start_times[i] == [self.is_start_tick[i, t] * critical_tick[t] for t in range(self.NT)]
        # self.end_times[i] == [self.is_end_tick[i, t] * critical_tick[t] for t in range(self.NT)]
        # ======================================================================

        # =====================================
        # Introduce slack variable v_start and v_end, and linearize the start and end time constraints
        # =====================================
        v_start = self.model.addVars(self.N, self.NT, vtype=GRB.CONTINUOUS, lb=0, name="V_Start")
        v_end = self.model.addVars(self.N, self.NT, vtype=GRB.CONTINUOUS, lb=0, name="V_End")

        M = self.T  # 大数 M，需确保大于 critical_tick 的上界
        for i in range(self.N):
            for t in range(self.NT):
                # v_start 约束
                self.model.addConstr(v_start[i, t] <= self.critical_tick[t], name=f"V_Start_Upper_{i}_{t}")
                self.model.addConstr(v_start[i, t] <= self.is_start_tick[i, t] * M, name=f"V_Start_Binary_Upper_{i}_{t}")
                self.model.addConstr(v_start[i, t] >= self.critical_tick[t] - (1 - self.is_start_tick[i, t]) * M, name=f"V_Start_Binary_Lower_{i}_{t}")
                
                # v_end 约束
                self.model.addConstr(v_end[i, t] <= self.critical_tick[t], name=f"V_End_Upper_{i}_{t}")
                self.model.addConstr(v_end[i, t] <= self.is_end_tick[i, t] * M, name=f"V_End_Binary_Upper_{i}_{t}")
                self.model.addConstr(v_end[i, t] >= self.critical_tick[t] - (1 - self.is_end_tick[i, t]) * M, name=f"V_End_Binary_Lower_{i}_{t}")

        for i in range(self.N):
            self.model.addConstr(self.start_times[i] == quicksum(v_start[i, t] for t in range(self.NT)), name=f"Start_Time_Definition_{i}")
            self.model.addConstr(self.end_times[i] == quicksum(v_end[i, t] for t in range(self.NT)), name=f"End_Time_Definition_{i}")

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
        M = self.T # Big M, uppper bound of the time
        for i, j in self.dependencies:
            self.model.addConstr(
                self.start_times[j] >= self.end_times[i] - M * self.is_edge_cross_period[i, j],
                name=f"Edge_No_Cross_{i}_{j}"
            )
            self.model.addConstr(
                self.start_times[j] + self.T >= self.end_times[i] - M * (1 - self.is_edge_cross_period[i, j]),
                name=f"Edge_Cross_{i}_{j}"
            )

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
            self.model.addConstr(self.speedup_rate[path[-1]] * (n_cross * self.T + self.end_times[path[-1]]) == self.end_constraints[final_node], 
                                 name=f"Speedup_Rate_Constraint_{path[0]}_{path[-1]}")
            

    def add_bin_constrs(self):
        # =================================================================================
        # (6) Partition related constraints:
        # """
        # 6-1. the summation of the resources used by each partition equals to the M 
        # 6-2. at any time, the maximum resource usage should never exceed the partition size
        # """
        # =================================================================================
        
        # 6-1. The summation of the resources used by each partition equals to the M 
        self.model.addConstr(
            quicksum(self.partition_size[s] for s in range(self.S)) == self.M,
            name=f"Partition_Size_Constraint"
        )
        
        # ======================================================================
        # 6-2. at any time, the maximum resource usage should never exceed the partition size
        # for each partition, we need to calculate sum along the spatial dim and calculate the max along the time dim
        # for s in range(self.S):
        #   max[sum(resources[i] * active_at_t[i, t] * is_task_in_bin[i, s] for i in range(self.N)) for t in self.critical_tick] <= partition_size[s]        # ======================================================================

        # add spatial-temporal constraints
        # if allocated on one partition, it should be active at least once
        for i in range(self.N):
            for s in range(self.S):
                self.model.addConstr(
                    quicksum(self.active_at_t[i, t] for t in range(self.NT)) >= self.is_task_in_bin[i, s],
                    name=f"Active_At_T_Partition_Constraint_{i}_{s}",
                )
        
        # step 1: replace active_at_t[i, t] * is_task_in_bin[i, s] with at_bin_t[i, s, t]
        self.at_bin_t = {}
        for i in range(self.N):
            for s in range(self.S):
                # for t in self.critical_tick:
                for t in range(self.NT):
                    self.at_bin_t[i, s, t] = self.model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{s}_{t}")

        for s in range(self.S):
            for t in range(self.NT):
                for i in range(self.N):
                    #  线性化 at_bin_t[i, s, t] = is_task_in_bin[i, s] * active_at_t[i, t]
                    self.model.addConstr(self.at_bin_t[i, s, t] <= self.is_task_in_bin[i, s], name=f"Z_Upper1_{i}_{s}_{t}")
                    self.model.addConstr(self.at_bin_t[i, s, t] <= self.active_at_t[i, t], name=f"Z_Upper2_{i}_{s}_{t}")
                    self.model.addConstr(self.at_bin_t[i, s, t] >= self.is_task_in_bin[i, s] + self.active_at_t[i, t] - 1, name=f"Z_Lower_{i}_{s}_{t}")
                    # self.model.addConstr(self.at_bin_t[i, s, t] == self.is_task_in_bin[i, s] * self.active_at_t[i, t], name=f"Z_Linear_{i}_{s}_{t}")
                    
        # step 2: list allocated tasks at all time points
        # active_at_t: Binds to cross-cycle and non-cross-cycle logic via slack conditions.
        
        # t<=end_times[i]-self.slackR
        self.Rsatisfied = self.model.addVars(
            [(i, t) for i in range(self.N) for t in range(self.NT)],
            vtype=GRB.BINARY,
            name="R_Satisfied",
        )
        # t>=start_times[i]
        self.Lsatisfied = self.model.addVars(
            [(i, t) for i in range(self.N) for t in range(self.NT)],
            vtype=GRB.BINARY,
            name="L_Satisfied",
        )
        self.active_at_t_cross_period = self.model.addVars(
            [(i, t) for i in range(self.N) for t in range(self.NT)],
            vtype=GRB.BINARY,
            name="Active_At_T_Cross_Period",
        )
        self.active_at_t_not_cross_period = self.model.addVars(
            [(i, t) for i in range(self.N) for t in range(self.NT)],
            vtype=GRB.BINARY,
            name="Active_At_T_Not_Cross_Period",
        )
        for s in range(self.S):
            for t in range(self.NT):
                for i in range(self.N):
                    tick = self.critical_tick[t]
                    self.model.addConstr(
                        self.start_times[i] <= tick + (1 - self.Lsatisfied[i, t]) * 1000,  # 1000 is a big M value to make sure the constraint is satisfied
                        name=f"Start_Active_Constraint_{i}_{t}",
                    )
                    self.model.addConstr(
                        tick <= self.end_times[i] - self.slackR + (1 - self.Rsatisfied[i, t]) * 1000,
                        name=f"End_Active_Constraint_{i}_{t}",
                    )
        
        # active_at_t_not_cross_period (non-cross-hyperperiod)：
        # active_at_t[i, t] = (start_times[i] <= t < start_times[i] + durations[i]) 
        for s in range(self.S):
            for t in range(self.NT):
                for i in range(self.N):
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
        for s in range(self.S):
            for t in range(self.NT):
                for i in range(self.N):
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
            vtype=GRB.INTEGER,
            name="Usage_Along_Ticks",
        )

        # Bringing at_bin_t[i, s, t] into 
        # max[sum(resources[i] * active_at_t[i, t] * is_task_in_bin[i, s] for i in range(self.N)) for t in self.critical_tick] <= partition_size[s]
        # we can get:
        # max[sum(resources[i] * at_bin_t[i, s, t] for i in range(self.N)) for t in self.critical_tick] <= partition_size[s]
        
        # define usage_along_ticks[s, t] == sum(resources[i] * at_bin_t[i, s, t] for i in range(self.N)) for t in self.critical_tick
        # define peak_bin_usage[s] == max(usage_along_ticks[s, t] for t in self.critical_tick)
        for s in range(self.S):
            for t in range(self.NT):
                # 在时间 t 上分区 s 的资源需求
                self.model.addConstr(
                    self.usage_along_ticks[s, t] == quicksum(self.resources[i] * self.at_bin_t[i, s, t] for i in range(self.N)),
                    name=f"Usage_Along_Ticks_{s}_{t}",
                )
            # Max usage in each partition
            self.model.addGenConstrMax(
                self.peak_bin_usage[s], [self.usage_along_ticks[s, t] for t in range(self.NT)],
                name=f"Max_Usage_{s}",                    
            )

        # Bring peak_bin_usage[s] into the objective function, we can get:
        # peak_bin_usage[s] <= partition_size[s]
        self.model.addConstrs((self.peak_bin_usage[s] <= self.partition_size[s] for s in range(self.S)), name="Peak_Bin_Usage_Constraint")

    def define_obj(self):
        # Define metric: utilization of each partition
        for s in range(self.S):
            self.model.addConstr(
                self.utilization[s] * self.T * self.partition_size[s] == quicksum(self.compute_lower_bounds[i]*self.is_task_in_bin[i, s] for i in range(self.N)),
                name=f"Utilization_{s}",
            )
        # if the partition is not used, its utilization is 0
        self.model.addConstrs((self.utilization[s] <= self.partition_size[s] for s in range(self.S)), name="Zero_Utilization_Constraint")
        # Define max/min utilization
        self.model.addGenConstrMax(
            self.max_util, [self.utilization[s] for s in range(self.S)],
            name=f"Max_Utilization_Constraint",
        )
        self.model.addGenConstrMin(
            self.min_util, [self.utilization[s] for s in range(self.S)],
            name=f"Min_Utilization_Constraint",
        )

        # Objective function: minimize the utilization difference among partitions
        # self.model.setObjective(quicksum(self.peak_bin_usage[s] for s in range(self.S)), GRB.MINIMIZE)
        self.model.setObjectiveN(self.max_util-self.min_util,index=1, priority=2, name="min_tot_size") 

        # Define affinity score, to be maximized
        # +1 if tasks connected by an edge (i, j) is assigned to the same partition
        self.model.addConstr(
        quicksum(quicksum(self.is_task_in_bin[i, s] * self.is_task_in_bin[j, s] for s in range(self.S)) for i, j in self.dependencies)/len(self.dependencies) == self.affinity_score,
            name="affinity_score")
        # self.model.setObjectiveN(-self.affinity_score, index=2, priority=1, name="max_affinity_score")
        # self.model.setObjective(self.affinity_score, GRB.MAXIMIZE)
        
        # define speedup rate, to be minimized
        # self.model.setObjectiveN(quicksum(self.speedup_rate.values())/len(self.path_info), index=3, priority=0, name="speedup_rate")
        
        # define spatial redundancy rate, to be minimized
        self.model.addConstrs((self.spatial_rda_ratio[s] * self.peak_bin_usage[s] == self.partition_size[s] for s in range(self.S)), name="spatial_redundancy_rate")
        # self.model.setObjectiveN(quicksum(self.spatial_rda_ratio.values())/self.S, index=4, priority=0, name="spatial_redundancy_rate")
        
        

    def solve(self):
        """求解模型"""
        try:
            # Optimize model
            self.model.optimize()

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
                print(f"Utilization: {[self.utilization[s].x for s in range(self.S)]}")
                print(f"Max/Min Utilization: {self.max_util.x:.2f}/{self.min_util.x:.2f}")
                print(f"Affinity Score: {self.affinity_score.x}")
                print(f"Speedup Rate: {sum(map(lambda x: self.speedup_rate[x].x, self.speedup_rate.keys()))/len(self.path_info):.2f}")
                print(f"Spatial Redundancy Rate: {sum(map(lambda x: self.spatial_rda_ratio[x].x, self.spatial_rda_ratio.keys()))/self.S:.2f}")

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
        
if __name__ == '__main__':
    import math
    # Input data
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

    T = 100  # Period
    N = 4  # Number of tasks
    M = 200  # Maximum number of resources
    S = 2  # Number of partitions

    # tag to id
    # D,C,A,B -> 0,1,2,3
    typical_lat = {
        2: 20,
        3: 25,
        1: 15,
        0: 45,
    }
    typical_size = 25
    assumed_efficiency = 0.9
    size_list = {
        2: [10, 20, 30, 40],
        3: [10, 20, 30, 40],
        1: [5, 10, 20, 30],
        0: [5, 10, 20, 30],
    }
    
    # load = typical_size * typical_lat
    ld = {i: typical_size * typical_lat[i] for i in range(N)}
    # load[i] / j/(assumed_efficiency*int((j-typical_size)/5))
    A = {i: {j: math.ceil(ld[i] / j/(assumed_efficiency**int((j-typical_size)/5))) for j in size_list[i]} for i in range(N)}    
    ld_sample_var = 2.5
    ld_ppf_value = dict(zip(range(N), (i * typical_size for i in [55, 25, 30, 40, ])))
    jitter_sample_var = 0
    jitter_ppf_value = {i: jitter_sample_var for i in range(N)}

    # Max(ppf(ld, x%), max(A[i][j] for j in A[i]))
    start_constraints = [jitter_ppf_value[0], jitter_ppf_value[1], jitter_ppf_value[2], None]  # Earliest start time for each task (None means no constraint)
    end_constraints = [90, None, None, 90]  # Latest completion time for each task
    compute_lower_bounds = [max(ld_ppf_value[i], max([k*v for k,v in A[i].items()])) for i in range(N)]
    
    # from logical graph to physical graph, scale the number of nodes
    scaling_factor= [3, 1, 2, 2]
    from functools import reduce
    start_constraints = reduce(lambda x,y: x+y, [[start_constraints[i]]*scaling_factor[i] for i in range(N)])
    end_constraints = reduce(lambda x,y: x+y, [([end_constraints[i]]*scaling_factor[i]) for i in range(N)])
    compute_lower_bounds = reduce(lambda x,y: x+y, [[compute_lower_bounds[i]]*scaling_factor[i] for i in range(N)])
    idx = sum(scaling_factor) - 1
    for i in range(N-1, -1, -1):
        for scale_idx in range(scaling_factor[i]):
            size_list[idx] = size_list[i]
            A[idx] = A[i]
            ld[idx] = ld[i]
            ld_ppf_value[idx] = ld_ppf_value[i]
            jitter_ppf_value[idx] = jitter_ppf_value[i]
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
        "NT": 2 * N + 2,
        "T": T,
        "A": A,
        "dependencies": dependencies,
        "compute_lower_bounds": compute_lower_bounds,
        "start_constraints": start_constraints,
        "end_constraints": end_constraints,
        "path_info": path_info,
        "time_format": "int",
        "timestep_size": 1,
    }

    print("Test Input:")
    print(test_input)
    
    # Initialize scheduler and solve
    scheduler = GurobiSemi2DClstMapping(**test_input)
    scheduler.solve()
