import networkx as nx
from queue import Queue
from task.task_cfg import creat_logical_graph
from example.bm4 import swt_lat
from collections import defaultdict
from typing import OrderedDict, Callable, Set, List
import math
from utils import core_distr
import functools
import types, json
from global_var import FLOPS_PER_CORE

time_unit = 1 
class MyGraph(nx.DiGraph):
    def __init__(self, srcs, ops, sinks, task_attr, src_attr, sink_attr=None):
        super(MyGraph, self).__init__()
        for src_n in srcs:
            self.add_node(src_n, type="src", **src_attr[src_n])
            for op_n in srcs[src_n]:
                self.add_node(op_n)
                self.add_edge(src_n, op_n, type="control")
        for op_n in ops:
            self.add_node(op_n, type="op", **task_attr[op_n])
            for sink_n in ops[op_n]:
                self.add_node(sink_n)
                self.add_edge(op_n, sink_n, type="data")
        for sink_n in sinks:
            if sink_attr is None:
                self.add_node(sink_n, type="sink")
            else:
                self.add_node(sink_n, type="sink", **sink_attr[sink_n])
            
        # self.n_pred_map trackes the non-ready, intermediate tasks in the graph
        # The value of n_pred_map is updated when tasks in graph are finished
        # A item is removed from the queue when it is moved to the ready queue
        # srcs are initially excluded from n_pred_map
        self.n_pred_map =  {node:len(list(self.predecessors(node))) for node in self.nodes() if node not in srcs}
        
        # self.srcs trackes the non-active srcs, 
        # which are removed when they are activated by external events
        self.srcs = [node for node in self.nodes() if node in srcs]

        self.sinks = [node for node in self.nodes() if node in sinks]
        self.ops = [node for node in self.nodes() if node in ops]
            

    def mark_finish(self, node):
        for succ in self.successors(node):
            self.n_pred_map[succ] -= 1
        self.remove_edges_from([(node, succ) for succ in self.successors(node)])
        self.remove_node(node)
        
    def mark_ready(self, node):
        self.n_pred_map.pop(node)


class GlobalEvent_t:
    def __init__(self, event_t) -> List:
        """
        Renew the event_t, which is a list of (event_time, event_type)
        assert event_type in ["external"]
        We enforce an event at inf is inserted 
        """
        self.event_t = []
        self.event_t.extend(event_t)
        self.event_t.append((float("inf"), "external"))
        self.event_t = list(set(self.event_t))
        self.event_t.sort()

    def get_next_event_time(self, curr_t):
        """返回下一个事件时间和类型"""
        # if not self.event_t:
        #     return float('inf'), 'finish'
        # 只返回大于当前时刻的事件
        # A good math format:
        # future_events = [e for e in self.event_t if e[0] > curr_t]
        # A high efficiency format:
        if self.event_t[0][0] <= curr_t:
            self.event_t.pop(0)
        # We assume an event at inf is all ready inserted to the event_t
        # if not self.event_t:
        #     return float('inf'), 'finish'
        next_timer_event = min(self.event_t, key=lambda x:x[0])
        return next_timer_event

class BaseProcessor:
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set=None):
        # 统一的基础属性
        self.id = id
        self.base_pwr = base_pwr
        self.G_ptr = G
        self.cap = cap
        self.mapped_node = mapped_node
        self.policy = 'N/A'


    def update_run(self, pred_t, curr_t) -> bool:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError
    
    def update_ready(self, curr_t) -> List:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError
    
    def sched(self, curr_t, **kwargs) -> float:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError
    
    def predict_next(self, curr_t) -> float:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError

    def __str__(self):
        return f"{self.id}, cap={self.cap}, base_pwr={self.base_pwr}, policy={self.policy}, mapped_ops={len(self.mapped_node)})"
    
    def __repr__(self):
        return self.__str__()


class Sen_p(BaseProcessor):
    """ Execution model of multi-sequential processor (MSSP), 
        where each processor only serves one task at a time,
        and scheduler will determine the which tasks are served, 
        and which processor is responsible for each task. 
    """
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set):
        super(Sen_p, self).__init__(id, cap, base_pwr, G, mapped_node)
        self.running = dict()
        self.ready = Queue(-1)
        self.policy = 'FCFS'

    def update_run(self, pred_t, curr_t) -> bool:
        for node in list(self.running.keys()):
            rem_t  = self.running[node] - (curr_t - pred_t) * self.base_pwr
            if rem_t <= 0:
                self.G_ptr.mark_finish(node)
                self.running.pop(node)
                print(f"\tsrc {node} arrives at {curr_t}")             
            else:
                self.running[node] = rem_t
        return False # No new task completion in Sen_p

    def update_ready(self, curr_t) -> List:
        """
        Scan mapped_nodes -> check if the node is ready -> 
        remove the node from the scan list -> 
        process the ready node depending their type and loads
        """
        # for node in list(self.G_ptr.srcs):
        for node in list(self.mapped_node):
            if curr_t == self.G_ptr.nodes[node]["offset"]:
                load = self.G_ptr.nodes[node]["exp_comp_t"]
                if load > 0:
                    self.ready.put((node, load))
                    print(f"\tsrc {node} is triggered at {curr_t}")
                else:
                    self.G_ptr.mark_finish(node)
                    print(f"\tsrc {node} arrives at {curr_t}")
                # self.G_ptr.srcs.remove(node)
                self.mapped_node.remove(node)
        return [] # No new ready tasks in Sen_p

    def sched(self, curr_t, **kwargs) -> float:
        """
        FCFS policy: serves new tasks until the last task is finished.
        Predicts the next event in this queue.
        """
        while not self.ready.empty() and len(self.running) < self.cap:
            node, rem_t = self.ready.get()
            self.running[node] = rem_t
            print(f"\tsen {self.id} starts task {node} at {self.G_ptr.nodes[node]['offset']}")

        # predict the next event in this queue
        duation_sen_p = float("inf")
        for node in self.running:
            duation_sen_p = min(duation_sen_p, self.running[node])
        if duation_sen_p == float("inf"):
            print(f"\tNo sensor event in future at {curr_t}")
        else:
            print(f"\tNext sensor event at {curr_t + duation_sen_p}")
        return duation_sen_p

class Acc_p(BaseProcessor):
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set):
        super(Acc_p, self).__init__(id, cap, base_pwr, G, mapped_node)

        # all keys in res_map should be in running
        # running: records the remaining load of task that is selected to be issued, including the system task "R"
        self.running = dict() 
        # res_map: records the actual resource allocation of each task, which only contains 'R' if reallocation is triggered
        self.res_map = dict()
        self.ready = dict()
        self.starving = False
        self.sys_state = "S"
        self.alloc_map_curr = dict()
        self.sys_state_emu = ["S", "R"]
        self.slack_map = dict()
        self.static_schedule_map = None
        # These will be bound by the factory function
        self.alloc_fn: Callable
        self.trigger_cond: Callable
            
    def update_run(self, pred_t, curr_t) -> bool:
        """Updates the remaining workload for running tasks and checks for finishing events."""
        new_complete_flag = False
        for node in list(self.res_map.keys()):
            rem_t  = self.running[node] - (curr_t - pred_t) * self.res_map[node] * self.base_pwr
            if rem_t <= 0:
                if node != "R":
                    self.G_ptr.mark_finish(node)
                    new_complete_flag |= True
                self.running.pop(node)
                self.res_map.pop(node)
                print(f"\tTask {node} finishes at {curr_t}") 
            else:
                self.running[node] = rem_t
        return new_complete_flag

    def update_ready(self, curr_t) -> List:
        """
        Puts newly ready tasks into the ready queue.
        Scan mapped_nodes -> check if the node is ready -> 
        remove the node from the scan list -> 
        process the ready node depending their type and loads
        """
        new_ready_list = []
        # for node, n_pred in list(self.G_ptr.n_pred_map.items()):
        for node in list(self.mapped_node):
            if not node in self.G_ptr.n_pred_map:
                continue
            n_pred = self.G_ptr.n_pred_map[node]
            if n_pred == 0:
                if node in self.G_ptr.sinks:
                    self.G_ptr.mark_finish(node)
                    print(f"\tsink {node} finish at {curr_t}")
                elif node in self.G_ptr.ops:
                    # illegal check
                    if node in self.running or node in self.ready:
                        assert False, "task should not be in running or ready queue"
                    self.ready[node] = self.G_ptr.nodes[node]["exp_comp_t"]* self.G_ptr.nodes[node]["base_size"]
                    new_ready_list.append(node)
                    print(f"\ttask {node} ready at {curr_t}")
                self.mapped_node.remove(node)
                self.G_ptr.mark_ready(node)
        return new_ready_list

    def sched(self, curr_t, new_comp, new_ready_list, **kwargs) -> float:
        """
        Allocation progress: 
            Check the event type: 
            If the event informs the completion of reallocation, 
            scheduler will directly use the cached allocation map. 
            Otherwise, scheduler will generate a new allocation map, 
            during which the allocation map in last round will be renamed as alloc_map_prev. 
            
            Two types of tasks: 
            - "system task" that stalls the accelerator
            - "user task" that can be executed by the accelerator
        """         
        realloc = self.trigger_cond(new_comp, new_ready_list)
        
        # These two lines belong to allocation progress. 
        # However, in order to maintain the running/ready queues, 
        # these two lines of logic are deliberately placed outside the conditional statement, which is not elegant.
        self.alloc_map_curr = self.alloc_fn(curr_t, realloc)
        self.update_queue()

        self.res_map.clear()
        if realloc: 
            self.running["R"] = swt_lat
            self.res_map.update({"R": 1})
        else:
            self.res_map.update(self.alloc_map_curr) 
        
        # predict the next event in this queue
        duation_acc_p = self.predict_next(curr_t)
        # state display 
        if self.sys_state == "R":
            print(f"\tEnter reallocation progress at {curr_t}")
            type_ = "reallocate"
        else:
            print(f"\texist reallocation state at {curr_t}")   
            type_ = "finish" 
        # TODO: the event info is not accurate, should be removed in the future.
        # if duation_acc_p == float("inf"):
        #     print(f"\tNo accelerator event in future at {curr_t}")
        # else:
        #     print(f"\tNext accelerator {type_} event at {curr_t + duation_acc_p}")

        return duation_acc_p
    
    def update_queue(self):
        """
        Moves tasks between running and ready queues based on the current allocation map.
        Tasks not in alloc_map_curr are preempted (moved to ready).
        Tasks in alloc_map_curr are moved to running.
        """
        for node in list(self.running.keys()):
            if node != "R" and node not in self.alloc_map_curr: 
                self.ready[node] = self.running.pop(node)
                print(f"\tTask {node} preempted and moved to ready queue.")
            
        for node in list(self.ready.keys()):
            if node in self.alloc_map_curr: 
                self.running[node] = self.ready.pop(node)
                print(f"\tTask {node} moved from ready to running queue.")

    def predict_next(self, curr_t) -> float:
        """Helper to predict the next event duration based on current resource map."""
        # predict the next event in this queue
        duation_acc_p = min([self.running[pid]/self.res_map[pid] 
                                        for pid in self.res_map  if self.res_map[pid] > 0]) if self.res_map else float("inf")
        if duation_acc_p < float("inf"):
            duation_acc_p = math.ceil(duation_acc_p * time_unit) / time_unit 
        return duation_acc_p

def trigger_cond_dyn(self, new_comp, new_ready_list):
    """_summary_
    compare the priorities of the new ready tasks with the running tasks:
    Main logic: 
        Under the assumption that all tiles are busy, reallocation is involed if and only if preemption is needed: 
            if the priority of the new ready tasks is smaller than the running tasks, trigger is need. 
        If there are idle tiles, reallocation is needed if there are tasks need more resources:
            1. ready tasks
            2. the running task with the lowest priority is starved.
    Method: 
        we use running as a mask to detect the reallcation progress, and res_map to detect the free tiles.
    """
    # cond1 = new_comp
    # cond2 = len(self.running) == 0 and len(new_ready_list) > 0
    # cond3 = len(self.running) > 0 and len(new_ready_list) > 0 \
    #     and min(new_ready_list, key=lambda x:self.G_ptr.nodes[x]['ddl']) < \
    #         max(self.running, key=lambda x:self.G_ptr.nodes[x]['ddl'])
    # cond = cond1 or cond2 or cond3

    # get free tiles
    free_tiles = self.cap - sum(self.res_map.values())
    # case 1: no free tiles
    cond1 = (free_tiles <= 0) and len(new_ready_list) > 0 \
        and min(new_ready_list, key=lambda x:self.G_ptr.nodes[x]['ddl']) < \
            max(self.running, key=lambda x:self.G_ptr.nodes[x]['ddl'])
    # case 2: free tiles
    cond2 = (free_tiles > 0) and (len(new_ready_list) > 0 or self.starving) 
    
    cond = cond1 or cond2      
    if cond:
        realloc = True
        self.sys_state = "R"
    else:
        realloc = False
        self.sys_state = "S"
    return realloc


def alloc_fn_pglb(acc_p, curr_t, realloc=True, 
                  # static parameters, which will be removed by lambda or functools.partial
                  reserv_en=False):
    """
    Allocation function for reservation-aware scheduler.
    Only allocates minimum required resources, respects EST.
    """

    # calculate slack 
    realloc_slack = 0 if not realloc else swt_lat
    if not reserv_en:
        acc_p.slack_map = {node:acc_p.G_ptr.nodes[node]['ddl'] - curr_t - realloc_slack
                        for node in list(acc_p.running.keys()) + list(acc_p.ready.keys())} 
    else:
        # In reservation-aware scheduler, only tasks with ERT >= curr_t can be allocated.
        acc_p.slack_map = {node:acc_p.G_ptr.nodes[node]['ddl'] - curr_t - realloc_slack
                        for node in list(acc_p.running.keys()) + list(acc_p.ready.keys()) if acc_p.G_ptr.nodes[node]['ert'] >= curr_t} 
        
    score = acc_p.slack_map.copy()
    alloc_map_curr = {}
    acc_p.starving = False
    # calculate min_rsc requirement
    score_dict = OrderedDict(); constr_dict = OrderedDict()
    curr_aval_rsc = acc_p.cap
    for node in sorted(score, key=score.get):
        slack = score[node]
        if slack <= 0:
            print(f"\t{node} is timeout at {curr_t}")
            req_rsc_size = curr_aval_rsc
        else:
            assert not (node in acc_p.ready and node in acc_p.running) 
            req_rsc_size = math.ceil((acc_p.running.get(node, 0) + acc_p.ready.get(node, 0))/slack)
            if req_rsc_size > curr_aval_rsc:
                print(f"\t{node} is hungry at {curr_t}: lack {req_rsc_size - curr_aval_rsc} tiles") 
                req_rsc_size = curr_aval_rsc
                acc_p.starving = True and (not realloc)
                
        curr_aval_rsc -= req_rsc_size
        alloc_map_curr[node] = req_rsc_size
        constr_dict[node] = "N/A"
        score_dict[node] = 1/score[node] if score[node] >0 else float("inf")
        score.pop(node)
        if curr_aval_rsc <= 0:
            break
            
    # allocate free resource
    # if there are still resources left, 
    # it means no late process is waiting for resources
    if curr_aval_rsc > 0 and len(alloc_map_curr) and not reserv_en:
        print(f"\tMinimum resource requirement at {curr_t}: {alloc_map_curr}")
        assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
        # also, there is no process waiting for resources in the ready queue
        assert len(score) == 0
        core_distr(alloc_map_curr, score_dict, curr_aval_rsc)
    
    if reserv_en and curr_aval_rsc > 0 and len(alloc_map_curr):
        print(f"\tReservation (en): {curr_aval_rsc} tiles reserved.") 
    return alloc_map_curr

# Cyclic specific alloc_fn and trigger_cond
def trigger_cond_cyclic(acc_p, new_comp, new_ready_list, curr_t, static_schedule_map=None):
    """
    Trigger condition for cyclic scheduler: 
        New ready tasks are in current map, or curr_t > next time in static_schedule_map.
    """    
    return False

def alloc_fn_cyclic(acc_p, curr_t, static_schedule_map=None, force=False):
    """
    Allocation function for cyclic scheduler.
    Allocates resources based on a pre-defined static schedule map.
    :param acc_p: The Acc_p instance.
    :param curr_t: Current time.
    :param static_schedule_map: A nested dictionary {time: {task_id: tile_count}}.
                                 Expected to be passed via the main sched method.
    :param force: Boolean, if True, asserts capacity is sufficient for static allocation.
    """
    alloc_map_curr = OrderedDict()

    if static_schedule_map is None:
        print("Error: 'static_schedule_map' must be provided for 'cyclic' policy's alloc_fn.")
        return alloc_map_curr

    current_static_allocation = static_schedule_map.get(curr_t)

    if force:
        if current_static_allocation:
            assert acc_p.cap >= sum(current_static_allocation.values()), \
                f"Cyclic scheduler (force=True): Capacity Error: {acc_p.cap} is less than required {sum(current_static_allocation.values())} at {curr_t}"
            alloc_map_curr.update(current_static_allocation)
        else:
            print(f"\tCyclic scheduler (force=True): No static allocation defined for time {curr_t}. Allocating nothing.")
    else:
        if current_static_allocation:
            print(f"\tCyclic scheduler: Applying static allocation for time {curr_t}: {current_static_allocation}")
            temp_aval_rsc = acc_p.cap
            for node, requested_rsc in current_static_allocation.items():
                if temp_aval_rsc <= 0:
                    break
                # ensure the expected task is ready or running
                if node in acc_p.running or node in acc_p.ready:
                    alloc_rsc = min(requested_rsc, temp_aval_rsc)
                    alloc_map_curr[node] = alloc_rsc
                    temp_aval_rsc -= alloc_rsc
                else:
                    print(f"\tCyclic scheduler: Task {node} not found in running/ready queues, skipping static allocation.")
        else:
            print(f"\tCyclic scheduler: No static allocation defined for time {curr_t}. Allocating nothing.")    
    return alloc_map_curr

# TODO: abstract the scheduler that follows some fixed schedule time events.

# factory function
class PartitionConfig:
    """
    用于描述所有分区的调度配置。
    Attributes:
        num_partitions: 分区数量
        cap_list: 每个分区的cap
        base_pwr_list: 每个分区的base_pwr
        G_list: 每个分区的MyGraph
        TSmap: 每个分区的static_schedule_map（可选）
        mapped_node_list: 每个分区的节点映射（可选）
    """
    def __init__(
        self, num_partitions, cap_list, base_pwr_list, 
        G, TSmap_list=None, mapped_node_list=None
        ):
        self.num_parts = num_partitions
        self.cap_list = cap_list
        self.base_pwr_list = base_pwr_list
        self.G = G
        self.TSmap_list = TSmap_list or [None]*num_partitions
        self.mapped_node_list = mapped_node_list or [None]*num_partitions


def acc_p_factory(
    policy: str,
    cfg: PartitionConfig,
    **kwargs
):
    """
    policy: pglb, glb, cyc, reserv
    cfg: PartitionConfig对象，包含所有分区的参数
    kwargs: 其他策略相关参数
    返回: acc_p实例列表
    """
    acc_p_list = []
    assert cfg.G is not None, "G_list is not None"
    for i in range(cfg.num_parts):
        # check the validity of the partition config
        assert cfg.cap_list[i] > 0, "cap_list is not None"
        assert cfg.base_pwr_list[i] > 0, "base_pwr_list is not None"
        assert cfg.mapped_node_list[i] is not None, "mapped_node_list is not None"
        
        # create acc_p instance
        acc_p = Acc_p(i, cfg.cap_list[i], cfg.base_pwr_list[i], cfg.G, cfg.mapped_node_list[i])
        
        # policy binding
        if policy in ["pglb", "glb"]:
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_pglb, reserv_en=False), acc_p)
            acc_p.trigger_cond = types.MethodType(trigger_cond_dyn, acc_p)
        elif policy in ["cyc"]:
            assert cfg.TSmap_list[i] is not None, "TSmap_list is not None"
            static_schedule_map = cfg.TSmap_list[i]
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_cyclic, static_schedule_map=static_schedule_map), acc_p)
            acc_p.trigger_cond = types.MethodType(functools.partial(trigger_cond_cyclic, static_schedule_map=static_schedule_map), acc_p)
        elif policy == "reserv":
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_pglb, reserv_en=True), acc_p)
            acc_p.trigger_cond = types.MethodType(trigger_cond_dyn, acc_p)
        else:
            raise ValueError(f"Unknown strategy: {policy}")
        acc_p_list.append(acc_p)
    return acc_p_list

def get_partition_info(bin_list):
    # 1. 分区的任务映射
    partition_task_map = {}
    for _bin in bin_list:
        task_set = set()
        for rsc_agent in _bin.scheduling_table:
            task_set.update(rsc_agent.rsc_map.keys())
        partition_task_map[_bin.name] = list(task_set)

    bin_name_list = [bin.name for bin in bin_list]
    # 2. 分区的大小
    partition_size = [_bin.num_resources for _bin in bin_list]

    # 3. cyclic方法下的静态调度表
    TSMap_list = []
    for _bin in bin_list:
        TSmap = []
        for cfg_slot_s, next_cfg, cfg_slot_num in getattr(_bin, "sparse_list", []):
            TSmap.append({
                "start_slot": cfg_slot_s,
                "alloc": next_cfg,
                "duration": cfg_slot_num
            })
        TSMap_list.append(TSmap)

    # 打印结果
    # print("Partition-Task Mapping:", partition_task_map)
    # print("Partition Size:", partition_size)
    # print("Cyclic Static Schedule:", cyclic_static_schedule)    

    return len(bin_list), partition_size, [FLOPS_PER_CORE]*len(bin_list), partition_task_map, TSMap_list 


# latency model: 
# 总执行时间分解为三个关键组成部分来建模延迟（执行时间）：计算、访存/传输和调度。

# 首先，资源分配过程是不aware 访存延迟和调度开销这些运行时数据的。调度器能看到的只有分区内带宽和tile两种资源。
# 任务的计算延迟：
# （NN 任务）等价算数操作数量*非阻塞执行时间占比/设备算力FLOPS 
# （其他程序）指令数量*CPI/CPU频率
# 为了同一抽象，这里统一为，
# 计算延迟=架构无关的操作数量/使用算力/(1-通信时间占比)

# 任务初始定义：基准指令数
# 设备定义：单位计算单元算力，计算单元容量
# 运行时参数：通信延迟占比

def load_graph_from_json(json_path):
    with open(json_path, 'r') as f:
        graph_data = json.load(f)

    # 假设json结构包含nodes和edges
    nodes = graph_data['nodes']
    edges = graph_data['edges']

    # 分类节点
    srcs, ops, sinks = {}, {}, {}
    task_attr, src_attr, sink_attr = {}, {}, {}

    # 先统计所有节点的入度和出度
    src_nodes = [n for n, x in graph_data.in_degree() if x == 0]
    sink_nodes = [n for n, x in graph_data.out_degree() if x == 0]

    # 分类
    for n in nodes:
        node_id = n['id']
        # 统一负载定义
        if 'flops' in n:
            exp_comp_t = n['flops']
        else:
            exp_comp_t = n['exp_comp_t']
        base_size = 1
        offset = n.get('offset', 0)
        # 源节点
        if node_id in src_nodes:
            # 找到所有后继
            succs = [e['target'] for e in edges if e['source'] == node_id]
            srcs[node_id] = succs
            src_attr[node_id] = {
                'offset': offset,
                'exp_comp_t': exp_comp_t,
                'base_size': base_size,
                # 'tgt_device': n.get('tgt_device', 'sen_p0')
            }
        # 汇节点
        elif node_id in sink_nodes:
            sinks[node_id] = []
            sink_attr[node_id] = {
                'exp_comp_t': exp_comp_t,
                'base_size': base_size,
                # 'tgt_device': n.get('tgt_device', 'sink')
            }
        # 中间节点
        else:
            succs = [e['target'] for e in edges if e['source'] == node_id]
            ops[node_id] = succs
            task_attr[node_id] = {
                'offset': offset,
                'exp_comp_t': exp_comp_t,
                'base_size': base_size,
                # 'tgt_device': n.get('tgt_device', 'acc_p0'),
                'ert': n.get('ert', 0),
                'ddl': n.get('ddl', 100)
            }

    return srcs, ops, sinks, task_attr, src_attr, sink_attr

# 实例化MyGraph
def instantiate_mygraph_from_json(json_path):
    srcs, ops, sinks, task_attr, src_attr, sink_attr = load_graph_from_json(json_path)
    G = MyGraph(srcs, ops, sinks, task_attr, src_attr, sink_attr)
    return G
