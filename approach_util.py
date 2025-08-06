import networkx as nx
from queue import Queue
from task.task_cfg import creat_logical_graph
# from example.bm4 import swt_lat
from collections import defaultdict
from typing import OrderedDict, Callable, Set, List
import math
from utils import core_distr, elim_nume_error
import functools
import types, json
from global_var import FLOPS_PER_CORE, GLB_BUFFER_SIZE_PER_CORE, BW_DRAM
from warnings import warn
from task_estimation import (
    estimate_task_execution_time, 
    estimate_resource_requirement, 
    normalize_time_to_unit,
    get_task_load_and_base_size,
    calculate_slack_time,
    estimate_task_progress
)

# TODO: These numbers are temporal magic numbers, which must be removed. 
old_timestep = 10e-6


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
        self.ddl_map = {node:task_attr[node]['ddl'] for node in self.ops}
        self.ddl_map.update({node:sink_attr[node]['ddl'] for node in self.sinks})
        self.ert_map = {node:task_attr[node]['ert'] for node in self.ops}
        self.ddl_map['R'] = -float('inf')
        self.ert_map['R'] = -float('inf')
            

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
            # rem_t  = self.running[node] - (curr_t - pred_t) * self.base_pwr
            rem_t = estimate_task_progress(self.running[node], curr_t - pred_t, 1, self.base_pwr)
            if rem_t <= 0:
                self.G_ptr.mark_finish(node)
                self.running.pop(node)
                print(f"\t[{self.id}] src {node} arrives at {curr_t}")             
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
                load = cal_load(self.G_ptr.nodes[node]["exp_comp_t"], self.G_ptr.nodes[node]["base_size"])
                if load > 0:
                    self.ready.put((node, load))
                    print(f"\t[{self.id}] src {node} is triggered at {curr_t}")
                else:
                    self.G_ptr.mark_finish(node)
                    print(f"\t[{self.id}] src {node} arrives at {curr_t}")
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
            print(f"\t[{self.id}] sen starts task {node} at {self.G_ptr.nodes[node]['offset']}")

        # predict the next event in this queue
        duation_sen_p = float("inf")
        for node in self.running:
            duation_sen_p = min(duation_sen_p, self.running[node])
        if duation_sen_p == float("inf"):
            print(f"\t[{self.id}] No sensor event in future at {curr_t}")
        else:
            print(f"\t[{self.id}] Next sensor event at {curr_t + duation_sen_p}")
        return duation_sen_p

class Acc_p(BaseProcessor):
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set):
        super(Acc_p, self).__init__(id, cap, base_pwr, G, mapped_node)

        # all keys in res_map should be in running
        # running: records the remaining load of task that is selected to be issued, including the system task "R"
        self.running = dict() 
        # res_map: records the actual resource allocation of each task, ** regardless system state**
        self.res_map = dict()
        self.ready = dict()
        self.starving = False
        self.sys_state = "S"
        self.sys_state_emu = ["S", "R"]
        self.slack_map = dict()
        # TODO: init the switch latency by cap and cap_sram, and DRAM_bw
        self.swt_lat: float
        self.static_schedule_map = None
        # These will be bound by the factory function
        self.alloc_fn: Callable
        self.trigger_cond: Callable
        self.curr_slot_index = 0
            
    def update_run(self, pred_t, curr_t) -> bool:
        """
        Updates the remaining workload for running tasks and checks for finishing events.
        All tasks depends on the system state, 
        the remining load is updated only if the system state is "S" 
        """
        new_complete_flag = False
        if self.sys_state == "R":
            assert "R" in self.running, \
                f"R should be in running at {curr_t}"
            assert "R" not in self.res_map, \
                f"R should not be in res_map at {curr_t}"
            rem_t = estimate_task_progress(self.running["R"], curr_t - pred_t, 1, 1)
            if rem_t <= 0:
                self.sys_state = "S"
                print(f"\t[{self.id}] exist reallocation state at {curr_t}") 
                self.running.pop("R")
                print(f"\t[{self.id}] Task R finishes at {curr_t}") 
            else:
                self.running["R"] = rem_t
        else:
            for node in list(self.res_map.keys()):
                # rem_t  = self.running[node] - (curr_t - pred_t) * self.res_map[node] * self.base_pwr
                rem_t = estimate_task_progress(self.running[node], curr_t - pred_t, self.res_map[node], self.base_pwr)
                if rem_t <= 0:
                    self.G_ptr.mark_finish(node)
                    new_complete_flag |= True
                    self.running.pop(node)
                    self.res_map.pop(node)
                    if curr_t > self.G_ptr.ddl_map[node]:
                        print(f"\t[{self.id}] {node} is timeout at {curr_t}")
                    print(f"\t[{self.id}] Task {node} finishes at {curr_t}") 
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
                    if curr_t > self.G_ptr.ddl_map[node]:
                        print(f"\t[{self.id}] {node} is timeout at {curr_t}")
                    print(f"\t[{self.id}] sink {node} finish at {curr_t}")
                elif node in self.G_ptr.ops:
                    # illegal check
                    if node in self.running or node in self.ready:
                        assert False, "task should not be in running or ready queue"
                    self.ready[node] = cal_load(self.G_ptr.nodes[node]["exp_comp_t"], self.G_ptr.nodes[node]["base_size"])
                    new_ready_list.append(node)
                    print(f"\t[{self.id}] task {node} ready at {curr_t}")
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

        # The elegent way to handle the reallocation progress,
        # but will not display which task are going to be allocated in the allocation function.
        # TODO: Not examined yet.
        # if realloc: 
        #     for node in list(self.running.keys()):
        #         if node != "R":
        #             self.ready[node] = self.running.pop(node)
        #             print(f"\t[{self.id}] Task {node} preempted and moved to ready queue.")
        #     self.running["R"] = self.swt_lat
        #     self.res_map.clear()
        #     self.res_map["R"] = 1
        # elif "R" in self.running:
        #     pass
        # else:
        #     alloc_map_curr = self.alloc_fn(curr_t, False)
        #     self.update_queue(alloc_map_curr)
        #     self.res_map.update(alloc_map_curr) 
        
        # These lines belong to allocation progress. 
        # However, in order to maintain the running/ready queues, 
        # these two lines of logic are deliberately placed outside the conditional statement, which is not elegant.
        # TODO: double check
        alloc_map_curr = self.alloc_fn(curr_t, realloc)
        self.res_map.clear()
        self.res_map.update(alloc_map_curr) 
        self.update_queue(alloc_map_curr)
        
        # predict the next event in this queue
        duation_acc_p = self.predict_next(curr_t)
        # state display 
        if self.sys_state == "R":
            print(f"\t[{self.id}] Enter reallocation progress at {curr_t}")
            type_ = "reallocate"
        else:
            type_ = "finish" 
        # TODO: the event info is not accurate, should be removed in the future.
        # if duation_acc_p == float("inf"):
        #     print(f"\tNo accelerator event in future at {curr_t}")
        # else:
        #     print(f"\tNext accelerator {type_} event at {curr_t + duation_acc_p}")

        return duation_acc_p
    
    def update_queue(self, alloc_map_curr):
        """
        Moves tasks between running and ready queues based on the current allocation map.
        Tasks not in alloc_map_curr are preempted (moved to ready).
        Tasks in alloc_map_curr are moved to running.
        """
        for node in list(self.running.keys()):
            if node != "R" and node not in alloc_map_curr: 
                self.ready[node] = self.running.pop(node)
                print(f"\t[{self.id}] Task {node} preempted and moved to ready queue.")
            
        for node in list(self.ready.keys()):
            if node in alloc_map_curr: 
                self.running[node] = self.ready.pop(node)
                print(f"\t[{self.id}] Task {node} moved from ready to running queue.")

    def predict_next(self, curr_t) -> float:
        """Helper to predict the next event duration based on current resource map."""
        if self.sys_state == "R":
            duation_acc_p = estimate_task_execution_time(self.running["R"], 1, 1)
        else:
            # predict the next event in this queue
            actual_running = [estimate_task_execution_time(
                        self.running[pid], 
                        self.res_map[pid], 
                        self.base_pwr
                    ) for pid in self.res_map  if self.res_map[pid] > 0]
            if not actual_running:
                return float("inf")
            duation_acc_p = min(actual_running)

        if duation_acc_p < float("inf"):
            duation_acc_p = normalize_time_to_unit(duation_acc_p, time_unit)
        return duation_acc_p

def trigger_cond_dyn(self, new_comp, new_ready_list):
    """_summary_
    The reallocation is triggered if and only if some tasks need more resources, and we can also find free tiles.
    Need more:
        - A. new ready tasks arrives
        - B. not allocated tasks, allocated by starving tasks
    Free tiles:
        - 1. running tasks are preempted
        - 2. finished tasks at the beginning of this round
        - 3. free tiles in the last round
            |   | 1 | 2 | 3 | 
            | A | o | √ | √ |
            | B | x | √ | - |
            x: must not trigger, √: must trigger, -: impossible case, o: on condition
    
    System state should not affect the judgement of reallocation. 
    Main logic: 
        - 1&A: Under the assumption that all tiles are busy, reallocation is involed if and only if preemption is triggered: 
            if the priority of the new ready tasks is smaller than the running tasks, trigger is need. 
        - 2+3) there are idle tiles, A) new ready tasks B) not allocated or starving tasks. 
        As 3B is impossible we can treat it as true, we have (2 or 3) & (A or B).
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
        and min(new_ready_list, key=lambda x:self.G_ptr.ddl_map[x]) < \
            max(self.running, key=lambda x:self.G_ptr.ddl_map[x])
    # case 2: free tiles
    cond2 = (free_tiles > 0) and (len(new_ready_list) > 0 or self.starving) 
    
    cond = cond1 or cond2      
    if cond:
        realloc = True
        self.sys_state = "R"
        self.running["R"] = cal_load(self.swt_lat, 1) 
    else:
        realloc = False
        # self.sys_state = "S"
        # the system exits reallocation progress until the "R" is finished
    return realloc


def alloc_fn_pglb(acc_p, curr_t, realloc=True, 
                  # static parameters, which will be removed by lambda or functools.partial
                  reserv_en=False):
    """
    Allocation function for reservation-aware scheduler.
    Only allocates minimum required resources, respects EST.
    """

    # calculate slack 
    realloc_slack = 0 if not realloc else estimate_task_execution_time(acc_p.swt_lat, 1, 1)
    if not reserv_en:
        acc_p.slack_map = {
            node: calculate_slack_time(acc_p.G_ptr.ddl_map[node], curr_t, realloc_slack)
            for node in list(acc_p.running.keys()) + list(acc_p.ready.keys())
            if node != "R"
        }
    else:
        # In reservation-aware scheduler, only tasks with ERT >= curr_t can be allocated.
        acc_p.slack_map = {
            node: calculate_slack_time(acc_p.G_ptr.ddl_map[node], curr_t, realloc_slack)
            for node in list(acc_p.running.keys()) + list(acc_p.ready.keys()) 
            if acc_p.G_ptr.ert_map[node] <= curr_t and node != "R"
        } 
        
    score = acc_p.slack_map.copy()
    alloc_map_curr = {}
    acc_p.starving = False
    # calculate min_rsc requirement
    score_dict = OrderedDict(); constr_dict = OrderedDict()
    curr_aval_rsc = acc_p.cap
    for node in sorted(score, key=score.get):
        slack = score[node]
        if slack <= 0:
            # if node != "R":
            #     print(f"\t[{acc_p.id}] {node} is timeout at {curr_t}")
            req_rsc_size = curr_aval_rsc
        else:
            assert not (node in acc_p.ready and node in acc_p.running) 
            task_load = acc_p.running.get(node, 0) + acc_p.ready.get(node, 0)
            req_rsc_size = estimate_resource_requirement(task_load, slack, acc_p.base_pwr)
            if req_rsc_size > curr_aval_rsc:
                print(f"\t[{acc_p.id}] {node} is hungry at {curr_t}: lack {req_rsc_size - curr_aval_rsc} tiles") 
                req_rsc_size = curr_aval_rsc
                curr_aval_rsc = 0
                alloc_map_curr[node] = req_rsc_size
                acc_p.starving = True and (not realloc)
                break
                
        curr_aval_rsc -= req_rsc_size
        alloc_map_curr[node] = req_rsc_size
        constr_dict[node] = "N/A"
        score_dict[node] = 1/score[node] if score[node] >0 else float("inf")
        score.pop(node)
            
    # allocate free resource
    # if there are still resources left, 
    # it means no late process is waiting for resources
    if curr_aval_rsc > 0 and len(alloc_map_curr) and not reserv_en:
        print(f"\t[{acc_p.id}] Minimum resource requirement at {curr_t}: {alloc_map_curr}")
        assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
        # also, there is no process waiting for resources in the ready queue
        assert len(score) == 0
        core_distr(alloc_map_curr, score_dict, curr_aval_rsc)
    
    if reserv_en and curr_aval_rsc > 0 and len(alloc_map_curr):
        print(f"\t[{acc_p.id}] Reservation (en): {curr_aval_rsc} tiles reserved.") 
    return alloc_map_curr

# Cyclic specific alloc_fn and trigger_cond
def trigger_cond_cyclic(acc_p, new_comp, new_ready_list, static_schedule_map=None):
    """
    Trigger condition for cyclic scheduler: 
        New ready tasks are in current map, or curr_t > next time in static_schedule_map.
    """    
    return False

def alloc_fn_cyclic(acc_p, curr_t, realloc=True, static_schedule_map=None, force=False):
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

    # 这里考虑到curr_t 可能不属于static_schedule_map的key，要怎么处理，可以保证数学上有较好的抽象
    # 保存一个状态变量，用来记录当前选中的静态调度表中的slot
    # 在数学上，是一个查表操作，循环的行为可以忽略，认为有一个足够长的表，足以覆盖所有的执行时间。
    # find the largest slot index that is less than curr_t
    assert acc_p.curr_slot_index < len(static_schedule_map), \
        f"Cyclic scheduler: curr_slot_index {acc_p.curr_slot_index} is out of range {len(static_schedule_map)}"
    # check the next slot
    if acc_p.curr_slot_index == len(static_schedule_map) - 1:
        t = float("inf")
        cfg = {}
    else:
        t, cfg = static_schedule_map[acc_p.curr_slot_index+1]
    if curr_t >= t:
        acc_p.curr_slot_index += 1
        # ensure not slot is skipped: next slot's start time is larger than curr_t
        t_next = static_schedule_map[acc_p.curr_slot_index+1][0]
        assert curr_t < t_next, \
            f"Cyclic scheduler: next slot's start time {t_next} is less than curr_t {curr_t}"
    else:
        t, cfg = static_schedule_map[acc_p.curr_slot_index]

    if force:
        if cfg:
            assert acc_p.cap >= sum(cfg.values()), \
                f"Cyclic scheduler (force=True): Capacity Error: {acc_p.cap} is less than required {sum(cfg.values())} at {curr_t}"
            temp_aval_rsc = acc_p.cap
            for node, requested_rsc in cfg.items():
                # ensure the expected task is ready or running
                if node in acc_p.running or node in acc_p.ready:
                    alloc_rsc = cfg[node]
                    alloc_map_curr[node] = alloc_rsc
                    temp_aval_rsc -= alloc_rsc
                    assert temp_aval_rsc >= 0
        else:
            print(f"\t[{acc_p.id}] Cyc-Sched  (force=True): No static allocation defined for time {curr_t}. Allocating nothing.")
    else:
        if cfg:
            print(f"\t[{acc_p.id}] Cyc-Sched: Applying static allocation for time {curr_t}: {cfg}")
            temp_aval_rsc = acc_p.cap
            for node, requested_rsc in cfg.items():
                if temp_aval_rsc <= 0:
                    break
                # ensure the expected task is ready or running
                if node in acc_p.running or node in acc_p.ready:
                    alloc_rsc = min(requested_rsc, temp_aval_rsc)
                    alloc_map_curr[node] = alloc_rsc
                    temp_aval_rsc -= alloc_rsc
                else:
                    print(f"\t[{acc_p.id}] Cyc-Sched: Task {node} not found in running/ready queues, skipping static allocation.")
        else:
            print(f"\t[{acc_p.id}] Cyc-Sched: No static allocation defined for time {curr_t}. Allocating nothing.")    
    return alloc_map_curr


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
        swt_lat_list: 每个分区的切换延迟（可选）
    """
    def __init__(
        self, num_partitions, cap_list, base_pwr_list, 
        G, TSmap_list=None, mapped_node_list=None, swt_lat_list=None
        ):
        self.num_parts = num_partitions
        self.cap_list = cap_list
        self.base_pwr_list = base_pwr_list
        self.G = G
        self.TSmap_list = TSmap_list or [None]*num_partitions
        self.mapped_node_list = mapped_node_list or [None]*num_partitions
        self.swt_lat_list = swt_lat_list or [None]*num_partitions


def acc_p_factory(
    policy: str,
    cfg: PartitionConfig,
    **kwargs
):
    """
    Define scheduler: 
    To define a scheduler, we need first to follow the basic interface of BaseProcessor.
    Then, for specific scheduler, we should define the internal state and static parameters/tables need by the scheduling. 
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
        acc_p = Acc_p(f"acc_p{i}", cfg.cap_list[i], cfg.base_pwr_list[i], cfg.G, cfg.mapped_node_list[i])
        
        # policy binding
        if policy in ["pglb", "glb"]:
            if policy == "pglb" and cfg.num_parts <= 1:
                warn("pglb policy is not supported for single partition, use glb instead")
                policy = "glb"
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_pglb, reserv_en=False), acc_p)
            acc_p.trigger_cond = types.MethodType(trigger_cond_dyn, acc_p)
        elif policy in ["cyc"]:
            assert cfg.TSmap_list[i] is not None, "TSmap_list is not None"
            static_schedule_map = cfg.TSmap_list[i]
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_cyclic, static_schedule_map=static_schedule_map, force=True), acc_p)
            acc_p.trigger_cond = types.MethodType(functools.partial(trigger_cond_cyclic, static_schedule_map=static_schedule_map), acc_p)

        elif policy == "reserv":
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_pglb, reserv_en=True), acc_p)
            acc_p.trigger_cond = types.MethodType(trigger_cond_dyn, acc_p)
        else:
            raise ValueError(f"Unknown strategy: {policy}")
        acc_p.policy = policy
        # 使用分区配置中的swt_lat，如果没有提供则使用默认计算
        assert cfg.swt_lat_list[i] is not None, f"swt_lat is not None for partition {i}"
        acc_p.swt_lat = cfg.swt_lat_list[i]
        acc_p_list.append(acc_p)
    return acc_p_list

def get_partition_info(bin_list, graph:MyGraph, pid2name=None):
    # 1. 分区的任务映射
    partition_task_map = []
    for _bin in bin_list:
        task_set = set()
        for rsc_agent in _bin.scheduling_table:
            task_set.update(rsc_agent.rsc_map.keys())
        if pid2name is not None:
            partition_task_map.append([pid2name[pid] for pid in task_set])
        else:
            partition_task_map.append(list(task_set))
    
    for node in graph.sinks: 
        preds = graph.pred[node]
        assert len(preds) == 1, f"sink node {node} should have only one predecessor, but got {len(preds)}"
        pred = list(preds.keys())[0]
        for i, partition_tasks in enumerate(partition_task_map):
            if pred in partition_tasks:
                partition_task_map[i].append(node)
                break
        else:
            assert False, "sink node should be assigned to a partition"


    bin_name_list = [bin.name for bin in bin_list]
    # 2. 分区的大小
    partition_size = [_bin.num_resources for _bin in bin_list]

    # 3. cyclic方法下的静态调度表
    TSMap_list = []
    for _bin in bin_list:
        TSmap = []
        for cfg_slot_s, next_cfg, cfg_slot_num in getattr(_bin, "sparse_list", []):
            if pid2name is not None:
                TSmap.append((elim_nume_error(cfg_slot_s*old_timestep), {pid2name[pid]: size for pid, size in next_cfg.items()}))
            else:
                TSmap.append((elim_nume_error(cfg_slot_s*old_timestep), next_cfg))
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

def set_time_unit(timestep, int_slot):
    global time_unit
    if int_slot:
        time_unit = 1
        normalize_factor = timestep
    else:
        time_unit = timestep
        normalize_factor = 1
    return time_unit, normalize_factor

# seting_size, time_unit, base_pwr
# if int_slot is True, the time_unit is 1, and all exe_comp_t should divide by timestep;
# otherwise, the time_unit is timestep
def load_graph_from_json(json_path, time_norm_factor):

    from task.task_cfg import load_json_graph_utils
    G = load_json_graph_utils(json_path)
    nodes = G.nodes
    edges = G.edges

    # get the pid2name map
    pid2name = {n_att['node_id']: n for n, n_att in G.nodes(data=True)}

    # 分类节点
    srcs, ops, sinks = {}, {}, {}
    task_attr, src_attr, sink_attr = {}, {}, {}

    # 先统计所有节点的入度和出度
    src_nodes = [n for n, x in G.in_degree() if x == 0]
    sink_nodes = [n for n, x in G.out_degree() if x == 0]

    # if int_slot is True, the time_unit is 1, and all exe_comp_t should divide by timestep;
    # otherwise, the time_unit is timestep

    # 分类
    for node_id in nodes:
        node = nodes[node_id]

        exp_comp_t, base_size = get_task_load_and_base_size(node, time_norm_factor)
        offset = elim_nume_error(node['offset'])
        # 源节点
        if node_id in src_nodes:
            # 找到所有后继
            srcs[node_id] = list(G.successors(node_id))
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
                'ddl': node['ddl'] + offset
                # 'tgt_device': n.get('tgt_device', 'sink')
            }
        # 中间节点
        else:
            ops[node_id] = list(G.successors(node_id))
            task_attr[node_id] = {
                'offset': offset,
                'exp_comp_t': exp_comp_t,
                'base_size': base_size,
                # 'tgt_device': n.get('tgt_device', 'acc_p0'),
                'ert': node['ert'],
                'ddl': node['ddl'] + offset
            }

    return srcs, ops, sinks, task_attr, src_attr, sink_attr, pid2name
    # TODO: duplicate nodes in the graph, add noise to their exp_comp_t as the simulation time expands

# 实例化MyGraph
def instantiate_mygraph_from_json(json_path, time_norm_factor):
    srcs, ops, sinks, task_attr, src_attr, sink_attr, pid2name = load_graph_from_json(json_path, time_norm_factor)
    G = MyGraph(srcs, ops, sinks, task_attr, src_attr, sink_attr)
    return G, pid2name

def cal_load(exp_comp_t, base_size):
    """
    统一计算任务load的函数
    
    Args:
        node: 任务节点
        G_ptr: MyGraph实例    
    Returns:
        float: 任务的load值
    """
    return exp_comp_t * base_size
