import numpy as np
import math, re, warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Tuple
from queue import Queue
from global_var import *
from utils import load_pickle, core_distr

from model.buffer import Buffer, EventCache, TriggerCache
from model.buffer import Buffer, Data
from model.resource_agent import Resource_model_int
from model.event_gen.e2e_latency import jitter_gen_biside
from model.task_queue_agent import TaskQueue 
from model.lru import LRUCache
from model.barrier_agent import Barrier
from model.message.msg_dispatcher import MsgDispatcher
from model.message.Context_message import ContextMsg
from model.message.data_pipe import DataPipe
from model.streaming_processing.wartermark_strategy import WatermarkStrategy
from model.performance import slack_comp, cal_lat

from task.task_agent import ProcessInt
from sched.scheduling_table import SchedulingTableInt, parse_event_msg
from sched.monitor_agent import Monitor
from sched.monitor_agent import get_rsc_2b_released
from sched.slack_estim import EstimCoreNums4Process

class Scheduler(object): 
    """
    Scheduler is responsible for the scheduling of the tasks:
    Task queue: 
    track ready tasks:
     (active): 
        Tasks are enqueued on some runqueue when they wake up and are dequeued when they are suspended.
        group processes into priority classes:  use priority scheduling among the classes but round-robin scheduling within each class
        track of deadlines of the earliest deadline tasks currently executing on each runqueue.
     (expired):
        Tasks are enqueued on some expired queue when they expire and are dequeued when they are refill. 
    track blocked tasks: 
        enqueue when they are blocked and are dequeued when they are unblocked.

    Preemption:
     condition: When a task is activated/increased priority on CPU k, which has higher priority than the executing one, 
     Operation: 
      a preemption happens, the preempted task is inserted at the head of the queue; 
      otherwise the wakenup task is inserted in the proper runqueue, depending on the state of the system. 

    Push: 
     condition: the head of the queue is modified, 
     operation: a push operation is executed to see if some task can be moved to another queue. 
    
    Pull:
     Condition: When a task suspends itself (due to blocking or sleeping) or lowers its priority on CPU k
     Operation: it looks at the other run-queues to see if some other higher priority tasks need to be migrated to the current CPU.

    1. maintain the scheduling table (SMT runable queue, for resources prevision) 
       enqueue new tasks, dequeue expired tasks, and adjust the position of the tasks in the runable queue
    2. maintain and monitor the task status: running, runnable, expired (throttled), suspended (blocked), terminated 
       update the task status when event happens 
       periodically (tile-level) checks whether it runs slower than expected due to resource contention
    3. maintain the event queue: 
        cache the pre-defined events (task release, task deadline, tick), 
        predict events (completion)

        record the runtime events (suspension, preemption, expiration, lag, spec update, delay, timeout) 
    4. make the scheduling decision:
        4.1. dispatch the task that can attain available resources
        4.2. make the preemption/pull/push decision
        4.3. enforce rule confinements

    track something: 
        number of Idle Cores by semaphore
        track of deadlines of the earliest deadline tasks currently executing on each runqueue.
    
    struct dl_rq {
        struct rb_root rb_root
        struct rb_node * rb_leftmost
        unsigned long dl_nr_running
        # ifdef CONFIG_SMP
        struct {
            / * two earliest tasks in queue * /
            u64 curr
            u64 next
            / * next earliest * /
        } earliest_dl
        int overloaded
        unsigned long dl_nr_migratory
        unsigned long dl_nr_total
        struct rb_root pushable_tasks_root
        struct rb_node * pushable_tasks_leftmost  # endif /* CONFIG_SMP */
    }
    • rb_root: the root of the red-black tree
    • rb_leftmost: the leftmost node of the red-black tree
    • dl_nr_running: the number of tasks in the run queue
    • earliest dl is a per-runqueue data structure used for “caching” the deadlines of the first two ready tasks, 
    so to facilitate migration-related decisions; 
    • dl_nr_migratory and dl_nr_total represent the number of queued tasks that can migrate and the total number of queued tasks, respectively; 
    • overloaded serves as a flag, and it is set when the queue contains more than one task; 
    • pushable_tasks_root is the root of the redblack tree of tasks that can be migrated, since they are queued but not running, 
    and it is ordered by increasing deadline; 
    • pushable_tasks_leftmost is a pointer to the node of pushable tasks root containing the task with the earliest deadline.
    """

    def __init__(self, 
                 _SchedTab: SchedulingTableInt, e2e_latency:float, hyper_period:int,
                 glb_p_list:List[ProcessInt],
                 budget_recoder:Dict[int, List]=None, rsc_recoder_his:Dict[int, LRUCache]=None, 
                 barrier_en:bool=True, res_cfg:Resource_model_int=None, exec_t_comp_ratioB=0, 
                 forbid_miss:bool=False, trace_path:str=None,
                 ) -> None:
        self.expired_queue: List = []
        self.blocked_queue: List = []

        # wait 
        # structure: (wait_time, task)
        self.weight_wait_queue = TaskQueue(sort_f=lambda x: x.io_time-x.waitTime, descending=False)
        self.input_wait_queue: List = []

        # buffer
        self.buffer:Buffer = Buffer()
        # monitor the deadline: (ascending)
        self.ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, descending=False)
        
        # Running queue: 
        # cache the running task list in an order of priority (here we use ddl)
        # monitor the deadline for pre-emption: (descending)
        # interrupt the task with the latest ddl
        self.running_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline)

        # the function is different with preallocation stage
        self.issue_list:List[ProcessInt]  = []
        # used to record the completed tasks before its expected completion time
        self.completed_list:List[ProcessInt] = []
        self.inactive_list:List[ProcessInt] = []
        self.miss_list:List[ProcessInt] = []
        self.preempt_list:List[ProcessInt] = []
        self.throttle_list:List[ProcessInt] = []
        self.active_list:List[ProcessInt] = []
        self.ctx_switch_list:List[ProcessInt] = []
        
        self.new_ready_flg:bool = False
        self.new_drop_flg:bool = False
        self._SchedTab = _SchedTab
        self._SchedTab_L0 = None
        self.core_map = []
        self.core_map_L0 = dict()
        self.drain_flg = False
        self.switch_border = 0

        self.curr_cfg:Resource_model_int = Resource_model_int(size=_SchedTab.num_resources)
        # self.process_dict: Dict[int, ProcessInt] = {pid:glb_p_list[pid] for pid in _SchedTab.index_occupy_by_id()}
        resident = list(_SchedTab.index_occupy_by_id().keys())
        self.process_dict: Dict[int, ProcessInt] = {_p.pid:_p for _p in glb_p_list if _p.pid in resident}
        self.res_cfg:Resource_model_int = res_cfg

        # event queue
        self.event_cache = EventCache(type='data')
        self.trigger_cache = TriggerCache(type='ctrl')
        for pid in self.process_dict:
            _p = self.process_dict[pid] 
            self.event_cache.new_process(_p)
            self.trigger_cache.new_process(_p)

        # create res_cfg, monitor, msg_queue, 
        # self.res_cfg = res_cfg
        # self.monitor = monitor
        # self.msg_queue = msg_queue

        # curr_cfg, budget_recoder, rsc_recoder_his, process_dict
        # curr_cfg_list, budget_recoder_list, rsc_recoder_his_list, process_dict_list
        self.budget_recoder = budget_recoder if budget_recoder else {}
        self.rsc_recoder_his = rsc_recoder_his if rsc_recoder_his else {}

        self.position_dict: Dict[int, int] = {}
        self.barrier_en = barrier_en
        self.barrier = Barrier(0)
        self.assert_barrier = False

        self.e2e_latency = e2e_latency
        self.hyper_period = hyper_period
        self.detail_alloc_info:Dict[str, Dict[float, Tuple]] = {}

        # Buffer size: 40MB, 256 cores, 156.25KB/core
        # bandwidth: 100GB/s
        # direction: off-chip -> on-chip, on-chip -> off-chip
        # latency: 100ns
        head_latency = AVG_HOP_NUM * LAT_PER_HOP
        self.ctx_size_gen = jitter_gen_biside(1, {'scale': 0.2, }, size=1, seed=_SchedTab.id)
        self.roll_size = lambda:(self.ctx_size_gen()+0.8)*GLB_BUFFER_SIZE_PER_CORE
        self.get_ctx_lat = lambda x=1: self.roll_size()*x/BW_DRAM + head_latency
        self.overprovision_rate = exec_t_comp_ratioB
        self.forbid_miss = forbid_miss
        self.trace_path = trace_path

    def res_release(self, pid, op_pos_dict:bool=True):
        self.res_cfg.release(pid, verbose=False)
        if op_pos_dict: 
            self.position_dict.pop(pid, None)

    def check_draining_state(self):
        queue_clear_flag = True
        for _p in  (self.active_list + self.ready_queue.queue + self.running_queue.queue):
            if _p.msg_cache[0].get_timestamp() < self.switch_border:
                queue_clear_flag = False
                break
        if queue_clear_flag:
            self.drain_flg = False


    def drain_old_cores(self, msg_dispatcher:MsgDispatcher):
        # release the resource and inform other schedulers to get the resource
        used_position = []
        for pid in self.position_dict.keys():
            for s, size in zip(*self.position_dict[pid][:-1]):
                e = s + size
                used_position += [i for i in range(s, e)]               
        aval_pos = [i for i in self.core_map if i not in used_position]
        for i in aval_pos: 
            if i not in self.core_map_L0:
                self.core_map.remove(i)
                msg_dispatcher.broadcast_message(f"CORE {i} is set free")
    
    def switch_sched_tab(self):
        self._SchedTab = self._SchedTab_L0
        self._SchedTab_L0 = None

    def get_queues(self):
        # wait_queue, ready_queue, running_queue, miss_list, preempt_list, issue_list, completed_list
        return self.weight_wait_queue, self.ready_queue, self.running_queue, \
            self.miss_list, self.preempt_list, self.issue_list, self.completed_list, self.throttle_list,\
            self.inactive_list, self.active_list
    
    def get_buffer(self):
        return self.buffer

    def get_state(self):
        # curr_cfg, _SchedTab, sched, budget_recoder, rsc_recoder_his, msg_queue, process_dict, 
        return self.curr_cfg, self._SchedTab, self.budget_recoder, self.rsc_recoder_his, self.process_dict

    def throttleToReady(self, curr_t, bin_event_flg):
        return throttleToReady(self, curr_t, 
                                self.budget_recoder, self.ready_queue, self.throttle_list, self._SchedTab.name,
                                bin_event_flg)

    def chk_release(self, event_range, curr_t, timestep, 
                    bin_event_flg:bool=False, ):
        return chk_release(self, event_range, curr_t, self.inactive_list, self.active_list, self._SchedTab, timestep, 
                           bin_event_flg, self._SchedTab.name)

    def check_miss(self, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                            curr_t, res_cfg, 
                            bin_event_flg:bool=False):

        # check_miss(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            # throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name)
        return check_miss(self, self.budget_recoder, msg_dispatcher, curr_t, res_cfg, self.weight_wait_queue, self.ready_queue, self.running_queue, self.miss_list,
                            self.throttle_list, self.active_list, self.inactive_list, self.buffer, bin_event_flg, self._SchedTab.name)

    def check_throttle(self,
                            curr_t, res_cfg, 
                            bin_event_flg:bool=False):
        # check_throttle(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
        #                             throttle_list, active_list, inactive_list, bin_event_flg, bin_name)
        return check_throttle(self, self.budget_recoder, curr_t, res_cfg, self.weight_wait_queue, self.ready_queue, self.running_queue, self.miss_list,
                            self.throttle_list, self.active_list, self.inactive_list, bin_event_flg, self._SchedTab.name)

    def check_complete(self, timestep, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                       a_data_pipe:DataPipe,
                        curr_t, res_cfg, 
                        bin_event_flg:bool=False,
                        save_trace:bool=True,
                        mode:str="current", 
                        bin_list:List[SchedulingTableInt]=None, 
                        n_slot:int=0, rsc_recoder=None,):
        # check_complete(budget_recoder, timestep, msg_dispatcher, curr_t, res_cfg, running_queue, completed_list, inactive_list, buffer, bin_event_flg, bin_name)
        return check_complete(self, self.budget_recoder, timestep, msg_dispatcher, a_data_pipe, curr_t, res_cfg, 
                              self.running_queue, self.completed_list, self.inactive_list, self.buffer, 
                              bin_event_flg, self._SchedTab.name, save_trace, mode, bin_list, n_slot, rsc_recoder) 

    def record_comp_bw_slot_by_slot(self, n_slot, pid):
        if pid in self.budget_recoder:
            alloc_slot_s:List[int]
            alloc_size:List[int]
            allo_slot:List[int]
            alloc_slot_s, alloc_size, allo_slot = self.budget_recoder[pid]
            # merge the allocation
            if alloc_slot_s[-1] + allo_slot[-1] == n_slot and self.curr_cfg.rsc_map[pid] == alloc_size[-1]:
                allo_slot[-1] += 1
                self.budget_recoder[pid] = [alloc_slot_s, alloc_size, allo_slot]
            else:
                alloc_slot_s.append(n_slot)
                alloc_size.append(self.curr_cfg.rsc_map[pid])
                allo_slot.append(1)
                self.budget_recoder[pid] = [alloc_slot_s, alloc_size, allo_slot]
        else:
            self.budget_recoder[pid] = [[n_slot,], [self.curr_cfg.rsc_map[pid],], [1,]]

    def updateRunningQueue(self, timestep, res_cfg:Resource_model_int):
        # updateRunningQueue(timestep, running_queue, res_cfg) 
        return res_cfg.updateRunningQueue(timestep, self.running_queue)
    
    def pendingToReady(self, curr_t, glb_n_task_dict:Dict[str, ProcessInt]):
        # pendingToReady(active_list, ready_queue, buffer, budget_recoder, throttle_list, curr_t, glb_name_p_dict, bin_name, ) 
        return pendingToReady_cbs(self, self.buffer, self.budget_recoder, self.active_list, 
                                  self.ready_queue, self.throttle_list, curr_t, 
                                  glb_n_task_dict, self.event_cache, self._SchedTab.name, self.process_dict)

    def scheduler_step(self, msg_dispatcher:MsgDispatcher, a_data_pipe: DataPipe, data_pipe: DataPipe,
                       n_slot: int, timestep: int, event_range: List[int], 
                        sim_slot_num: int, curr_t: int, glb_name_p_dict: Dict[str, List[int]], 
                        res_cfg: Dict[str, int], 
                        msg_queue:Queue, a_msg_queue:TaskQueue,
                        sensor_msg_queue:Queue,
                        monitor:Monitor,
                        DEBUG_FG: bool) -> None:

        # scheduler_step(sched, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG)
        return scheduler_step(self, msg_dispatcher, a_data_pipe, data_pipe, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, 
                              res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, monitor, DEBUG_FG)
    
    def cyclic_step(self, msg_dispatcher:MsgDispatcher, a_data_pipe: DataPipe, data_pipe: DataPipe,
                       n_slot: int, timestep: int, event_range: List[int], 
                        sim_slot_num: int, curr_t: int, glb_name_p_dict: Dict[str, List[int]], 
                        res_cfg: Dict[str, int], 
                        msg_queue:Queue, a_msg_queue:TaskQueue,
                        sensor_msg_queue:Queue,
                        monitor:Monitor,
                        DEBUG_FG: bool) -> None:

        # scheduler_step(sched, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG)
        return scheduler_step_cyclic(self, msg_dispatcher, a_data_pipe, data_pipe, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, 
                              res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, monitor, DEBUG_FG)
    
    def fifo_step(self, msg_dispatcher:MsgDispatcher, a_data_pipe: DataPipe, data_pipe: DataPipe,
                       n_slot: int, timestep: int, event_range: List[int], 
                        sim_slot_num: int, curr_t: int, glb_name_p_dict: Dict[str, List[int]], 
                        res_cfg: Dict[str, int], 
                        msg_queue:Queue, a_msg_queue:TaskQueue,
                        sensor_msg_queue:Queue,
                        monitor:Monitor,
                        DEBUG_FG: bool) -> None:

        # scheduler_step(sched, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG)
        return scheduler_step_fifo(self, msg_dispatcher, a_data_pipe, data_pipe, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, 
                              res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, monitor, DEBUG_FG)

    def pglb_step(self, msg_dispatcher:MsgDispatcher, a_data_pipe: DataPipe, data_pipe: DataPipe,
                       n_slot: int, timestep: int, event_range: List[int], 
                        sim_slot_num: int, curr_t: int, glb_name_p_dict: Dict[str, List[int]], 
                        res_cfg: Dict[str, int], 
                        msg_queue:Queue, a_msg_queue:TaskQueue,
                        sensor_msg_queue:Queue,
                        monitor:Monitor,
                        DEBUG_FG: bool) -> None:

        # scheduler_step(sched, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG)
        return scheduler_step_pglb(self, msg_dispatcher, a_data_pipe, data_pipe, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, 
                              res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, monitor, DEBUG_FG)


# =================== intergrated into scheduler class ===================
def throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name:str="", bin_event_flg:bool=False):
    l_res_ready:List[ProcessInt] = []
    for _p in throttle_list:
        if _p.pid in budget_recoder and sched._SchedTab.id in _p.rem_flop_budget:
            rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
            if rem_flop_budget> numerical_error_tol_abs:
                l_res_ready.append(_p)
        
    if bin_name and l_res_ready and not bin_event_flg:
        bin_event_flg = True 
        print(f"({bin_name})")

    if len(l_res_ready):
        sched.new_ready_flg = True

    for _p in l_res_ready: 
        throttle_list.remove(_p)
        _p.ready_util(curr_t, ready_queue)
        print("		TASK {:d}:{:s}({:d}) THROTTLE -> READY!!".format(_p.task.id, _p.task.name, _p.pid))
    return bin_event_flg

def chk_release(sched, event_range, curr_t, inactive_list:List[ProcessInt], active_list, 
                _SchedTab, timestep, event_cache:EventCache=None, trigger_cache:TriggerCache=None,
                bin_event_flg:bool=False, 
                bin_name:str="", DEBUG_FG:bool=False,):
    """
    check release
        1. check the dependencies of the tasks in inactive list
        2. if the dependencies are satisfied, move the task to the wait queue
    """

    l_active = []
    # if curr_t <= event_range:
    #     # simulate the event trigger
    #     trigger_state = message_trigger_event(_SchedTab.sim_triggered_list, sched.jitter_sim_en, sched.jitter_sim_para, 
    #                                           inactive_list, timestep, curr_t, True)
    #     if bin_name and trigger_state and not bin_event_flg:
    #         bin_event_flg = True
    #         print(f"({bin_name})")

    for _p in inactive_list:
        if _p.check_depends(event_cache=event_cache, trigger_cache=trigger_cache):
            l_active.append(_p)

    if bin_name and len(l_active) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
        
    for _p in l_active:
        active_list.append(_p)
        inactive_list.remove(_p)
        _p.release_time = curr_t
        _p.released = True
        _p.remburst += _p.task.flops
        _p.set_state("active")
        # _p.release_time = curr_t 
        # _p.deadline = curr_t + _p.task.ddl
        # _p.deadline += _p.task.period
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is activated @ {curr_t:.6f}!!"
        print(_str)
    return bin_event_flg


def check_miss(sched: Scheduler,
               budget_recoder, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
               curr_t, res_cfg, wait_queue, ready_queue,
               running_queue, miss_list:List[ProcessInt], throttle_list, active_list, inactive_list, buffer,
               bin_event_flg: bool = False,
               bin_name: str = "", mode: str = "current",
               bin_list: List[SchedulingTableInt] = None,
               n_slot: int = 0,
               rsc_recoder=None,
               process_dict: Dict[int, ProcessInt] = None,
               show_warnings=True):

    bin_id = sched._SchedTab.id
    for _p in sorted(active_list + ready_queue.queue + running_queue.queue, key=lambda x: x.deadline):
        _p:ProcessInt
        # !!!!!!!!!!!! Bug Here !!!!!!!!!!!!!
        # task miss: hard -> stop, soft -> warning
        # chain miss: output task hard -> stop, soft -> warning 
        inst_hard_miss = _p.deadline <= curr_t 
        chain_miss = _p.get_chain_deadline(sched) <= curr_t and _p.task.chain_criticality == "hard"
        if chain_miss:
            miss_list.append(_p)
        elif inst_hard_miss:
            if _p.task.criticality == "hard":
                miss_list.append(_p)
            elif show_warnings:
                _str = f"Task {_p.task.id}:{_p.task.name}({_p.pid}) violate timing constraint @ {_p.deadline:.6f}/{_p.msg_cache[0].get_timestamp():.6f}!!"
                warnings.warn(_str)

    if bin_name and len(miss_list) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
    
    if len(miss_list)>0:
        sched.new_drop_flg = True

    for _p in miss_list:
        # release the resource and move to the wait list
        # buffer.pop(_p.pid)
        if _p in ready_queue.queue:
            ready_queue.remove(_p)
        elif _p in active_list:
            active_list.remove(_p)
        elif _p in running_queue.queue:
            assert mode in ["future", "current", "none"]
            if mode == "future":
                bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                _SchedTab:SchedulingTableInt = bin_list[bin_id_t]
                _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
            elif mode == "current":
                sched.res_release(_p.pid)
            else:
                pass
            try:
                _p.rem_flop_budget[bin_id] -= (_p.totcpu - _p.totburst)
            except KeyError:
                print("20231126: CodingError, attempt to remove budget of missed tasks")
            if rsc_recoder is not None:
                rsc_recoder.pop(_p.pid)

            running_queue.remove(_p)
        print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.msg_cache[0].get_timestamp():.6f}!!")
        if sched.forbid_miss: 
            print("forbid_miss is True, exit the simulation")
            import sys; sys.exit(1)

        if msg_dispatcher is not None:
            msg_dispatcher.broadcast_message(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.msg_cache[0].get_timestamp():.6f}({bin_name})!!")

        _p.task.missed_deadline_count += 1
        # _p.release_time += _p.task.period
        # _p.deadline += _p.task.period

        _p.reset_state_vars()            

        if _p.is_fork_inst:
            _p.kill_fork(process_dict)
            del _p
        else:
            inactive_list.append(_p)
        # _p.reset_depends()

        # print("Scheduling Table:")
        # print(SchedTab.print_scheduling_table())
    miss_list.clear()
    return bin_event_flg


def check_throttle(sched:Scheduler,
                   budget_recoder, curr_t, res_cfg, wait_queue, ready_queue,
                   running_queue, miss_list, throttle_list, active_list, inactive_list,
                   bin_event_flg: bool = False,
                   bin_name: str = ""):

    # for _p in (active_list + wait_queue.queue + ready_queue.queue + running_queue.queue):
    bin_id = sched._SchedTab.id
    l_throttle = []
    for _p in (running_queue.queue+ready_queue.queue):
        budget_usedup_flg = False
        # determine if the budget is used up, considering the numerical error
        # less than 1 OPS
        if _p.rem_flop_budget[bin_id] < numerical_error_tol_abs:
            budget_usedup_flg = True
        if budget_usedup_flg and _p.cbs_en: 
            l_throttle.append(_p)
            _p.set_state("throttled")

    if bin_name and len(l_throttle) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
    
    for _p in l_throttle: 
        _p:ProcessInt
        _p.throttle_util(running_queue, ready_queue, throttle_list, sched,
                      curr_t)
    return bin_event_flg

def check_complete(sched:Scheduler, budget_recoder, timestep, 
                   msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                   a_data_pipe:DataPipe,
                   curr_t, res_cfg, 
                    running_queue:TaskQueue, 
                    completed_list: List[ProcessInt], 
                    inactive_list:List[ProcessInt],
                    buffer:Buffer, 
                    bin_event_flg:bool=False, 
                    bin_name:str="", 
                    save_trace:bool=True,
                    mode:str="current", 
                    bin_list:List[SchedulingTableInt]=None, 
                    n_slot:int=0, rsc_recoder=None,
                    detail_alloc_info: Dict[str, Dict[float, List[Tuple]]]=None,
                    process_dict:Dict[int, ProcessInt]=None,
                    ):
    bin_id = sched._SchedTab.id
    for _p in sorted(running_queue.queue, key=lambda x: x.is_fork_inst, reverse=True): 
        # check whether the task is completed
        """
        3. check complete:
            both `n_fork == 0` and totburst
        """
        fork_complete_flg = (_p.n_fork == 0 and _p.fork_pid_list == []) or sum([_forked_p in completed_list for _forked_p in _p.fork_p_inst]) == _p.n_fork
        if elim_nume_error(_p.totburst - _p.totcpu)>=0 and fork_complete_flg:
            completed_list.append(_p)

    if bin_name and len(completed_list) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")

    if len(completed_list)>0:
        sched.new_drop_flg = True

    for _p in completed_list:
        # update statistics
        # TODO: add lock 
        _p.task.completion_count += 1
        _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
        _p.end_time = curr_t

        # cache processing info for ctx message and attach to the data
        msg:ContextMsg = _p.msg_cache.pop(0)
        msg.cache_processing(_p)
        _p.event_time = msg.get_timestamp()
        data = Data(_p.pid, _p.io_time, (0,), "output", curr_t, 1/_p.task.freq, _p.task.period)
        data.ctx = msg
        data.cache_data_info()

        # TODO: communication scheduling
        # cache the message sending time when the bus is allocated
        data.cache_msg_transfer(curr_t)

        off_size = _p.required_resource_size * 1/3
        x = (off_size) 
        data.size = sched.get_ctx_lat(x)

        a_data_pipe.put(data,)
        # if succ_ctrl is not empty, 
        # redirect print(data.ctx.serialize()) to the trace_file path
        if len(_p.succ_ctrl) and save_trace:
            trace_list.append(data.ctx.serialize())
            # save_chunk(sched.trace_path.replace(".pkl", ".h5"), trace_list)

        if detail_alloc_info is not None:
            for bin_id, _SchedTab in enumerate(bin_list):
                time_slot_s = int(np.ceil(_p.release_time/timestep))
                time_slot_e = int(_p.deadline//timestep)
                alloc_rsc_record = _SchedTab.index_occupy_by_id(time_slot_s, time_slot_e)
                if not _p.pid in alloc_rsc_record:
                    continue
                _recrd = detail_alloc_info.get(_p.task.name, {})
                _recrd.update({round(_p.event_time, numerical_tol_bit): (bin_id, *alloc_rsc_record[_p.pid])})
                detail_alloc_info.update({_p.task.name: _recrd})

        # reset the task
        # release the resource and move to the wait list
        assert mode in ["future", "current", "none"]
        if mode == "future": 
            # release the resource and move to the wait list
            bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                
            _SchedTab:SchedulingTableInt = bin_list[bin_id_t]
            _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)

        elif mode == "current":
            sched.res_release(_p.pid)
        
        else:
            pass

        # detect the lateness 
        _str = f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}!!"
        if _p.deadline < curr_t:
            _str = "(lateness detected)" + _str
        else:
            _str = "		" + _str
        print(_str)
        
        _p.release_time += _p.task.period
        # _p.deadline += _p.task.period

        # if budget_recoder is not None:
        #     per_slot_flops = FLOPS_PER_CORE*timestep*budget_recoder[_p.pid][1]
        #     # if _p.rem_flop_budget[bin_id] >= per_slot_flops-numerical_error_tol_abs:
        #     #     curr_cfg = sched.curr_cfg
        #     #     # truncate the curr_cfg
        #     #     curr_cfg.slot_num = curr_cfg.slot_num - (n_slot - curr_cfg.slot_s)
        #     #     curr_cfg.slot_s = n_slot
        #     # budget_recoder.pop(_p.pid)
        #     # _p.rem_flop_budget.pop(bin_id)
        #     # TODO: a large number of tasks are truncated, remaing budget ~= 1 slot, check why
        #     if _p.rem_flop_budget[bin_id] >= per_slot_flops+numerical_error_tol_abs and \
        #         _p.task.name.startswith("Steering_speed"):
        #         budget_recoder[_p.pid][2] -= n_slot - budget_recoder[_p.pid][0] 
        #         budget_recoder[_p.pid][0] = n_slot
        #         print(f"\t\tTruncate the curr_cfg @{n_slot}({curr_t:.6f}), \
        #                 \n\t\trem_flop_budget: {_p.rem_flop_budget[bin_id]/per_slot_flops:.6f}(x{per_slot_flops:.6f})")
        #     else:
        #         budget_recoder.pop(_p.pid)
        #         _p.rem_flop_budget.pop(bin_id)

        if rsc_recoder is not None:
            rsc_recoder.pop(_p.pid)

        _p.reset_state_vars()
        running_queue.remove(_p)

        if msg_dispatcher is not None:
            msg_dispatcher.broadcast_message(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}({bin_name})!!")

        if _p.is_fork_inst:
            _p.kill_fork(process_dict)
            del _p
        else:
            inactive_list.append(_p)
        # _p.reset_depends()

    # update_depend(task_dict, completed_list)
    
    
    completed_list.clear()
    return bin_event_flg

def record_comp_bw_slot_by_slot(rsc_recoder, n_slot, curr_cfg, pid):
    if pid in rsc_recoder:
        alloc_slot_s:List[int]
        alloc_size:List[int]
        allo_slot:List[int]
        alloc_slot_s, alloc_size, allo_slot = rsc_recoder[pid]
        # merge the allocation
        if alloc_slot_s[-1] + allo_slot[-1] == n_slot and curr_cfg.rsc_map[pid] == alloc_size[-1]:
            allo_slot[-1] += 1
            rsc_recoder[pid] = [alloc_slot_s, alloc_size, allo_slot]
        else:
            alloc_slot_s.append(n_slot)
            alloc_size.append(curr_cfg.rsc_map[pid])
            allo_slot.append(1)
            rsc_recoder[pid] = [alloc_slot_s, alloc_size, allo_slot]
    else:
        rsc_recoder[pid] = [[n_slot,], [curr_cfg.rsc_map[pid],], [1,]]

def pendingToReady_cbs(sched, buffer:Buffer, budget_recoder, 
                       active_list:List[ProcessInt], ready_queue, throttle_list, 
                       curr_t, glb_n_task_dict:Dict[str, ProcessInt], event_cache:EventCache=None,
                       bin_name="", process_dict:Dict[int, ProcessInt]=None,
                       show_warnings=True):
    # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
    l_ready:List[ProcessInt] = []
    for _p in active_list:
        # check data availability
        w_avail = _p.pid in buffer.buffer_w
        # in_avail = _p.check_depends_data(buffer, glb_n_task_dict=glb_n_task_dict, event_cache)

        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0: 
            matched_pair, in_avail = WatermarkStrategy.check_data_depends(_p, buffer, glb_n_task_dict, 
                                                                                      _p.msg_cache[0].get_timestamp(), event_cache)
        else:
            in_avail = True

        if w_avail and in_avail:
            l_ready.append(_p)

    for _p in l_ready:
        # cache the context of the upstream weight node and src node
        _p.update_ctx('weight', buffer=buffer)
        # _p.update_ctx('upstream', buffer=buffer, glb_n_task_dict=glb_n_task_dict)
        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0:
            _p.update_ctx('upstream', matched_pair=matched_pair)
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) "
        if bin_name:
            _str = f"({bin_name})\n" + _str
        # if _p.rem_flop_budget > 0: 
        if _p.pid in budget_recoder and sched._SchedTab.id in _p.rem_flop_budget:
            rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
            if rem_flop_budget> numerical_error_tol_abs:
                _p.ready_util(curr_t, ready_queue)
                _str += "READY!!"
                print(_str)
                active_list.remove(_p)
                sched.new_ready_flg = True

            _p.pre_fork(True)
            fork_list = _p.p_fork()
            process_dict.update({p.pid:p for p in fork_list})
            _p.fork_p_inst.extend(fork_list)
            for _forked_p in fork_list:
                budget_recoder[_forked_p.pid] = budget_recoder[_p.pid]
                _forked_p.ready_util(curr_t, ready_queue)
                print(f"{_str}" + f" (FORKED {_forked_p.pid:d})")
        else:
            active_list.remove(_p)
            throttle_list.append(_p)
            _p.set_state("throttled")
            if show_warnings:
                _str += "data ready, but throttled!!"
                warnings.warn(_str)

def scheduler_step(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()

    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    # clear the marginal budget
    for pid in list(budget_recoder.keys()):
        _p = process_dict[pid]
        rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
        if rem_flop_budget < numerical_error_tol_abs and _p.pid in budget_recoder:
            # Each partition has splited budget recoder 
            budget_recoder.pop(pid)
            _p.rem_flop_budget[bin_id] = 0

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, budget_recoder, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, budget_recoder, msg_dispatcher, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    bin_event_flg = check_throttle(sched, budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, bin_event_flg, bin_name)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)
    # buffer.recyle_no_ref("output", True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)

    # # instruction prefetching
    # cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]         
    
    # # if the pointer reaches the start of the next cfg, update the cfg
    # if tab_pointer == cfg_slot_s:
    #     # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
    #     curr_cfg.update(cached_cfg)
    #     # move the pointer to the next cfg
    #     _SchedTab.idx_plus_1()

    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.next_item()
        if _SchedTab.alloc_mod != 'exactly':
            curr_cfg.update(_SchedTab.sparse_cores[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.update(next_cfg)
        # update the deadline
        if cfg_slot_s < tab_pointer: 
            curr_cfg.slot_s = (hyper_p_n + 1) * tab_temp_size + cfg_slot_s
        else:
            curr_cfg.slot_s = hyper_p_n * tab_temp_size + cfg_slot_s
        curr_cfg.slot_e = curr_cfg.slot_s + cfg_slot_num - 1 
        curr_cfg.slot_num = cfg_slot_num
        curr_cfg.flops_dict.clear()
        if _SchedTab.alloc_mod != 'exactly':
            curr_cfg.flops_dict.update(_SchedTab.sparse_flops[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.flops_dict.update({pid: n_cores * cfg_slot_num * timestep * FLOPS_PER_CORE for pid, n_cores in next_cfg.items()})
        
        # process the events
        # process old events -> clear -> process new events
        # NOTE: should not directly rob the budget from all the other partitions
        # CASE: suppose a task A executes on 1 -> 2 -> 3
        # and the task is late and misses the budeget on (1), then at arrival of A, 
        # it should be executed on 2, with the budget of sum of 1 and 2, without 3
        # CASE: 2 -> 1
        # scheduler scan (1), take from bk but got nothing
        # then scheduler scan (2), put the budget to (2) ruther than bk
        if _SchedTab.alloc_mod != 'exactly':
            # 1. process old events
            for msg in curr_cfg.event_list:
                results = parse_event_msg(msg)
                # {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
                # start, complete, migrate
                if results['event_type'] == "migrate":
                    if 'to' in results:
                        _p = process_dict[results['pid']]
                        rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
                        assert len(rem_flop_budget) <= 2
                        if _p in (running_queue.queue+ready_queue.queue):
                            _p.throttle_util(running_queue, ready_queue, throttle_list, sched, curr_t)
                        # backup the _p.rem_flop_budget
                        rem_flop_budget = _p.rem_flop_budget[bin_id]
                        if results['to'] in _p.rem_flop_budget:
                            _p.rem_flop_budget[results['to']] += rem_flop_budget
                        else:
                            _p.rem_flop_budget['bk'] = rem_flop_budget
                        _p.rem_flop_budget[bin_id] = 0
                elif results['event_type'] == "start":
                    pass
                elif results['event_type'] == "complete":
                    pass
                else:
                    raise ValueError(f"Unknown event type {results['event_type']}")
            
            # 2. clear the old events
            curr_cfg.event_list.clear()
            # 3. process new events
            # (slot_idx, Dict[pid, event_set])
            from functools import reduce
            if _SchedTab.sparse_event[_SchedTab.sparse_idx][1]:
                event_set = reduce(lambda x,y: x+y, map(lambda x:list(x), _SchedTab.sparse_event[_SchedTab.sparse_idx][1].values()))
                curr_cfg.event_list.extend(list(event_set))

        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))


    # ******************************************************
    # Mechanism of budget and progress recoder
    # 1. rem_flop_budget
    # Record the expected operators to be executed in the next few moments
    # 
    # 2. budget_recoder: 
    # record the upper bound of resource consumption (spatial and temporal)
    # The previous budget is covered, when the new chunk is entered.
    # Explanation: 
    # If the load of privious chunk is uncompleted,
    # the previous timeout budget is useless, 
    # because the comming computation should be allocated with resources as soon as ponssible
    # ******************************************************

    # At the beginning of each cfg chunk
    if curr_cfg.slot_s == n_slot:
        # cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
        cfg_slot_s, next_cfg, cfg_slot_num = curr_cfg.slot_s, curr_cfg.rsc_map, curr_cfg.slot_num
        cfg_flops_dict = curr_cfg.flops_dict 
        cfg_event_list = curr_cfg.event_list
        # replenish the budget
        for pid in next_cfg.keys():
            _p = process_dict[pid]
                    
            if bin_id not in _p.rem_flop_budget:
                _p.rem_flop_budget[bin_id] = 0
                        
            flops_tbd = cfg_flops_dict[pid]
            rem_flop_budget=_p.rem_flop_budget[bin_id]
            if rem_flop_budget> numerical_error_tol_abs or flops_tbd>numerical_error_tol_abs: 
                _p.rem_flop_budget[bin_id] += flops_tbd # * _p.var_scale_factor
                budget_recoder[pid] = [cfg_slot_s, next_cfg[pid], cfg_slot_num, True]
            
            # TODO: move the hit recoder to issue stage
            if _p.pid not in rsc_recoder_his:
                rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)

        for msg in curr_cfg.event_list:
            results = parse_event_msg(msg)
            # {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
            # start, complete, migrate
            if results['event_type'] == "migrate":
                if 'from' in results:
                    _p = process_dict[results['pid']]
                    rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
                    try:
                        assert len(rem_flop_budget) <= 2
                    except AssertionError:
                        print(f"20231126: CodingError, try to gurrante the budget only on one partition at a time")
                    # restore the _p.rem_flop_budget from BK
                    _p.rem_flop_budget[bin_id] += _p.rem_flop_budget.pop('bk', 0.)

        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady_cbs(sched, buffer, budget_recoder, active_list, ready_queue, throttle_list, curr_t, glb_name_p_dict, event_cache, bin_name, process_dict, show_warnings=show_warnings)
    # move the task to the ready queue
    bin_event_flg = throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg)

    # for _p in ready_queue.queue:
    #     # NOTE:cross cancelation
    #     _p:ProcessInt
    #     rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
    #     if len(rem_flop_budget) > 1:
    #         planned_flops = sum(rem_flop_budget.values())
    #         assert len(rem_flop_budget) == 2
    #         # move the remained budget to the current partition
    #         previous_partition_id = (set(rem_flop_budget.keys()) - {bin_id,}).pop()
    #         print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is cancled from bin {previous_partition_id:d} \
    #                 and moved to bin {bin_id:d} with budget {planned_flops:.6f}")
    #         _p.rem_flop_budget[previous_partition_id] = 0
    #         _p.rem_flop_budget[bin_id] = planned_flops

    #         # judge whether current slot is executed on other patition
    #         # if so, pass the current slot
    #         if previous_partition_id < bin_id:
    #             skiped_tasks.append(_p)

    if sched.drain_flg:
        sched.check_draining_state()

    # check the running tasks
    for _p in running_queue.queue:
        # detect the lateness of the tasks
        # if _p.pid not in curr_cfg.rsc_map: 
        if budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2] < n_slot:
            if show_warnings:
                warnings.warn("Execution lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    # build the local running configuration
    # Try to allocate the resource to the ready tasks
        # combine the ready tasks, running tasks, budget, and deadline together
        # co-operate the designed properties with EDF algorithm
            # stationary/movable
            # realtime/deadline
            # w/ or w/o need to be scaled
        # first stationary, then movable
        # tackle 3 cases:
            # 1. arrival lateness
            # 2. tighter deadline
            # 3. workload scaling
    # Two components:
        # 1. when to trigger
        # 2. compenstation algorithm
    
    # intution:
        # 1. as soon as possible (ASAP)
        # 2. as evenly as possible (AEAP)        
    # here, we adapt the ASAP to simplify the scheduling process

    # trigger condition:
        # detect risk of current block execution timeout

    # calculate the criticity of the task
    # the most critical one is the first one, with the smallest value, use the ascending order

    is_not_hard = lambda x: x.task.criticality != "hard"

    fn_crit = lambda x: x.deadline
    # fn_crit = lambda _p: budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2]

    fn_task_flag = lambda x: 0 if x.task.task_flag=="stationary" else 1

    # =============== cheduler trigger condition ===============

    # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
    # Positive cases that trigger the scheduler: 
    # A. running queue status changes: some task finish/miss/throttled & release cores
    # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
    #   idle resources, or preempt others
    # C. cfg of running tasks changes: want to extend/release some cores
    # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
    # E. resource ++: some task is waiting to get cores
    # F. some running task is starving 

    # Negative cases that not trigger the scheduler:
    # N.A: some free resources but all other thing not changes, 
    #   which means scheduler will get the same allocation for a new calling 

    # If no execution lateness in previous cfg, 
    # then the running queue is empty all the tasks chunks shares the same deadline; 
    # without spec changes, 
    # the execution sequence not matter the schedulibility. 
    # Otherwise, the task is sorted by the 
    # deadline, host or guest, running or not

    pre_rsc = res_cfg.rsc_map
    task_queue = running_queue.queue + ready_queue.queue
    # ready queue status
    trigger_condA = sched.new_ready_flg
    # running queue status && resource ++ 
    trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
    # budget indicator
    trigger_condC = sum([budget_recoder[_p.pid][3] for _p in task_queue])


    if trigger_condA or trigger_condB or trigger_condC:

        # =============== build the scehduling candidate list ===============
        preemptable_list = []
        curr_aval_rsc = aval_rsc
        if preemption_en:
            if quantum_check_en: 
                assert quantumSize is not None
                for _p_2b_preempt in running_queue.queue:
                    cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                    reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
                    if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                        continue
                    else:
                        preemptable_list.append(_p_2b_preempt)
                curr_aval_rsc = aval_rsc + sum([_p.required_resource_size for _p in preemptable_list])
            else:
                curr_aval_rsc = res_cfg.size
                preemptable_list = running_queue.queue

        if sched.drain_flg:
            score_fn = lambda x: (x.msg_cache[0].get_timestamp()>=sched.switch_border, fn_crit(x), fn_task_flag(x), x not in preemptable_list)
        else:
            score_fn = lambda x: (fn_crit(x), fn_task_flag(x), x not in preemptable_list)
        # if len(preemptable_list) > 0:
        #     threshold_item = min(preemptable_list, key=lambda x: score_fn(x))
        #     threshold_score = score_fn(threshold_item)
        # else:
        #     threshold_score = (np.inf, 1)
        threshold_score = (np.inf, 1, True)
        filtered_ready_queue = [_p for _p in ready_queue.queue if score_fn(_p) < threshold_score]
        sorted_queue = sorted(filtered_ready_queue+preemptable_list, key=score_fn,)
        # ===================================================================

        rsc_map = OrderedDict()

        while len(sorted_queue) > 0 and curr_aval_rsc > 0:
            _p:ProcessInt = sorted_queue[0]
            _p.is_starving = False

            # =============== calculate the required resource size ===============
            # 
            # get the newest assigned budget
            # planned_rsc_size = budget_recoder[_p.pid][1] # curr_cfg.rsc_map[_p.pid]
            # planned_slot_num = budget_recoder[_p.pid][2] # curr_cfg.slot_num
            
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            chunk_flops = chunk_alloc * chunk_slot_num * timestep * FLOPS_PER_CORE # _p.rem_flop_budget[bin_id]
            chunk_e = chunk_s + chunk_slot_num
            # late_slot_num = fn_trig(_p) 
            planned_flops = sum([v for v in _p.rem_flop_budget.values() if v > numerical_error_tol_abs])

            # - We discuss this issue in two scenarios:
            #     1. with data arriving on time: allocate the resources according to the budget
            #     2. with data arriving late: allocate the resources following the "EDF", and estimate the resources at runtime

            # lateness detection mechanism:
            #  both task-level and chunk-level
            # case 1: release late, i.e., the task is not released at the beginning of the current configuration
            # case 2: previous chunk is late, i.e., the task is not finished at the end of the previous configuration
            # case 3: current chunk is late, i.e., the task is not resumed at the beginning of the current configuration

            assert chunk_s <= n_slot, "chunk_s {:d} > n_slot {:d}".format(chunk_s, n_slot)
            if sched.drain_flg and _p.msg_cache[0].get_timestamp() < sched.switch_border:
                req_rsc_size = curr_aval_rsc
            elif chunk_s < n_slot < chunk_e:
                # case 1: release late !!! the running task that is identified as preemptable [chunk_s, chunk_e] 
                assert chunk_e == curr_cfg.slot_e + 1
                # req_rsc_size = math.ceil(planned_flops/(chunk_e - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate)) 
                slack = slack_comp((chunk_e - n_slot)*timestep, 0, sched.over_provision_rate)
                req_rsc_size = math.ceil(planned_flops/slack/FLOPS_PER_CORE) 
            elif n_slot >= chunk_e:
                # case 3: current chunk is late
                #   newest assigned budget is skipped
                req_rsc_size = curr_aval_rsc
            else:
                if round(planned_flops, flop1u_error_tol_bit) > round(chunk_flops, flop1u_error_tol_bit):
                    # case 2: previous chunk is late
                    #   newest assigned budget is still available but not enough
                    # req_rsc_size = math.ceil(planned_flops/(chunk_e + 1 - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate))
                    # req_rsc_size = math.ceil(planned_flops/(chunk_e - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate)) 
                    slack = slack_comp((chunk_e - n_slot)*timestep, 0, sched.over_provision_rate)
                    req_rsc_size = math.ceil(planned_flops/slack/FLOPS_PER_CORE) 
                else:
                    # newest assigned budget is still available                    
                    # tries to finish the remaining work assigned by the configuration chunk until the now
                    req_rsc_size = chunk_alloc 

            # **************************************************************
            # check the rsc_size is valid
            # compare with the core_max, core_min, core_list, parallel_mode
            # **************************************************************

            if req_rsc_size > curr_aval_rsc:
                if show_warnings: 
                    warnings.warn(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is starving {req_rsc_size-curr_aval_rsc:d} cores")
                req_rsc_size = curr_aval_rsc
                _p.is_starving = True

            if _p.parallel_mode in ["upb","range"]:
                if req_rsc_size > _p.core_max:
                    req_rsc_size = _p.core_max
            elif _p.parallel_mode in ["lwb", "range"]:
                if req_rsc_size < _p.core_min:
                    if _p.core_min > curr_aval_rsc:
                        # no available solution
                        if o3_boost_util_en:
                            skiped_tasks.append(_p)
                            sorted_queue.pop(0)
                            continue
                        else:
                            break
                    else:
                        req_rsc_size = _p.core_min
            elif _p.parallel_mode == "list":
                # select the nearest one
                # filter the core_list by the current available resource
                core_list = [x for x in _p.core_list if x <= curr_aval_rsc]
                if len(core_list) == 0:
                    # no available solution
                    if o3_boost_util_en:
                        skiped_tasks.append(_p)
                        sorted_queue.pop(0)
                        continue
                    else:
                        break
                req_rsc_size = min(core_list, key=lambda x:abs(x-req_rsc_size))

            if _p.totburst == 0 and chunk_s < n_slot:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is deteted a lateness of {(n_slot-chunk_s):d} slots")

            assert isinstance(req_rsc_size, (int, np.integer)), "req_rsc_size is not integer"
            if req_rsc_size == 0:
                if o3_boost_util_en:
                    skiped_tasks.append(_p)
                    sorted_queue.pop(0)
                    continue
                else:
                    break
            assert req_rsc_size > 0

            if _p.load_var == 0:
                req_rsc_size = 0
            
            # TODO: if the req_rsc_size is larger than the aval_rsc, then add a flag to indicate the task is late                
            sorted_queue.pop(0)
            # if curr_aval_rsc >= req_rsc_size: # and req_rsc_size > 0:
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            _p.required_resource_size = req_rsc_size

        sched.new_ready_flg = False

        to_assert_flag = False
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # update the position dict
            used_position = []
            # for pid in old_pid:
            #     p_size = rsc_map[pid]
            #     # set the is_new flag to False
            #     position_dict[pid][-1] = False
            #     for s, size in zip(*position_dict[pid][:-1]):
            #         e = s + size
            #         used_position += [i for i in range(s, e)]

            # remove the expired task from the position dict
            for pid in expired_pid:
                position_dict.pop(pid)
                preempt_list.append(process_dict[pid])

            for pid in position_dict.keys():
                for s, size in zip(*position_dict[pid][:-1]):
                    e = s + size
                    used_position += [i for i in range(s, e)]               
            aval_pos = [i for i in sched.core_map if i not in used_position]
            
            # check if the old task's allocation is changed
            # if so, release the old position and allocate the new one
            # to release the data transfering overhead, we try to allocate the new position as close as possible to the old one
            # TODO: consider the data transfering overhead
            size_plus = []
            size_minus = []
            for pid in old_pid:
                old_size = pre_rsc[pid]
                new_size = rsc_map[pid]
                if old_size != new_size:
                    ctx_switch_list.append(process_dict[pid])
                if new_size > old_size:
                    size_plus.append(pid)
                elif new_size < old_size:
                    size_minus.append(pid)

            for group in [size_minus, size_plus]:
                for pid in group: 
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    # release the old position
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        aval_pos += [i for i in range(s, e)]
                    aval_pos.sort()
                    # get the start position of the old task
                    cum_pos = position_dict[pid][0][0]
                    # divide the available position into two parts
                    left_pos = aval_pos[:aval_pos.index(cum_pos)]
                    right_pos = aval_pos[aval_pos.index(cum_pos):]
                    # select the leftmost position from cum_pos
                    interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                    if len(interval_picked) < new_size:
                        # select the leftmost position from left_pos
                        interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(new_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, position_dict[pid][-1]]
            
            # pick a proper position for the new task in the available position
            for pid in new_pid:
                p_size = rsc_map[pid]
                # search for a gap in the available positions that can accommodate the task's new size
                gap_start = None
                gap_size = 0
                for pos in aval_pos:
                    if gap_start is None:
                        gap_start = pos
                    gap_size += 1
                    if gap_size == p_size:
                        break
                    if pos + 1 not in aval_pos:
                        gap_start = None
                        gap_size = 0

                if gap_start is not None and gap_size == p_size:
                    # allocate the task to the found gap
                    start = [gap_start]
                    size = [gap_size]
                    position_dict[pid] = [start, size, True]
                    # remove the selected positions from aval_pos
                    aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                else:
                    # select the leftmost position
                    interval_picked = aval_pos[:p_size]
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(p_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, True]
                issue_list.append(process_dict[pid])

            # logic to judge whether the swithing is planned or decided by the scheduler at runtime
            # the swithing out of plan features:
            # 1. some tasks release core and other tasks take the core
            # 2. this overtake behavior is not planned in the scheduling table
            # step: 
            # 1. check whether some cores are released
            # 2. check whether this plan is in the scheduling table
            off_discount = 0
            on_discount = 0
            for _p in ctx_switch_list:
                old_size = pre_rsc[_p.pid]
                new_size = rsc_map[_p.pid]
                if old_size > new_size: 
                    if budget_recoder[_p.pid][3]:
                        chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                        if new_size != chunk_alloc:
                            to_assert_flag = True
                        else:
                            off_discount += chunk_alloc
                    else:
                        to_assert_flag = True
                else:
                    chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                    if new_size == chunk_alloc:
                        on_discount += chunk_alloc

            
            if len(preempt_list):
                to_assert_flag = True

            if to_assert_flag:
                for _p in issue_list:
                    chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                    if chunk_alloc == rsc_map[_p.pid]:
                        on_discount += chunk_alloc

            # update the resource configuration
            for _p in preempt_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
                running_queue.remove(_p)
                ready_queue.put(_p)
                sched.new_ready_flg = True
                sched.res_release(_p.pid, False)
            preempt_list.clear()
            
            for _p in ctx_switch_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) ctx switch at {curr_t:.6f}({pre_rsc[_p.pid]} -> {rsc_map[_p.pid]});")
                sched.res_release(_p.pid, False)
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
            ctx_switch_list.clear()

            # if issue the task to runnning list
            for _p in issue_list:
                running_queue.put(_p)
                ready_queue.remove(_p)
                # NOTE:cross cancelation (removed)
                _p.set_state("running")
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
                _p.waitTime = 0 
                _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued and "
                if _p.totburst==0:
                    _p.start_time = curr_t
                    _str += f"start at {curr_t:.6f}; "
                else:
                    _str += f"resume at {curr_t:.6f}; "
                _p.curr_start_time = curr_t
                if bin_name and not bin_event_flg:
                    bin_event_flg = True 
                    print(f"({bin_name})")
                print(_str)
            issue_list.clear()

        for pid in budget_recoder:
            budget_recoder[pid][3] = False
        
        if to_assert_flag:
            # assert a barrier, sim data movement: 
            if sched.barrier_en:
                off_size = (sum(pre_rsc.values())-off_discount) * 1/3
                on_size = (sum(rsc_map.values()) - on_discount) * 2/3
                x = (on_size+off_size) 
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    if sched.drain_flg:
        sched.drain_old_cores()
    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue, True, bin_id, skiped_tasks=skiped_tasks) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

def read_msg_queue(sched:Scheduler, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id, res_cfg=None, budget_recoder=None,):
    # 定义正则表达式模式
    cancel_pattern = r'TASK (\d+):([\w_]+)\((\d+)\) (?:COMPLETED|MISSED DEADLINE) @ ([\d.]+)/([\d.]+)\(([\w_]+)\)!!'
    core_release_pattern = r'CORE ([\d]+) is set free'
    msg_list = []
    # read out all message and clear the message pipe
    while not msg_queue.empty():
        msg_list.append(msg_queue.get())
    if msg_list:
        for msg in msg_list:
        # 使用正则表达式进行匹配
            if match:= re.match(cancel_pattern, msg):
                # 提取匹配结果
                pid = int(match.group(3))
                name = match.group(2)
                event_time = float(match.group(5))
                bin_name_t = match.group(6)
                if pid not in process_dict or bin_name_t == bin_name:
                    continue
                _p = process_dict[pid]

                # pop the task from ready queue, active list, throttle list and running queue
                if _p in active_list:
                    active_list.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from active list @ {bin_name:s} {curr_t:.6f}")
                    inactive_list.append(_p)
                if _p in ready_queue:
                    ready_queue.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from ready queue @ {bin_name:s} {curr_t:.6f}")
                    inactive_list.append(_p)
                if _p in throttle_list:
                    throttle_list.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from throttle list @ {bin_name:s} {curr_t:.6f}")
                    inactive_list.append(_p)
                if _p in running_queue:
                    running_queue.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from running queue @ {bin_name:s} {curr_t:.6f}")
                    sched.res_release(_p.pid)
                    inactive_list.append(_p)
                    # if budget_recoder is not None:
                    #     budget_recoder.pop(_p.pid)
                    # _p.rem_flop_budget.pop(bin_id)
            # elif match := re.match(core_release_pattern, msg):
            #     core_id = int(match.group(1))
            #     if core_id in sched.core_map_L0:
            #         sched.core_map_L0[core_id] = True
            #         if sched.drain_flg:
            #             sched.core_map.append(core_id)
            #             sched.core_map.sort()
            #             sched.core_map_L0.pop(core_id)
                    
def scheduler_step_budget_old(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()

    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)


    for pid in list(budget_recoder.keys()):
        _p = process_dict[pid]
        rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
        if rem_flop_budget < numerical_error_tol_abs and _p.pid in budget_recoder:
            # Each partition has splited budget recoder 
            budget_recoder.pop(pid)
            _p.rem_flop_budget[bin_id] = 0

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, budget_recoder, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, budget_recoder, msg_dispatcher, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    bin_event_flg = check_throttle(sched, budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, bin_event_flg, bin_name)


    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)


    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.next_item()
        if _SchedTab.alloc_mod != 'exactly':
            curr_cfg.update(_SchedTab.sparse_cores[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.update(next_cfg)
        # update the deadline
        if cfg_slot_s < tab_pointer: 
            curr_cfg.slot_s = (hyper_p_n + 1) * tab_temp_size + cfg_slot_s
        else:
            curr_cfg.slot_s = hyper_p_n * tab_temp_size + cfg_slot_s
        curr_cfg.slot_e = curr_cfg.slot_s + cfg_slot_num - 1 
        curr_cfg.slot_num = cfg_slot_num
        curr_cfg.flops_dict.clear()
        if _SchedTab.alloc_mod != 'exactly':
            curr_cfg.flops_dict.update(_SchedTab.sparse_flops[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.flops_dict.update({pid: n_cores * cfg_slot_num * timestep * FLOPS_PER_CORE for pid, n_cores in next_cfg.items()})
        
        # process the events
        # process old events -> clear -> process new events
        # NOTE: should not directly rob the budget from all the other partitions
        # CASE: suppose a task A executes on 1 -> 2 -> 3
        # and the task is late and misses the budeget on (1), then at arrival of A, 
        # it should be executed on 2, with the budget of sum of 1 and 2, without 3
        # CASE: 2 -> 1
        # scheduler scan (1), take from bk but got nothing
        # then scheduler scan (2), put the budget to (2) ruther than bk
        if _SchedTab.alloc_mod != 'exactly':
            # 1. process old events
            for msg in curr_cfg.event_list:
                results = parse_event_msg(msg)
                # {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
                # start, complete, migrate
                if results['event_type'] == "migrate":
                    if 'to' in results:
                        _p = process_dict[results['pid']]
                        rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
                        assert len(rem_flop_budget) <= 2
                        if _p in (running_queue.queue+ready_queue.queue):
                            _p.throttle_util(running_queue, ready_queue, throttle_list, sched, curr_t)
                        # backup the _p.rem_flop_budget
                        rem_flop_budget = _p.rem_flop_budget[bin_id]
                        if results['to'] in _p.rem_flop_budget:
                            _p.rem_flop_budget[results['to']] += rem_flop_budget
                        else:
                            _p.rem_flop_budget['bk'] = rem_flop_budget
                        _p.rem_flop_budget[bin_id] = 0
                elif results['event_type'] == "start":
                    pass
                elif results['event_type'] == "complete":
                    pass
                else:
                    raise ValueError(f"Unknown event type {results['event_type']}")
            
            # 2. clear the old events
            curr_cfg.event_list.clear()
            # 3. process new events
            # (slot_idx, Dict[pid, event_set])
            from functools import reduce
            if _SchedTab.sparse_event[_SchedTab.sparse_idx][1]:
                event_set = reduce(lambda x,y: x+y, map(lambda x:list(x), _SchedTab.sparse_event[_SchedTab.sparse_idx][1].values()))
                curr_cfg.event_list.extend(list(event_set))

        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))


    # ******************************************************
    # Mechanism of budget and progress recoder
    # 1. rem_flop_budget
    # Record the expected operators to be executed in the next few moments
    # 
    # 2. budget_recoder: 
    # record the upper bound of resource consumption (spatial and temporal)
    # The previous budget is covered, when the new chunk is entered.
    # Explanation: 
    # If the load of privious chunk is uncompleted,
    # the previous timeout budget is useless, 
    # because the comming computation should be allocated with resources as soon as ponssible
    # ******************************************************

    # At the beginning of each cfg chunk
    if curr_cfg.slot_s == n_slot:
        # cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
        cfg_slot_s, next_cfg, cfg_slot_num = curr_cfg.slot_s, curr_cfg.rsc_map, curr_cfg.slot_num
        cfg_flops_dict = curr_cfg.flops_dict 
        cfg_event_list = curr_cfg.event_list
        # replenish the budget
        for pid in next_cfg.keys():
            _p = process_dict[pid]
                    
            if bin_id not in _p.rem_flop_budget:
                _p.rem_flop_budget[bin_id] = 0
                        
            flops_tbd = cfg_flops_dict[pid]
            rem_flop_budget=_p.rem_flop_budget[bin_id]
            if rem_flop_budget> numerical_error_tol_abs or flops_tbd>numerical_error_tol_abs: 
                _p.rem_flop_budget[bin_id] += flops_tbd # * _p.var_scale_factor
                budget_recoder[pid] = [cfg_slot_s, next_cfg[pid], cfg_slot_num, True]
            
            # TODO: move the hit recoder to issue stage
            if _p.pid not in rsc_recoder_his:
                rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)

        for msg in curr_cfg.event_list:
            results = parse_event_msg(msg)
            # {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
            # start, complete, migrate
            if results['event_type'] == "migrate":
                if 'from' in results:
                    _p = process_dict[results['pid']]
                    rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
                    try:
                        assert len(rem_flop_budget) <= 2
                    except AssertionError:
                        print(f"20231126: CodingError, try to gurrante the budget only on one partition at a time")
                    # restore the _p.rem_flop_budget from BK
                    _p.rem_flop_budget[bin_id] += _p.rem_flop_budget.pop('bk', 0.)

        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady_cbs(sched, buffer, budget_recoder, active_list, ready_queue, throttle_list, curr_t, glb_name_p_dict, event_cache, bin_name, process_dict, show_warnings=show_warnings)
    # move the task to the ready queue
    bin_event_flg = throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg)

    # check the running tasks
    for _p in running_queue.queue:
        # detect the lateness of the tasks
        # if _p.pid not in curr_cfg.rsc_map: 
        if budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2] < n_slot:
            if show_warnings:
                warnings.warn("Execution lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    # Scheduler is triggered when:
    # either the aval_rsc or the candidate changes, i.e.,
    # A. task release
    # 1. new tasks join the ready queue, preemption may happen
    # 2. Budget updated

    # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
    # Positive cases that trigger the scheduler: 
    # A. running queue status changes: some task finish/miss/throttled & release cores
    # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
    #   idle resources, or preempt others
    # C. cfg of running tasks changes: want to extend/release some cores
    # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
    # E. resource ++: some task is waiting to get cores
    # F. some running task is starving 

    # Negative cases that not trigger the scheduler:
    # N.A: some free resources but all other thing not changes, 
    #   which means scheduler will get the same allocation for a new calling 

    pre_rsc = res_cfg.rsc_map
    task_queue = running_queue.queue + ready_queue.queue
    # ready queue status
    trigger_condA = sched.new_ready_flg
    # running queue status && resource ++ 
    trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
    # budget indicator
    trigger_condC = sum([budget_recoder[_p.pid][3] for _p in task_queue])


    if trigger_condA or trigger_condB or trigger_condC:
        curr_aval_rsc = res_cfg.size
        res_cfg.clear()
        while len(task_queue) > 0:
            _p:ProcessInt = task_queue[0]
            
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            chunk_e = chunk_s + chunk_slot_num
            # late_slot_num = fn_trig(_p) 

            # case 1: release late, i.e., the task is not released at the beginning of the current configuration
            # case 2: previous chunk is late, i.e., the task is not finished at the end of the previous configuration
            # case 3: current chunk is late, i.e., the task is not resumed at the beginning of the current configuration

            # if there is resource available, allocate the task as planned
            # if not late read the bandwidth from the current configuration
            # if the task is timeout, read from the _p.required_resource_size

            assert chunk_s <= n_slot, "chunk_s {:d} > n_slot {:d}".format(chunk_s, n_slot)
            if chunk_s < n_slot < chunk_e:
                # case 1: release late !!! the running task that is identified as preemptable [chunk_s, chunk_e] 
                assert chunk_e == curr_cfg.slot_e + 1
                # newest assigned budget is still available                    
                # tries to finish the remaining work assigned by the configuration chunk until the now
                req_rsc_size = chunk_alloc 
            else:
                # case 2: the task is late 
                # case 3: current chunk is late
                if curr_aval_rsc >= chunk_alloc and running_queue.queue:
                    req_rsc_size = _p.required_resource_size
                else:
                    req_rsc_size = 0
                

            # **************************************************************
            # check the rsc_size is valid
            # compare with the core_max, core_min, core_list, parallel_mode
            # **************************************************************

            if _p.totburst == 0 and chunk_s < n_slot:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is deteted a lateness of {(n_slot-chunk_s):d} slots")

            if req_rsc_size == 0:
                skiped_tasks.append(_p)
                task_queue.remove(_p)
                continue
            assert req_rsc_size > 0

            if _p.load_var == 0:
                req_rsc_size = 0
            
            # TODO: if the req_rsc_size is larger than the aval_rsc, then add a flag to indicate the task is late                
            task_queue.remove(_p)
            issue_list.append(_p)
            _p.required_resource_size = req_rsc_size
            curr_aval_rsc -= req_rsc_size

        sched.new_ready_flg = False
        
        running_queue.queue.clear()
        # if issue the task to runnning list
        for _p in issue_list:
            running_queue.put(_p)
            if _p in ready_queue.queue:
                ready_queue.queue.remove(_p)
            _p.set_state("running")
            res_cfg.allocate(_p.pid, _p.required_resource_size)
            _p.waitTime = 0 
            _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued and "
            if _p.totburst==0:
                _p.start_time = curr_t
                _str += f"start at {curr_t:.6f}; "
            else:
                _str += f"resume at {curr_t:.6f}; "
            _p.curr_start_time = curr_t
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(_str)
        issue_list.clear()

        for pid in budget_recoder:
            budget_recoder[pid][3] = False

    # execute the task in running list
    # update the running task
    res_cfg.updateRunningQueue(timestep, running_queue, True, bin_id, skiped_tasks=skiped_tasks) 

    monitor.add_a_record(res_cfg)

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

def scheduler_step_cyclic(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, None, msg_dispatcher, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)


    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.next_item()
        curr_cfg.update(next_cfg)
        # update the deadline
        if cfg_slot_s < tab_pointer: 
            curr_cfg.slot_s = (hyper_p_n + 1) * tab_temp_size + cfg_slot_s
        else:
            curr_cfg.slot_s = hyper_p_n * tab_temp_size + cfg_slot_s
        curr_cfg.slot_e = curr_cfg.slot_s + cfg_slot_num - 1 
        curr_cfg.slot_num = cfg_slot_num
        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))

    new_cfg_ld = False
    if curr_cfg.slot_s == n_slot:
        # cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
        cfg_slot_s, next_cfg, cfg_slot_num = curr_cfg.slot_s, curr_cfg.rsc_map, curr_cfg.slot_num

        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    if curr_cfg.slot_s <= n_slot and n_slot <= curr_cfg.slot_e:

        # Scheduler is triggered when:
        # either the aval_rsc or the candidate changes, i.e.,
        # A. task release
        # 1. new tasks join the ready queue, preemption may happen
        # 2. Budget updated

        curr_aval_rsc = res_cfg.size
        pre_rsc = res_cfg.rsc_map
        task_queue = running_queue.queue + ready_queue.queue
        # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
        # cases trigger the scheduler: 
        # A. running queue status changes: some task finish/miss/throttled & release cores
        # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
        #   idle resources, or preempt others
        # C. cfg of running tasks changes: want to extend/release some cores
        # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
        # E. resource ++: some task is waiting to get cores
        # F. some running task is starving 

        # case that not trigger the scheduler:
        # N.A: some free resources but all other thing not changes, 
        #   which means scheduler will get the same allocation for a new calling 

        # ready queue status
        trigger_condA = sched.new_ready_flg
        # running queue status && resource ++ 
        # trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
        # budget indicator
        trigger_condC = (curr_cfg.slot_s == n_slot) 

        # 20240731: remove condition B, because in this scheduler, we never change the resource bandwidth, 
        # so no starving flage will be set. 
        if trigger_condA or trigger_condC:

            while len(task_queue) > 0:
                _p:ProcessInt = task_queue.pop(0)
                if _p.pid in curr_cfg.rsc_map.keys():
                    issue_list.append(_p)
                    req_rsc_size = curr_cfg.rsc_map[_p.pid]
                    assert req_rsc_size > 0
                    _p.required_resource_size = req_rsc_size
                    curr_aval_rsc -= req_rsc_size
                    assert curr_aval_rsc >= 0 

            # new_pid = set(curr_cfg.rsc_map.keys()) - set(res_cfg.rsc_map.keys())
            expired_pid = set(res_cfg.rsc_map.keys()) - set(curr_cfg.rsc_map.keys())
            old_pid = set(res_cfg.rsc_map.keys()) - expired_pid
            
            sched.new_ready_flg = False
            running_queue.queue.clear()
            res_cfg.clear()
            
            # move the expired tasks into ready queue
            for pid in expired_pid: 
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
                ready_queue.put(process_dict[pid])

            running_queue.queue.clear()
            # if issue the task to runnning list
            for _p in issue_list:
                running_queue.put(_p)
                if _p in ready_queue.queue:
                    ready_queue.queue.remove(_p)
                _p.set_state("running")
                res_cfg.allocate(_p.pid, _p.required_resource_size)
                _p.waitTime = 0 
                _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued "
                if _p.totburst==0:
                    _p.start_time = curr_t
                    _str += f"and start at {curr_t:.6f}; "
                elif _p.pid not in old_pid:
                    _str += f"and resume at {curr_t:.6f}; "
                _p.curr_start_time = curr_t
                if bin_name and not bin_event_flg:
                    bin_event_flg = True 
                    print(f"({bin_name})")
                print(_str)
            issue_list.clear()
        
    
    # execute the task in running list
    # update the running task
    res_cfg.updateRunningQueue(timestep, running_queue) 

    monitor.add_a_record(res_cfg)

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

def scheduler_step_fifo(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()

    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    for pid in list(budget_recoder.keys()):
        _p = process_dict[pid]
        rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
        if rem_flop_budget < numerical_error_tol_abs and _p.pid in budget_recoder:
            # Each partition has splited budget recoder 
            budget_recoder.pop(pid)
            _p.rem_flop_budget[bin_id] = 0

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, budget_recoder, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, budget_recoder, msg_dispatcher, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    bin_event_flg = check_throttle(sched, budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, bin_event_flg, bin_name)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)
    # buffer.recyle_no_ref("output", True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)

    # # instruction prefetching
    # cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]         
    
    # # if the pointer reaches the start of the next cfg, update the cfg
    # if tab_pointer == cfg_slot_s:
    #     # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
    #     curr_cfg.update(cached_cfg)
    #     # move the pointer to the next cfg
    #     _SchedTab.idx_plus_1()

    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.next_item()
        if _SchedTab.alloc_mod != 'exactly':
            curr_cfg.update(_SchedTab.sparse_cores[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.update(next_cfg)
        # update the deadline
        if cfg_slot_s < tab_pointer: 
            curr_cfg.slot_s = (hyper_p_n + 1) * tab_temp_size + cfg_slot_s
        else:
            curr_cfg.slot_s = hyper_p_n * tab_temp_size + cfg_slot_s
        curr_cfg.slot_e = curr_cfg.slot_s + cfg_slot_num - 1 
        curr_cfg.slot_num = cfg_slot_num
        curr_cfg.flops_dict.clear()
        if _SchedTab.alloc_mod != 'exactly':
            curr_cfg.flops_dict.update(_SchedTab.sparse_flops[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.flops_dict.update({pid: n_cores * cfg_slot_num * timestep * FLOPS_PER_CORE for pid, n_cores in next_cfg.items()})
        
        # process the events
        # process old events -> clear -> process new events
        # NOTE: should not directly rob the budget from all the other partitions
        # CASE: suppose a task A executes on 1 -> 2 -> 3
        # and the task is late and misses the budeget on (1), then at arrival of A, 
        # it should be executed on 2, with the budget of sum of 1 and 2, without 3
        # CASE: 2 -> 1
        # scheduler scan (1), take from bk but got nothing
        # then scheduler scan (2), put the budget to (2) ruther than bk
        if _SchedTab.alloc_mod != 'exactly':
            # 1. process old events
            for msg in curr_cfg.event_list:
                results = parse_event_msg(msg)
                # {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
                # start, complete, migrate
                if results['event_type'] == "migrate":
                    if 'to' in results:
                        _p = process_dict[results['pid']]
                        rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
                        assert len(rem_flop_budget) <= 2
                        if _p in (running_queue.queue+ready_queue.queue):
                            _p.throttle_util(running_queue, ready_queue, throttle_list, sched, curr_t)
                        # backup the _p.rem_flop_budget
                        rem_flop_budget = _p.rem_flop_budget[bin_id]
                        if results['to'] in _p.rem_flop_budget:
                            _p.rem_flop_budget[results['to']] += rem_flop_budget
                        else:
                            _p.rem_flop_budget['bk'] = rem_flop_budget
                        _p.rem_flop_budget[bin_id] = 0
                elif results['event_type'] == "start":
                    pass
                elif results['event_type'] == "complete":
                    pass
                else:
                    raise ValueError(f"Unknown event type {results['event_type']}")
            
            # 2. clear the old events
            curr_cfg.event_list.clear()
            # 3. process new events
            # (slot_idx, Dict[pid, event_set])
            from functools import reduce
            if _SchedTab.sparse_event[_SchedTab.sparse_idx][1]:
                event_set = reduce(lambda x,y: x+y, map(lambda x:list(x), _SchedTab.sparse_event[_SchedTab.sparse_idx][1].values()))
                curr_cfg.event_list.extend(list(event_set))

        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))


    # ******************************************************
    # Mechanism of budget and progress recoder
    # 1. rem_flop_budget
    # Record the expected operators to be executed in the next few moments
    # 
    # 2. budget_recoder: 
    # record the upper bound of resource consumption (spatial and temporal)
    # The previous budget is covered, when the new chunk is entered.
    # Explanation: 
    # If the load of privious chunk is uncompleted,
    # the previous timeout budget is useless, 
    # because the comming computation should be allocated with resources as soon as ponssible
    # ******************************************************

    # At the beginning of each cfg chunk
    if curr_cfg.slot_s == n_slot:
        # cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
        cfg_slot_s, next_cfg, cfg_slot_num = curr_cfg.slot_s, curr_cfg.rsc_map, curr_cfg.slot_num
        cfg_flops_dict = curr_cfg.flops_dict 
        cfg_event_list = curr_cfg.event_list
        # replenish the budget
        for pid in next_cfg.keys():
            _p = process_dict[pid]
                    
            if bin_id not in _p.rem_flop_budget:
                _p.rem_flop_budget[bin_id] = 0
                        
            flops_tbd = cfg_flops_dict[pid]
            rem_flop_budget=_p.rem_flop_budget[bin_id]
            if rem_flop_budget> numerical_error_tol_abs or flops_tbd>numerical_error_tol_abs: 
                _p.rem_flop_budget[bin_id] += flops_tbd # * _p.var_scale_factor
                budget_recoder[pid] = [cfg_slot_s, next_cfg[pid], cfg_slot_num, True]
            
            # TODO: move the hit recoder to issue stage
            if _p.pid not in rsc_recoder_his:
                rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)

        for msg in curr_cfg.event_list:
            results = parse_event_msg(msg)
            # {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
            # start, complete, migrate
            if results['event_type'] == "migrate":
                if 'from' in results:
                    _p = process_dict[results['pid']]
                    rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
                    try:
                        assert len(rem_flop_budget) <= 2
                    except AssertionError:
                        print(f"20231126: CodingError, try to gurrante the budget only on one partition at a time")
                    # restore the _p.rem_flop_budget from BK
                    _p.rem_flop_budget[bin_id] += _p.rem_flop_budget.pop('bk', 0.)

        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady_cbs(sched, buffer, budget_recoder, active_list, ready_queue, throttle_list, curr_t, glb_name_p_dict, event_cache, bin_name, process_dict, show_warnings=show_warnings)
    # move the task to the ready queue
    bin_event_flg = throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg)

    # for _p in ready_queue.queue:
    #     # NOTE:cross cancelation
    #     _p:ProcessInt
    #     rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
    #     if len(rem_flop_budget) > 1:
    #         planned_flops = sum(rem_flop_budget.values())
    #         assert len(rem_flop_budget) == 2
    #         # move the remained budget to the current partition
    #         previous_partition_id = (set(rem_flop_budget.keys()) - {bin_id,}).pop()
    #         print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is cancled from bin {previous_partition_id:d} \
    #                 and moved to bin {bin_id:d} with budget {planned_flops:.6f}")
    #         _p.rem_flop_budget[previous_partition_id] = 0
    #         _p.rem_flop_budget[bin_id] = planned_flops

    #         # judge whether current slot is executed on other patition
    #         # if so, pass the current slot
    #         if previous_partition_id < bin_id:
    #             skiped_tasks.append(_p)

    if sched.drain_flg:
        sched.check_draining_state()

    # check the running tasks
    for _p in running_queue.queue:
        # detect the lateness of the tasks
        # if _p.pid not in curr_cfg.rsc_map: 
        if budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2] < n_slot:
            if show_warnings:
                warnings.warn("Execution lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    # build the local running configuration
    # Try to allocate the resource to the ready tasks
        # combine the ready tasks, running tasks, budget, and deadline together
        # co-operate the designed properties with EDF algorithm
            # stationary/movable
            # realtime/deadline
            # w/ or w/o need to be scaled
        # first stationary, then movable
        # tackle 3 cases:
            # 1. arrival lateness
            # 2. tighter deadline
            # 3. workload scaling
    # Two components:
        # 1. when to trigger
        # 2. compenstation algorithm
    
    # intution:
        # 1. as soon as possible (ASAP)
        # 2. as evenly as possible (AEAP)        
    # here, we adapt the ASAP to simplify the scheduling process

    # trigger condition:
        # detect risk of current block execution timeout

    # calculate the criticity of the task
    # the most critical one is the first one, with the smallest value, use the ascending order

    is_not_hard = lambda x: x.task.criticality != "hard"

    fn_crit = lambda x: x.deadline
    # fn_crit = lambda _p: budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2]

    fn_task_flag = lambda x: 0 if x.task.task_flag=="stationary" else 1

    # =============== cheduler trigger condition ===============

    # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
    # Positive cases that trigger the scheduler: 
    # A. running queue status changes: some task finish/miss/throttled & release cores
    # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
    #   idle resources, or preempt others
    # C. cfg of running tasks changes: want to extend/release some cores
    # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
    # E. resource ++: some task is waiting to get cores
    # F. some running task is starving 

    # Negative cases that not trigger the scheduler:
    # N.A: some free resources but all other thing not changes, 
    #   which means scheduler will get the same allocation for a new calling 

    # If no execution lateness in previous cfg, 
    # then the running queue is empty all the tasks chunks shares the same deadline; 
    # without spec changes, 
    # the execution sequence not matter the schedulibility. 
    # Otherwise, the task is sorted by the 
    # deadline, host or guest, running or not

    pre_rsc = res_cfg.rsc_map
    task_queue = running_queue.queue + ready_queue.queue
    # ready queue status
    trigger_condA = sched.new_ready_flg
    # running queue status && resource ++ 
    trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
    # budget indicator
    trigger_condC = sum([budget_recoder[_p.pid][3] for _p in task_queue])


    # Compared with the original scheduler, this scheduler remove the ability to 
    # adjust the resource allocation based on the workload. 
    # Consequntly, the scheduler only try to allocate the resource as planned, 
    # if there are enough resources, otherwise, the task is starving. 
    if trigger_condA or trigger_condB or trigger_condC:

        # =============== build the scehduling candidate list ===============
        preemptable_list = []
        curr_aval_rsc = aval_rsc
        if preemption_en:
            if quantum_check_en: 
                assert quantumSize is not None
                for _p_2b_preempt in running_queue.queue:
                    cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                    reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
                    if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                        continue
                    else:
                        preemptable_list.append(_p_2b_preempt)
                curr_aval_rsc = aval_rsc + sum([_p.required_resource_size for _p in preemptable_list])
            else:
                curr_aval_rsc = res_cfg.size
                preemptable_list = running_queue.queue

        # =============== no need to filter the tasks ===============
        score_fn = lambda x: (fn_crit(x), x not in preemptable_list)
        # ===================================================================
        threshold_score = (np.inf, 1, True)
        filtered_ready_queue = [_p for _p in ready_queue.queue if score_fn(_p) < threshold_score]
        sorted_queue = sorted(filtered_ready_queue+preemptable_list, key=score_fn,)

        rsc_map = OrderedDict()

        while len(sorted_queue) > 0 and curr_aval_rsc > 0:
            _p:ProcessInt = sorted_queue[0]
            _p.is_starving = False

            # =============== calculate the required resource size ===============
            # 
            # get the newest assigned budget
            # planned_rsc_size = budget_recoder[_p.pid][1] # curr_cfg.rsc_map[_p.pid]
            # planned_slot_num = budget_recoder[_p.pid][2] # curr_cfg.slot_num
            
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            # chunk_flops = chunk_alloc * chunk_slot_num * timestep * FLOPS_PER_CORE # _p.rem_flop_budget[bin_id]
            chunk_e = chunk_s + chunk_slot_num
            # late_slot_num = fn_trig(_p) 
            # planned_flops = sum([v for v in _p.rem_flop_budget.values() if v > numerical_error_tol_abs])

            # - We discuss this issue in two scenarios:
            #     1. with data arriving on time: allocate the resources according to the budget
            #     2. with data arriving late: allocate the resources following the "EDF", and estimate the resources at runtime

            # lateness detection mechanism:
            #  both task-level and chunk-level
            # case 1: release late, i.e., the task is not released at the beginning of the current configuration
            # case 2: previous chunk is late, i.e., the task is not finished at the end of the previous configuration
            # case 3: current chunk is late, i.e., the task is not resumed at the beginning of the current configuration

            req_rsc_size = chunk_alloc 

            # **************************************************************
            # check the rsc_size is valid
            # compare with the core_max, core_min, core_list, parallel_mode
            # **************************************************************

            if req_rsc_size > curr_aval_rsc:
                if show_warnings: 
                    warnings.warn(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is starving {req_rsc_size-curr_aval_rsc:d} cores")
                req_rsc_size = curr_aval_rsc
                _p.is_starving = True

            if _p.parallel_mode in ["upb","range"]:
                if req_rsc_size > _p.core_max:
                    req_rsc_size = _p.core_max
            elif _p.parallel_mode in ["lwb", "range"]:
                if req_rsc_size < _p.core_min:
                    if _p.core_min > curr_aval_rsc:
                        # no available solution
                        if o3_boost_util_en:
                            skiped_tasks.append(_p)
                            sorted_queue.pop(0)
                            continue
                        else:
                            break
                    else:
                        req_rsc_size = _p.core_min
            elif _p.parallel_mode == "list":
                # select the nearest one
                # filter the core_list by the current available resource
                core_list = [x for x in _p.core_list if x <= curr_aval_rsc]
                if len(core_list) == 0:
                    # no available solution
                    if o3_boost_util_en:
                        skiped_tasks.append(_p)
                        sorted_queue.pop(0)
                        continue
                    else:
                        break
                req_rsc_size = min(core_list, key=lambda x:abs(x-req_rsc_size))

            if _p.totburst == 0 and chunk_s < n_slot:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is deteted a lateness of {(n_slot-chunk_s):d} slots")

            assert isinstance(req_rsc_size, (int, np.integer)), "req_rsc_size is not integer"
            if req_rsc_size == 0:
                if o3_boost_util_en:
                    skiped_tasks.append(_p)
                    sorted_queue.pop(0)
                    continue
                else:
                    break
            assert req_rsc_size > 0

            if _p.load_var == 0:
                req_rsc_size = 0
            
            # TODO: if the req_rsc_size is larger than the aval_rsc, then add a flag to indicate the task is late                
            sorted_queue.pop(0)
            # if curr_aval_rsc >= req_rsc_size: # and req_rsc_size > 0:
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            _p.required_resource_size = req_rsc_size

        sched.new_ready_flg = False

        to_assert_flag = False
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # update the position dict
            used_position = []
            # for pid in old_pid:
            #     p_size = rsc_map[pid]
            #     # set the is_new flag to False
            #     position_dict[pid][-1] = False
            #     for s, size in zip(*position_dict[pid][:-1]):
            #         e = s + size
            #         used_position += [i for i in range(s, e)]

            # remove the expired task from the position dict
            for pid in expired_pid:
                position_dict.pop(pid)
                preempt_list.append(process_dict[pid])

            for pid in position_dict.keys():
                for s, size in zip(*position_dict[pid][:-1]):
                    e = s + size
                    used_position += [i for i in range(s, e)]               
            aval_pos = [i for i in sched.core_map if i not in used_position]
            
            # check if the old task's allocation is changed
            # if so, release the old position and allocate the new one
            # to release the data transfering overhead, we try to allocate the new position as close as possible to the old one
            # TODO: consider the data transfering overhead
            size_plus = []
            size_minus = []
            for pid in old_pid:
                old_size = pre_rsc[pid]
                new_size = rsc_map[pid]
                if old_size != new_size:
                    ctx_switch_list.append(process_dict[pid])
                if new_size > old_size:
                    size_plus.append(pid)
                elif new_size < old_size:
                    size_minus.append(pid)

            for group in [size_minus, size_plus]:
                for pid in group: 
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    # release the old position
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        aval_pos += [i for i in range(s, e)]
                    aval_pos.sort()
                    # get the start position of the old task
                    cum_pos = position_dict[pid][0][0]
                    # divide the available position into two parts
                    left_pos = aval_pos[:aval_pos.index(cum_pos)]
                    right_pos = aval_pos[aval_pos.index(cum_pos):]
                    # select the leftmost position from cum_pos
                    interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                    if len(interval_picked) < new_size:
                        # select the leftmost position from left_pos
                        interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(new_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, position_dict[pid][-1]]
            
            # pick a proper position for the new task in the available position
            for pid in new_pid:
                p_size = rsc_map[pid]
                # search for a gap in the available positions that can accommodate the task's new size
                gap_start = None
                gap_size = 0
                for pos in aval_pos:
                    if gap_start is None:
                        gap_start = pos
                    gap_size += 1
                    if gap_size == p_size:
                        break
                    if pos + 1 not in aval_pos:
                        gap_start = None
                        gap_size = 0

                if gap_start is not None and gap_size == p_size:
                    # allocate the task to the found gap
                    start = [gap_start]
                    size = [gap_size]
                    position_dict[pid] = [start, size, True]
                    # remove the selected positions from aval_pos
                    aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                else:
                    # select the leftmost position
                    interval_picked = aval_pos[:p_size]
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(p_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, True]
                issue_list.append(process_dict[pid])

            # logic to judge whether the swithing is planned or decided by the scheduler at runtime
            # the swithing out of plan features:
            # 1. some tasks release core and other tasks take the core
            # 2. this overtake behavior is not planned in the scheduling table
            # step: 
            # 1. check whether some cores are released
            # 2. check whether this plan is in the scheduling table
            off_discount = 0
            on_discount = 0
            for _p in ctx_switch_list:
                old_size = pre_rsc[_p.pid]
                new_size = rsc_map[_p.pid]
                if old_size > new_size: 
                    if budget_recoder[_p.pid][3]:
                        chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                        if new_size != chunk_alloc:
                            to_assert_flag = True
                        else:
                            off_discount += chunk_alloc
                    else:
                        to_assert_flag = True
                else:
                    chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                    if new_size == chunk_alloc:
                        on_discount += chunk_alloc

            
            if len(preempt_list):
                to_assert_flag = True

            if to_assert_flag:
                for _p in issue_list:
                    chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                    if chunk_alloc == rsc_map[_p.pid]:
                        on_discount += chunk_alloc

            # update the resource configuration
            for _p in preempt_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
                running_queue.remove(_p)
                ready_queue.put(_p)
                sched.new_ready_flg = True
                sched.res_release(_p.pid, False)
            preempt_list.clear()
            
            for _p in ctx_switch_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) ctx switch at {curr_t:.6f}({pre_rsc[_p.pid]} -> {rsc_map[_p.pid]});")
                sched.res_release(_p.pid, False)
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
            ctx_switch_list.clear()

            # if issue the task to runnning list
            for _p in issue_list:
                running_queue.put(_p)
                ready_queue.remove(_p)
                # NOTE:cross cancelation (removed)
                _p.set_state("running")
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
                _p.waitTime = 0 
                _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued and "
                if _p.totburst==0:
                    _p.start_time = curr_t
                    _str += f"start at {curr_t:.6f}; "
                else:
                    _str += f"resume at {curr_t:.6f}; "
                _p.curr_start_time = curr_t
                if bin_name and not bin_event_flg:
                    bin_event_flg = True 
                    print(f"({bin_name})")
                print(_str)
            issue_list.clear()

        for pid in budget_recoder:
            budget_recoder[pid][3] = False
        
        if to_assert_flag:
            # assert a barrier, sim data movement: 
            if sched.barrier_en:
                off_size = (sum(pre_rsc.values())-off_discount) * 1/3
                on_size = (sum(rsc_map.values()) - on_discount) * 2/3
                x = (on_size+off_size) 
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    if sched.drain_flg:
        sched.drain_old_cores()
    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue, True, bin_id, skiped_tasks=skiped_tasks) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")


def scheduler_step_pglb(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                           n_slot:int, timestep:float, 
                           event_range:float, sim_slot_num:int, curr_t:float, 
                           glb_name_p_dict, res_cfg:Resource_model_int, msg_queue:Queue, a_msg_queue, sensor_msg_queue, 
                           monitor:Monitor, DEBUG_FG=False, 
                    quantum_check_en:bool = False, quantumSize=None, 
		    preemption_en:bool=True,
                    o3_boost_util_en:bool=False,
		    show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    # tab_temp_size = len(_SchedTab.scheduling_table)
    # tab_pointer = n_slot % tab_temp_size
    # hyper_p_n = int(n_slot/tab_temp_size)
    # curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, None, msg_dispatcher, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    # sort the tasks in the ready queue and the running queue
    sort_fn = lambda x: x.deadline

    # Scheduler is triggered when:
    # either the aval_rsc or the candidate changes, i.e.,
        # 1. new tasks join the ready queue, preemption may happen
        # 2. some tasks leave the running queue, replenishment and curveup may happen
        # 1+2. both 1 and 2 happen: ressignment-in-turn, preemption, curveup, and replenishment may happen

    # compare the new cfg with the old one to decide the preemption
    pre_rsc = res_cfg.rsc_map
    trigger_condA = sched.new_ready_flg
    trigger_condB = set(pre_rsc.keys()) != set(pre_rsc_bk.keys()) 

    if trigger_condA or trigger_condB: 
        # filtter the preemptable jobs
        preemptable_list = []
        if quantum_check_en: 
            assert quantumSize is not None
            for _p_2b_preempt in running_queue.queue:
                cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
                if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                    continue
                else:
                    preemptable_list.append(_p_2b_preempt)
        else:
            preemptable_list = running_queue.queue

        sorted_queue = sorted(ready_queue.queue + preemptable_list, key=sort_fn)

        rsc_map = OrderedDict() # record the resource allocation
        score_dict = OrderedDict() # record the process allocated with resources and unbouned by constraints
        constr_dict = OrderedDict() # record the resource constraint applied to current allocation
        slack_dict = OrderedDict()
        curr_aval_rsc = res_cfg.size
       
        # build score dict refer to remaining slack
        for _p in sorted_queue:
            time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep) 
            if time_slot_s >= time_slot_e:
                slack_dict[_p.pid] = 0
            else:
                slack_dict[_p.pid] = time_slot_e - time_slot_s
        
        while len(sorted_queue) > 0:
            _p:ProcessInt = sorted_queue.pop(0)
            assert _p.remburst > 0
            # alloc resource depending to the number of resources
            # if curr_t < _p.deadline:
            if slack_dict[_p.pid] != 0:
                # estimate the runtime and the resource requirement
                req_rsc_size, got_latency, constr = EstimCoreNums4Process(_p, _p.remburst, slack_dict[_p.pid]*timestep, 
                                                                            "ceil", curr_aval_rsc)
            else:
                req_rsc_size, constr = _p.get_available_cfg(curr_aval_rsc, curr_aval_rsc)
                got_latency = _p.remburst/req_rsc_size/FLOPS_PER_CORE
            # check if process has no solution
            if req_rsc_size == 0 and constr == "N/A":
                # the process (we call it as failure) get no cores due to the resource constraint
                # but it is the process with higher priority rather the lower one blocking it
                # so we allow the search to continue, 
                # the process with lower priority is allowed to get the resource. 
                # Once the higher priority process release the resource,
                # and failures can get enough resource to run, 
                # failures can preempt the lower priority process anytime.
                continue
            assert req_rsc_size > 0
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            constr_dict[_p.pid] = constr
            score_dict[_p.pid] = 1/slack_dict[_p.pid] if slack_dict[_p.pid] != 0 else float("inf")
            if curr_aval_rsc <= 0:
                break
        
        if curr_aval_rsc > 0:
            # if there are still resources left, 
            # it means no late process is waiting for resources
            assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
            # also, there is no process waiting for resources in the ready queue
            assert len(sorted_queue) == 0

            while curr_aval_rsc > 0 and len(score_dict) > 0:
                for pid in list(score_dict.keys()):
                    if constr_dict[pid] == "upb":
                        score_dict.pop(pid)
                
                if len(score_dict) > 0:
                    # allocate the remaining resources proportionally to the score
                    cum_score_reverse = np.cumsum(list(reversed(score_dict.values())))
                    cum_size = [curr_aval_rsc * s / cum_score_reverse[-1] for s in cum_score_reverse]
                    for i, pid in enumerate(reversed(score_dict.keys())):
                        if i == 0:
                            size = int(cum_size[0])
                            rsc_map[pid] += size
                            cum_size[0] = size
                        else:
                            size = int(cum_size[i] - cum_size[i - 1])
                            rsc_map[pid] += size
                            cum_size[i] = size + cum_size[i - 1]
                    curr_aval_rsc = 0
                # check the rsc_size is valid
                # compare with the core_max, core_min, core_list, parallel_mode
                for pid in score_dict:
                    _p = process_dict[pid]
                    aval_size, constr = _p.get_available_cfg(rsc_map[pid], rsc_map[pid])
                    assert constr in ["upb", "none"]
                    curr_aval_rsc += rsc_map[pid] - aval_size
                    constr_dict[pid] = constr
                    rsc_map[pid] = aval_size

        sched.new_ready_flg = False
        
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # update the position dict
            used_position = []
            for pid in old_pid:
                p_size = rsc_map[pid]
                # set the is_new flag to False
                position_dict[pid][-1] = False
                for s, size in zip(*position_dict[pid][:-1]):
                    e = s + size
                    used_position += [i for i in range(s, e)]

            # remove the expired task from the position dict
            for pid in expired_pid:
                position_dict.pop(pid)
                
                preempt_list.append(process_dict[pid])
            
            aval_pos = [i for i in range(bin_spatial_size) if i not in used_position]
            
            # check if the old task's allocation is changed
            # if so, release the old position and allocate the new one
            # to release the data transfering overhead, we try to allocate the new position as close as possible to the old one
            # TODO: consider the data transfering overhead
            size_plus = []
            size_minus = []
            for pid in old_pid:
                old_size = pre_rsc[pid]
                new_size = rsc_map[pid]
                if new_size > old_size:
                    size_plus.append(pid)
                elif new_size < old_size:
                    size_minus.append(pid)

            for group in [size_minus, size_plus]:
                for pid in group: 
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    # release the old position
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        aval_pos += [i for i in range(s, e)]
                    aval_pos.sort()
                    # get the start position of the old task
                    cum_pos = position_dict[pid][0][0]
                    # divide the available position into two parts
                    left_pos = aval_pos[:aval_pos.index(cum_pos)]
                    right_pos = aval_pos[aval_pos.index(cum_pos):]
                    # select the leftmost position from cum_pos
                    interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                    if len(interval_picked) < new_size:
                        # select the leftmost position from left_pos
                        interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(new_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, True]
                
                    ctx_switch_list.append(process_dict[pid])

            # pick a proper position for the new task in the available position
            for pid in new_pid:
                p_size = rsc_map[pid]
                # select the leftmost position
                interval_picked = aval_pos[:p_size]
                # check if the position is continuous
                interval_picked.sort()
                # remove selected position from aval_pos
                aval_pos = [i for i in aval_pos if i not in interval_picked]
                start = [interval_picked[0]]
                size = []
                for i in range(p_size-1):
                    if interval_picked[i] != interval_picked[i+1]-1:
                        size.append(interval_picked[i]-start[-1]+1)
                        start.append(interval_picked[i+1])
                size.append(interval_picked[-1]-start[-1]+1)
                position_dict[pid] = [start, size, True]

                issue_list.append(process_dict[pid])

            # update the resource configuration
            for _p in preempt_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
                running_queue.remove(_p)
                ready_queue.put(_p)
                res_cfg.release(_p.pid)
            preempt_list.clear()
            
            for _p in ctx_switch_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) ctx switch at {curr_t:.6f}({pre_rsc[_p.pid]} -> {rsc_map[_p.pid]});")
                res_cfg.release(_p.pid)
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
            ctx_switch_list.clear()

            for _p in issue_list:
                running_queue.put(_p)
                ready_queue.remove(_p)
                _p.set_state("running")
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
                _p.waitTime = 0 
                _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued and "
                if _p.totburst==0:
                    _p.start_time = curr_t
                    _str += f"start at {curr_t:.6f}; "
                else:
                    _str += f"resume at {curr_t:.6f}; "
                _p.curr_start_time = curr_t
                if bin_name and not bin_event_flg:
                    bin_event_flg = True 
                    print(f"({bin_name})")
                print(_str)
            issue_list.clear()

            # assert a barrier
            # data movement: 
            # size: 40MB
            # bandwidth: 100GB/s
            # direction: off-chip -> on-chip, on-chip -> off-chip
            # latency: 100ns
            if sched.barrier_en:
                off_size = sum(pre_rsc.values()) * 1/3
                on_size = sum(rsc_map.values()) * 2/3
                x = (on_size+off_size)
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()


def trigger_read(inactive_list:List[ProcessInt], sensor_msg_queue:List, 
                trigger_cache:TriggerCache, process_dict:Dict,
                timestep, curr_t, DEBUG_FG):
    """
        1. Read the sensor message queue and update the trigger cache
        2. Update the ctrl dependency flags
    """

    # for pid, next_ingestion_time, next_event_time in sensor_msg_queue:
    for pid, msg in sensor_msg_queue:
        if pid in process_dict:
            trigger_cache.sensor_cache[pid].append(msg)
    sensor_msg_queue.clear()

    # read the trigger cache
    for _p in inactive_list:
        if trigger_cache is None:
            pred_ctrl = _p.pred_ctrl
            event_triggers = _p.event_triggers
        else:
            pred_ctrl = trigger_cache[_p.pid]
            event_triggers = trigger_cache.sensor_cache[_p.pid]
        
        _p.sim_trigger(curr_t, timestep, pred_ctrl, event_triggers)

def data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue: List[Data], 
                   event_cache:EventCache=None):
    msg_dict:Dict[int, list[Data]] = {}
    # read out all message and clear the message pipe
    for data in a_msg_queue:
        tgt_p_name_l = data.track_downstream()
        for key in tgt_p_name_l:
            tgt_pid = glb_name_p_dict[key].pid
            if tgt_pid in process_dict:
                if tgt_pid not in msg_dict:
                    msg_dict[tgt_pid] = []
                msg_dict[tgt_pid].append(data)
                # add the ref_pid to the data
                data.ref_pid.append(tgt_pid)
    for data in set([data for data_l in msg_dict.values() for data in data_l]):
        buffer.put(data)
    a_msg_queue.clear()

    for tgt_pid, data_l in msg_dict.items():
        _p = process_dict[tgt_pid]
        if event_cache is None:
            pred_data = _p.pred_data
        else:
            pred_data = event_cache[tgt_pid]
        for key, attr in pred_data.items():
            for data in data_l:
                if data.pid == glb_name_p_dict[key].pid:
                    attr["event_queue"].put(data)
                    attr["valid"] = True
                    attr["time"] = curr_t
                    attr["data"] = data
                    
                    # TODO: fix the event time as the actual time
                    if bin_name and not bin_event_flg:
                        bin_event_flg = True
                        print(f"({bin_name})")
                    print(f"		{_p.task.name} received event {key:s} @ {curr_t:.6f}/{data.ctx.get_timestamp():.6f}")
    return bin_event_flg

# =================== intergrated into scheduler class ===================

def glb_dynamic_sched_step(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                           n_slot:int, timestep:float, 
                           event_range:float, sim_slot_num:int, curr_t:float, 
                           glb_name_p_dict, res_cfg:Resource_model_int, msg_queue:Queue, 
                           monitor:Monitor, DEBUG_FG=False, quantum_check_en:bool = False, quantumSize=None,
                           show_warnings=True):

    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
        
    # extract the scheduling table
    bin_name = ""
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[0]
    _SchedTab = sched._SchedTab
    bin_spatial_size = _SchedTab.num_resources

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    pre_rsc_bk = deepcopy(res_cfg.rsc_map)
    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    bin_event_flg = check_miss(sched, None, None, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)
    # buffer.recyle_no_ref("output", True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    # a_data_pipe.data_tranfer_sim(curr_t)
    # cache all the src and weight data
    while a_data_pipe.buffer.queue:
        data:Data
        mode, data, dest = a_data_pipe.buffer.queue[0]
        a_data_pipe.remain_cap += data.size
        data.valid = True
        data.update_receive_time(curr_t)
        a_data_pipe.buffer.get()
        a_data_pipe.broadcast_message(data, prefix="  ")
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list)

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    # sort the tasks in the ready queue and the running queue
    sort_fn = lambda x: x.deadline

    # Scheduler is triggered when:
    # either the aval_rsc or the candidate changes, i.e.,
        # 1. new tasks join the ready queue, preemption may happen
        # 2. some tasks leave the running queue, replenishment and curveup may happen
        # 1+2. both 1 and 2 happen: ressignment-in-turn, preemption, curveup, and replenishment may happen

    # compare the new cfg with the old one to decide the preemption
    pre_rsc = res_cfg.rsc_map
    trigger_condA = sched.new_ready_flg
    trigger_condB = set(pre_rsc.keys()) != set(pre_rsc_bk.keys()) 

    if trigger_condA or trigger_condB: 
        # filtter the preemptable jobs
        preemptable_list = []
        if quantum_check_en: 
            assert quantumSize is not None
            for _p_2b_preempt in running_queue.queue:
                cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
                if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                    continue
                else:
                    preemptable_list.append(_p_2b_preempt)
        else:
            preemptable_list = running_queue.queue

        sorted_queue = sorted(ready_queue.queue + preemptable_list, key=sort_fn)

        rsc_map = OrderedDict() # record the resource allocation
        score_dict = OrderedDict() # record the process allocated with resources and unbouned by constraints
        constr_dict = OrderedDict() # record the resource constraint applied to current allocation
        slack_dict = OrderedDict()
        curr_aval_rsc = res_cfg.size
       
        # build score dict refer to remaining slack
        for _p in sorted_queue:
            time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep) 
            if time_slot_s >= time_slot_e:
                slack_dict[_p.pid] = 0
            else:
                slack_dict[_p.pid] = time_slot_e - time_slot_s
        
        while len(sorted_queue) > 0:
            _p:ProcessInt = sorted_queue.pop(0)
            assert _p.remburst > 0
            # alloc resource depending to the number of resources
            # if curr_t < _p.deadline:
            if slack_dict[_p.pid] != 0:
                # estimate the runtime and the resource requirement
                req_rsc_size, got_latency, constr = EstimCoreNums4Process(_p, _p.remburst, slack_dict[_p.pid]*timestep, 
                                                                            "ceil", curr_aval_rsc)
            else:
                req_rsc_size, constr = _p.get_available_cfg(curr_aval_rsc, curr_aval_rsc)
                got_latency = _p.remburst/req_rsc_size/FLOPS_PER_CORE
            # check if process has no solution
            if req_rsc_size == 0 and constr == "N/A":
                # the process (we call it as failure) get no cores due to the resource constraint
                # but it is the process with higher priority rather the lower one blocking it
                # so we allow the search to continue, 
                # the process with lower priority is allowed to get the resource. 
                # Once the higher priority process release the resource,
                # and failures can get enough resource to run, 
                # failures can preempt the lower priority process anytime.
                continue
            assert req_rsc_size > 0
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            constr_dict[_p.pid] = constr
            score_dict[_p.pid] = 1/slack_dict[_p.pid] if slack_dict[_p.pid] != 0 else float("inf")
            if curr_aval_rsc <= 0:
                break
        
        if curr_aval_rsc > 0:
            # if there are still resources left, 
            # it means no late process is waiting for resources
            assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
            # also, there is no process waiting for resources in the ready queue
            assert len(sorted_queue) == 0

            while curr_aval_rsc > 0 and len(score_dict) > 0:
                for pid in list(score_dict.keys()):
                    if constr_dict[pid] == "upb":
                        score_dict.pop(pid)
                
                if len(score_dict) > 0:
                    # allocate the remaining resources proportionally to the score
                    core_distr(rsc_map, score_dict, curr_aval_rsc)
                    curr_aval_rsc = 0
                # check the rsc_size is valid
                # compare with the core_max, core_min, core_list, parallel_mode
                for pid in score_dict:
                    _p = process_dict[pid]
                    aval_size, constr = _p.get_available_cfg(rsc_map[pid], rsc_map[pid])
                    assert constr in ["upb", "none"]
                    curr_aval_rsc += rsc_map[pid] - aval_size
                    constr_dict[pid] = constr
                    rsc_map[pid] = aval_size

        sched.new_ready_flg = False
        
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # update the position dict
            used_position = []
            for pid in old_pid:
                p_size = rsc_map[pid]
                # set the is_new flag to False
                position_dict[pid][-1] = False
                for s, size in zip(*position_dict[pid][:-1]):
                    e = s + size
                    used_position += [i for i in range(s, e)]

            # remove the expired task from the position dict
            for pid in expired_pid:
                position_dict.pop(pid)
                
                preempt_list.append(process_dict[pid])
            
            aval_pos = [i for i in range(bin_spatial_size) if i not in used_position]
            
            # check if the old task's allocation is changed
            # if so, release the old position and allocate the new one
            # to release the data transfering overhead, we try to allocate the new position as close as possible to the old one
            # TODO: consider the data transfering overhead
            size_plus = []
            size_minus = []
            for pid in old_pid:
                old_size = pre_rsc[pid]
                new_size = rsc_map[pid]
                if new_size > old_size:
                    size_plus.append(pid)
                elif new_size < old_size:
                    size_minus.append(pid)

            for group in [size_minus, size_plus]:
                for pid in group: 
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    # release the old position
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        aval_pos += [i for i in range(s, e)]
                    aval_pos.sort()
                    # get the start position of the old task
                    cum_pos = position_dict[pid][0][0]
                    # divide the available position into two parts
                    left_pos = aval_pos[:aval_pos.index(cum_pos)]
                    right_pos = aval_pos[aval_pos.index(cum_pos):]
                    # select the leftmost position from cum_pos
                    interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                    if len(interval_picked) < new_size:
                        # select the leftmost position from left_pos
                        interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(new_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, True]
                
                    ctx_switch_list.append(process_dict[pid])

            # pick a proper position for the new task in the available position
            for pid in new_pid:
                p_size = rsc_map[pid]
                # select the leftmost position
                interval_picked = aval_pos[:p_size]
                # check if the position is continuous
                interval_picked.sort()
                # remove selected position from aval_pos
                aval_pos = [i for i in aval_pos if i not in interval_picked]
                start = [interval_picked[0]]
                size = []
                for i in range(p_size-1):
                    if interval_picked[i] != interval_picked[i+1]-1:
                        size.append(interval_picked[i]-start[-1]+1)
                        start.append(interval_picked[i+1])
                size.append(interval_picked[-1]-start[-1]+1)
                position_dict[pid] = [start, size, True]

                issue_list.append(process_dict[pid])

            # update the resource configuration
            for _p in preempt_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
                running_queue.remove(_p)
                ready_queue.put(_p)
                res_cfg.release(_p.pid)
            preempt_list.clear()
            
            for _p in ctx_switch_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) ctx switch at {curr_t:.6f}({pre_rsc[_p.pid]} -> {rsc_map[_p.pid]});")
                res_cfg.release(_p.pid)
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
            ctx_switch_list.clear()

            for _p in issue_list:
                running_queue.put(_p)
                ready_queue.remove(_p)
                _p.set_state("running")
                res_cfg.allocate(_p.pid, rsc_map[_p.pid])
                _p.waitTime = 0 
                _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued and "
                if _p.totburst==0:
                    _p.start_time = curr_t
                    _str += f"start at {curr_t:.6f}; "
                else:
                    _str += f"resume at {curr_t:.6f}; "
                _p.curr_start_time = curr_t
                if bin_name and not bin_event_flg:
                    bin_event_flg = True 
                    print(f"({bin_name})")
                print(_str)
            issue_list.clear()

            # assert a barrier
            # data movement: 
            # size: 40MB
            # bandwidth: 100GB/s
            # direction: off-chip -> on-chip, on-chip -> off-chip
            # latency: 100ns
            if sched.barrier_en:
                off_size = sum(pre_rsc.values()) * 1/3
                on_size = sum(rsc_map.values()) * 2/3
                x = (on_size+off_size)
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()

def pendingToReady(sched, active_list:List[ProcessInt], ready_queue, 
                   buffer:Buffer, curr_t, glb_n_task_dict:Dict[str, ProcessInt], 
                   bin_name=""):
    # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
    l_ready = []
    for _p in active_list:
        # check data availability
        # w_avail = _p.pid in buffer.buffer_w
        # in_avail = _p.check_depends_data(buffer, glb_n_task_dict=glb_n_task_dict)

        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0: 
            matched_pair, in_avail = WatermarkStrategy.check_data_depends(_p, buffer, glb_n_task_dict, _p.msg_cache[0].get_timestamp())
        else:
            in_avail = True

        # if w_avail and in_avail:
        if in_avail:
            l_ready.append(_p)
    if len(l_ready)>0:
        sched.new_ready_flg = True

    for _p in l_ready:
        # cache the context of the upstream weight node and src node
        # _p.update_ctx('upstream', buffer=buffer, glb_n_task_dict=glb_n_task_dict)
        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0:
            _p.update_ctx('upstream', matched_pair=matched_pair)
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) "
        if bin_name:
            _str = f"({bin_name})\n" + _str
        ready_queue.put(_p)
        active_list.remove(_p)
        _p.ready_time = curr_t
        _p.ready = True
        _p.set_state("ready")
        _str += "READY!!"
        print(_str)

# =================== functions related to data transfer ===================

def data_prefetching(sched:Scheduler, init_p_list, wait_queue:DataPipe, curr_t, bin_id, cached_cfg=None):    
    # infinite bandwidth, buffer size, constant latency
    for pid in cached_cfg.keys():
        _p = init_p_list[pid]
        msg:ContextMsg = ContextMsg.create_weight_ctx()
        data = Data(_p.pid, _p.io_time, (0,), "weight")
        data.ctx = msg
        data.cache_msg_transfer(curr_t)

        on_size = cached_cfg[pid] * 1/3
        x = on_size
        data.size = sched.get_ctx_lat(x)
        wait_queue.put(data, "unicast", [bin_id,])
        _p.set_state("wait")


# unused functions
def RunningQueueToWait(running_queue, wait_queue:DataPipe):
    # CPU[i]->running->burst == CPU[i]->running->cpu
    l_wait = []
    for _p in running_queue:
        exe_io_tile = _p.burst / _p.cpu_time
        exe_io_tile_r = round(exe_io_tile)
        exe_comp = math.isclose(exe_io_tile, exe_io_tile_r, abs_tol=1e-2)
        if exe_comp:
            l_wait.append(_p)
    for _p in l_wait:
        _p.burst = 0
        _p.ready = False
        l_wait.remove(_p)
        running_queue.remove(_p)
        wait_queue.put(_p)
        print("		TASK {:d}:{:s}({:d}) WAIT!!".format(_p.task.id, _p.task.name, _p.pid))

def update_depend(tasc_dict:Dict[str, ProcessInt], completed_task:List[ProcessInt]): 
    """
    update the dependency list
    input: 
        task_dict: the dictionary of tasks
        completed_task: the list of completed tasks
    action:
        update the dependency list of the successor tasks of the completed tasks
    output:
        None
    """
    for _p in completed_task:
        for _s in _p.succ_data: 
            succ_task = tasc_dict[_s]
            succ_task.pred_data[_p.task.name] = True
        for _s in _p.succ_ctrl:
            succ_task = tasc_dict[_s]
            succ_task.pred_ctrl[_p.task.name] = True

def check_depends(task_list:List[ProcessInt])->List[ProcessInt]:
    """
    check the denpendency of task in the list
    input:
        task_list: the list of tasks
    output:
        ready_task: the list of tasks that all denpendencies are satisfied
    """
    active = []
    for _p in task_list:
        if _p.check_depends():
            active.append(_p)
            print("		TASK {:d}:{:s}({:d}) is avtivated!!".format(_p.task.id, _p.task.name, _p.pid))
    return active

def load_sched_tab(num_cores, e2e_latency:float, aux_scale_factor:int, bin_path_format:str, scheduler_list:List[Scheduler]):
    bin_list_save_path = bin_path_format.format(aux_scale_factor, e2e_latency, num_cores)
    bin_list = load_pickle(bin_list_save_path)
    for _SchedTab, scheduler in zip(bin_list, scheduler_list):
        scheduler._SchedTab_L0 = _SchedTab
        assert scheduler._SchedTab_L0.id == _SchedTab.id
    old_cores = [scheduler._SchedTab.num_resources for scheduler in scheduler_list]
    new_cores = [scheduler._SchedTab_L0.num_resources for scheduler in scheduler_list]
    
    assert len(old_cores) == len(new_cores)
    old_map = core_mapping_1d(old_cores)
    new_map = core_mapping_1d(new_cores)
    for scheduler in scheduler_list:
        scheduler.core_map_L0 = dict.fromkeys(new_map[scheduler._SchedTab_L0.id])
        for k in scheduler.core_map_L0.keys():
            if k in old_map[scheduler._SchedTab.id]:
                scheduler.core_map_L0[k] = True
            
    # reverse the old map
    old_map_rev = {}
    for k, items in old_map.items():
        for item in items:
            old_map_rev[item] = k
    # reverse the new map
    new_map_rev = {}
    for k, items in new_map.items():
        for item in items:
            new_map_rev[item] = k
    # create the mapping from the new scheduler to the old one
    map_new_to_old = {}
    for k, items in new_map.items():
        new_to_old_t = {}
        for item in items:
            src_partition = old_map_rev[item]
            if src_partition not in new_to_old_t:
                new_to_old_t[src_partition] = []
            new_to_old_t[src_partition].append(item)
        map_new_to_old[k] = new_to_old_t
    return map_new_to_old, new_map_rev
    

def core_mapping_1d(core_num_list:List[int]):
    """
    map the cores to the partitions
    """
    core_mapping = {}
    start = 0
    for i in range(len(core_num_list)):
        end = start + core_num_list[i]
        core_mapping[i] = list(range(start, end))
    return core_mapping





if __name__ == "__main__":
    pass
    

    