import numpy as np
from typing import Dict, List
from task_queue_agent import TaskQueue
from scheduling_table import SchedulingTableInt
from task_agent import ProcessInt
from buffer import Buffer
import warnings

from buffer import Buffer, Data
from msg_dispatcher import MsgDispatcher, msg_filter
from multiprocessing import Queue
from scheduling_table import SchedulingTableInt
from resource_agent import Resource_model_int
from global_var import *

from task_queue_agent import TaskQueue 
from task_agent import ProcessInt
from lru import LRUCache

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
                 _SchedTab: SchedulingTableInt, glb_p_list:List[ProcessInt],
                 budget_recoder:Dict[int, List]=None, rsc_recoder_his:Dict[int, LRUCache]=None, 
                 ) -> None:
        self.ready_queue: TaskQueue = TaskQueue()
        self.expired_queue: List = []
        self.blocked_queue: List = []

        # wait 
        # structure: (wait_time, task)
        self.weight_wait_queue = TaskQueue(sort_f=lambda x: x.io_time-x.waitTime, decending=False)
        self.input_wait_queue: List = []

        # buffer
        self.buffer:Buffer = Buffer()
        # monitor the deadline: (ascending)
        self.ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, decending=False)
        
        # Running queue: 
        # cache the running task list in an order of priority (here we use ddl)
        # monitor the deadline for pre-emption: (decending)
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
        
        self._SchedTab = _SchedTab
        self.curr_cfg = Resource_model_int(size=_SchedTab.scheduling_table[0].size)
        self.process_dict: Dict[int, ProcessInt] = {pid:glb_p_list[pid] for pid in _SchedTab.index_occupy_by_id()}

        # create res_cfg, monitor, msg_queue, 
        # self.res_cfg = res_cfg
        # self.monitor = monitor
        # self.msg_queue = msg_queue

        # curr_cfg, budget_recoder, rsc_recoder_his, process_dict
        # curr_cfg_list, budget_recoder_list, rsc_recoder_his_list, process_dict_list
        self.budget_recoder = budget_recoder if budget_recoder else {}
        self.rsc_recoder_his = rsc_recoder_his if rsc_recoder_his else {}


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
        return throttleToReady(curr_t, 
                                self.budget_recoder, self.ready_queue, self.throttle_list, self._SchedTab.name,
                                bin_event_flg)

    def chk_release(self, event_range, curr_t, 
                    bin_event_flg:bool=False, ):
        return chk_release(event_range, curr_t, self.inactive_list, self.active_list, bin_event_flg, self._SchedTab.name)

    def check_miss(self,
                            curr_t, res_cfg, 
                            bin_event_flg:bool=False):

        # check_miss(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            # throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name)
        return check_miss(self.budget_recoder, curr_t, res_cfg, self.weight_wait_queue, self.ready_queue, self.running_queue, self.miss_list,
                            self.throttle_list, self.active_list, self.inactive_list, self.buffer, bin_event_flg, self._SchedTab.name)

    def check_throttle(self,
                            curr_t, res_cfg, 
                            bin_event_flg:bool=False):
        # check_throttle(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
        #                             throttle_list, active_list, inactive_list, bin_event_flg, bin_name)
        return check_throttle(self.budget_recoder, curr_t, res_cfg, self.weight_wait_queue, self.ready_queue, self.running_queue, self.miss_list,
                            self.throttle_list, self.active_list, self.inactive_list, bin_event_flg, self._SchedTab.name)

    def check_complete(self, timestep, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                    curr_t, res_cfg, 
                        bin_event_flg:bool=False):
        # check_complete(budget_recoder, timestep, msg_dispatcher, curr_t, res_cfg, running_queue, completed_list, inactive_list, buffer, bin_event_flg, bin_name)
        return check_complete(self.budget_recoder, timestep, msg_dispatcher, curr_t, res_cfg, self.running_queue, self.completed_list, self.inactive_list, self.buffer, bin_event_flg, self._SchedTab.name) 

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

    def updateRunningQueue(self, timestep, rsc_cfg):
        # updateRunningQueue(timestep, running_queue, res_cfg) 
        return updateRunningQueue(timestep, self.running_queue, rsc_cfg)
    
    def pendingToReady(self, curr_t, glb_n_task_dict:Dict[str, ProcessInt]):
        # pendingToReady(active_list, ready_queue, buffer, budget_recoder, throttle_list, curr_t, glb_name_p_dict, bin_name, ) 
        return pendingToReady(self.active_list, self.ready_queue, self.buffer, self.budget_recoder, self.throttle_list, curr_t, glb_n_task_dict, self._SchedTab.name)

    def scheduler_step(self, msg_dispatcher: MsgDispatcher, n_slot: int, timestep: int, event_range: List[int], 
                    sim_slot_num: int, curr_t: int, glb_name_p_dict: Dict[str, List[int]], 
                    res_cfg: Dict[str, int], 
                    msg_queue:Queue, DEBUG_FG: bool) -> None:

        # scheduler_step(sched, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG)
        return scheduler_step(self, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG)
# =================== intergrated into scheduler class ===================
def throttleToReady(curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg):
    l_res_ready = []
    for _p in throttle_list:
        if _p.pid in budget_recoder:
            l_res_ready.append(_p)
        
    if l_res_ready and not bin_event_flg:
        bin_event_flg = True 
        print(f"({bin_name})")

    for _p in l_res_ready: 
            throttle_list.remove(_p)
            ready_queue.put(_p)
            _p.ready_time = curr_t
            _p.ready = True
            _p.set_state("ready")
            print("		TASK {:d}:{:s}({:d}) THROTTLE -> READY!!".format(_p.task.id, _p.task.name, _p.pid))
    return bin_event_flg
        # data prefetching
        # position candidate 

def chk_release(event_range, curr_t, inactive_list:List[ProcessInt], active_list, 
                bin_event_flg:bool=False, 
                bin_name:str=""):
    """
    check release
        1. check the dependencies of the tasks in inactive list
        2. if the dependencies are satisfied, move the task to the wait queue
    """

    l_active = []
    if curr_t <= event_range:
        for _p in inactive_list:
            if _p.check_depends():
                # if curr_t >= _p.task.ERT and _p.trigger_mode != "N": # constraint the fisrt release time of the event triggered task
                l_active.append(_p)

    if len(l_active) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
        
    for _p in l_active:
        active_list.append(_p)
        inactive_list.remove(_p)
        _p.release_time = curr_t
        _p.released = True
        _p.remburst += _p.task.flops
        # _p.set_state("runnable")
        # _p.release_time = curr_t 
        # _p.deadline = curr_t + _p.task.ddl
        # _p.deadline += _p.task.period
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is activated @ {curr_t:.6f}!!"
        print(_str)
    return bin_event_flg

def check_miss(
                        rsc_recoder, curr_t, res_cfg, wait_queue, ready_queue, 
                        running_queue, miss_list, throttle_list, active_list, inactive_list, buffer,
                        bin_event_flg:bool=False, 
                        bin_name:str=""):

    for _p in (active_list + ready_queue.queue + running_queue.queue):
    # for _p in (running_queue.queue):
        if _p.deadline < curr_t and _p.task.criticality == "hard":
            miss_list.append(_p)
            _p.set_state("suspend")

    if len(miss_list) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")

    for _p in miss_list:
        # release the resource and move to the wait list
        # buffer.pop(_p.pid)
        if _p in ready_queue.queue:
            ready_queue.remove(_p)
        elif _p in running_queue.queue:
            # bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
            # _SchedTab = bin_list[bin_id_t]
            # _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
            res_cfg.release(_p.pid, verbose=False)
            rsc_recoder.pop(_p.pid)
            running_queue.remove(_p)
        print("		TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid))
        _p.task.missed_deadline_count += 1
        # _p.release_time += _p.task.period
        _p.deadline += _p.task.period
        _p.remburst = 0
        _p.rem_flop_budget = 0

        _p.ready_time = -1
        _p.ready = False
            
        # kill & drop the total execution 
        _p.currentburst = 0
        _p.burst = 0
        _p.totburst = 0
        _p.waitTime = 0
        _p.cumulative_executed_time = 0

        # _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
        _p.set_state("suspend")
        inactive_list.append(_p)
        # _p.reset_depends()

        # print("Scheduling Table:")
        # print(SchedTab.print_scheduling_table())
    miss_list.clear()
    return bin_event_flg

def check_throttle(
                        rsc_recoder, curr_t, res_cfg, wait_queue, ready_queue, 
                        running_queue, miss_list, throttle_list, active_list, inactive_list, 
                        bin_event_flg:bool=False, 
                        bin_name:str=""):

    # for _p in (active_list + wait_queue.queue + ready_queue.queue + running_queue.queue):
    l_throttle = []
    for _p in (running_queue.queue):
        budget_usedup_flg = False
        # determine if the budget is used up, considering the numerical error
        # less than 1 OPS
        if _p.rem_flop_budget < 1e-12:
            budget_usedup_flg = True
        if budget_usedup_flg and _p.cbs_en: 
            l_throttle.append(_p)
            _p.set_state("throttled")

    if len(l_throttle) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
    
    for _p in l_throttle: 
        # warnings.warn("		TASK {:d}:{:s}({:d}) THROTTLED!!".format(_p.task.id, _p.task.name, _p.pid))
        print("		TASK {:d}:{:s}({:d}) is THROTTLED @ {:.6f} !!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
        
        # update statistics 
        _p.task.throttle_count += 1

        # _p.release_time += _p.task.period
        # _p.deadline += _p.task.period
        # _p.remburst += _p.task.flops

        # suppose kill strategy
        # current tile should be reloaded and re-executed
        # other wise, modify the io time
        _p.ready = False
        _p.ready_time = -1
        _p.currentburst = 0
        # _p.burst = 0

        _p.waitTime = 0
        # _p.cumulative_executed_time = 0

        # _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
        res_cfg.release(_p.pid, verbose=False)
        rsc_recoder.pop(_p.pid)
        running_queue.remove(_p)
        throttle_list.append(_p)
    return bin_event_flg

def check_complete(rsc_recoder, timestep, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                   curr_t, res_cfg, 
                    running_queue:TaskQueue, 
                    completed_list: List[ProcessInt], 
                    inactive_list:List[ProcessInt],
                    buffer:Buffer, 
                    bin_event_flg:bool=False, 
                    bin_name:str=""):
    for _p in running_queue:
        # check whether the task is completed
        if (_p.totburst >= _p.totcpu):
            completed_list.append(_p)
            rsc_recoder.pop(_p.pid)
            _p.set_state("suspend")

    if len(completed_list) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
        
    for _p in completed_list:
        # release the resource and move to the wait list
        # bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, tab_pointer, _p)
        # _SchedTab = bin_list[bin_id_t]
        # _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
        res_cfg.release(_p.pid, verbose=False)
        # buffer.pop(_p.pid)

        # detect the lateness 
        if _p.deadline < curr_t:
            print("Complete lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        else: 
            print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}")
        
        # update statistics
        # TODO: add lock 
        _p.task.completion_count += 1
        _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
            
        _p.release_time += _p.task.period
        _p.deadline += _p.task.period
        _p.remburst = 0
        _p.rem_flop_budget = 0

        _p.ready_time = -1 
        _p.ready = False
        _p.end_time = curr_t * timestep
        _p.currentburst = 0
        _p.burst = 0
        _p.totburst = 0
        _p.waitTime = 0
        _p.cumulative_executed_time = 0

        # _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
        running_queue.remove(_p)
        inactive_list.append(_p)
        # _p.reset_depends()

    # print("Scheduling Table:")
    # print(SchedTab.print_scheduling_table())
    # update_depend(task_dict, completed_list)
    for _p in completed_list:
        # msg_pipe.send(f"{_p.task.name}_completed", prefix="		")
        msg_dispatcher.broadcast_message(f"{_p.task.name}_completed", prefix="		")
    
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

def updateRunningQueue(timestep, running_queue, rsc_cfg):
    _p_dict = {p.pid:p for p in running_queue} 
    for pid in rsc_cfg.rsc_map.keys():
        _p = _p_dict[pid]
        _p.currentburst += rsc_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
        _p.burst += rsc_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
        _p.totburst += rsc_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
        _p.remburst -= rsc_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
        _p.cumulative_executed_time += timestep
        _p.rem_flop_budget -= rsc_cfg.rsc_map[_p.pid] * timestep * FLOPS_PER_CORE

def pendingToReady(active_list, ready_queue, buffer:Buffer, budget_recoder, throttle_list, curr_t, glb_n_task_dict:Dict[str, ProcessInt], bin_name=""):
    # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
    l_ready = []
    for _p in active_list:
        # check data availability
        w_avail = _p.pid in buffer.buffer_w
        in_avail = _p.check_depends_data(buffer, glb_n_task_dict=glb_n_task_dict)
        if w_avail and in_avail:
            l_ready.append(_p)

    for _p in l_ready:
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) "
        if bin_name:
            _str = f"({bin_name})\n" + _str
        # if _p.rem_flop_budget > 0: 
        if _p.pid in budget_recoder:
            ready_queue.put(_p)
            active_list.remove(_p)
            _p.ready_time = curr_t
            _p.ready = True
            _p.set_state("ready")
            _str += "READY!!"
            print(_str)
        else:
            active_list.remove(_p)
            throttle_list.append(_p)
            _str += "data ready, but throttled!!"
            warnings.warn(_str)

def scheduler_step(sched, msg_dispatcher, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, DEBUG_FG):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    buffer:Buffer = sched.get_buffer()
        
        # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False

        # (running_queue)
        # check running tasks
    bin_event_flg = check_complete(budget_recoder, timestep, msg_dispatcher, curr_t, res_cfg, running_queue, completed_list, inactive_list, buffer, bin_event_flg, bin_name)

        # check whether the task is miss
        # TODO: other ready tasks shoud be checked
        # TODO: cache eviction
    bin_event_flg = check_miss(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name)

    bin_event_flg = check_throttle(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, bin_event_flg, bin_name)

        # spill out the data of type "output", which is expired
    buffer.pop_timeout("output", curr_t, True)

        # simulate the event trigger
    for _p in _SchedTab.sim_triggered_list:
        trigger_state = _p.sim_trigger(curr_t, timestep)
        if trigger_state and not bin_event_flg and curr_t >= _p.task.ERT:
            bin_event_flg = True
            print(f"({bin_name})")

        
        # tackle the event in message pipe, set the valid flag in pred_data of each process
        # update barrier status
        # update the data status
        # if not msg_pipe.empty():

        # clear the message pipe
    msg_list = []
    while not msg_queue.empty():
        msg_list.append(msg_queue.get())
    if msg_list:
        for _p in process_dict.values():
            for key, attr in _p.pred_data.items():
                    # if msg_pipe.filter(key):
                if msg_filter(msg_list, key):
                    attr["valid"] = True
                    attr["time"] = curr_t
                        # TODO: fix the event time as the actual time
                    if not bin_event_flg:
                        bin_event_flg = True
                        print(f"({bin_name})")
                    print(f"		{_p.task.name} received event {key:s} @ {curr_t:.6f}")
                    buffer.put(Data(glb_name_p_dict[key].pid, 1, (0,), "output", glb_name_p_dict[key].io_time, curr_t, 1/glb_name_p_dict[key].task.freq))

        # check release
        # check the dependencies of the tasks in inactive list
        # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = chk_release(event_range, curr_t, inactive_list, active_list, bin_event_flg, bin_name) 

        # # instruction prefetching
        # cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]         
        
        # # if the pointer reaches the start of the next cfg, update the cfg
        # if tab_pointer == cfg_slot_s:
        #     # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        #     curr_cfg.update(cached_cfg)
        #     # move the pointer to the next cfg
        #     _SchedTab.idx_plus_1()
            

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
            if not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))

    if curr_cfg.slot_s == n_slot:
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
            # replenish the budget
        for pid in next_cfg.keys():
            _p = process_dict[pid]
                # _p.deadline += _p.task.period
            _p.rem_flop_budget += next_cfg[pid] * cfg_slot_num * timestep * FLOPS_PER_CORE

            budget_recoder[pid] = [cfg_slot_s, next_cfg[pid], cfg_slot_num]
            if _p.pid in rsc_recoder_his:
                rsc_recoder_his[_p.pid].put(bin_id)
            else:
                rsc_recoder_his[_p.pid] = LRUCache(3)
                rsc_recoder_his[_p.pid].put(bin_id)

            # TODO: Queue for the weight prefetching
            # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

            # weight prefetching based on the scheduling table
            # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(process_dict, weight_wait_queue, cached_cfg=cached_cfg)

        # TODO: simulate the congestion and the latency of the network
    data_tranfer_sim(weight_wait_queue, buffer, bin_name)

        # check data availability: some tasks may be prefetched
        # TODO: model the runtime weight and feature map transfering 
    pendingToReady(active_list, ready_queue, buffer, budget_recoder, throttle_list, curr_t, glb_name_p_dict, bin_name, ) 
        # move the task to the ready queue
    bin_event_flg = throttleToReady(curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg)

        # free resource index
        # check the running tasks
    for _p in running_queue.queue:
            # detect the lateness of the tasks
        if _p.pid not in curr_cfg.rsc_map: 
            warnings.warn("Execution lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
    aval_rsc = res_cfg.get_available_rsc()

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

    if curr_cfg.slot_s <= n_slot and n_slot <= curr_cfg.slot_e:
            # calculate the criticity of the task
            # remaining slack/remaining cpu time
            # fn_crit = lambda x: (x.deadline - curr_t)/(x.exp_comp_t - x.cumulative_executed_time)
        fn_crit = lambda x: x.deadline
        fn_task_flag = lambda x: 0 if x.task.task_flag=="stationary" else 1
            # the most critical one is the first one, with the smallest value, use the ascending order
        sorted_ready_queue = sorted(ready_queue.queue, key=lambda x: (fn_crit(x), fn_task_flag(x)))
            
            # trigger condition
            # release time round up: task should not be released earlier than the release time
        fn_release_slot = lambda x: max(int(np.ceil(x.release_time/timestep)), n_slot)
        fn_ddl_slot = lambda x: int(x.deadline//timestep)
        fn_trig = lambda x: fn_release_slot(x) - curr_cfg.slot_s 

            # if there is no execution lateness in previous cfg, then the running queue is empty
            # all the tasks chunks shares the same deadline; without spec changes, 
            # the execution sequence not matter the schedulibility. 

        while len(sorted_ready_queue) > 0:
            _p = sorted_ready_queue[0]

                # calculate the required resource size
                # get the current configuration
            planned_rsc_size = curr_cfg.rsc_map[_p.pid]
            late_slot_num = fn_trig(_p)
            if late_slot_num > 0:
                    # TO-CHECK: suppose previous chunk is not late
                req_rsc_size =min(curr_cfg.slot_num/(fn_ddl_slot(_p) - fn_release_slot(_p)) * planned_rsc_size, aval_rsc)
            else:
                req_rsc_size = planned_rsc_size
                # TODO: if the req_rsc_size is larger than the aval_rsc, then add a flag to indicate the task is late
            _p.required_resource_size = req_rsc_size

            if  aval_rsc >= req_rsc_size and req_rsc_size > 0:
                issue_list.append(_p)
                sorted_ready_queue.remove(_p)
                    # TODO: update the resource allocation
                aval_rsc -= req_rsc_size
            else:
                break

            # if issue the task to runnning list
        for _p in issue_list:
            running_queue.put(_p)
            ready_queue.get()
            _p.set_state("running")
            res_cfg.allocate(_p.pid, _p.required_resource_size)
            _p.waitTime = 0 
            _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued and "
            if _p.totburst==0:
                _p.start_time = curr_t
                _str += f"start at {curr_t:.6f}; "
            else:
                _str += f"resume at {curr_t:.6f}; "
            if not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(_str)
        issue_list.clear()

        # execute the task in running list
        # update the running task
    updateRunningQueue(timestep, running_queue, res_cfg) 
        # update the wait task
    updateWaitQueue(timestep, weight_wait_queue)

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

# =================== intergrated into scheduler class ===================


# =================== functions related to data transfer ===================

def data_prefetching(init_p_list, wait_queue, active_list=None, cached_cfg=None):
    # for pid in cached_cfg.keys():
    #     _p = init_p_list[pid]
    #     if _p in active_list:
    #         active_list.remove(_p)
    #         wait_queue.put(_p)
    #         _p.set_state("wait")
    #     else: 
    #         print("		Arriving lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
    
    # infinite bandwidth, buffer size, constant latency
    for pid in cached_cfg.keys():
        _p = init_p_list[pid]
        wait_queue.put(Data(_p.pid, 1, (0,), "weight", _p.io_time))
        _p.set_state("wait")

def updateWaitQueue(timestep, wait_queue:TaskQueue):
    """
    Update the waiting time used in waiting queue
    """
    # for _p in wait_queue:
    #     _p.waitTime += timestep
    for data in wait_queue: 
        data.waitTime += timestep 

def data_tranfer_sim(wait_queue, buffer:Buffer, bin_name):
    # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
    l_ready = []
    # for _p in wait_queue:
    #     trans_io_tile = _p.waitTime / _p.io_time
    for data in wait_queue: 
        trans_io_tile = data.waitTime / data.io_time
        # trans_io_tile_r = round(trans_io_tile)
        # trans_comp = np.allclose(trans_io_tile, trans_io_tile_r, atol=1e-2)
        trans_io_tile_r = int(trans_io_tile)
        trans_comp = trans_io_tile_r > 1
        # TODO: _p.waitTime > 0 
        if trans_comp and data.waitTime > 0: 
            wait_queue.remove(data)
            data.valid = True
            buffer.put(data)
            _str = f"		TASK {data.pid:d}:{data.data_id}({data.data_type}/{data.size}M) is transfered({bin_name})!!"
            # print(_str)

# unused functions
def RunningQueueToWait(running_queue, wait_queue):
    # CPU[i]->running->burst == CPU[i]->running->cpu
    l_wait = []
    for _p in running_queue:
        exe_io_tile = _p.burst / _p.cpu_time
        exe_io_tile_r = round(exe_io_tile)
        exe_comp = np.allclose(exe_io_tile, exe_io_tile_r, atol=1e-2)
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

def rsc_req_estm(_p, n_slot, timestep, FLOPS_PER_CORE):
    # release time round up: task should not be released earlier than the release time
    time_slot_s = int(np.ceil(_p.release_time/timestep))
    if time_slot_s < n_slot:
        time_slot_s = n_slot
    # deadline round down: task should not be finised later than the deadline
    time_slot_e = int(_p.deadline//timestep)
    req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
    return time_slot_s,time_slot_e,req_rsc_size


if __name__ == "__main__":
    pass
    

    