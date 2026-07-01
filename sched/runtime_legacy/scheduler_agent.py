from typing import Dict, List, Tuple
from queue import Queue
from global_var import *

from model.buffer import Buffer, EventCache, TriggerCache
from model.buffer import Buffer
from model.resource_agent import Resource_model_int
from model.event_gen.e2e_latency import jitter_gen_biside
from model.task_queue_agent import TaskQueue 
from model.lru import LRUCache
from model.barrier_agent import Barrier
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe

from task.task_agent import ProcessInt
from sched.scheduling_table import SchedulingTableInt
from sched.monitor_agent import Monitor
from sched.runtime_legacy.sched_fn import *
from collections import defaultdict

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
                 forbid_miss:bool=False, trace_path:str=None, progress_aware:bool=True, 
                 allow_realloc:bool=True,
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
        self.budget_recoder = budget_recoder if budget_recoder else {}
        self.rsc_recoder_his = rsc_recoder_his if rsc_recoder_his else defaultdict(lambda: LRUCache(3))

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
        
        # progress aware
        self.progress_aware = progress_aware
        self.allow_realloc = allow_realloc

    def res_release(self, pid, op_pos_dict:bool=True):
        self.res_cfg.release(pid, verbose=False)
        if op_pos_dict: 
            self.position_dict.pop(pid, None)

    def check_draining_state(self):
        queue_clear_flag = True
        for _p in  (self.active_list + self.ready_queue.queue + self.running_queue.queue):
            if _p.get_timestamp() < self.switch_border:
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

    def check_miss(self, timestep, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                    a_data_pipe:DataPipe,
                    curr_t, res_cfg, 
                    bin_event_flg:bool=False):

        # check_miss(budget_recoder, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            # throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name)
        return check_miss(self, self.budget_recoder, timestep, msg_dispatcher, a_data_pipe, curr_t, res_cfg, self.weight_wait_queue, self.ready_queue, self.running_queue, self.miss_list,
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


    # sort metrices
    # excess_switching_border = lambda x: x.get_timestamp()>=sched.switch_border
    # is_running = x not in preemptable_list
    # is_not_hard = lambda x: x.task.criticality != "hard"
    # fn_crit = lambda x: x.deadline
    # # fn_crit = lambda _p: budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2]
    # fn_task_flag = lambda x: 0 if x.task.task_flag=="stationary" else 1

    def handle_taskqueue(self, curr_t, bin_event_flg, pre_rsc, rsc_map): 
        """ update the resource configuration by preempt_list, issue_list, ctx_switch_list

        Args:
            curr_t (_type_): _description_
            bin_event_flg (_type_): _description_
            pre_rsc (_type_): _description_
            rsc_map (_type_): _description_

        Returns:
            _type_: _description_
        """
        return handle_taskqueue(
            self, curr_t, self.res_cfg, self.ready_queue, self.running_queue, 
            self.preempt_list, self.issue_list, self.ctx_switch_list, 
            self._SchedTab.name, bin_event_flg, pre_rsc, rsc_map
            )
        

def handle_taskqueue(sched:Scheduler, curr_t, res_cfg, ready_queue, running_queue, preempt_list, issue_list, ctx_switch_list, bin_name, bin_event_flg, pre_rsc, rsc_map):
    for _p in preempt_list:
        print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
        running_queue.remove(_p)
        ready_queue.put(_p)
        sched.res_release(_p.pid, False)
        _p.set_state("ready")
        _p.task.preemption_count += 1
    preempt_list.clear()
            
    for _p in ctx_switch_list:
        print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) ctx switch at {curr_t:.6f}({pre_rsc[_p.pid]} -> {rsc_map[_p.pid]});")
        sched.res_release(_p.pid, False)
        res_cfg.allocate(_p.pid, rsc_map[_p.pid])
        _p.task.context_switch_count += 1
    ctx_switch_list.clear()

    # if issue the task to runnning list
    for _p in issue_list:
        _p:ProcessInt
        assert _p.get_state_name() != "running"
        running_queue.put(_p)
        if _p.pid not in sched.rsc_recoder_his:
            sched.rsc_recoder_his[_p.pid] = LRUCache(3)
        sched.rsc_recoder_his[_p.pid].put(sched._SchedTab.id)
        
        ready_queue.remove(_p)
        # NOTE:cross cancelation (removed)
        _p.set_state("running")
        res_cfg.allocate(_p.pid, rsc_map[_p.pid])
        _p.waitTime = 0 
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is issued and "
        if _p.totburst==0:
            _p.start_time = curr_t
            _str += f"started at {curr_t:.6f}; "
        else:
            _str += f"resumed at {curr_t:.6f}; "
        _p.curr_start_time = curr_t
        if bin_name and not bin_event_flg:
            bin_event_flg = True 
            print(f"({bin_name})")
        print(_str)
    issue_list.clear()



if __name__ == "__main__":
    pass
    

    