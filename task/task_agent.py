from __future__ import annotations
from typing import Dict, List, Tuple, Union, Any, OrderedDict
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from model.buffer import Buffer, EventCache, TriggerCache
    from model.buffer import Data

from dataclasses import dataclass, field, InitVar, asdict
import bisect
import copy
import numpy as np
import math
from scipy.stats import truncnorm

from global_var import *
from model.message.Context_message import ContextMsg
from model.resource_agent import DDL_reservation, RT_reservation, dummy_reservation
from model.event_gen.e2e_latency import jitter_gen_biside
from model.performance import cal_lat, slack_comp

# preemptable?/able to preempt others
scheduling_attr = {
    "fixed": 0,
    "hard": 1,
    "stationary": 2,
    "moveable": 3,
    }
# classify the task in to realtime task and deadline task
# enumerate value of timing_flag
task_timing_type = {
    "realtime": 0,
    "deadline": 1,
    }
task_freq_type = {
    "periodic": 0,
    "aperiodic": 1,
}
criticality = {
    "soft": 0,
    "hard": 1,
}
# lifetime of the task
# task_lifetime = {
#     "terminated": 0, # terminated
#     "suspend": 1, # inactive, wait for activation
#     "active": 2, # active, but not ready to be executed
#     "running": 3,
#     "throttled": 4, # ready by has no budget
#     "wait": 5, # waiting for I/O
#     "preempted": 6, # preempted by other tasks
#     "ready": 7, # ready to be executed
# }
from enum import Enum
class TaskState(Enum):
    terminated = 0
    suspend = 1
    active = 2
    running = 3
    throttled = 4
    wait = 5
    preempted = 6
    ready = 7

class ProcessBase(object): 
    def __init__(self, task, release_t, deadline_abs, pid):
        self.task:TaskBase = task
        self.pred_ctrl = task.pred_ctrl # used
        self.succ_data = task.succ_data # used
        self.pred_data = task.pred_data # used
        self.succ_ctrl = task.succ_ctrl # used
        self.trigger_mode = task.trigger_mode # event-triggered or periodic
        self.event_triggers = []  # list to hold event triggers


        # =============== 2. runtime state but schedule related ===============
        self.exp_comp_t = task.exp_comp_t # total cpu time, ++ when cpu burst
        self.release_time = release_t     # recored when the process is released
        self.deadline = deadline_abs      # recored when the process is generated  

        # =============== 2. runtime state ===============
        self.pid = pid # process id
        self._state = "terminated" # task state: running, terminated, suspend, runnable, throttled
        self.prio = task.prio      # priority

        self.cpu_time = task.cpu_time     # execution time per I/O burst
        self.totio = task.totio           # transfer time per I/O burst
        self.io_time = task.io_time       # I/O time
        self.totcpu = task.totcpu         # total cpu time, ++ when cpu burst

        self.i_offset = task.i_offset     # offset of the trigger time
        self.cbs_en = task.cbs_en         # whether the task has a constant bandwidth, i.e., applys resource reservation algorithm

        
        self.released = False
        self.remburst = 0                 # Record the remaining cpu time for current I/O op, -- when cpu burst, set when moved from waiting to running
        self.rem_flop_budget = {}          # the predefined budget of execution time, -- when cpu burst excceds the budget, moved from running to throttled, set when process is activated

        self.ready_time = -1 
        self.ready = False
        self.start_time = 0   # record at begining
        self.end_time = 0     # record at endding
        self.curr_start_time = 0 # record at getting cpu
        self.currentburst = 0 # For preemption, clear once context switches or preemption judgment happens, ++ when cpu burst
        self.burst = 0        # Record the current cpu time for current I/O op, ++ when cpu burst, clear from waiting to running
        self.totburst = 0     # Sync with clocks, ++ when cpu burst, 
        self.waitTime = 0     # Time in wait queue, ++ when in wait queue
        self.cumulative_executed_time = 0 
        
        self.req_queue = []    # cache the arrival requests
        self.msg_cache:List[ContextMsg] = []    # cache the generated messages
        self.event_time = -1    # record the event_time of the last process 
        self.next_event_time = None
        self.next_ingestion_time = None
        self.ddl_sharing_cross_chain = False

        self.n_fork = 0
        self.fork_pid_list = []
        self.fork_pid_candi = [] 
        self.fork_p_inst = []
        self.is_fork_inst = False
        self.parent_pid = None
        self.var_scale_factor = 1
        self.load_var = None

    def update_from_sched_task(self, task):
        self.exp_comp_t = task.exp_comp_t
        self.deadline = elim_nume_error(task.get_deadline_time())
        self.release_time = elim_nume_error(task.get_release_time())

    def reset_state_vars(self,):
        self.released = False
        self.ready_time = float("inf")
        self.ready = False

        # kill & drop the total execution 
        self.remburst = 0
        self.currentburst = 0
        self.burst = 0
        self.totburst = 0
        self.waitTime = 0
        self.cumulative_executed_time = 0
        self.set_state("suspend")

    def update_deadline(self): 
        if not self.ddl_sharing_cross_chain:
            if self.trigger_mode == "event":
                self.deadline = self.next_ingestion_time + self.task.ERT + self.task.ddl
            else:
                self.deadline += self.task.period
        else:
            raise NotImplementedError
    
    def update_deadline_from_timestamp(self):
        if not self.ddl_sharing_cross_chain:
            self.deadline = self.get_timestamp()+ self.task.ERT + self.task.ddl
        else:
            raise NotImplementedError
    
    def get_timestamp(self):
        return self.msg_cache[0].get_timestamp()

    def get_chain_deadline(_p, sched):
        event_time = _p.get_timestamp()
        e2e_latency = sched.e2e_latency if _p.task.timing_flag == "deadline" else sched.hyper_period
        chain_deadline = event_time + e2e_latency 
        return chain_deadline

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        if isinstance(new_state, TaskState):
            self._state = new_state
        else:
            raise ValueError("New state must be an instance of TaskState")

    def set_state(self, state_name):
        """
        Sets the state using the name of the TaskState member.
        """
        if state_name in TaskState.__members__:
            self.state = TaskState[state_name]
        else:
            raise ValueError(f"Invalid state: {state_name}")

    def get_state_name(self):
        """
        Returns the name of the current state as a string.
        """
        if self._state is not None:
            return self._state.name
        else:
            return None

    def check_depends(self, event_cache:EventCache=None, 
                      trigger_cache:TriggerCache=None, event_triggers:List[Tuple]=None):
        """
        if all the predecessor tasks are completed, return True
        """
        if len(self.pred_ctrl):
            if trigger_cache is None:
                pred_ctrl = self.pred_ctrl
            else:
                pred_ctrl = trigger_cache[self.pid]

            if event_triggers is None:
                event_triggers = self.event_triggers

            for key in pred_ctrl.keys():
                if not pred_ctrl[key]["valid"]:
                    return False
            if len(self.msg_cache) == 0:
                self.build_ctx()
            self.update_ctx("trigger")
            # clear the pred_ctrl valid flag
            self.reset_depends(type="ctrl", pred_ctrl=pred_ctrl)
            event_triggers.pop(0)
            return True
        else:
            if event_cache is None:
                pred_data = self.pred_data
            else:
                pred_data = event_cache[self.pid]

            if self.check_depends_data(None, True, event_cache=event_cache):
                self.reset_depends(pred_ctrl=pred_ctrl, pred_data=pred_data)
                if len(self.msg_cache) == 0:
                    self.build_ctx()
                return True
            else:
                return False

    def check_depends_data(self, buffer=None, barrier=False, glb_n_task_dict=None, event_cache:EventCache=None):
        assert buffer is not None if not barrier else True
        dict_t = {}
        if event_cache is None:
            pred_data = self.pred_data
        else:
            pred_data = event_cache[self.pid]

        for key in pred_data:
            attr_dict = pred_data[key]
            if barrier:
                valid = attr_dict["valid"]
            else:
                valid = glb_n_task_dict[key].pid in buffer.buffer_mux("output")
            if attr_dict["reDistPattn"] == "downscaling":
                # name parse
                # remove the thread number at the end of the name
                thread_n = key.split('_')[-2]
                troughput_n = key.split('_')[-1]
                task_n = key.replace("_"+thread_n, "").replace("_"+troughput_n, "")
                dict_t.update({task_n:dict_t.get(task_n, False) or valid})
            elif not valid:
                return False
        return np.array(list(dict_t.values())).all()

    def get_upstream_ctx(self, glb_n_task_dict:Dict, buffer:Buffer, pred_data:Dict[int, Dict]):
        """
        extract the serialized context of the upstream tasks
        """
        src_dict = {}
        for key in self.pred_data:
            attr_dict = pred_data[key]
            pid = glb_n_task_dict[key].pid
            tgt_buffer = buffer.buffer_mux("output")
            valid = pid in tgt_buffer
            if valid:
                data:Data = tgt_buffer[pid][0]
                src_dict.update({key:data.ctx.serialize()})
        if len(self.pred_data):
            assert len(src_dict) > 0
        return src_dict

    def get_downstream_ctx(self):
        """
        return the downstream context
        """
        return list(self.succ_data.keys())

    def get_trigger_ctx(self, pred_ctrl:Dict[int, Dict]=None) -> Dict[Any, Dict]:
        trigger_dict = {}
        for key in self.pred_ctrl: 
            # msg:ContextMsg = ContextMsg.create_sensor_ctx(pred_ctrl[key]["ingestion_time"], 
            #                                               pred_ctrl[key]["event_time"],
            #                                               pred_ctrl[key]["period"])
            msg:ContextMsg = pred_ctrl[key]["event_queue"].get()
            trigger_dict.update({key:msg.serialize()})
        return trigger_dict
    
    def build_ctx(self, ):
        """
        build the context of the task
        """
        # _p.msg_cache.append(ContextMsg().cache_upstreaming(_p, buffer, glb_n_task_dict))
        # create the context of the task
        ctx = ContextMsg.create_p_ctx(self)
        # generate and cache the context before firing the data to the next node
        self.msg_cache.append(ctx)
        return ctx

    def update_ctx(self, ctx_type, **kwargs):
        """
        update the context of the task
        """
        if ctx_type == "upstream":
            self.msg_cache[0].cache_upstreaming(**kwargs)
        elif ctx_type == "weight":
            self.msg_cache[0].cache_weight(self, **kwargs)
        elif ctx_type == "trigger":
            if len(self.pred_ctrl):
                self.msg_cache[0].cache_trigger(self, **kwargs)

    def sim_trigger(self, time=None, time_step=1e-6, pred_ctrl:Dict[int, Dict]=None, event_triggers:List[Tuple]=None):
        """
        if the task is activated, return True
        """
        if pred_ctrl is None:
            pred_ctrl = self.pred_ctrl
        if event_triggers is None:
            event_triggers = self.event_triggers
        if self.task.trigger_mode == "timer":
            if math.isclose(time%self.task.period, self.i_offset, abs_tol=time_step*0.99):
                for key in pred_ctrl.keys():
                    pred_ctrl[key]["valid"] = True
                    pred_ctrl[key]["ingestion_time"] = time
                return True
        elif self.trigger_mode == "event":
            if event_triggers:
                # self.event_triggers.pop(0)
                # next_ingestion_time, next_event_time = event_triggers[0]
                msg = event_triggers[0]
                for key in pred_ctrl.keys():
                    pred_ctrl[key]["valid"] = True
                    # TODO: replace these attribute by the context, 
                    # from event_triggers to event_queue
                    pred_ctrl[key]["event_queue"].put(msg)
                    # pred_ctrl[key]["ingestion_time"] = next_ingestion_time
                    # pred_ctrl[key]["event_time"] = next_event_time
                    # pred_ctrl[key]["period"] = self.task.period
                return True
        return False
    
    def reset_depends(self, type="all", pred_ctrl:Dict[int, Dict]=None, pred_data:Dict[int, Dict]=None):
        """
        reset the values of the predecessors to False
        """
        clear_pred_ctrl = False
        clear_pred_data = False
        if type == "all":
            clear_pred_ctrl = True
            clear_pred_data = True
        elif type == "ctrl":
            clear_pred_ctrl = True
        elif type == "data":
            clear_pred_data = True
        if clear_pred_ctrl:
            if pred_ctrl is None:
                pred_ctrl = self.pred_ctrl
            for key in pred_ctrl.keys():
                pred_ctrl[key]["valid"] = False
        if clear_pred_data:
            if pred_data is None:
                pred_data = self.pred_data
            for key in pred_data.keys():
                pred_data[key]["valid"] = False

    def parse_name(self):
        name = self.task.name
        thread_n = name.split('_')[-2]
        troughput_n = name.split('_')[-1]
        task_n = name.replace("_"+thread_n, "").replace("_"+troughput_n, "")
        return task_n,thread_n,troughput_n

    def pre_fork(self, verbose:bool=False) -> List[ProcessBase]:
        for fork_num in range(self.var_scale_factor-1):
            new_pid = self.fork_pid_candi.pop(0)
            self.fork_pid_list.append(new_pid)
        if verbose and self.var_scale_factor > 1:
            print(f"                {self.task.name}({self.pid}) forked {self.var_scale_factor-1} processes({self.fork_pid_list})")

    def p_fork(self) -> List[ProcessBase]:
        if self.var_scale_factor == 1:
            return []
        fork_list:List[ProcessBase] = []
        task_n, thread_n, troughput_n = self.parse_name()
        rem_load = self.load_var - 1
        for fork_num in range(self.var_scale_factor-1):
            _p_fork = copy.deepcopy(self)
            _p_fork.task.name = task_n + f"_fork_{fork_num+1}_{thread_n}_{troughput_n}"
            _p_fork.pid = self.fork_pid_list[fork_num]

            _p_fork.fork_pid_list = []
            _p_fork.fork_pid_candi = [] 
            _p_fork.fork_p_inst = []
            _p_fork.n_fork = 0
            _p_fork.is_fork_inst = True
            _p_fork.parent_pid = self.pid
            _p_fork.var_scale_factor = 1

            assert rem_load >= 0, f"rem_load({rem_load}) should be greater than 0"
            _p_fork.load_var = rem_load
            _p_fork.totcpu = self.task.totcpu if rem_load >= 1 else self.task.totcpu * rem_load
            self.n_fork += 1
            rem_load -= 1
            fork_list.append(_p_fork)
        return fork_list

    def kill_fork(self, process_dict:List[ProcessBase]):
        """       
            terminate forked process:
                set the property of process with `parent_pid`
                append pid to fork_pid_candi
                remove the pid from fork_pid_list
                minus n_fork by 1
                delete the process
        """        
        _p_parent:ProcessBase = process_dict[self.parent_pid]
        _p_parent.fork_pid_candi.append(self.pid)
        _p_parent.fork_pid_list.remove(self.pid)
        _p_parent.fork_p_inst.remove(self)
        _p_parent.n_fork -= 1
        process_dict.pop(self.pid)

    def handle_process_load_var(self):
        """
        fork the task @ release:
            copy the process, rename the process and change the process id
            set the `parent_pid, is_fork_inst, pid` for the new process
            set `n_fork, new_pid, fork_pid_list` for the parent process
        """
        workload_var_info = self.msg_cache[0].get_load_var()
        involve_times = 0
        var_scale_factor = 1

        task_n, thread_n, troughput_n = self.parse_name()
        for var_item, var_param in workload_var_info.items():
            tgt_list = var_param['tgt_name']
            if task_n in tgt_list:
                var_scale_factor = max(math.ceil(var_param['size']/var_param['typical']), 1)
                load_var = var_param['size']/var_param['typical']
                involve_times += 1            
        if involve_times > 1:
            raise ValueError("a single task should not involve multiple workload scaling processes")
        elif involve_times == 1:
            self.var_scale_factor = var_scale_factor 
            self.load_var = load_var

    def ready_util(self, curr_t, ready_queue):
        ready_queue.put(self)
        self.ready_time = curr_t
        self.ready = True
        self.set_state("ready")

    def release_util(self, curr_t, active_list, verbose=True):
        active_list.append(self)
        self.set_state("active")
        # _p.totcpu = _p.task.totcpu if _p.load_var is None else _p.task.totcpu * _p.load_var
        if self.load_var is None or self.load_var >= 1:
            self.totcpu = self.task.totcpu
        else:
            self.totcpu = self.task.totcpu * self.load_var
        self.release_time = curr_t
        self.released = True
        # a task is initialized in multiple queues,
        # we ensure that the remburst is only released once
        if self.remburst == 0:
            self.remburst += self.totcpu
        if verbose:
            _str = f"		TASK {self.task.id:d}:{self.task.name:s}({self.pid:d}) is activated @ {curr_t:.6f}/{self.get_timestamp():.6f}!!"
            print(_str)

    def throttle_util(_p, throttle_list, curr_t=None, 
                      mode="start", 
                      
                      verbose=True):
        assert mode in ["start", "pend", "migrate"]
        if verbose:
            assert curr_t is not None
            # warnings.warn("		TASK {:d}:{:s}({:d}) THROTTLED!!".format(_p.task.id, _p.task.name, _p.pid))
            print("		TASK {:d}:{:s}({:d}) is THROTTLED @ {:.6f} !!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
        # update statistics 
        throttle_list.append(_p)
        _p.set_state("throttled")

        # suppose kill strategy
        # current tile should be reloaded and re-executed
        # other wise, modify the io time
        _p.ready = False
        _p.task.throttle_count += 1
        _p.currentburst = 0
        # _p.burst = 0
        # _p.cumulative_executed_time = 0

        if mode == "start":
            pass
        elif mode == "pend":
            _p.ready_time = -1
            _p.waitTime = 0
        elif mode == "migrate":
            _p.migration_count = 0


class ProcessInt(ProcessBase):
    def __init__(self, task:TaskBase, release_t, deadline_abs, pid):
        super().__init__(task, release_t, deadline_abs, pid)
        # task_id -> (main_num, RDA_num)
        self.task:TaskInt
        self.allocated_resource:OrderedDict[int, Tuple[int, int]] = OrderedDict()
        # runtime state but schedule related
        self.required_resource_size:int = task.required_resource_size

        self.input_ready:bool = False
        self.output_ready:bool = False
        self.weight_ready:bool = False

        self.core_max = task.core_max
        self.core_min = task.core_min
        self.core_list = task.core_list
        self.parallel_mode = task.parallel_mode

        self.is_starving = False
        self.expexted_rsc_size = -1

    def reset_state_vars(self):
        super().reset_state_vars()
        self.allocated_resource.clear()
        self.required_resource_size = self.task.required_resource_size
        self.input_ready = False
        self.output_ready = False
        self.weight_ready = False
        self.is_starving = False
    
    def update_from_sched_task(self, task:TaskInt):
        super().update_from_sched_task(task)
        self.required_resource_size = self.task.required_resource_size
        

    def rsc_req_estm(_p, n_slot, timestep, FLOPS_PER_CORE, time_slot_s=None, time_slot_e=None, mode='rt-wsc', over_provision_rate=0., max_size=float("inf")):
        assert mode in ['rt-wsc', 'expected']
        if time_slot_e is None or time_slot_s is None:
            time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep)
        if mode == 'expected':
            num_slot = int(_p.exp_comp_t/timestep)
            req_rsc_size = int(np.ceil(_p.remburst/num_slot/timestep/FLOPS_PER_CORE))
        else:
            if time_slot_e <= time_slot_s:
                req_rsc_size = 0
            else:
                # req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE/(1-over_provision_rate)))
                slack = slack_comp(time_slot_e - time_slot_s, 0, over_provision_rate)
                req_rsc_size = int(np.ceil(_p.remburst/slack/timestep/FLOPS_PER_CORE))
                if req_rsc_size > max_size and max_size != float("inf"):
                    req_rsc_size = max_size
                    Warning(f"req_rsc_size({req_rsc_size}) is greater than max_size({max_size})")
                    time_slot_e = time_slot_s + int(np.ceil(_p.remburst/req_rsc_size/timestep/FLOPS_PER_CORE))
        return time_slot_s,time_slot_e,req_rsc_size

    def quant_release_deadline(_p, n_slot, timestep):
        # release time round up: task should not be released earlier than the release time
        time_slot_s = int(np.ceil(_p.release_time/timestep))
        if time_slot_s < n_slot:
            time_slot_s = n_slot
        # deadline round down: task should not be finised later than the deadline
        time_slot_e = int(_p.deadline//timestep)
        return time_slot_s,time_slot_e

    def get_available_cfg(self, req_rsc_size:int, curr_aval_rsc:Union[int, None]=None, show_warnings=False): 
        applied_constraint = "none"         
        # apply constraints based on parallel_mode
        if self.parallel_mode in ["upb","range"]:
            if req_rsc_size > self.core_max:
                req_rsc_size = self.core_max
                applied_constraint = "upb"
        elif self.parallel_mode in ["lwb", "range"]:
            if curr_aval_rsc is not None:
                if self.core_min > curr_aval_rsc:
                    # no available solution
                    return 0, "N/A"
            if self.core_min > req_rsc_size: 
                req_rsc_size = self.core_min
                applied_constraint = "lwb"
        elif self.parallel_mode == "list":
            # select the nearest one
            # filter the core_list by the current available resource
            if curr_aval_rsc is not None:
                core_list = [x for x in self.core_list if 0 < x <= curr_aval_rsc] 
                if len(core_list) == 0:
                    # no available solution
                    return 0, "N/A"
            else:
                core_list = self.core_list
            if req_rsc_size > max(core_list):
                req_rsc_size = max(core_list)
            else:
                # return i, s.t., all e in a[i:] have e >= x
                idx = bisect.bisect_left(sorted(core_list), req_rsc_size)
                req_rsc_size = core_list[idx] 
            req_rsc_size = min(core_list, key=lambda x:abs(x-req_rsc_size))
            applied_constraint = "list"
        
        if req_rsc_size == 0: 
            return 0, "N/A"
        if curr_aval_rsc is not None:
            if req_rsc_size > curr_aval_rsc:
                if show_warnings: 
                    warnings.warn(f"TASK {self.task.id:d}:{self.task.name:s}({self.pid:d}) is starving {req_rsc_size-curr_aval_rsc:d} cores")
                applied_constraint = "partial"
            req_rsc_size = min(req_rsc_size, curr_aval_rsc)

        return req_rsc_size, applied_constraint

    def get_available_cfg_vector(self, req_rsc_size_arr: np.ndarray, curr_aval_rsc_arr: np.ndarray = None):
        assert req_rsc_size_arr.ndim == 1
        assert curr_aval_rsc_arr.ndim == 1
        applied_constraint_arr = np.array(["none"] * len(req_rsc_size_arr))

        # Apply constraints based on parallel_mode
        if self.parallel_mode in ["upb", "range"]:
            req_rsc_size_arr = np.minimum(req_rsc_size_arr, self.core_max)
            applied_constraint_arr[req_rsc_size_arr == self.core_max] = "upb"
        elif self.parallel_mode in ["lwb", "range"]:
            # No available solution
            idx = (self.core_min > curr_aval_rsc_arr)
            req_rsc_size_arr[idx] = 0
            applied_constraint_arr[idx] = "N/A"
            idx = (self.core_min > req_rsc_size_arr)
            req_rsc_size_arr[idx] = self.core_min
            applied_constraint_arr[idx] = "lwb"
        elif self.parallel_mode == "list":
            # Select the nearest one
            # Filter the core_list by the current available resource
            # if curr_aval_rsc_arr is not None:
            #     core_list = np.array([x for x in self.core_list if x <= curr_aval_rsc_arr.max()])
            # else:
            #     core_list = np.array(self.core_list)
            # core_list = np.min(req_rsc_size_arr, curr_aval_rsc_arr)
            # not_aval_pos = np.nonzero(curr_aval_rsc_arr >= core_list)
            curr_aval_rsc_arr_T = curr_aval_rsc_arr.reshape(-1, 1)
            core_list = np.array(self.core_list).reshape(1, -1)

            diff_arr = np.abs(curr_aval_rsc_arr_T - core_list)
            if curr_aval_rsc_arr is not None:
                diff_arr[curr_aval_rsc_arr_T < core_list] = np.inf
            # exclude the case diff_arr[i, j] == np.inf 
            min_diff_idx = np.argmin(diff_arr, axis=1)
            req_rsc_size_arr = core_list[min_diff_idx]
            not_available_pos = np.nonzero(diff_arr[np.arange(len(diff_arr)), min_diff_idx] == np.inf)
            not_applied_pos = np.nonzero(diff_arr[np.arange(len(diff_arr)), min_diff_idx] == 0)
            applied_constraint_arr[~not_applied_pos and ~not_available_pos] = "list"
            applied_constraint_arr[not_available_pos] = "N/A"
        return req_rsc_size_arr, applied_constraint_arr


@dataclass
class TaskAttr:
    name: str
    freq: float

    timing_flag: str
    criticality: str
    trigger_mode: str
    
    chain_criticality: str = 'hard'
    core_max: int = 0  # Maximum core
    core_min: int = 0  # Minimum core
    core_list: List[int] = None  # Core list
    parallel_mode: str = None  # Parallel mode
    core_max_compile: int = 0  # Maximum core for compilation
    core_min_compile: int = 0  # Minimum core for compilation
    core_list_compile: List[int] = None  # Core list for compilation
    thread_scaling_factor: int = 1  # Thread scaling factor
    freq_division_factor: int = 1  # Frequency division factor
    var_factor: int = 1  # Variable factor

    jitter_max: int = 0  # Maximum jitter

    ERT:Union[float, int] = field(init=False) # ERT (Earliest Release Time)
    ddl:Union[float, int] = field(init=False) # Deadline
    exp_comp_t:Union[float, int] = field(init=False) # Expected Completion Time

@dataclass
class TaskIntAttr(TaskAttr):
    flops:Union[int, float]=0
    io_time:Union[int, float]=0
    task_flag:str="moveable"
    pre_assigned_resource_flag:bool=False

    num_exec: int = field(init=False)
    no_stall_latency: float = field(init=False)
    min_tot_rsc: int = field(init=False)
    max_tot_rsc: int = field(init=False)
    flops_ModelSum: float = field(init=False)
    flops_ModelSumMax: float = field(init=False)
    equiv_core: float = field(init=False)
    util: float = field(init=False)
    main_size: float = field(init=False)
    rda_size: float = field(init=False)
    # database: InitVar[Dict] = {'f_gcd': 10, 'hyper_p': 0.1, 'exec_t_comp_ratioA': 0.05, 'wsc_slack_ratio': 0.8}

    def __post_init__(self):
        self.task_flag_num = scheduling_attr[self.task_flag]

class TaskBase(object):
    def __init__(self, task_name:str, task_id:int, timing_flag:str,
                 period:int, i_offset:int, jitter_max:int,
                 ERT:Union[int, float, None]=None, ddl:Union[int, float, None]=None, exp_comp_t:Union[int, float, None]=None,
                 op_io_time:int=0, op_cpu_time:int=0, seq_io_time:int=0, seq_cpu_time:int=0, priority:int=0, 
                 criti_flag:str="soft", cbs_en:bool=False, 
                 trigger_mode:bool=False, 
                 parallel_cfg:dict={}, parallel_cfg_compile:dict={},
                 chain_criti_flag:str="hard",
                 ):
        self.id = task_id
        self.name = task_name

        # =============== 1. task properties ===============
        # timing spec 
        self.freq = 1
        self.timing_flag = timing_flag
        self.timing_flag_num = task_timing_type[timing_flag]
        self.criticality = criti_flag # soft or hard
        self.chain_criticality = chain_criti_flag # soft or hard
        assert self.criticality in criticality.keys()
        assert self.timing_flag in task_timing_type.keys()
        self.trigger_mode = trigger_mode # event-triggered or periodic

        self.core_max = parallel_cfg["max"] if "max" in parallel_cfg else 1e3
        self.core_min = parallel_cfg["min"] if "min" in parallel_cfg else 0
        self.core_list = parallel_cfg["list"] if "list" in parallel_cfg else None
        self.parallel_mode = parallel_cfg["mode"] if "mode" in parallel_cfg else None

        self.core_max_compile = parallel_cfg_compile["max"] if "max" in parallel_cfg else 1e3
        self.core_min_compile = parallel_cfg_compile["min"] if "min" in parallel_cfg else 0
        self.core_list_compile = parallel_cfg_compile["list"] if "list" in parallel_cfg else None

        self.thread_scaling_factor = 1
        self.freq_division_factor = 1
        self.var_factor = 1

        self.jitter_max = jitter_max # max jitter

        self.update_sched_timing(ERT, ddl, exp_comp_t)
        # e.g. the task with 30hz but be divided into 3 tasks with 10hz and 1/30s offset
        self.i_offset = i_offset # offset of the sub-period
        self.period = period
        self.aval_sub_period = []
        self.hyper_period_size = None 
        self.sub_cycle_cnt = 0     

        self.pred_ctrl = {} # used
        self.succ_data = {} # used
        self.pred_data = {} # used
        self.succ_ctrl = {} # used

        # =============== 2. runtime state ===============
        # self.pid = 0 # process id
        self.prio = priority      # priority

        # self.release_time = self.ERT + self.i_offset # recored when the process is generated
        # self.deadline = self.release_time + self.ddl # recored when the process is generated  
        self.cpu_time = op_cpu_time                  # execution time per I/O burst
        self.io_time = op_io_time                    # I/O time
        self.totcpu = seq_cpu_time                   # total cpu time
        self.totio = seq_io_time                     # tot io transfer
        self.cbs_en = cbs_en

        # =============== 3. statistical properties ===============
        self.missed_deadline_count = 0 # used 

        # every time the task is scheduled
        self.cum_trunAroundTime = 0 # used
        self.completion_count = 0 # used

        # for interrupt
        self.context_switch_count = 0
        self.preemption_count = 0
        self.migration_count = 0
        self.throttle_count = 0

        # unused properties
        # # L1 resource allocation
        # self.allocated_resource:OrderedDict[int, List[Any]] = OrderedDict()
        # self.required_resource_size:int = 0
        # self.cumulative_waiting_time = 0
        # self.response_time = 0
        # # self.fault = False
        # # self.fault_time = 0

        # # statistics
        # self.missed_deadline = False
        # self.missed_deadline_time = 0
        # self.turnaround_time = 0
        # self.jitter = 0

    def update_sched_timing(self, ERT, ddl, exp_comp_t):
        # deadline and earlist release time without considering the offset of its sub-period
        self._ERT = ERT # relative earliest release time in each sub-period
        self._ddl = ddl # relative deadline in each sub-period
        self._exp_comp_t = exp_comp_t
    
    # getter and setter for ERT, ddl, exp_comp_t
    @property
    def ERT(self):
        if self._ERT is None:
            raise ValueError("ERT is not set yet")
        return self._ERT

    @ERT.setter
    def ERT(self, value):
        self._ERT = value

    @property
    def ddl(self):
        if self._ddl is None:
            raise ValueError("ddl is not set yet")
        return self._ddl

    @ddl.setter
    def ddl(self, value):
        self._ddl = value

    @property
    def exp_comp_t(self):
        if self._exp_comp_t is None:
            raise ValueError("exp_comp_t is not set yet")
        return self._exp_comp_t

    @exp_comp_t.setter
    def exp_comp_t(self, value):
        self._exp_comp_t = value

    def make_process(self, release_t, deadline_abs, pid):
        """
        make a process for the task
        """
        return ProcessBase(self, release_t, deadline_abs, pid)

    def add_event_trigger(self, trigger):
        self.event_triggers.append(trigger)

    def __str__(self) -> str:
        _str = f"Task {self.id}: {self.name}\n"
        _str += f"\ttiming_flag: {self.timing_flag}, criticality: {self.criticality}, trigger_mode: {self.trigger_mode}\n"
        if self.parallel_mode == "upb":
            _str += f"\tnum of cores (compile): 1 ~ {self.core_max_compile}\n"
            _str += f"\tnum of cores (run): 1 ~ {self.core_max}\n"
        elif self.parallel_mode == "list":
            _str += f"\tnum of cores (compile): {self.core_list_compile}\n"
            _str += f"\tnum of cores (run): {self.core_list}\n"
        elif self.parallel_mode == "range":
            _str += f"\tnum of cores (compile): {self.core_min_compile} ~ {self.core_max_compile}\n"
            _str += f"\tnum of cores (run): {self.core_min} ~ {self.core_max}\n"
        elif self.parallel_mode == "lwb":
            _str += f"\tnum of cores (compile): {self.core_min_compile} ~ Max\n"
            _str += f"\tnum of cores (run): {self.core_min} ~ Max\n"
        else:
            _str += f"\tnum of cores: no constraint\n"
        _str += f"\tspatial factor: thread {self.thread_scaling_factor}, freq {self.freq_division_factor}, var {self.var_factor}\n"
        _str += f"\tslack info: {self.ERT:.2e} ~ {(self.ERT+self.ddl):.2e} ({self.exp_comp_t:.2e})\n"
        _str += f"\tperiod: {self.period:.2e}, i_offset: {self.i_offset:.2e}\n"
        _str += f"\tjitter_max: {self.jitter_max}\n"
        _str += f"\tcpu_time: {self.cpu_time:.2e}, io_time: {self.io_time:.2e}, totcpu: {self.totcpu:.2e}\n"
        return _str

    def print_stat(self):
        _str += f"state: {self.state}, "
        _str += f"start_time: {self.start_time}, end_time: {self.end_time}, currentburst: {self.currentburst}, burst: {self.burst}, totburst: {self.totburst}, waitTime: {self.waitTime}\n"
        _str += f"missed_deadline_count: {self.missed_deadline_count}\n"
        _str += f"cumulative_executed_time: {self.cumulative_executed_time}, cum_trunAroundTime: {self.cum_trunAroundTime}, completion_count: {self.completion_count}\n"
        _str += f"context_switch_count: {self.context_switch_count}, preemption_count: {self.preemption_count}, migration_count: {self.migration_count}\n"

    def freq_division(self, factor, hyper_p, mode) -> List[TaskBase]:
        """
        generate a series of sub-tasks that hold the different parts of execution in a hyper-period
        example: 
        Task 240Hz,hyper_p=0.1s, factor=3
        mode A: [0-7], [8-15],[15-23]
        mode B: [0, 3, 6, ..., 21] [1, 4, 7, ..., 22], [2, 5, 8, ..., 23]
        """
        if factor == 1:
            self.name  = f"{self.name}_{0}"
            return [self]
        assert factor >= 1

        assert self.period <= hyper_p
        assert round(hyper_p % self.period, numerical_tol_bit) == 0
        n_cycle = int(hyper_p / self.period)

        assert round(n_cycle % factor, numerical_tol_bit) == 0
        interval = int(n_cycle / factor)
        
        sub_tasks:List[TaskBase] = [copy.deepcopy(self) for _ in range(factor)]
        assert mode in ['interleave', 'repeat']
        if mode == "interleave":
            for i in range(factor):
                new_period = self.period * factor
                new_i_offset = self.period * i
                sub_tasks[i].period = new_period
                sub_tasks[i].i_offset = new_i_offset
                sub_tasks[i].id = self.id + i
                sub_tasks[i].name  = f"{self.name}_{i}"
        elif mode == "repeat":
            for i in range(factor):
                sub_tasks[i].aval_sub_period = [i * interval + j for j in range(interval)]
                sub_tasks[i].hyper_period_size = n_cycle
                sub_tasks[i].id = self.id + i
                sub_tasks[i].name  = f"{self.name}_{i}"

        return sub_tasks

    def gen_event_modA_endless(self, ):
        i = 0
        event_time = self.i_offset
        jitter = 0
        jitter = yield 
        while True:
            jitter = yield event_time + jitter
            if jitter is None: 
                return i
            event_time += self.period
            i += 1      
    
    def gen_event_modB_endless(self, ):
        i = 0
        event_time = self.i_offset
        sub_period_offset = -np.inf
        jitter = yield event_time + sub_period_offset 
        while True:
            for j in self.aval_sub_period:
                jitter = yield event_time + self.period * j + jitter
                if jitter is None: 
                    return i*len(self.aval_sub_period)+j
            event_time += self.hyper_period_size * self.period
            i += 1
    
    # def extract_sensor_event(_p, event_range, jitter_sim_en=False, jitter_sim_para=None, seed=0):
    #     n_event = int(event_range/_p.task.period)
    #     if jitter_sim_en:
    #         jitter_gen_inst = jitter_gen(1/_p.freq, jitter_sim_para, size=1, seed=seed)
    #     event_gen = _p.event_generator()
    #     next(event_gen)
    #     for i in range(n_event):
    #         if jitter_sim_en:
    #             jitter = jitter_gen_inst()
    #             assert abs(jitter.max()) < 0.5*_p.period, "jitter is too large"
    #             event_gen.send(jitter)
    #         else:
    #             event_gen.send(0)
    #     event_gen.send(None)

    def gen_event_modA(self, event_range, jitter_sim_en=False, jitter_sim_para=None, seed=0):
        n_event = int(event_range/self.period)
        if jitter_sim_en:
            jitter_gen_inst = jitter_gen_biside(1/self.freq, jitter_sim_para, size=1, seed=seed)

        i = 0
        event_time = self.i_offset
        while True:
            if jitter_sim_en:
                jitter = jitter_gen_inst()
                assert abs(jitter.max()) < 0.5*self.period, "jitter is too large"
                out = round(event_time + jitter, numerical_tol_bit)
                yield out
            else:
                yield round(event_time, numerical_tol_bit)
            i += 1
            if i >= n_event:
                yield np.inf
                return i
            event_time += self.period
    
    def gen_event_modB(self, event_range, jitter_sim_en=False, jitter_sim_para=None, seed=0):
        n_p = round(event_range/self.period)
        n_event = int(n_p//self.hyper_period_size) * len(self.aval_sub_period) + len([i for i in range(n_p%len(self.aval_sub_period)) if i in self.aval_sub_period])
        if jitter_sim_en:
            jitter_gen_inst = jitter_gen_biside(1/self.freq, jitter_sim_para, size=1, seed=seed)

        event_no = 0
        event_time = self.i_offset
        while True:
            for j in self.aval_sub_period:
                if jitter_sim_en:
                    jitter = jitter_gen_inst()
                    assert abs(jitter.max()) < 0.5*self.period, "jitter is too large"
                    yield round(event_time + self.period * j + jitter, numerical_tol_bit)
                else:
                    yield round(event_time + self.period * j, numerical_tol_bit)
                event_no += 1
                if event_no >= n_event:
                    yield np.inf
                    return event_no
            event_time += self.hyper_period_size * self.period

    def event_generator(self, event_range=None, jitter_sim_en=False, jitter_sim_para=None, seed=0):
        """
        return a generator that generates the events of the task
        """
        if event_range is None:
            if self.hyper_period_size is not None:
                return self.gen_event_modB_endless()
            else:
                return self.gen_event_modA_endless()
        else:
            if self.hyper_period_size is not None:
                return self.gen_event_modB(event_range, jitter_sim_en, jitter_sim_para, seed)
            else:
                return self.gen_event_modA(event_range, jitter_sim_en, jitter_sim_para, seed)

    def delegate_event_generator(self, verbose=True, **kwargs):
        while True:
            n_event = yield from self.event_generator(**kwargs)
            if verbose:
                print(f"{n_event} events of {self.name} are generated")

    @classmethod
    def get_event_generator(cls, glb_p_list:List[ProcessBase], hyper_p, n_p, warmup, **kwargs): 
        event_range = hyper_p * (n_p+warmup)
        event_iter_dict = {}
        for _p in glb_p_list:
            _task = _p.task
            task_n = _task.name
            if _task.trigger_mode!='N':
                # filter the processes with trigger_mode is not "N"
                event_time_iter = _task.delegate_event_generator(event_range=event_range)
                ingestion_time_iter = _task.delegate_event_generator(event_range=event_range, **kwargs)
                event_iter_dict[task_n] = [ingestion_time_iter, event_time_iter]
        return event_iter_dict

    def extract_sensor_event(_task, event_range, verbose=True):
        event_gen = _task.delegate_event_generator(verbose, event_range=event_range)
        l = [next(event_gen)]
        while True:
            event_time = event_gen.send(0)
            # print(event_time)
            if event_time >= event_range:
                break
            l.append(event_time) 
        event_gen.send(None)
        return l

class TaskInt(TaskBase): 
    def __init__(
                    self, task_name:str, task_id:int, timing_flag:str,
                    # ERT:Union[int, float], ddl:Union[int, float], 
                    # exp_comp_t:Union[int, float],                     
                    period:Union[int, float], 
                    i_offset:Union[int, float], jitter_max:Union[int, float]=0,
                    flops:Union[int, float]=0, task_flag:str="moveable",
                    pre_assigned_resource_flag:bool=False, 
                    op_io_time:int=0, op_cpu_time:int=0, seq_io_time:int=0, seq_cpu_time:int=0, priority:int=0, 
                    criti_flag:str="soft", cbs_en:bool=False, 
                    trigger_mode:str="N",
                    parallel_cfg:dict={}, parallel_cfg_compile:dict={},
                    chain_criti_flag:str="hard",
                    **kwargs
                ) -> None:
        super().__init__(
                            task_name=task_name, task_id=task_id, timing_flag=timing_flag,
                            period=period, i_offset=i_offset, jitter_max=jitter_max, 
                            op_cpu_time=op_cpu_time, op_io_time=op_io_time, seq_io_time=seq_io_time,
                            seq_cpu_time=seq_cpu_time, priority=priority, 
                            criti_flag=criti_flag, cbs_en=cbs_en,
                            trigger_mode=trigger_mode, parallel_cfg=parallel_cfg,
                            parallel_cfg_compile=parallel_cfg_compile,
                            chain_criti_flag=chain_criti_flag,
                            ERT=kwargs.get("ERT", None), ddl=kwargs.get("ddl", None), exp_comp_t=kwargs.get("exp_comp_t", None),
                        )
        
        # =============== 1. task properties ===============
        self.flops = flops
        # designed scheduling attribute
        self.affinity = []
        self.affinity_n = []
        self.task_flag_num = scheduling_attr[task_flag]
        self.task_flag = task_flag 
        
        # =============== 2. Node attribution ===============
        self.pre_assigned_resource_flag = pre_assigned_resource_flag
        
        # =============== 3. runtime attribute ===============
        # L1 resource allocation
        # task_id -> (main_num, RDA_num)
        self.allocated_resource:OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.update_sched_size(main_size=kwargs.get("main_size", None), RDA_size=kwargs.get("RDA_size", None))

    def update_sched_size(self, main_size=None, RDA_size=None):
        if main_size is not None and RDA_size is not None:
            if RDA_size > 0:
                self.pre_assigned_resource = DDL_reservation(main_size, RDA_size)
            else:
                self.pre_assigned_resource = RT_reservation(main_size, RDA_size)
            self.required_resource_size:int = main_size

    # ======== vanilla properties setter and getter ========
    def set_task_name(self, task_name):
        self.task_name = task_name
    
    def set_task_id(self, task_id):
        self.id = task_id
    
    # ======== non-trival properties setter and getter ========

    def get_release_time(self, period_n=0, jitter=0):
        return self.ERT + self.i_offset + period_n * self.period + jitter

    def get_deadline_time(self, period_n=0, jitter=0):
        return self.ERT + self.ddl + self.i_offset + period_n * self.period + jitter

    def get_release_event(self, time_interval, period_n=0, i_offset=0):
        # TODO: miss the event at the remaining time
        for i in range(int(time_interval//self.period)):
            yield self.ERT + self.i_offset + (i+period_n) * self.period + i_offset

    def get_deadline_event(self, time_interval, period_n=0, i_offset=0):
        # TODO: miss the event at the remaining time
        for i in range(int(time_interval//self.period)):
            yield self.ERT + self.ddl + self.i_offset + (i+period_n) * self.period + i_offset
    
    def get_finish_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.exp_comp_t +self.ERT+ self.i_offset + (i+period_n) * self.period + i_offset

    def get_available(self):
        assert self.pre_assigned_resource_flag
        return self.pre_assigned_resource.get_available_rsc()

    # def release(self, task_id:int=0, main_num:int=0, RDA_num:int=0, verbose:bool=False):
    #     if not task_id:
    #         task_id = self.id
    #     if self.pre_assigned_resource_flag: 
    #         self.pre_assigned_resource.release(task_id, main_num, RDA_num, verbose)
    #         return {"id":self.id, "resource":self.allocated_resource, "released": True}
    #     else:
    #         return {"id":self.id, "resource":self.allocated_resource, "released": False}
    
    # def allocate(self, task_id:int=0, main_num:int=0, RDA_num:int=0, verbose:bool=False):
    #     if self.pre_assigned_resource_flag: 
    #         self.pre_assigned_resource.allocate(task_id, main_num, RDA_num, verbose)
    #         return {"id":self.id, "resource":self.allocated_resource, "allocated": True}
    #     else:
    #         return {"id":self.id, "resource":self.allocated_resource, "allocated": False}

    # def can_execute(self, task:TaskInt, verbose:bool=False) -> bool:
    #     """
    #     A task is executable if it has enough resource to execute the task before the deadline
    #     if the task is pre-assigned, it will always return true
    #     if the task is not pre-assigned, it will play a insert-based scheduling 
    #     """
    #     # check if the task can be scheduled
    #     if task.pre_assigned_resource_flag:
    #         return True
    #     else:
    #         # check if the task can be scheduled
    #         if self.pre_assigned_resource.can_execute(task, verbose):
    #             # allocate the resource
    #             self.allocated_resource[task.id] = (task.main_size, task.RDA_size)
    #             self.required_resource_size += task.main_size + task.RDA_size
    #             return True
    #         else:
    #             return False

    def query_rsc(self, task_id:int=0, verbose:bool=False): 
        # query the rsc allocated by task (id) 
        return self.pre_assigned_resource.rsc_map[task_id]

    def is_stationary(self) -> bool:
        return self.task_flag == "stationary"

    def get_rsc(self) -> int:
        return self.required_resource_size

    def __str__(self) -> str:
        _str = super().__str__()
        _str += f"\tflops: {self.flops:.2e}, req.: {self.required_resource_size}\n" 
        _str += f"\tmain_size: {self.pre_assigned_resource.main_size}, RDA_size: {self.pre_assigned_resource.RDA_size}, pre_assigned: {self.pre_assigned_resource_flag}\n"
        return _str

    def make_process(self, release_t, deadline_abs, pid):
        """
        make a process for the task
        """
        return ProcessInt(self, release_t, deadline_abs, pid)


def load_task_from_cfg(verbose:bool=False):
    """
    load task from config file
    """
    from task_cfg import task_attr_dict
    task_list = []
    task_id = 0
    # print(task_attr_dict)
    for task_n, task_attr in task_attr_dict.items():
        for i in range(task_attr["Thread_factor"]):
            T = task_attr["period"]*task_attr["Thread_factor"]
            phase = task_attr["period"]*i

            task = TaskInt(
                task_name=task_n+"_"+str(i), task_id=task_id, timing_flag=task_attr["timing_flag"], 
                ERT=task_attr["release_t"]/1000, ddl=task_attr['ddl']/1000, period=T, 
                exp_comp_t=task_attr['exe_t']/1000, i_offset=phase, jitter_max=0,
                flops=task_attr["flops"]/1e3, task_flag=task_attr["task_flag"], 
                pre_assigned_resource_flag=task_attr["pre_signed"]>0, 
                RDA_size=task_attr['Redundent_req'], main_size=task_attr['Cores_req']
            )
            task_id += 1
            task_list.append(task)
    return task_list

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="task1", help="task name")
    args = parser.parse_args() 
    if args.test_case == "base_task":
        task1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                    ERT=10, ddl=20, period=30, exp_comp_t=10, i_offset=0, jitter_max=0,
                    flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
        verbose = True
        task1.allocate(task_id=1, main_num=15, RDA_num=4, verbose=verbose)
        print(task1.get_available())
        task1.release(task_id=1, main_num=0, RDA_num=4, verbose=verbose)
    elif args.test_case == "load_task_cfg":
         task_list = load_task_from_cfg(verbose=args.verbose)
         for task in task_list:
             print(str(task))