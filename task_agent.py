from __future__ import annotations
from typing import Dict, List, Tuple, Union, Any, OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from buffer import Buffer
    from buffer import Data
from Context_message import ContextMsg

import numpy as np
import math
# from hw_rsc import FLOPS_PER_CORE
# from scheduler_global_cfg import *
from resource_agent import DDL_reservation, RT_reservation, dummy_reservation
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
criticality = {
    "soft": 0,
    "hard": 1,
}
# lifetime of the task
task_lifetime = {
    "terminated": 0, # terminated
    "suspend": 1, # inactive, wait for activation
    "runnable": 2, # ready to be executed
    "running": 3,
    "throttled": 4, # ready by has no budget
    "wait": 5, # waiting for I/O
    "preempted": 6, # preempted by other tasks
    "ready": 7, # ready to be executed
    
}

class ProcessBase(object): 
    def __init__(self, task, release_t, deadline_abs, pid):
        self.task = task
        self.pred_ctrl = task.pred_ctrl # used
        self.succ_data = task.succ_data # used
        self.pred_data = task.pred_data # used
        self.succ_ctrl = task.succ_ctrl # used
        self.trigger_mode = task.trigger_mode # event-triggered or periodic
        self.event_triggers = []  # list to hold event triggers

        # =============== 2. runtime state ===============
        self.pid = pid # process id
        self.state = "terminated" # task state: running, terminated, suspend, runnable, throttled
        self.prio = task.prio      # priority

        self.cpu_time = task.cpu_time     # execution time per I/O burst
        self.io_time = task.io_time       # I/O time
        self.totcpu = task.totcpu         # total cpu time, ++ when cpu burst

        self.released = False
        self.i_offset = task.i_offset     # offset of the trigger time
        self.release_time = release_t     # recored when the process is released
        self.deadline = deadline_abs      # recored when the process is generated  
        self.exp_comp_t = task.exp_comp_t # total cpu time, ++ when cpu burst
        self.remburst = task.totcpu       # Record the remaining cpu time for current I/O op, -- when cpu burst, set when moved from waiting to running
        self.cbs_en = task.cbs_en         # whether the task has a constant bandwidth, i.e., applys resource reservation algorithm
        self.rem_flop_budget = 0            # the predefined budget of execution time, -- when cpu burst excceds the budget, moved from running to throttled, set when process is activated

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
        
        self.req_queue = []   # cache the arrival requests
        self.msg_cache:List[ContextMsg] = []    # cache the generated messages

    def set_state(self, state):
        assert state in task_lifetime.keys()
        self.state = state
    
    def check_depends(self):
        """
        if all the predecessor tasks are completed, return True
        """
        # if not len(self.pred_data):
        #     return np.allclose(time%self.task.period, self.task.ERT+self.task.i_offset, atol=time_step)
        # else:
        #     return np.array(list(self.pred_ctrl.values())+ list(self.pred_data.values())).all()
        # reDistPattn="downscaling"
        if len(self.pred_ctrl):
            for key in self.pred_ctrl.keys():
                if not self.pred_ctrl[key]["valid"]:
                    return False
            # clear the pred_ctrl valid flag
            self.reset_depends(type="ctrl")
            return True
        else:
            if self.check_depends_data(None, True):
                self.reset_depends()
                return True
            else:
                return False

    def check_depends_data(self, buffer=None, barrier=False, glb_n_task_dict=None):
        assert buffer is not None if not barrier else True
        dict_t = {}
        for key in self.pred_data:
            attr_dict = self.pred_data[key]
            if barrier:
                valid = attr_dict["valid"]
            else:
                valid = glb_n_task_dict[key].pid in buffer.buffer_mux("output")
            if attr_dict["reDistPattn"] == "downscaling":
                # name parse
                # remove the thread number at the end of the name
                thread_n = key.split('_')[-1]
                task_n = key.replace("_"+thread_n, "")
                dict_t.update({task_n:dict_t.get(task_n, False) or valid})
            elif not valid:
                return False
        return np.array(list(dict_t.values())).all()

    def get_upstream_ctx(self, glb_n_task_dict:Dict, buffer:Buffer):
        """
        extract the serialized context of the upstream tasks
        """
        src_dict = {}
        for key in self.pred_data:
            attr_dict = self.pred_data[key]
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

    def get_trigger_ctx(self):
        trigger_dict = {}
        for key in self.pred_ctrl: 
            msg:ContextMsg = ContextMsg.create_sensor_ctx(self.pred_ctrl[key]["trigger_time"])
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
            self.msg_cache[0].cache_upstreaming(self, **kwargs)
        elif ctx_type == "weight":
            self.msg_cache[0].cache_weight(self, **kwargs)
        elif ctx_type == "trigger":
            if len(self.pred_ctrl):
                self.msg_cache[0].cache_trigger(self)
        
            

    def sim_trigger(self, time=None, time_step=1e-6):
        """
        if the task is activated, return True
        """
        if self.task.trigger_mode == "timer":
            if math.isclose(time%self.task.period, self.i_offset, abs_tol=time_step*0.99):
                for key in self.pred_ctrl.keys():
                    self.pred_ctrl[key]["valid"] = True
                    self.pred_ctrl[key]["trigger_time"] = time
                return True
        elif self.trigger_mode == "event":
            if self.event_triggers:
                self.event_triggers.pop(0)
                for key in self.pred_ctrl.keys():
                    self.pred_ctrl[key]["valid"] = True
                return True
        return False
    
    def reset_depends(self, type="all"):
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
            for key in self.pred_ctrl.keys():
                self.pred_ctrl[key]["valid"] = False
        if clear_pred_data:
            for key in self.pred_data.keys():
                self.pred_data[key]["valid"] = False

class ProcessInt(ProcessBase):
    def __init__(self, task, release_t, deadline_abs, pid):
        super().__init__(task, release_t, deadline_abs, pid)
        # task_id -> (main_num, RDA_num)
        self.allocated_resource:OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.required_resource_size:int = task.required_resource_size
        self.input_ready:bool = False
        self.output_ready:bool = False
        self.weight_ready:bool = False

        self.core_max = task.core_max
        self.core_min = task.core_min
        self.core_list = task.core_list
        self.parallel_mode = task.parallel_mode

    def rsc_req_estm(_p, n_slot, timestep, FLOPS_PER_CORE):
        # release time round up: task should not be released earlier than the release time
        time_slot_s = int(np.ceil(_p.release_time/timestep))
        if time_slot_s < n_slot:
            time_slot_s = n_slot
        # deadline round down: task should not be finised later than the deadline
        time_slot_e = int(_p.deadline//timestep)
        if time_slot_e <= time_slot_s:
            req_rsc_size = 0
        else:
            req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
        return time_slot_s,time_slot_e,req_rsc_size

class TaskBase(object):
    def __init__(self, task_name:str, task_id:int, timing_flag:str,
                 ERT:int, ddl:int, period:int, exp_comp_t:int, i_offset:int, jitter_max:int,
                 op_io_time:int=0, op_cpu_time:int=0, seq_cpu_time:int=0, priority:int=0, 
                 criti_flag:str="soft", cbs_en:bool=False, 
                 trigger_mode:bool=False, 
                 parallel_cfg:dict={},
                 ):
        self.id = task_id
        self.name = task_name

        # =============== 1. task properties ===============
        # timing spec 
        self.timing_flag = timing_flag
        self.timing_flag_num = task_timing_type[timing_flag]
        self.criticality = criti_flag # soft or hard
        assert self.criticality in criticality.keys()
        assert self.timing_flag in task_timing_type.keys()
        self.trigger_mode = trigger_mode # event-triggered or periodic
        self.core_max = parallel_cfg["max"] if "max" in parallel_cfg else None
        self.core_min = parallel_cfg["min"] if "min" in parallel_cfg else None
        self.core_list = parallel_cfg["list"] if "list" in parallel_cfg else None
        self.parallel_mode = parallel_cfg["mode"] if "mode" in parallel_cfg else None

        # deadline in each hyper-period (task that have multiple sub-periods in a hyper-period)
        # e.g. the task with 30hz but be divided into 3 tasks with 10hz and 1/30s offset
        self.i_offset = i_offset # offset of the sub-period
        self.ERT = ERT # relative earliest release time in each sub-period
        self.ddl = ddl # relative deadline in each sub-period

        self.exp_comp_t = exp_comp_t
        self.period = period
        self.jitter_max = jitter_max # max jitter

        self.pred_ctrl = {} # used
        self.succ_data = {} # used
        self.pred_data = {} # used
        self.succ_ctrl = {} # used

        # =============== 2. runtime state ===============
        # self.pid = 0 # process id
        self.prio = priority      # priority

        self.release_time = self.ERT + self.i_offset # recored when the process is generated
        self.deadline = self.release_time + self.ddl # recored when the process is generated  
        self.cpu_time = op_cpu_time                  # execution time per I/O burst
        self.io_time = op_io_time                    # I/O time
        self.totcpu = seq_cpu_time                   # total cpu time, ++ when cpu burst
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

    def make_process(self, release_t, deadline_abs, pid):
        """
        make a process for the task
        """
        return ProcessBase(self, release_t, deadline_abs, pid)

    def add_event_trigger(self, trigger):
        self.event_triggers.append(trigger)

    def __str__(self) -> str:
        _str = f"Task {self.id}: {self.name}\n"
        _str += f"\ttiming_flag: {self.timing_flag}, prio: {self.prio}\n"
        _str += f"\tperiod: {self.period:.2e}, i_offset: {self.i_offset:.2e}, ERT: {self.ERT:.2e}, ddl: {self.ddl:.2e}, exp_comp_t: {self.exp_comp_t:.2e}\n"
        _str += f"\trelease_time: {self.release_time:.2e}, deadline: {self.deadline:.2e}, jitter_max: {self.jitter_max}\n"
        _str += f"\tcpu_time: {self.cpu_time:.2e}, io_time: {self.io_time:.2e}, totcpu: {self.totcpu:.2e}\n"
        # _str += f"state: {self.state}, "
        # _str += f"start_time: {self.start_time}, end_time: {self.end_time}, currentburst: {self.currentburst}, burst: {self.burst}, totburst: {self.totburst}, waitTime: {self.waitTime}\n"
        # _str += f"missed_deadline_count: {self.missed_deadline_count}\n"
        # _str += f"cumulative_executed_time: {self.cumulative_executed_time}, cum_trunAroundTime: {self.cum_trunAroundTime}, completion_count: {self.completion_count}\n"
        # _str += f"context_switch_count: {self.context_switch_count}, preemption_count: {self.preemption_count}, migration_count: {self.migration_count}\n"
        return _str


class TaskInt(TaskBase): 
    def __init__(
                    self, task_name:str, task_id:int, timing_flag:str,
                    ERT:Union[int, float], ddl:Union[int, float], period:Union[int, float], 
                    exp_comp_t:Union[int, float], i_offset:Union[int, float], jitter_max:Union[int, float]=0,
                    flops:Union[int, float]=0, task_flag:str="moveable",
                    pre_assigned_resource_flag:bool=False, 
                    op_io_time:int=0, op_cpu_time:int=0, seq_cpu_time:int=0, priority:int=0, 
                    criti_flag:str="soft", cbs_en:bool=False, 
                    trigger_mode:str="N",
                    parallel_cfg:dict={},
                    **kwargs
                ) -> None:
        super().__init__(
                            task_name=task_name, task_id=task_id, timing_flag=timing_flag,
                            ERT=ERT, ddl=ddl, period=period, exp_comp_t=exp_comp_t, i_offset=i_offset, jitter_max=jitter_max, 
                            op_cpu_time=op_cpu_time, op_io_time=op_io_time, seq_cpu_time=seq_cpu_time, priority=priority, 
                            criti_flag=criti_flag, cbs_en=cbs_en,
                            trigger_mode=trigger_mode, parallel_cfg=parallel_cfg,
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

        # resource mapping
        if pre_assigned_resource_flag:
            assert "main_size" in kwargs.keys(), "main_size is not provided"
            assert "RDA_size" in kwargs.keys(), "RDA_size is not provided"
            src_type:DDL_reservation = DDL_reservation if timing_flag == "deadline" else RT_reservation
            self.pre_assigned_resource = src_type(kwargs["main_size"], kwargs["RDA_size"])
        else:
            self.pre_assigned_resource:DDL_reservation = dummy_reservation(0, 0)
        
        # =============== 3. runtime attribute ===============
        # L1 resource allocation
        # task_id -> (main_num, RDA_num)
        self.allocated_resource:OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.required_resource_size:int = 0

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
        _str += f"\tflops: {self.flops:.2e}, main_size: {self.pre_assigned_resource.main_size}, RDA_size: {self.pre_assigned_resource.RDA_size}, pre_assigned: {self.pre_assigned_resource_flag}" 
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