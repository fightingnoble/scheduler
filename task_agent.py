from __future__ import annotations
from typing import Dict, List, Tuple, Union, Any, OrderedDict
import numpy as np
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
# lifetime of the task
task_lifetime = {
    "running": 0,
    "terminated": 3,
    "suspend": 2,
    "runnable": 1,
    "throttled": 4,
}


class TaskBaseInt(object):
    def __init__(self, task_name:str, task_id:int, timing_flag:str,
                 ERT:int, ddl:int, period:int, exp_comp_t:int, i_offset:int, jitter_max:int,
                 ):
        self.id = task_id
        self.task_name = task_name

        # =============== 1. task properties ===============
        # timing spec 
        self.i_offset = i_offset

        # in each sub-period
        self.ERT = ERT # w.r.t period 0s
        self.ddl = ddl # w.r.t period ERT
        self.timing_flag = timing_flag
        self.timing_flag_num = task_timing_type[timing_flag]

        self.exp_comp_t = exp_comp_t
        self.period = period
        # performance
        self.jitter_max = jitter_max

        # =============== 2. runtime state ===============
        # task state
        self.state = "terminated"
        # recored when the task is released
        # deadline in each hyper-period (task that have multiple sub-periods in a hyper-period)
        # e.g. the task with 30hz but be divided into 3 tasks with 10hz and 1/30s offset
        self.release_time = self.ERT + self.i_offset 
        self.deadline = self.release_time + self.ddl + self.i_offset 
        # record at begining:
        self.start_time = 0
        # record at endding:
        self.end_time = 0
        self.latency = 0

        # =============== 3. statistical properties ===============
        self.missed_deadline_count = 0 # used 

        # every time the task is scheduled
        self.cumulative_executed_time = 0 # used
        self.cumulative_response_time = 0 # used
        self.completion_count = 0 # used

        # for interrupt
        self.context_switch_count = 0
        self.preemption_count = 0
        self.migration_count = 0

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

# task_name="task1", task_id=1, task_flag="periodic", timing_flag="deadline", 
#                 ERT=10, ddl=20, period=30, exp_comp_t=10, i_offset=0, jitter_max=0, 
#                 pre_assigned_resource_flag=False, main_size=100, RDA_size=20


class TaskInt(TaskBaseInt): 
    def __init__(
                    self, task_name:str, task_id:int, timing_flag:str,
                    ERT:Union[int, float], ddl:Union[int, float], period:Union[int, float], 
                    exp_comp_t:Union[int, float], i_offset:Union[int, float], jitter_max:Union[int, float]=0,
                    flops:Union[int, float]=0, task_flag:str="moveable",
                    pre_assigned_resource_flag:bool=False, 
                    **kwargs
                ) -> None:
        super().__init__(
                            task_name=task_name, task_id=task_id, timing_flag=timing_flag,
                            ERT=ERT, ddl=ddl, period=period, exp_comp_t=exp_comp_t, i_offset=i_offset, jitter_max=jitter_max,
                        )
        
        # =============== 1. task properties ===============
        self.flops = flops
        # designed scheduling attribute
        self.affinity = []
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
    def set_state(self, state):
        assert state in task_lifetime.keys()
        self.state = state

    def get_release_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.ERT + self.i_offset + (i+period_n) * self.period + i_offset

    def get_deadline_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.ddl + self.i_offset + (i+period_n) * self.period + i_offset
    
    def get_finish_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.exp_comp_t +self.ERT+ self.i_offset + (i+period_n) * self.period + i_offset

    def get_available(self):
        assert self.pre_assigned_resource_flag
        return self.pre_assigned_resource.get_available_rsc()

    def release(self, task_id:int=0, main_num:int=0, RDA_num:int=0, verbose:bool=False):
        if not task_id:
            task_id = self.id
        if self.pre_assigned_resource_flag: 
            self.pre_assigned_resource.release(task_id, main_num, RDA_num, verbose)
            return {"id":self.id, "resource":self.allocated_resource, "released": True}
        else:
            return {"id":self.id, "resource":self.allocated_resource, "released": False}
    
    def allocate(self, task_id:int=0, main_num:int=0, RDA_num:int=0, verbose:bool=False):
        if self.pre_assigned_resource_flag: 
            self.pre_assigned_resource.allocate(task_id, main_num, RDA_num, verbose)
            return {"id":self.id, "resource":self.allocated_resource, "allocated": True}
        else:
            return {"id":self.id, "resource":self.allocated_resource, "allocated": False}
    
    def can_execute(self, task:TaskInt, verbose:bool=False) -> bool:
        """
        A task is executable if it has enough resource to execute the task before the deadline
        if the task is pre-assigned, it will always return true
        if the task is not pre-assigned, it will play a insert-based scheduling 
        """
        # check if the task can be scheduled
        if task.pre_assigned_resource_flag:
            return True
        else:
            # check if the task can be scheduled
            if self.pre_assigned_resource.can_execute(task, verbose):
                # allocate the resource
                self.allocated_resource[task.id] = (task.main_size, task.RDA_size)
                self.required_resource_size += task.main_size + task.RDA_size
                return True
            else:
                return False

    def query_rsc(self, task_id:int=0, verbose:bool=False): 
        # query the rsc allocated by task (id) 
        return self.pre_assigned_resource.rsc_map[task_id]

    def is_stationary(self) -> bool:
        return self.task_flag == "stationary"

    def get_rsc(self) -> int:
        return self.required_resource_size


if __name__ == "__main__": 
    task1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                ERT=10, ddl=20, period=30, exp_comp_t=10, i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    verbose = True
    task1.allocate(task_id=1, main_num=15, RDA_num=4, verbose=verbose)
    print(task1.get_available())
    task1.release(task_id=1, main_num=0, RDA_num=4, verbose=verbose)
    