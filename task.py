from typing import Dict, List, Tuple, Union, Any, OrderedDict
import numpy as np

# preemptable?/able to preempt others
task_attr = {
    "fixed": 0,
    "hard": 1,
    "stationary": 2,
    "moveable": 3,
    }
# classify the task in to realtime task and deadline task
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
}

class Task(object):
    def __init__(self, # cores:OrderedDict, 
                flops:Union[int, float],
                ERT:Union[int, float], 
                i_offset:Union[int, float], 
                exp_comp_t:Union[int, float], 
                period:Union[int, float], 
                ddl:Union[int, float],
                jitter_max:Union[int, float]=0,
                task_flag:str="moveable",
                timing_flag:str="deadline") -> None:
        
        self.id = None
        self.task_name = None

        # 1. task properties
        self.flops = 0
        self.task_type_flag = timing_flag
        self.task_type_num = task_timing_type[timing_flag]

        # 2. timing spec of the static scheduling table
        self.ERT = ERT
        self.i_offset = i_offset
        self.exp_comp_t = exp_comp_t
        self.period = period
        
        # for event driven 
        # deadline in each sub-period
        self.ddl = ddl

        # performance
        self.jitter_max = jitter_max

        # 3. designed scheduling attribute
        self.task_flag_num = task_attr[task_flag]
        self.task_flag = task_flag        
        self.group_no = 'Na'

        # 4. L0 resource partition
        self.pre_assigned_resource_flag = False
        self.main_resource_size = 0
        self.expected_redundancy_size = 0
        self.main_resource:List = []
        self.redundant_resource:List = []
        self.overlaped_resource:OrderedDict[int, List[Any]] = []
        
        # 5. runtime attribute
        self.state = "terminated"
        # L1 resource allocation
        self.required_resource_size:int = 0
        self.available_resource:List = []
        self.allocated_resource:OrderedDict[int, List[Any]] = {}

        # record at begining:
        self.start_time = 0
        self.release_time = self.ERT + self.i_offset 
        # deadline in each hyper-period (task that have multiple sub-periods in a hyper-period)
        # e.g. the task with 30hz but be divided into 3 tasks with 10hz and 1/30s offset
        self.deadline = self.ddl + self.i_offset 

        # record at endding:
        self.end_time = 0
        self.latency = 0

        # record at checkpoint/tiling point:
        self.executed = False
        self.executed_time = 0
        self.completion_count = 0

        # every time the task is scheduled
        self.cumulative_executed_time = 0 # used
        self.cumulative_response_time = 0 # used
        self.waiting_time = 0
        self.response_time = 0
        self.jitter = 0
        # self.fault = False
        # self.fault_time = 0

        # statistics
        self.missed_deadline = False
        self.missed_deadline_time = 0
        self.missed_deadline_count = 0
        self.turnaround_time = 0
        self.jitter = 0

        # for preemption
        self.context_switch_n = 0
        self.preemption_n = 0
        self.migration_n = 0

    def get_comm_latency(self, s_id, d_id):
        basic_latency = 0
        interference = 0
        return basic_latency + interference

    def get_remain_exec_t(self, time) -> Union[int, float]:
        # TODO: take preemption, tiling etc into account
        return self.exp_comp_t - self.cumulative_executed_time - (time - self.start_time)

    def count_allocated_resource(self):
        return sum(len(i) for i in self.allocated_resource.values())

    def set_task_name(self, task_name):
        self.task_name = task_name
    
    def set_task_id(self, task_id):
        self.id = task_id
            
    def add_resource(self, tgt_name, rsc_list):
        assert isinstance(self.__dict__[tgt_name], OrderedDict) 
        temp_l = []
        for rsc in rsc_list:
            if rsc in self.__dict__[tgt_name]:
                Warning("Resource {} already in {} of task {}".format(rsc, tgt_name, self.task_name))
            else: 
                temp_l.append(rsc)
        self.__dict__[tgt_name].fromkeys(temp_l)
    
    def remove_resource(self, tgt_name, rsc_list):
        assert isinstance(self.__dict__[tgt_name], OrderedDict) 
        temp_l = []
        for rsc in rsc_list:
            if rsc in self.__dict__[tgt_name]:
                Warning("Resource {} not in {} of task {}".format(rsc, tgt_name, self.task_name))
            else: 
                temp_l.pop(rsc)
    
    def get_release_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.ERT + self.i_offset + (i+period_n) * self.period + i_offset

    def get_deadline_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.ddl + self.i_offset + (i+period_n) * self.period + i_offset
    
    def get_finish_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.exp_comp_t +self.ERT+ self.i_offset + (i+period_n) * self.period + i_offset
    
    def allocate(self, task_id, num_cores:int, allocator):
        if not self.pre_assigned_resource_flag:
            assert "Directly allocating resource is not allowed for this task!"
        core_list = allocator.allocate(self, task_id, num_cores)
        self.allocated_resource[task_id] += core_list
    
    def get_guest_task(self):
        return [id for id in self.allocate_resource.keys() if id!=self.id]
    

# define a task group, that contains a list of task with same group_no
class Task_group(object):
    def __init__(self, group_no:int, task_list:List) -> None:
        self.group_no = group_no
        self.task_list = task_list
