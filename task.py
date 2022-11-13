from typing import Dict, List, Tuple, Union, Any, OrderedDict
import numpy as np
from hw_rsc import FLOPS_PER_CORE
from scheduler_global_cfg import *

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


class RscNode(object):
    def __init__(self):
        self.id = None
        self.task_name = None
        # recording the rsc state which rsc is allocated by which task (id) 
        self.rsc_map:OrderedDict[Any, int] = OrderedDict()
    
    def allocate(self, task_id:int, rsc_list:List[Any]=None):
        for rsc in rsc_list:
            self.rsc_map[rsc] = task_id
    
    def release(self, task_id:int=0, rsc_list:List[Any]=None):
        assert (task_id != 0 and rsc_list == None) or (task_id == 0 and rsc_list)
        if task_id:
            for rsc in self.rsc_map.keys():
                if self.rsc_map[rsc] == task_id:
                    self.rsc_map[rsc] = None
        else:
            for rsc in rsc_list:
                self.rsc_map[rsc] = None

    def get_available_rsc(self):
        available_rsc = []
        for rsc in self.rsc_map.keys():
            if self.rsc_map[rsc] == None:
                available_rsc.append(rsc)
        return available_rsc


class Task(RscNode):
    def __init__(self, # cores:OrderedDict, 
                flops:Union[int, float],
                ERT:Union[int, float], 
                i_offset:Union[int, float], 
                exp_comp_t:Union[int, float], 
                period:Union[int, float], 
                ddl:Union[int, float],
                jitter_max:Union[int, float]=0,
                task_flag:str="moveable",
                timing_flag:str="deadline",
                ) -> None:

        super().__init__()
        
        # =============== 1. task properties ===============
        self.flops = 0

        # timing spec 
        self.i_offset = i_offset

        # in each sub-period
        self.ERT = ERT
        self.ddl = ddl
        self.task_type_flag = timing_flag
        self.task_type_num = task_timing_type[timing_flag]

        self.exp_comp_t = exp_comp_t
        self.period = period
        # performance
        self.jitter_max = jitter_max

        # designed scheduling attribute
        self.task_flag_num = task_attr[task_flag]
        self.task_flag = task_flag        

        # =============== 2. Node attribution ===============
        self.pre_assigned_resource_flag = False

        # resource map
        # static rsc_map
        self.main_resource_size = 0
        # who use
        self.main_resource:OrderedDict[Any, int] = OrderedDict()

        # dynamic rsc_map
        self.expected_redundancy_size = 0
        # who use
        self.redundant_resource:OrderedDict[Any, int] = OrderedDict()
        # from where
        self.overlaped_resource:OrderedDict[int, List[Any]] = OrderedDict()
        
        # =============== 3. runtime attribute ===============
        # L1 resource allocation
        self.allocated_resource:OrderedDict[int, List[Any]] = OrderedDict()
        self.required_resource_size:int = 0
        self.release_time = self.ERT + self.i_offset 
        # deadline in each hyper-period (task that have multiple sub-periods in a hyper-period)
        # e.g. the task with 30hz but be divided into 3 tasks with 10hz and 1/30s offset
        self.deadline = self.ddl + self.i_offset 
        # record at begining:
        self.start_time = 0
        # record at endding:
        self.end_time = 0
        self.latency = 0

        # every time the task is scheduled
        self.state = "terminated"
        self.cumulative_executed_time = 0 # used
        self.cumulative_response_time = 0 # used
        self.completion_count = 0
        self.cumulative_waiting_time = 0
        self.response_time = 0
        # self.fault = False
        # self.fault_time = 0

        # statistics
        self.missed_deadline = False
        self.missed_deadline_time = 0
        self.missed_deadline_count = 0
        self.turnaround_time = 0
        self.jitter = 0

        # for preemption
        self.context_switch_count = 0
        self.preemption_count = 0
        self.migration_count = 0

    # ======== vanilla properties setter and getter ========
    def set_task_name(self, task_name):
        self.task_name = task_name
    
    def set_task_id(self, task_id):
        self.id = task_id
    
    # ======== non-trival properties setter and getter ========
    def set_state(self, state):
        assert state in task_lifetime.keys()
        self.state = state

    def add_resource(self, tgt_name, key_list, value_list=None):
        assert isinstance(self.__dict__[tgt_name], OrderedDict) 
        temp_l = []
        for rsc in key_list:
            if rsc in self.__dict__[tgt_name]:
                Warning("Resource {} already in {} of task {}".format(rsc, tgt_name, self.task_name))
            else: 
                temp_l.append(rsc)
        if value_list is None:
            self.__dict__[tgt_name].fromkeys(temp_l)
        else:
            self.__dict__[tgt_name].update(zip(temp_l, value_list))
    
    def remove_resource(self, tgt_name, rsc_list):
        assert isinstance(self.__dict__[tgt_name], OrderedDict) 
        temp_l = []
        for rsc in rsc_list:
            if rsc in self.__dict__[tgt_name]:
                Warning("Resource {} not in {} of task {}".format(rsc, tgt_name, self.task_name))
            else: 
                temp_l.pop(rsc)
    
    def get_remain_exec_t(self, time) -> Union[int, float]:
        # TODO: take preemption, tiling etc into account
        return self.exp_comp_t - self.cumulative_executed_time - (time - self.start_time)

    def count_allocated_resource(self):
        return sum(len(i) for i in self.allocated_resource.values())

    def get_release_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.ERT + self.i_offset + (i+period_n) * self.period + i_offset

    def get_deadline_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.ddl + self.i_offset + (i+period_n) * self.period + i_offset
    
    def get_finish_event(self, time_interval, period_n=0, i_offset=0):
        for i in range(int(time_interval//self.period)):
            yield self.exp_comp_t +self.ERT+ self.i_offset + (i+period_n) * self.period + i_offset

    def get_guest_task(self):
        return list(set(self.main_resource.values()).union(set(self.redundant_resource.values())))
    
    def get_guset_in_overlap(self, overlap_node_id):
        dict_t = OrderedDict()
        for rsc in self.overlaped_resource[overlap_node_id]: 
            id = self.redundant_resource[rsc]
            if dict_t[id]:
                dict_t[id].append(rsc)
            else:
                dict_t[id] = [rsc]
        return dict_t

    def get_competetor_task(self):
        guset_dict_t = {}
        host_dict_t = {}
        for overlap_node_id in self.overlaped_resource.keys():
            node:RscNode = task_index_table_by_id[overlap_node_id][1]
            assert isinstance(node, RscNode)
            host_dict_t[overlap_node_id]=node.get_available_rsc()
            for vs_host_id in set(node.rsc_map.values()): 
                if vs_host_id is None:
                    continue
                vs_host = task_index_table_by_id[vs_host_id][1]
                guset_dict_t.update(vs_host.get_guset_in_overlap(overlap_node_id))
        return host_dict_t, guset_dict_t


    def release(self, allocator, mode="all", rsc_list:List[Any]=[]):
        if mode == "all":
            # release all resources
            # update available resource of host
            allocator.release(self)
            self.allocated_resource.clear()
        else:
            # release specific resources
            assert rsc_list != []
            # update available resource of host
            allocator.release(self, rsc_list)
            for rsc in rsc_list:
                self.allocated_resource[self.id].remove(rsc)
            # update expected completion time
            # TODO: replace this strategic retreat by a performance model of a profiling table
            self.exp_comp_t = self.flops / self.allocated_resource[self.id].count() / FLOPS_PER_CORE
        
    def allocate(self, allocator, rsc_list:List[Any]=[]):
        # if mode == "all":
        # allocate all resources
        # update available resource of host
        allocator.allocate(self, rsc_list)
        # update expected completion time
        self.exp_comp_t = self.flops / self.allocated_resource[self.id].count() / FLOPS_PER_CORE
    
    def get_available_rsc(self):
        available_rsc = []
        if not self.pre_assigned_resource_flag:
            for rsc in self.main_resource.keys():
                if self.rsc_map[rsc] == None:
                    available_rsc.append(rsc)
            for rsc in self.redundant_resource.keys():
                if self.rsc_map[rsc] == None:
                    available_rsc.append(rsc)
        return available_rsc
    
    def check_guest_task(self, comm_latency_slack=0):
        # check the allocated resource
        guest_task_id_list = self.get_guest_task()

        conflit_free = {}
        conflit_wait = {}
        conflit_preemptable = {}
        # check the guest tasks
        for guest_id in guest_task_id_list: 
            guest = task_index_table_by_id[guest_id][1]
            remained_t = guest.get_remain_exec_t()
            if comm_latency_slack < remained_t:
                # means the task is conflict free
                rsc_l = guest.allocated_resource[guest.id]
                conflit_free[guest_id] = rsc_l
            else:
                if guest.task_flag == "moveable":
                    # means the task is conflict wait
                    rsc_l = guest.allocated_resource[guest.id]
                    conflit_wait[guest_id] = rsc_l
                else:
                    # means the task is conflict preemptable
                    rsc_l = guest.allocated_resource[guest.id]
                    conflit_preemptable[guest_id] = rsc_l
        return conflit_free, conflit_wait, conflit_preemptable

    def check_competetor(self, comm_latency_slack):
        # check the allocated resource
        vs_host_dict, vs_guest_dict = self.get_competetor_task(comm_latency_slack)

        conflit_free = vs_host_dict
        conflit_wait = {}
        conflit_preemptable = {}
        # check the guest tasks
        for guest_id in vs_guest_dict: 
            guest = task_index_table_by_id[guest_id][1]
            remained_t = guest.get_remain_exec_t()
            if comm_latency_slack < remained_t:
                # means the task is conflict free
                rsc_l = vs_guest_dict[guest_id]
                conflit_free[guest_id] = rsc_l
            else:
                if guest.task_flag == "moveable":
                    # means the task is conflict wait
                    rsc_l = vs_guest_dict[guest_id]
                    conflit_wait[guest_id] = rsc_l
                else:
                    # means the task is conflict preemptable
                    rsc_l = guest.allocated_resource[guest.id]
                    conflit_preemptable[guest_id] = rsc_l
        return conflit_free, conflit_wait, conflit_preemptable


        

# define a task group, that contains a list of task with same group_no
class Task_group(object):
    def __init__(self, group_no:int, task_list:List, mode="id") -> None:
        self.group_no = group_no
        self.task_list = task_list
        assert mode in ['id', 'name']
        self.mode = mode
        self.task_dict = OrderedDict()
        if mode == "id":
            for task in task_list:
                self.task_dict[task.id] = task
        else:
            for task in task_list:
                self.task_dict[task.task_name] = task

