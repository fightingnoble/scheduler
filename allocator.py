from functools import reduce
from typing import Dict, List, Tuple
from task import Task, RscNode
from .hw_rsc import Spatial
from networkx import DiGraph

class Allocator(object):
    def __init__(self) -> None:
        pass

    def estimate_PE_number(self, task, latency, ):
        pass 

    def locate_resources(self, task, num_cores):


        if task.task_flag == "hard":
            # hard task
            # check the avalable resource allocated by the task group
            # if avalable resources is enough, allocate it
            # else decide to wait or interrupt other tasks
            pass
        elif task.task_flag == "stationary":
            # stationary task
            # scan from level 0 to level 3
            for level in [0, 1, 2, 3]:
                self.get_available_from_pe_set(task,)

            pass
        elif task.task_flag == "fixed":
            # fixed task
            # allocate the resource directly
            pass
        elif task.task_flag == "moveable":
            # moveable task
            # check the avalable resource allocated by the task group
            # if avalable resources is enough, allocate it
            # else decide to wait, release other tasks, preetempt other tasks or migrate
            pass
        else:
            raise "Unedpected task_flag:{}".format(task.task_flag)

        idx = task.group_no
        for competitor in task_grouping[idx]:
            pass
    
                # available_redandency += guest_cores
                # if len(available_redandency) >= num_new_cores:

        # available_tasks = []
        # wait_tasks = []
        # preemptable_tasks = []
        # rsc_list = []
        # # Task with pre-assigened rcs: check the available resource locally
        # if task.pre_assigned_resource_flag:

        # rsc_list = [task.id]
        # # check the pre-assigned resource
        # available_redandency = task.available_resource
        # if len(available_redandency) >= num_new_cores:
        #     return rsc_list


        # # source from rsc target：rsc_affinity_graph.neighbors(id)
        # for rsc_task_id in rsc_affinity_graph.neighbors(task.id):
        #     # source from L0
        #     task_index_table_by_id[rsc_task_id]

        
        
                        
                    


            

        # for competitor in competitor_list:
        #     if competitor.task_flag == "hard" or competitor.task_flag == "fiexd":
        #         pass
        #     elif competitor.task_flag == "stationary":
        #         if task in competitor.guest_task:
        #             if len(task.allocated_resource + competitor.available_resource) >= task.required_resource_size:
        #                 # available
        #                 available_tasks.append(competitor)
        #             else: 
        #                 comm_latency = task.get_comm_latency(competitor)
        #                 if comm_latency < competitor.remain_exec_t:
        #                     # preemptable
        #                     available_tasks.append(competitor)
        #                 else:
        #                     pass

    def get_available_from_pe_set(
            self, task:Task, task_index_table_by_id: Dict[int, Tuple[str, Task, RscNode]], 
            scheduling_table, rsc_affinity_graph:DiGraph, num_new_cores:int,
            level:int=-1
            ) -> List[Tuple[str, Task]]:
        """
        level: -1: all levels
        level: 0: check the main @main redundency and remote
        level: 1: check the redundency @main redundency and remote
        level: 2: check the remote @main redundency and remote
        """
        assert task.pre_assigned_resource_flag == True

        available_tasks = []
        wait_tasks = []
        preemptable_tasks = []
        rsc_list = []

        if level in [0, 1, -1]:

            # level 0
            l0_free += list(set(task.available_resource).intersection(task.main_resource.keys()))
            l1_free += list(set(task.available_resource).intersection(task.redundant_resource.keys(), task.overlaped_resource.keys()))

            l0_waitable = {}
            l0_preemptable = {}
            l1_waitable = {}
            l1_preemptable = {}

            # check the allocated resource
            guest_task_id_list = task.get_guest_task()

            conflit_free = []
            conflit_wait = []
            conflit_preemptable = []
            # check the guest tasks
            for guest_id in guest_task_id_list: 
                guest = task_index_table_by_id[guest_id][1]
                # compare the transfer time to the candidate task and the remaining time of the candidate task
                # TODO: add the input of s_point and e_point for latency estimation
                transfer_t = task.get_comm_latency()
                remained_t = guest.get_remain_exec_t()
                if transfer_t < remained_t:
                    # means the task is conflict free
                    guest_cores = task.allocated_resource(task.id)
                    conflit_free.append(guest_id)
                else:
                    if guest.task_flag == "moveable":
                        conflit_preemptable.append(guest_id)
                    else:
                        conflit_wait.append(guest_id)

            # get allocated resource from the available task group
            available_rsc_list = []
            for guest_id in conflit_free:
                guest = task_index_table_by_id[guest_id][1]
                guest_cores = guest.allocated_resource(task.id)
                # available_rsc_list += guest_cores
                l0_free += list(set(guest_cores).intersection(task.main_resource.keys()))
                l1_free += list(set(guest_cores).intersection(task.redundant_resource.keys(), task.overlaped_resource.keys()))

            # get allocated resource from the preemptable task group
            preemptable_rsc_list = []
            for guest_id in conflit_preemptable:
                guest = task_index_table_by_id[guest_id][1]
                guest_cores = guest.allocated_resource(task.id)
                # preemptable_rsc_list += guest_cores
                l0_preemptable[guest_id] += list(set(guest_cores).intersection(task.main_resource.keys()))
                l1_preemptable[guest_id] += list(set(guest_cores).intersection(task.redundant_resource.keys(), task.overlaped_resource.keys()))

            # get allocated resource from the wait task group
            wait_rsc_list = []
            for guest_id in conflit_wait:
                guest = task_index_table_by_id[guest_id][1]
                guest_cores = guest.allocated_resource(task.id)
                wait_rsc_list += guest_cores
                l0_waitable[guest_id] += list(set(guest_cores).intersection(task.main_resource.keys()))
                l1_waitable[guest_id] += list(set(guest_cores).intersection(task.redundant_resource.keys(), task.overlaped_resource.keys()))


            # for overlaped_rsc in task.overlaped_resource.keys():
            #     for competitor in rsc_affinity_graph.predecessors(overlaped_rsc):
            #         if guest.task_flag == "moveable":
            #             guest = task_index_table_by_id[guest_id][1]
            #             guest_cores = guest.allocated_resource(task.id)
            #         else:
            #             conflit_wait.append(guest_id)
        if level in [2, -1]:
            # level 3
            l2_waitable = {}
            l2_free = {}
            l2_preemptable = {}
        
        if level == 0:
            return l0_free, l0_waitable, l0_preemptable
        elif level == 1:
            return l1_free, l1_waitable, l1_preemptable
        elif level == 2:
            return l2_free, l2_waitable, l2_preemptable
        


        
    

    def allocate(self, task_obj, task_id, num_cores):
        pass
        

    def preemption(self, task_owner, task2_competitor):
        # check preemptablity
        if task_owner.task_flag_no > task2_competitor.task_flag_no:
            # preemption
            pass

    def release(self, task_owner, task2_competitor):
        pass

    

# create a set associative task management
class set_associative_memory(Allocator):
    def __init__(self,) -> None:
        pass

    def allocate(self, task_list:List[Task]):
        pass

    def deallocate(self, task_list:List[Task]):
        pass

    def get_available_space(self):
        pass

    def get_available_space(self):
        pass
