from typing import Dict, List, Tuple, Union, Any, OrderedDict
from collections import OrderedDict
import numpy as np
from task_agent import TaskInt


class AllocatorInt(object):
    def __init__(self, ):
        pass

    def release(self, task_list: List[TaskInt], host_list: List[TaskInt]):
        """
        release the resource allocated to the task
        invoke the task.release() method for each task
        read the returned massage whether the task is released successfully locally
        get the host task id and the num of the released cores
        if not, invoke the task.release() method for the task on the remote node
        """
        for task in task_list:
            msg:dict = task.release()
            status = msg["released"]
            if not status:
                task_id = msg["id"]
                rsc = msg["resource"] # (main_num, RDA_num)
                # suppose the task only get resource from one node
                assert len(rsc) == 1, \
                    "Not implemented: suppose one task only get resource from one node"
                host_id = list(rsc.keys())[0]
                num = rsc[host_id] 
                host = host_list[host_id] 
                host_record_num = host.query_rsc(task_id)
                assert host_record_num == num, \
                    f"(host:{host_id}):{host_record_num} != (guset{task_id}){num}, \nNot implemented: suppose one task only get resource from one node"
                host.release(task_id, num)

    def allocate(self, task_list: List[TaskInt], host_idx: Dict[int, TaskInt], task_idx: Dict[int, TaskInt], verbose:bool=False):
        """
        allocate resource to the task in three steps:
        1. check the affinity of the task
            1.1 if task is starionary, get free reource from the host
            1.2 if task is movable, get free resource from all the hosts
        2. select and allocate the rsc to the task
        """
        for task in task_list: 
            # 1. check the affinity of the task
            if task.is_stationary():
                # 1.1 if task is starionary, get free reource from the host
                # check the legality of the affinity
                assert len(task.affinity) == 1
                host = host_idx[task.affinity[0]]
                num = task.get_rsc()
                host.allocate(task.id, num, verbose=verbose)
            else:
                # 1.2 if task is movable, get free resource from all the hosts
                # check the legality of the affinity
                assert len(task.affinity) <= len(host_idx)
                num = task.get_rsc()
                # TODO: free resource selection based on the fdl_mask
                # TODO: push operation based on the max heap
                for cluster_id in task.affinity:
                    host = host_idx[cluster_id]
                    if self.can_execute(host, task, verbose=verbose): 
                        host.allocate(task.id, num, verbose=verbose)
                        break
    
    def can_execute(self, host:TaskInt, guest:TaskInt, verbose:bool=False) -> bool:
        """
        A task is executable if it has enough resource to execute the task before the deadline
        if the task is pre-assigned, it will always return true
        if the task is not pre-assigned, it will play a insert-based scheduling 
        if the item with size (task.rsc, task.deadline) can be inserted into the host's queue,
        """ 
        if guest.is_pre_assigned():
            return True
        else:


    
