from typing import List, Dict, Tuple, Union, Optional, Iterable, Iterator, Collection, Set
from collections import OrderedDict, defaultdict

from functools import reduce
import math
import bisect
import os
import numpy as np
from model.resource_agent import Resource_model_int
from task.task_agent import ProcessInt, TaskInt
from global_var import fork_pid_base
from global_var import *
from utils import load_pickle

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
# import networkx as nx
# import matplotlib as mpl
from model.lru import LRUCache

class SchedulingTableInt(object): 
    """
    =============== 1. scheduling table ===============
    scheduling table is a 2D array, each row is a resource, each column is a time slot
    the value of each cell is the task id that occupies the resource in the time slot
    the size of the scheduling table is determined by the number of resources and the number of time slots
    the number of time slots is determined by the hyper-period of the tasks
    the number of resources is determined by the number of resources that are available
    """
    def __init__(self, num_resources: int, num_time_slots:int, id: int = None, name: str = None, hp:int=None):
        # self.scheduling_table = np.zeros((num_resources, num_time_slots), dtype=int)
        # self.scheduling_table = np.full((num_time_slots), Resource_model_int(num_resources, id, name), dtype=Resource_model_int)
        self.scheduling_table = np.array([Resource_model_int(num_resources, ) for _ in range(num_time_slots)], dtype=Resource_model_int)
        self.flops_list = list()
        self.event_list = list()
        self.core_list = list()
        self.id = id
        self.name = name
        self.num_resources = num_resources
        self.temp_size = num_time_slots
        self.locker = None
        self.lock_mask = np.ones((num_time_slots), dtype=bool)
        self.sparse_list = list() # OrderedDict()
        # current index of the sparse list
        self.sparse_idx = 0
        # previous index of the sparse list
        self.sparse_idx_prev = 0
        # next index of the sparse list
        self.sparse_idx_next = 0
        self.sparse_mode = False
        self.alloc_mod = "N/A" 
        
        # need by overtime
        self.sparse_flops = list()
        # need by compress
        self.sparse_cores = list()
        # need by compact
        # list of event blocks, each block is a list of events grouped by pid, which has the form like (head_event_t, {pid: {event_msg1, ..., event_msgn}})
        self.sparse_event:List[List[Tuple[int, Dict[int, Set[str]]]]] = list()

        # available when used as a recorder
        self.wr_pointer = 0
        self.hypper_period = hp
    
    def set_alloc_mod(self, mod:str):
        """_summary_

        Args:
            mod (str): 
            compress (coleasing_alloc_1bin): the actual core usage is expected to be less than the allocated cores
            compact (push_task_into_bins_new): allocate process with less budget
        """
        assert mod in ["N/A", "overtime", "compress", "compact"]
        self.alloc_mod = mod

    def is_empty(self):
        return np.all([rsc.is_empty() for rsc in self.scheduling_table]) or len(self.scheduling_table) == 0

    def append(self, rsc_agent: Resource_model_int):
        # Too slow
        # self.scheduling_table = np.append(self.scheduling_table, rsc_agent)
        
        if self.wr_pointer >= self.temp_size:
            assert self.hypper_period is not None
            self.temp_size += self.hypper_period
            self.scheduling_table = np.append(self.scheduling_table, np.array([Resource_model_int(self.num_resources, ) for _ in range(self.hypper_period)], dtype=Resource_model_int))
        self.scheduling_table[self.wr_pointer].update(rsc_agent.rsc_map)
        self.wr_pointer += 1

    def add_rsc_num(self, num:int):
        self.num_resources += num
        for rsc in self.scheduling_table:
            rsc.add_rsc_num(num)

    def index_occupy_by_id(self, time_slot_s:int=None, time_slot_e:int=None) -> Dict[int, List[int]]:
        """
        get the task id that occupies the resource
        """
        # rsc_array = self.scheduling_table[:, time_slot_s:time_slot_e]
        # rsc_occp = []
        # for id in np.unique(rsc_array[rsc_array != 0]):
        #     rsc_occp.append((id, np.where(rsc_array == id)[0]))
        # return np.where(rsc_array == 0)[0], rsc_occp
        
        # get task_id set
        task_id_set = set()
        assert not (time_slot_s is None and time_slot_e is not None)
        if time_slot_s is None:
            time_slot_s = 0

        if time_slot_e is None:
            rsc_agents_arr = self.scheduling_table[time_slot_s:]
        else:
            rsc_agents_arr = self.scheduling_table[time_slot_s:time_slot_e]
        
        # get task_id set
        for rsc_agent in rsc_agents_arr:
            # print(rsc_agent.rsc_map.keys())
            task_id_set.update(rsc_agent.rsc_map.keys())
        # return list(task_id_set)
        
        Scheduling_table_index_by_task_id = OrderedDict()
        for task_id in task_id_set: 
            Scheduling_table_index_by_task_id[task_id] = []
        
        for rsc_agent in rsc_agents_arr:
            for task_id in task_id_set:
                Scheduling_table_index_by_task_id[task_id].append(rsc_agent.rsc_map.get(task_id, 0))

        # divide the time slot into intervals
        # get the size and the start and length of each interval
        for task_id, rsc_occ in Scheduling_table_index_by_task_id.items():
            rsc_occ = np.array(rsc_occ)
            boader = (rsc_occ[0:-1] != rsc_occ[1:]).nonzero()[0] + 1
            s = [0] + boader.tolist() 
            e = boader.tolist() + [len(rsc_occ)] 
            l = [e[i] - s[i] for i in range(len(s))]
            size = [rsc_occ[s[i]] for i in range(len(s))]
            # get non zero intervals
            s = [s[i]+time_slot_s for i in range(len(s)) if size[i] != 0]
            l = [l[i] for i in range(len(l)) if size[i] != 0]
            size = [size[i] for i in range(len(size)) if size[i] != 0]            
            Scheduling_table_index_by_task_id[task_id] = (s, size, l)


        return Scheduling_table_index_by_task_id

    def idx_free_by_slot(self, time_slot_s, time_slot_e, key=None):
        """
        get the available resources in the time slot
        """
        # culculate available resources at each time slot
        rsc_maps_arr = self.scheduling_table[time_slot_s:time_slot_e]
        rsc_avl = []
        # check lock: 
        #   if the lock is free in the target interval or the query is from the locker, return all available resources 
        if np.all(self.lock_mask[time_slot_s:time_slot_e]) or key == self.locker:
            for idx, rsc_map in enumerate(rsc_maps_arr):
                rsc_avl.append(rsc_map.get_available_rsc())
        else: 
            for rsc_map, lock in zip(rsc_maps_arr, self.lock_mask[time_slot_s:time_slot_e]):
                if lock:
                    rsc_avl.append(rsc_map.get_available_rsc())
                else:
                    rsc_avl.append(0)
        return rsc_avl
    
    def insert_task(self, task:ProcessInt, req_rsc_size:int, time_slot_s:int, time_slot_e:int, expected_slot_num:int, 
                    verbose=False, DEBUG=False)->Tuple[bool, Union[int,List[int]], Union[int,List[int]], Union[int,List[int]]]: 
        """
        play insert-based scheduling: 
        1. search available tensor cores at each slot
        2. insert the task into the scheduling table at a proper interval (here we adapt First-Fit)
            return success or not, the start time slot, the allocated resources, the allocated time slots

        mechanism:
        1. Try to match the number of resources originally requested by the job,
            which can avoid complex calculation of parallelism rules; 
        2. else, if there is enough rsc in [0, expected_slot_num] + time_slot_s
            try to distribute the resources to the intervals as evenly as possible; 
        3. else, try to allocate the resources to the intervals as soon as possible;
        """ 
        rsc_avl = self.idx_free_by_slot(time_slot_s, time_slot_e, key=task.pid)
        rsc_avl = np.array(rsc_avl)
        
        # check if the task can be scheduled with the expected resources
        # The task can be scheduled at any time slot
        # bug
        if np.all(rsc_avl >= req_rsc_size) and (time_slot_e - time_slot_s) >= expected_slot_num:
            # allocate resources 
            for rsc_map in self.scheduling_table[time_slot_s:time_slot_s+expected_slot_num]:
                rsc_map.allocate(task.pid, req_rsc_size, verbose)
            return True, time_slot_s, req_rsc_size, expected_slot_num
        
        else: 
            # divide the rsc_avl into intervals
            boader = (rsc_avl[0:-1] != rsc_avl[1:]).nonzero()[0] + 1
            s = [0] + boader.tolist() 
            e = boader.tolist() + [len(rsc_avl)] 
            slot_n = [rsc_avl[s[i]] for i in range(len(s))]

            # current allocation (C)
            curr_alloc = np.zeros(len(s), dtype=int)
            curr_slot = np.zeros(len(s), dtype=int)

            # required (R)
            expected_req_rsc_size = req_rsc_size * expected_slot_num

            if (rsc_avl > req_rsc_size).sum() * req_rsc_size > expected_req_rsc_size:

                # check if the task can be executed on the single interval
                for i in range(len(s)):
                    if (e[i] - s[i]) >= expected_slot_num and slot_n[i] > req_rsc_size:
                        # allocate resources 
                        alloc_slot_s = time_slot_s+s[i]
                        for rsc_map in self.scheduling_table[alloc_slot_s:alloc_slot_s+expected_slot_num]:
                            rsc_map.allocate(task.pid, req_rsc_size, verbose)
                        return True, [alloc_slot_s,], [req_rsc_size,], [expected_slot_num,]

                cum_rsc_alloc = 0
                # divide the rsc_avl into intervals
                for i in range(len(s)): 
                    # rsc size
                    if slot_n[i] > req_rsc_size:
                        curr_alloc[i] = req_rsc_size
                        # slot length
                        if cum_rsc_alloc + req_rsc_size * (e[i] - s[i]) >= expected_req_rsc_size:
                            curr_slot[i] = math.ceil((expected_req_rsc_size - cum_rsc_alloc)/req_rsc_size)
                            break
                        else:
                            curr_slot[i] = int(e[i] - s[i])
                            cum_rsc_alloc += req_rsc_size * curr_slot[i]
            # warning: expected_slot_num may be larger than the available slots
            elif rsc_avl[:expected_slot_num].sum() < expected_req_rsc_size: 
                # as soon as possible
                self.asap_insert(task, DEBUG, s, e, slot_n, curr_alloc, curr_slot, expected_req_rsc_size)
            else: 
                if DEBUG:
                    print("Enough resources: as evenly as possible")

                self.aeap_insert(task, expected_slot_num, DEBUG, s, e, slot_n, curr_alloc, curr_slot, expected_req_rsc_size)

            # allocate resources
            idx = curr_alloc.nonzero()[0]
            for i in range(len(idx)):
                alloc_slot_s = time_slot_s + s[idx[i]]
                for rsc_map in self.scheduling_table[alloc_slot_s:alloc_slot_s+int(curr_slot[idx[i]])]:
                    rsc_map.allocate(task.pid, curr_alloc[idx[i]], verbose)

            self.sparse_mode = False
            return True, (time_slot_s+np.array(s)[idx]).tolist(), curr_alloc[idx].tolist(), curr_slot[idx].tolist()

    def aeap_insert(self, task, expected_slot_num, DEBUG, s, e, size, curr_alloc, curr_slot, expected_req_rsc_size):
        # available (A)
        rsc_avl_tmp = np.zeros(len(s), dtype=int) 
        rsc_lack = expected_req_rsc_size
        cum_slot_length = 0

        for i in range(len(s)):
            rsc_avl_tmp[i] = size[i]
                    # slot length                        
            if cum_slot_length + (e[i] - s[i]) >= expected_slot_num:
                curr_slot[i] = int(expected_slot_num - cum_slot_length)
                break
            else:
                curr_slot[i] = int(e[i] - s[i])
                cum_slot_length += curr_slot[i]

                # Try to distribute the rsc_lack to the intervals as evenly as possible
        n_iter = 0
        cum_size = 0
                # find the available slots (a vector)
        avl_slot_idx = np.where(rsc_avl_tmp> 0)[0]
                # apply the size constraint based on parallelism cfg files
        if task.parallel_mode in ["lwb", "range"]: 
            avl_slot_idx &= np.where(rsc_avl_tmp>task.core_min)[0] 
                
        while rsc_lack>0 and avl_slot_idx.size>0: 
                    # the max step of the size ++
            max_step = rsc_avl_tmp[avl_slot_idx].min() 
                    # the size of the current available slots
            cum_size_t = max_step + cum_size
                    # apply the size constraint 
            cum_size_t, _constr = task.get_available_cfg(cum_size_t, cum_size_t)
                    # judge whether the size_t is valid
            size_t = cum_size_t - cum_size                            
            if size_t > 0:
                slot_size_sum = np.sum(curr_slot[avl_slot_idx])
                if slot_size_sum * size_t >= rsc_lack:
                    size_t = np.ceil(rsc_lack / slot_size_sum).astype(int)
                    rsc_lack = 0
                else: 
                    rsc_lack -= slot_size_sum * size_t
                    
                cum_size += size_t

                curr_alloc[avl_slot_idx] += size_t
                rsc_avl_tmp[avl_slot_idx] -= size_t
            else:
                rsc_avl_tmp[rsc_avl_tmp == max_step] = 0

            upb_flg = _constr == "upb"
            list_upb_flg = _constr == "list" and cum_size_t == max(task.core_list)
            if upb_flg or list_upb_flg:
                break
                    # subtract the slots idx that cannot be allocated from the available slots
                    # Find indices where rsc_avl_tmp is greater than 0
            zero_indices = np.where(rsc_avl_tmp <= 0)[0]

                    # Subtract the non-zero indices from avl_slot_idx
            avl_slot_idx = np.setdiff1d(avl_slot_idx, zero_indices)
                    
            n_iter += 1
            if n_iter > 1000:
                assert False, "Infinite loop"
                
        if rsc_lack > 0:
            curr_alloc.fill(0)
            curr_slot.fill(0)
                    # as soon as possible
            self.asap_insert(task, DEBUG, s, e, size, curr_alloc, curr_slot, expected_req_rsc_size)
        else:
                    # Check whether the task allocate too much resources
            i=1
            while True:
                non_zero_indices = np.where(curr_alloc > 0)[0]
                non_zero_min_idx = curr_alloc[non_zero_indices].argmin()
                slot_idx = non_zero_indices[non_zero_min_idx]
                if rsc_lack + curr_alloc[slot_idx] <= 0:
                    curr_slot[slot_idx] -= 1
                    rsc_lack += curr_alloc[slot_idx] * 1
                    if curr_slot[slot_idx] == 0:
                        curr_alloc[slot_idx] = 0
                else:
                    break

    def block_insert(self, rsc_avl:np.ndarray,
                     expected_req_rsc_size, req_rsc_size:int, 
                     expected_slot_num:int, preempt_en:bool=True, strategy:str='asap', 
                     verbose=False, DEBUG=False)->Tuple[bool, Union[int,List[int]], Union[int,List[int]], Union[int,List[int]]]: 
        """
            Input: 
                preemptable: preempt_en
                strategy: asap

            Output:
                placement offset: phi_pos
        """
        
        # divide the rsc_avl into intervals, w.r.t. whether the rsc_avl >= req_rsc_size or not
        direct_aval_idx = rsc_avl >= req_rsc_size
        s, e, size = self.interval_sparsifier(direct_aval_idx)
        size = [req_rsc_size if size[i] else 0 for i in range(len(size))]
        # current allocation (C)
        curr_slot = np.zeros(len(s), dtype=int)

        # no enough slots with size >= req_rsc_size
        if (direct_aval_idx).sum() * req_rsc_size < expected_req_rsc_size:
            return False, [], []
        else: 
            # mapped to single time single interval or 
            if not preempt_en:
                # check if the task can be executed on the single interval
                for i in range(len(s)):
                    if (e[i] - s[i]) >= expected_slot_num and size[i] >= req_rsc_size:
                        # allocate resources 
                        return True, [s[i],], [expected_slot_num,]
                return False, [], []
            # broken into multiple chunk
            else:
                cum_rsc_alloc = 0
                # divide the rsc_avl into intervals as soon as possible
                for i in range(len(s)): 
                    # rsc size
                    if size[i] >= req_rsc_size:
                        # slot length
                        if cum_rsc_alloc + req_rsc_size * (e[i] - s[i]) >= expected_req_rsc_size:
                            curr_slot[i] = math.ceil((expected_req_rsc_size - cum_rsc_alloc)/req_rsc_size)
                            break
                        else:
                            curr_slot[i] = int(e[i] - s[i])
                            cum_rsc_alloc += req_rsc_size * curr_slot[i]
                # allocate resources
                idx = curr_slot.nonzero()[0]
                return True, np.array(s)[idx].tolist(), curr_slot[idx].tolist()

    @staticmethod
    def interval_sparsifier(rsc_avl):
        boader = (rsc_avl[0:-1] != rsc_avl[1:]).nonzero()[0] + 1
        s = [0] + boader.tolist() 
        e = boader.tolist() + [len(rsc_avl)] 
        size = [rsc_avl[s[i]] for i in range(len(s))]
        return s,e,size


    def asap_insert(self, task:ProcessInt, DEBUG, s, e, slot_n, curr_alloc, curr_slot, expected_req_rsc_size):
        if DEBUG:
            print("Not enough resources: as soon as possible")
        cum_rsc_alloc = 0
        for i in range(len(s)): 
            if slot_n[i] > 0:
                # rsc size 
                curr_alloc[i], _constr = task.get_available_cfg(slot_n[i], slot_n[i])
                if _constr != "N/A":
                    # slot length
                    if cum_rsc_alloc + curr_alloc[i] * (e[i] - s[i]) >= expected_req_rsc_size:
                        curr_slot[i] = math.ceil((expected_req_rsc_size - cum_rsc_alloc)/curr_alloc[i])
                        cum_rsc_alloc += curr_alloc[i] * curr_slot[i]
                        break
                    else:
                        curr_slot[i] = int(e[i] - s[i])
                        cum_rsc_alloc += curr_alloc[i] * curr_slot[i]
                else:
                    curr_alloc[i] = 0
                    curr_slot[i] = 0

        # Check whether the task allocate too much resources
        i=1
        if cum_rsc_alloc >= expected_req_rsc_size:
            while True:
                non_zero_indices = np.where(curr_alloc > 0)[0]
                non_zero_min_idx = curr_alloc[non_zero_indices].argmin()
                slot_idx = non_zero_indices[non_zero_min_idx]
                if cum_rsc_alloc - curr_alloc[slot_idx] >= expected_req_rsc_size:
                    curr_slot[slot_idx] -= 1
                    cum_rsc_alloc -= curr_alloc[slot_idx] * 1
                    if curr_slot[slot_idx] == 0:
                        curr_alloc[slot_idx] = 0
                else:
                    break

    def release(self, task: ProcessInt, time_slot_s:Union[int,List[int]], curr_alloc:Union[int,List[int]], curr_slot:Union[int,List[int]], verbose: bool = False):
        if isinstance(curr_alloc, int) and isinstance(curr_slot, int) and isinstance(time_slot_s, int):
            for rsc_map in self.scheduling_table[time_slot_s:time_slot_s+curr_slot]:
                rsc_map.release(task.pid, curr_alloc, verbose)
        else:
            assert len(curr_alloc) == len(curr_slot) == len(time_slot_s)
            for i in range(len(curr_alloc)):
                for rsc_map in self.scheduling_table[time_slot_s[i]:time_slot_s[i]+curr_slot[i]]:
                    rsc_map.release(task.pid, curr_alloc[i], verbose)
    
    def clear(self):
        for rsc_map in self.scheduling_table:
            rsc_map:Resource_model_int
            rsc_map.clear()

    def allocate(self, pid:int, time_slot_s:List[int], curr_alloc:List[int], curr_slot:List[int], verbose: bool = False):
        assert len(curr_alloc) == len(curr_slot) == len(time_slot_s)
        for i in range(len(curr_alloc)):
            for rsc_map in self.scheduling_table[time_slot_s[i]:time_slot_s[i]+curr_slot[i]]:
                rsc_map:Resource_model_int
                rsc_map.allocate(pid, curr_alloc[i], verbose)

    def step(self, mode:str="cyclic"): 
        assert mode in ["cyclic", "dynamic"]
        running = self.scheduling_table[0]
        self.scheduling_table = np.roll(self.scheduling_table, -1, axis=0)
        if mode == "dynamic": 
            self.scheduling_table[-1].clear()
        return running

    def print_scheduling_table(self, pid2name:Dict[int,str]=None, timestep=None, time_slot_s:int=None, time_slot_e:int=None):
        if time_slot_s is None:
            time_slot_s = 0

        if time_slot_e is None:
            rsc_agents_arr = self.scheduling_table[time_slot_s:]
        else:
            rsc_agents_arr = self.scheduling_table[time_slot_s:time_slot_e]

        empty_boader_s = []
        empty_boader_e = []
        title_line = False

        pre_rsc = rsc_agents_arr[0].rsc_map
        pre_idx = 0
        empty_flag = len(pre_rsc) == 0
        if empty_flag:
            empty_boader_s.append(0)
        
        for rsc_map_idx in range(len(rsc_agents_arr)):
            rsc_map = rsc_agents_arr[rsc_map_idx].rsc_map

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    if timestep is not None:
                        _str = f"time:[{pre_idx*timestep:.6f}-{rsc_map_idx*timestep:.6f}), slot:[{pre_idx}-{rsc_map_idx})\n"
                    else:
                        _str = f"slot:[{pre_idx}-{rsc_map_idx})\n"
                    if title_line: 
                        _str += pre_rsc.title_line
                        title_line = True
                    _str += f"{pre_rsc.print_simple(pid2name)}"

                    print(_str)
                pre_idx = rsc_map_idx
                pre_rsc = rsc_map
                empty_flag = len(pre_rsc) == 0
                if empty_flag:
                    empty_boader_s.append(rsc_map_idx)
        if empty_flag:
            empty_boader_e.append(len(rsc_agents_arr))
        else:
            print(f"slot:[{pre_idx}-{len(rsc_agents_arr)})\n{str(pre_rsc)}")

        _str = [f"[{empty_boader_s[i]}-{empty_boader_e[i]})" for i in range(len(empty_boader_s))]
        print("slot:{} Empty".format(",".join(_str,)))

    def print_alloc_detail(self, pid2name:Dict[int,str], timestep, core_max_dict:Dict[str,int]=dict(), max_core_stat:bool=False):
        bin_pack_result = self.index_occupy_by_id()

        # sort the result by the start time
        # item[1] is alloc_slot_s_t, alloc_size_t, allo_slot_t
        # item[1][0] is alloc_slot_s_t
        sorted_task_pid = [k for k, v in sorted(bin_pack_result.items(), key=lambda item: item[1][0])]

        # replace the pid with the task name
        # and print the result
        print(f"bin: {self.name}({self.id})")
        for pid in list(sorted_task_pid):
            _result = bin_pack_result.pop(pid)
            if pid < fork_pid_base:
                _name = pid2name[pid]
                thread_n = _name.split('_')[-2]
                troughput_n = _name.split('_')[-1]
                task_n = _name.replace("_"+thread_n, "").replace("_"+troughput_n, "")                
            else:
                task_n = pid2name[int(pid/fork_pid_base)]
                thread_n = task_n.split('_')[-2]
                troughput_n = task_n.split('_')[-1]
                task_n = task_n.replace("_"+thread_n, "").replace("_"+troughput_n, "")                
                fork_num = 1
                while True:
                    _name = task_n + f"_fork_{fork_num+1}_{thread_n}_{troughput_n}"
                    if _name not in bin_pack_result:
                        pid2name[pid] = _name
                        break
                    fork_num += 1
            
            bin_pack_result[_name] = _result
            print("task: {:s}({:d})".format(_name, pid))
            print("\tstart time: {:s}".format(", ".join([f"{x*timestep:.6f}" for x in _result[0]])))
            print("\talloc cores: {:s}".format(", ".join([f"{x:d}" for x in _result[1]])))
            print("\tused time: {:s}".format(", ".join([f"{x*timestep:.6f}" for x in _result[2]])))
            if max_core_stat:
                if task_n not in core_max_dict:
                    core_max_dict[task_n] = set()
                core_max_dict[task_n] = core_max_dict[task_n] | set(_result[1])
        print("=====================================\n")

    def to_sparse_dict(self, init_pos: int = 0, verbose: bool = False):
        # sparst_dict = {}
        sparse_list = dense_to_sparse(self.scheduling_table)
        # _str = [f"[{empty_boader_s[i]}-{empty_boader_e[i]})" for i in range(len(empty_boader_s))]
        # print("slot:{} Empty".format(",".join(_str,)))
        self.sparse_list = sparse_list
        self.sparse_idx = init_pos
        self.sparse_idx_next = init_pos + 1
        self.sparse_mode = True
    
    def to_sparse(self, init_pos: int = 0, timestep:float=float('nan'), verbose: bool = False):
        self.to_sparse_dict(init_pos, verbose)
        if self.alloc_mod == "compact":
            self.build_sparse_event()
        self.build_sparse_flops(timestep)

    def build_sparse_event(self):
        # event format: alloc_slot_s[0], _p.get_timestamp(), _p.pid, evet_msg
        # sparse_list format: cfg_slot_s, next_cfg, cfg_slot_num
        if not self.event_list:
            return
        self.event_list.sort(key=lambda x: x[0])
        sparse_event_dict = defaultdict(lambda:defaultdict(set)) # key: slot_s, value: [ts, pid, event_msg]
        for event_s, ts, pid, event_msg in self.event_list:
            sparse_event_dict[event_s][pid].update({event_msg})
        sparse_event = list(sorted(sparse_event_dict.items()))
            
        self.sparse_event.clear()
        for sparse_idx in range(len(self.sparse_list)):
            cfg_slot_s, next_cfg, cfg_slot_num = self.sparse_list[sparse_idx]
            next_cfg_slot_s = self.sparse_list[sparse_idx+1][0] if sparse_idx+1 < len(self.sparse_list) else float('inf')
            # (slot_idx, Dict[pid, event_set])
            event_block = []
            while sparse_event:
                head_event_t, head_event_dict = sparse_event[0]
                # ensure match the range from cfg_slot_s -> next_cfg_slot_s
                assert head_event_t >= cfg_slot_s
                if head_event_t < next_cfg_slot_s:
                    sparse_event.pop(0)
                else:
                    break
                # update event_block
                event_block.append((head_event_t, head_event_dict))
            if event_block:
                self.sparse_event.append((cfg_slot_s, event_block))
            else:
                self.sparse_event.append((cfg_slot_s, []))

        assert len(self.sparse_list) == len(self.sparse_event)
        assert len(sparse_event) == 0
        self.event_list.clear()
            
    def build_sparse_flops(self, timestep:float):
        assert not math.isnan(timestep) and timestep > 0
        # flops format: alloc_slot_s, _p.get_timestamp(), _p.pid, evet_msg
        # sparse_list format: cfg_slot_s, next_cfg, cfg_slot_num
        if not self.flops_list:
            return
        # sort the flops_list by the start time
        self.flops_list.sort(key=lambda x: x[0])
        
        # merge the sparse_cores and sparse_flops refer to the new spase_list
        # assume that the head and the tail of a chunk is always the start or end of a cfg block in the sparse_list
        # match the flops_list with the sparse_list
        sparse_flops = defaultdict(dict) # key: slot_s, value: Dict[pid, chunk_flops]
        for flops_idx in range(len(self.flops_list)):
            # range A
            flops_slot_s, pid, (chunk_len, size, chunk_flops) = self.flops_list[flops_idx]
            rem_flops = chunk_flops
            # find the index of range B in A that satisfies:
            # flops_slot_s <= cfg_slot_s and flops_slot_s + chunk_len > cfg_slot_s+cfg_slot_num
            start_idx = bisect.bisect_left([it[0] for it in self.sparse_list], flops_slot_s)
            end_idx = bisect.bisect_left([it[0] for it in self.sparse_list], flops_slot_s + chunk_len)
            for cfg_slot_s, cfg, cfg_slot_num in self.sparse_list[start_idx:end_idx]:
                assert size == cfg[pid]
                flops_tbd = elim_nume_error(size * cfg_slot_num * timestep * FLOPS_PER_CORE)
                sparse_flops[cfg_slot_s][pid] = flops_tbd = min(flops_tbd, rem_flops)
                rem_flops = elim_nume_error(rem_flops - flops_tbd)                
            assert rem_flops == 0
        self.sparse_flops.clear()
        self.sparse_flops = list(sorted(sparse_flops.items()))
        self.flops_list.clear()
    
    # add spase inded by 1
    def idx_plus_1(self,):
        assert self.sparse_mode
        self.sparse_idx_prev = self.sparse_idx
        self.sparse_idx = self.sparse_idx_next
        self.sparse_idx_next += 1
        if self.sparse_idx_next == len(self.sparse_list):
            self.sparse_idx_next = 0

    def idx_minus_1(self,):
        assert self.sparse_mode
        self.sparse_idx_next = self.sparse_idx
        self.sparse_idx = self.sparse_idx_prev
        self.sparse_idx_prev -= 1
        if self.sparse_idx_prev == -1:
            self.sparse_idx_prev = len(self.sparse_list) - 1
    
    def next_item(self,):
        self.idx_plus_1()
        # return self.sparse_list[list(self.sparse_dict.keys())[self.sparse_dict_idx]]
        return self.sparse_list[self.sparse_idx]

    def update(self, scheduling_table:np.ndarray):
        self.scheduling_table = scheduling_table
        self._SchedTab.to_sparse_dict(-1)

    def get_plot_frame(self, start:int=0, end:int=-1):
        frame = []
        for rsc_map in self.scheduling_table[start:end]:
            frame.append(rsc_map.get_plot_col())
        frame = np.array(frame).T
        return frame

    def __eq__(self, __o: object) -> bool:
        for i in range(len(self.scheduling_table)):
            if self.scheduling_table[i] != __o.scheduling_table[i]:
                return False
        return True
    
    def add_lock(self, task:ProcessInt, time_slot_s, time_slot_e):
        # check if the lock is valid
        if self.locker is not None:
            return False
        # set lock mask
        self.lock_mask[time_slot_s:time_slot_e] = False
        self.locker = task.pid
        return True

    def release_lock(self, task:ProcessInt, time_slot_s, time_slot_e):
        if self.locker == task.pid:
            self.lock_mask[time_slot_s:time_slot_e] = True
            self.locker = None
            return True
        else:
            return False

    @staticmethod
    def get_core_size(_p, timestep, FLOPS_PER_CORE):
        # release time round up: task should not be released earlier than the release time
        time_slot_s = int(np.ceil(_p.release_time/timestep))
        # deadline round down: task should not be finised later than the deadline
        time_slot_e = int(_p.deadline//timestep)
        req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
        return req_rsc_size
        
def dense_to_sparse(scheduling_table:np.ndarray):
    sparse_list = []
    empty_boader_s = []
    empty_boader_e = []
    title_line = False

    pre_rsc = scheduling_table[0].rsc_map
    pre_idx = 0
    empty_flag = len(pre_rsc) == 0
    if empty_flag:
        empty_boader_s.append(0)
    
    for rsc_map_idx in range(len(scheduling_table)):
        rsc_map = scheduling_table[rsc_map_idx].rsc_map

        if rsc_map == pre_rsc:
            continue
        else:
            if empty_flag:
                empty_boader_e.append(rsc_map_idx)
            else:
                sparse_list.append([pre_idx, pre_rsc, rsc_map_idx-pre_idx])
            pre_idx = rsc_map_idx
            pre_rsc = rsc_map
            empty_flag = len(pre_rsc) == 0
            if empty_flag:
                empty_boader_s.append(rsc_map_idx)
    if empty_flag:
        empty_boader_e.append(len(scheduling_table))
    else:
        # print(f"slot:[{pre_idx}-{len(scheduling_table)})\n{str(pre_rsc)}")
        # sparse_dict[pre_idx] = [pre_rsc, len(self.scheduling_table)-pre_idx]
        sparse_list.append([pre_idx, pre_rsc, len(scheduling_table)-pre_idx])
    return sparse_list

# event msg parser

def init_event(_p, curr_t:float, type:str, src:int=-1, tgt:int=-1) -> str:
    if type == "migrate_from":
        assert src >= 0
        return f"{_p.task.name:s}({_p.pid:d}) migrate from {src:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    elif type == "migrate_to":
        assert tgt >= 0
        return f"{_p.task.name:s}({_p.pid:d}) migrate to {tgt:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    elif type == "start":
        assert src >= 0
        return f"{_p.task.name:s}({_p.pid:d}) start on {src:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    elif type == "complete":
        assert src >= 0
        return f"{_p.task.name:s}({_p.pid:d}) complete on {src:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    else:
        raise ValueError(f"Unknown event type {type}")

tab_event_type = ["migrate_from", "start", "complete", "migrate_to"]
tab_event_re = r"(.*)\((\d+)\) (migrate from|migrate to|start on|complete on) (\d+) @ (\d+\.\d+)/(\d+\.\d+)!!"
tab_event_pattern_keys = ["name", "pid", "event_type", "bin_id", "curr_t", "ts"]
tab_event_pattern_type = [str, int, str, int, float, float]


def parse_event_msg(msg:str):
    """
    use re to match event pattern, and extract 
     start, ts, complete, migrate
    f"start-{pid:d}_{j:d}", 
    f"complete-{pid:d}_{j:d}", 
    f"migrate-{pid:d}_from_{pre_bin_idx}", 
    f"migrate-{pid:d}_to_{next_bin_idx}"
    """ 
    import re
    # pattern = re.compile(r"^(?P<event_type>\w+)-(?P<pid>\d+)(_(?P<ts>\d+))?(_from_(?P<from>\d+))?(_to_(?P<to>\d+))?$")
    pattern = re.compile(tab_event_re)
    match = pattern.match(msg)
    # {k: t(v) for k,v,t in zip(pattern_keys, match.groups(), pattern_type) if v is not None}
    try:
        result = {k: t(v) for k,v,t in zip(tab_event_pattern_keys, match.groups(), tab_event_pattern_type)}
    except TypeError:
        assert False, f"TypeError: {msg}"
    # replace "migrate from" with "migrate_from", "migrate to" with "migrate_to"
    result['event_type'] = result['event_type'].replace("on", "")
    result['event_type'] = result['event_type'].replace(" ", "_")
    return result

def filter_msg(tab_event_msg:str, msg_filter:Optional[Union[None, Dict[str, str]]]): 
    """
    Parse the event message and filter the message by the filter dict
    Return True if the message is filtered, False otherwise
    """
    results = parse_event_msg(tab_event_msg)
    for k,v in msg_filter.items(): 
        if k not in results or (results[k] != '?' and results[k]!= v):
            return False, {}
    return True, results
    
def process_tab_event(sched, curr_t, ready_queue, running_queue, throttle_list, event_list, _SchedTab, process_dict, bin_id, 
                      msg_filter:Optional[Union[None, Dict[str, str]]]):
    """
    process old events -> clear -> process new events
    NOTE: should not directly rob the budget from all the other partitions
    CASE: suppose a task A executes on 1 -> 2 -> 3
    and the task is late and misses the budeget on (1), then at arrival of A, 
    it should be executed on 2, with the budget of sum of 1 and 2, without 3
    CASE: 2 -> 1
    scheduler scan (1), take from bk but got nothing
    then scheduler scan (2), put the budget to (2) ruther than bk
    Event format: 
        {
            "name": str,
            "pid": int,
            "event_type": str,
            "bin_id": int,
            "curr_t": float,
            "ts": float
        }
    """
    if _SchedTab.alloc_mod == 'N/A':
        return 

    matched_events = []
    for msg in event_list:
        match, results = filter_msg(msg, msg_filter) 
        if not match:
            continue
        matched_events.append(msg)
        # "migrate_from", "start", "complete", "migrate_to"
        if "migrate" in results['event_type']:
            if 'to' in results['event_type']:
                print(msg)
                event_handle_migrate_to(sched, curr_t, ready_queue, running_queue, throttle_list, process_dict, bin_id, results)
            elif 'from' in results['event_type']:
                print(msg)
                event_handle_migrate_from(process_dict, bin_id, results)
        elif results['event_type'] == "start":
            pass
        elif results['event_type'] == "complete":
            pass
        else:
            raise ValueError(f"Unknown event type {results['event_type']}")
    # clear the matched events
    for idx in reversed(matched_events):
        event_list.remove(idx)
        
def event_handle_migrate_from(process_dict, bin_id, results):
    _p = check_legality(process_dict, results)
    # restore the _p.rem_flop_budget from BK
    _p.rem_flop_budget[bin_id] += _p.rem_flop_budget.pop('bk', 0.)

def check_legality(process_dict, results):
    _p = process_dict[results['pid']]
    # rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
    # try:
    #     assert len(rem_flop_budget) <= 2
    # except AssertionError:
    #     print(f"20231126: CodingError, try to gurrante the budget only on one partition at a time")
    return _p

def event_handle_migrate_to(sched, curr_t, ready_queue, running_queue, throttle_list, process_dict, bin_id, results):
    _p:ProcessInt = check_legality(process_dict, results)
    # throttle the task to be migrated
    if _p in (running_queue.queue+ready_queue.queue):
        _p.throttle_util(throttle_list, curr_t)
        _p.task.migration_count += 1
        if _p in running_queue.queue:
            sched.res_release(_p.pid)
            # budget_recoder.pop(_p.pid)
            running_queue.remove(_p)
        elif _p in ready_queue.queue:
            ready_queue.remove(_p)
        else:
            raise ValueError("Task is not in the running queue or ready queue")
    # backup the _p.rem_flop_budget
    rem_flop_budget = _p.rem_flop_budget[bin_id]
    if results['bin_id'] in _p.rem_flop_budget:
        _p.rem_flop_budget[results['bin_id']] += rem_flop_budget
    else:
        _p.rem_flop_budget['bk'] = rem_flop_budget
    _p.rem_flop_budget[bin_id] = 0

def calc_free_spaces(items, bin_height, bin_width):
    # Sort by start  
    items.sort(key=lambda x: x[0]) 

    free_spaces = []
    prev_end = 0
    prev_height = 0

    for item in items:
        start, height, length = item 
        
        # Calculate free space before this item
        if start > prev_end:
            free_spaces.append([prev_end, start - prev_end, bin_height - prev_height])

        # Calculate free space above this item
        if height < bin_height: 
            free_spaces.append([start, length, bin_height - height])

        # Update prev end and height
        prev_end = start + length
        prev_height = height

    # Add trailing free space
    if prev_end < bin_width:
        free_spaces.append([prev_end, bin_width - prev_end, bin_height - prev_height])

    return free_spaces

def get_freespace_features(free_spaces):
    # Calculate total free area
    free_area = sum(w*h for x,y,w,h in free_spaces)

    # Calculate x and y barycenters 
    # weighted_x = np.array([x*w*h for x,w,h in free_spaces])
    # weighted_y = np.array([y*w*h for y,w,h in free_spaces])
    weighted_x = np.array([(x+w/2)*w*h for x,y,w,h in free_spaces])
    weighted_y = np.array([(y+h/2)*w*h for x,y,w,h in free_spaces])

    bary_x = weighted_x.sum()/free_area
    bary_y = weighted_y.sum()/free_area 
    return free_area, bary_x, bary_y

class BinGenSelInt(object):
    def __init__(self, tab_temp_size:int):
        self.tab_temp_size = tab_temp_size
        self.init_bin_list()
    
    def _new_bin(self, id, size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, self.tab_temp_size, id=id, name=name)
    
    def init_bin_list(self, bin_list=[], bin_name_list=[]):
        self.bin_list = bin_list
        self.bin_name_list = bin_name_list

    def init_gen(self, gen:Iterator):
        self.iter_next_bin_obj = gen

class BinSelInt(BinGenSelInt): 
    """
    Given a set of bins, select a bin to allocate a task
    """
    pass
    
class BinGenInt(BinGenSelInt):
    """
    Given a generator, find a proper bins to allocate a task, if not found, create a new bin
    """
    def pick(self,):
        bin = next(self.iter_next_bin_obj)
        self.bin_list.append(bin)
        self.bin_name_list.append(bin.name)
        return bin

def new_bin(spatial_size:int, temporal_size:int, id:int = 0, name:str = "bin"):
    SchedTab = SchedulingTableInt(spatial_size, temporal_size, id=id, name=name)
    return SchedTab
            
def extend_dummy_bins(bin_list, min_num_bins=-1):
    min_num_bins = len(bin_list) if min_num_bins == -1 else min_num_bins
    for bin_id in range(min_num_bins):
        if bin_id >= len(bin_list):
            bin_list.append(SchedulingTableInt(0, bin_id, 0, f'dummy_bin_{bin_id}'))
    return bin_list

if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.') 
    parser.add_argument('--test_case', type=str, default="no constrants", help='test case name')
    args = parser.parse_args()
    # create a task set
    # first branch: have free cores and free slots at beginning
    # t1 [15, 19] 25 rsc and 2 slot
    # t2 [3, 12] 24 rsc and 7 slot
    # second branch: select a interval with enough free cores and free slots
    # t3 [0, 20] 10 rsc and 4 slot
    # thrird branch: evently distribute the resources in the expected interval
    # t4 [0, 7] 10 rsc and 4 slot
    # last branch: As soon as possible
    # t5 [0, 16] 10 rsc and 8 slot
    # t6 [9, 20] 9 rsc and 9 slot

    N_task = 6
    t1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                ERT=15, ddl=19, period=30, exp_comp_t=2, 
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t2 = TaskInt(task_name="task2", task_id=2, task_flag="moveable", timing_flag="deadline",
                ERT=3, ddl=12, period=30, exp_comp_t=7,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t3 = TaskInt(task_name="task3", task_id=3, task_flag="moveable", timing_flag="deadline", 
                ERT=0, ddl=20, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t4 = TaskInt(task_name="task4", task_id=4, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=7, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t5 = TaskInt(task_name="task5", task_id=5, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=16, period=30, exp_comp_t=8,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t6 = TaskInt(task_name="task6", task_id=6, task_flag="moveable", timing_flag="deadline",
                ERT=9, ddl=20, period=30, exp_comp_t=9,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)

    task_list:List[TaskInt] = [None for i in range(10)]
    alloc_info = [None for i in range(10)]
    require_rsc_size = [0 for i in range(10)]
    task_list[0:N_task] = [t1, t2, t3, t4, t5, t6]
    require_rsc_size[0:N_task] = [25, 24, 10, 10, 10, 9]

    if args.test_case == "no constrants":
        pass
    elif args.test_case == "upb":
        for i in range(N_task):
            task_list[i].parallel_mode = "upb"
            task_list[i].core_max = int(require_rsc_size[i] * 1.2)
    elif args.test_case == "list":
        for i in range(N_task):
            task_list[i].parallel_mode = "list"
            task_list[i].core_list = [i for i in range(0, require_rsc_size[i], 4)]
            task_list[i].core_list.append(require_rsc_size[i])
            task_list[i].core_list.append(int(require_rsc_size[i] * 1.5))
    elif args.test_case == "lwb":
        for i in range(N_task):
            task_list[i].parallel_mode = "lwb"
            task_list[i].core_min = int(require_rsc_size[i] * 0.8)



    # create a scheduling table
    scheduling_table = SchedulingTableInt(30, 20)
    # allocate resources

    # alloc_info[0] = scheduling_table.insert_task(t1, 25, t1.ERT, t1.ddl, t1.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[1] = scheduling_table.insert_task(t2, 24, t2.ERT, t2.ddl, t2.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[2] = scheduling_table.insert_task(t3, 10, t3.ERT, t3.ddl, t3.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[3] = scheduling_table.insert_task(t4, 10, t4.ERT, t4.ddl, t4.exp_comp_t, verbose=True)
    # alloc_info[4] = scheduling_table.print_scheduling_table()
    # alloc_info[5] = scheduling_table.insert_task(t5, 10, t5.ERT, t5.ddl, t5.exp_comp_t, verbose=True)
    # # print the scheduling table
    # scheduling_table.print_scheduling_table()

    pid2name = []
    pid = 0
    for task in task_list[0:N_task]: 
        # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
        r = task.get_release_time()
        d = task.get_deadline_time()
        p = task.make_process(r, d, pid)
        pid += 1
        pid2name.append(p)

    for i in range(N_task):
        print(f"task {i} allocation\n")
        alloc_info[i] = scheduling_table.insert_task(pid2name[i], require_rsc_size[i], 
                                                     pid2name[i].release_time, pid2name[i].deadline, 
                                                     pid2name[i].exp_comp_t, verbose=True)
        print("="*20)
        scheduling_table.print_scheduling_table()
    
    print("occupy by id:", scheduling_table.index_occupy_by_id())
    
    # release resources
    for i in range(N_task):
        print("before release")
        scheduling_table.print_scheduling_table()
        if alloc_info[i] is not None:
            print(f"task {pid2name[i].pid}({i}) release:")
            print([*alloc_info[i]])
            scheduling_table.release(pid2name[i], *alloc_info[i][1:], verbose=True)
        print("after release")
        scheduling_table.print_scheduling_table()