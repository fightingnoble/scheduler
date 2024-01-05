from typing import List, Dict, Tuple, Union, Optional, Iterable, Iterator, Collection
from collections import OrderedDict
import math
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
# from bokeh.plotting import figure, show
# from bokeh.models import ColumnDataSource, HoverTool, Range1d, LabelSet, Label, Legend
# from bokeh.layouts import row, column, gridplot
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

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
        self.alloc_mod = "exactly" 
        self.sparse_flops = list()
        self.sparse_cores = list()
        self.sparse_event = list()

        # available when used as a recorder
        self.wr_pointer = 0
        self.hypper_period = hp

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

        if (direct_aval_idx).sum() * req_rsc_size < expected_req_rsc_size:
            return False, [], []
        else: 
            if not preempt_en:
                # check if the task can be executed on the single interval
                for i in range(len(s)):
                    if (e[i] - s[i]) >= expected_slot_num and size[i] >= req_rsc_size:
                        # allocate resources 
                        return True, [s[i],], [expected_slot_num,]
                return False, [], []
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
        # check attribute self.alloc_mod, sparse_flops
        if not hasattr(self, "alloc_mod"):
            self.alloc_mod = "exactly"
        if not hasattr(self, "sparse_flops"):
            self.sparse_flops = list()
        if not hasattr(self, "sparse_cores"):
            self.sparse_cores = list()
        if not hasattr(self, "sparse_event"):
            self.sparse_event = list()


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

    # def generate_dependency_table(bin_list, job_graph_nx:nx.DiGraph, pname2pid, pid2pname): 
    #     num_processors = len(bin_list)
    #     num_timesteps = len(bin_list[0])
    #     dependency_table = np.empty_like(bin_list, dtype=object)
    #     dependency_table.fill([])  # Initialize all items with an empty list
        
    #     num_processors = bin_list.shape[1]

    #     for _Sched_tab in bin_list:
    #         for cfg_slot_s, next_cfg, cfg_slot_num in _Sched_tab.sparse_list: 
    #             next_cfg:RscMapInt
    #             for pid in next_cfg.keys():
    #                 job_n = pid2pname[pid]
    #                 for succ_n, datadict in job_graph_nx.succ[job_n].items(): 
    #                     succ_pid = pname2pid[succ_n]
        
    #     # Iterate over each timestep and processor in the scheduling table
    #     for timestep, processor in np.ndindex(bin_list.shape):
    #         jobs_executed = bin_list[timestep, processor]
            
    #         # Iterate over each job executed by the current processor
    #         for job_id in jobs_executed:
    #             dependencies = job_graph.get_dependencies(job_id)  # Get the dependencies of the current job
                
    #             # Iterate over each dependency of the current job
    #             for dependency in dependencies:
    #                 # Find the processors that executed the predecessor jobs
    #                 predecessor_processors = np.where(bin_list[:, :] == dependency)[1]
                    
    #                 # Append the predecessor processors to the dependency table
    #                 dependency_table[timestep, processor].extend(predecessor_processors)
        
    #     return dependency_table

def get_task_layout_compact(bin_list:List[SchedulingTableInt], pid2name:Dict[int, str], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = hyper_p * (warmup)
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

    colors=list(mcolors.XKCD_COLORS.keys())
    
    base_vertical_offset = 0
    y_margin = 0.5 
    time_grid_size = 0.004
    x_margin = time_grid_size

    # plot timeline and task name bin by bin
    # and select color for the task automatically
    if plot_legend:
        fig_size = (40, 50)
    else:
        fig_size = (60, 30)
    fig, axes = plt.subplots(nrows=len(bin_list),ncols=1,sharex=True,figsize=fig_size) 

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    for bin_idx, _SchedTab in enumerate(bin_list): 
        # bin_temp_size = len(_SchedTab.scheduling_table)
        # bin_spatial_size = _SchedTab.scheduling_table[0].size
        bin_spatial_size = _SchedTab.num_resources
        bin_temp_size = _SchedTab.temp_size

        ax = axes[len(bin_list)-bin_idx-1]                        

        empty_boader_s = []
        empty_boader_e = []
        title_line = False
        pre_rsc = _SchedTab.scheduling_table[0].rsc_map
        # build a position dict
        position_dict = {}
        cum_pos = 0
        for k,v in pre_rsc.items():
            position_dict[k] = [[cum_pos], [v], True] 
            cum_pos += v

        pre_idx = 0
        empty_flag = len(pre_rsc) == 0

        if empty_flag:
            empty_boader_s.append(0)
        
        for rsc_map_idx in range(bin_temp_size):
            rsc_map = _SchedTab.scheduling_table[rsc_map_idx].rsc_map

            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            s, e = pre_idx*time_step, rsc_map_idx*time_step
            if s < plot_start:
                s = plot_start

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    # plot the task layout
                    for pid, size in pre_rsc.items():
                        _p_name = pid2name[pid]
                        _p_color = mcolors.XKCD_COLORS[colors[pid%len(colors)]]
                        is_new = position_dict[pid][-1]

                        for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                            # culculate the position of the bar
                            bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                            text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                            # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                            ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)
                            # plot the text
                            if not plot_legend:
                                if is_new:
                                    position_dict[pid][-1] = False
                                    ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

                # the vertical grid at the end of the bar
                ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

                # update the position dict
                new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
                expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
                old_pid = set(pre_rsc.keys()) - expired_pid

                used_position = []
                for pid in old_pid:
                    p_size = rsc_map[pid]
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        used_position += [i for i in range(s, e)]

                # remove the expired task from the position dict
                for pid in expired_pid:
                    position_dict.pop(pid)
                
                aval_pos = [i for i in range(bin_spatial_size) if i not in used_position]
                # check if the old task's allocation is changed

                size_plus = []
                size_minus = []
                for pid in sorted(old_pid):
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    if new_size > old_size:
                        size_plus.append(pid)
                    elif new_size < old_size:
                        size_minus.append(pid)

                for group in [size_minus, size_plus]:
                    for pid in group: 
                        old_size = pre_rsc[pid]
                        new_size = rsc_map[pid]

                        # release the old position
                        for s, size in zip(*position_dict[pid][:-1]):
                            e = s + size
                            aval_pos += [i for i in range(s, e)]
                        aval_pos.sort()
                        # get the start position of the old task
                        cum_pos = position_dict[pid][0][0]
                        # divide the available position into two parts
                        left_pos = aval_pos[:aval_pos.index(cum_pos)]
                        right_pos = aval_pos[aval_pos.index(cum_pos):]
                        # select the leftmost position from cum_pos
                        interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                        if len(interval_picked) < new_size:
                            # select the leftmost position from left_pos
                            interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(new_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, position_dict[pid][-1]]

                # pick a proper position for the new task in the available position
                for pid in new_pid:
                    p_size = rsc_map[pid]
                    # search for a gap in the available positions that can accommodate the task's new size
                    gap_start = None
                    gap_size = 0
                    for pos in aval_pos:
                        if gap_start is None:
                            gap_start = pos
                        gap_size += 1
                        if gap_size == p_size:
                            break
                        if pos + 1 not in aval_pos:
                            gap_start = None
                            gap_size = 0

                    if gap_start is not None and gap_size == p_size:
                        # allocate the task to the found gap
                        start = [gap_start]
                        size = [gap_size]
                        position_dict[pid] = [start, size, True]
                        # remove the selected positions from aval_pos
                        aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                    else:
                        # select the leftmost position
                        interval_picked = aval_pos[:p_size]
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(p_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, True]
                
                # update the pre_rsc                                                
                pre_idx = rsc_map_idx
                pre_rsc = rsc_map
                empty_flag = len(pre_rsc) == 0
                if empty_flag:
                    empty_boader_s.append(rsc_map_idx)


        # set start and end time for each task: 
        #   if part of the task is in the warmup cycle or drain cycle, 
            # set the start and end time to the start and end time of the plot
        s, e = pre_idx*time_step, bin_temp_size*time_step
        if s < plot_end or e > plot_start: 
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end

            if empty_flag:
                empty_boader_e.append(bin_temp_size)
            else:
                # plot the task layout
                for pid, size in pre_rsc.items():
                    _p_name = pid2name[pid]
                    _p_color = mcolors.XKCD_COLORS[[colors[pid%len(colors)]]]
                    is_new = position_dict[pid][-1]

                    for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                        # culculate the position of the bar
                        bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                        text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                        # plot the bar
                        ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)

                        # plot the text
                        if not plot_legend:
                            if is_new:
                                position_dict[pid][-1] = False
                                ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

            # the vertical grid at the end of the bar
            ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

        # set the axis and title
        vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
        ticks = np.linspace(vs, vs+bin_spatial_size-1, 4, dtype=int)
        ax.set_ylim(bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=txt_size, rotation=45)

        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)

    # only set x axis for the bottom plot
    ax = axes[len(bin_list)-1]
    ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
    ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
    ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
    ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
    ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
    ax.set_xlabel("Time (s)", fontsize=txt_size) 

    if plot_legend:
        # add legend to the top plot
        ax = axes[0]
        from matplotlib.lines import Line2D
        legend_elements = []
        # for i in range(len(init_p_list)): 
        for i, pid in enumerate(pid2name.keys()):
            legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=pid2name[pid]))
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.2),
        ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
            

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
    # save the figure
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if tool == "matplotlib":
            if "format" in kwargs and isinstance(kwargs["format"], list):
                fmt_list = kwargs.pop("format")
                if path_parse[-1] not in fmt_list:
                    kwargs["format"].append(path_parse[-1])
                for f in fmt_list:
                    save_path = ".".join(path_parse[:-1]) + "." + f
                    if tool == "matplotlib":
                        plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
            elif "format" not in kwargs and len(path_parse) > 1: 
                kwargs["format"] = path_parse[-1]
            else:
                kwargs["format"] = "pdf"
                save_path = save_path + ".pdf"        
                if tool == "matplotlib":
                    plt.savefig(save_path, bbox_inches='tight', **kwargs)

def get_task_layout_compact1bin(bin_list:List[SchedulingTableInt], pid2name:Dict[int, str], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = hyper_p * (warmup)
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

    colors=list(mcolors.XKCD_COLORS.keys())
    
    base_vertical_offset = 0
    y_margin = 0.5 
    time_grid_size = 0.004
    x_margin = time_grid_size

    # plot timeline and task name bin by bin
    # and select color for the task automatically
    tot_cores = sum([_SchedTab.num_resources for _SchedTab in bin_list])
    height = math.ceil(tot_cores//256)
    if plot_legend:
        fig_size = (40, 20+30*height)
    else:
        fig_size = (60, 30*height)
    fig, axes = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=fig_size) 

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    ax = axes           
    for bin_idx, _SchedTab in enumerate(bin_list): 
        # bin_temp_size = len(_SchedTab.scheduling_table)
        # bin_spatial_size = _SchedTab.scheduling_table[0].size
        bin_spatial_size = _SchedTab.num_resources
        bin_temp_size = _SchedTab.temp_size


        empty_boader_s = []
        empty_boader_e = []
        title_line = False
        pre_rsc = _SchedTab.scheduling_table[0].rsc_map
        # build a position dict
        position_dict = {}
        cum_pos = 0
        for k,v in pre_rsc.items():
            position_dict[k] = [[cum_pos], [v], True] 
            cum_pos += v

        pre_idx = 0
        empty_flag = len(pre_rsc) == 0

        if empty_flag:
            empty_boader_s.append(0)
        
        for rsc_map_idx in range(bin_temp_size):
            rsc_map = _SchedTab.scheduling_table[rsc_map_idx].rsc_map

            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            s, e = pre_idx*time_step, rsc_map_idx*time_step
            if s < plot_start:
                s = plot_start

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    # plot the task layout
                    for pid, size in pre_rsc.items():
                        _p_name = pid2name[pid]
                        _p_color = mcolors.XKCD_COLORS[colors[pid%len(colors)]]
                        is_new = position_dict[pid][-1]

                        for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                            # culculate the position of the bar
                            bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                            text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                            # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                            ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)
                            # plot the text
                            if not plot_legend:
                                if is_new:
                                    position_dict[pid][-1] = False
                                    ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

                # the vertical grid at the end of the bar
                ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

                # update the position dict
                new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
                expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
                old_pid = set(pre_rsc.keys()) - expired_pid

                used_position = []
                for pid in old_pid:
                    p_size = rsc_map[pid]
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        used_position += [i for i in range(s, e)]

                # remove the expired task from the position dict
                for pid in expired_pid:
                    position_dict.pop(pid)
                
                aval_pos = [i for i in range(bin_spatial_size) if i not in used_position]
                # check if the old task's allocation is changed

                size_plus = []
                size_minus = []
                for pid in sorted(old_pid):
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    if new_size > old_size:
                        size_plus.append(pid)
                    elif new_size < old_size:
                        size_minus.append(pid)

                for group in [size_minus, size_plus]:
                    for pid in group: 
                        old_size = pre_rsc[pid]
                        new_size = rsc_map[pid]

                        # release the old position
                        for s, size in zip(*position_dict[pid][:-1]):
                            e = s + size
                            aval_pos += [i for i in range(s, e)]
                        aval_pos.sort()
                        # get the start position of the old task
                        cum_pos = position_dict[pid][0][0]
                        # divide the available position into two parts
                        left_pos = aval_pos[:aval_pos.index(cum_pos)]
                        right_pos = aval_pos[aval_pos.index(cum_pos):]
                        # select the leftmost position from cum_pos
                        interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                        if len(interval_picked) < new_size:
                            # select the leftmost position from left_pos
                            interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(new_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, position_dict[pid][-1]]

                # pick a proper position for the new task in the available position
                for pid in new_pid:
                    p_size = rsc_map[pid]
                    # search for a gap in the available positions that can accommodate the task's new size
                    gap_start = None
                    gap_size = 0
                    for pos in aval_pos:
                        if gap_start is None:
                            gap_start = pos
                        gap_size += 1
                        if gap_size == p_size:
                            break
                        if pos + 1 not in aval_pos:
                            gap_start = None
                            gap_size = 0

                    if gap_start is not None and gap_size == p_size:
                        # allocate the task to the found gap
                        start = [gap_start]
                        size = [gap_size]
                        position_dict[pid] = [start, size, True]
                        # remove the selected positions from aval_pos
                        aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                    else:
                        # select the leftmost position
                        interval_picked = aval_pos[:p_size]
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(p_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, True]
                
                # update the pre_rsc                                                
                pre_idx = rsc_map_idx
                pre_rsc = rsc_map
                empty_flag = len(pre_rsc) == 0
                if empty_flag:
                    empty_boader_s.append(rsc_map_idx)


        # set start and end time for each task: 
        #   if part of the task is in the warmup cycle or drain cycle, 
            # set the start and end time to the start and end time of the plot
        s, e = pre_idx*time_step, bin_temp_size*time_step
        if s < plot_end or e > plot_start: 
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end

            if empty_flag:
                empty_boader_e.append(bin_temp_size)
            else:
                # plot the task layout
                for pid, size in pre_rsc.items():
                    _p_name = pid2name[pid]
                    _p_color = mcolors.XKCD_COLORS[[colors[pid%len(colors)]]]
                    is_new = position_dict[pid][-1]

                    for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                        # culculate the position of the bar
                        bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                        text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                        # plot the bar
                        ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)

                        # plot the text
                        if not plot_legend:
                            if is_new:
                                position_dict[pid][-1] = False
                                ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

            # the vertical grid at the end of the bar
            ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

        # add a horizontal line to seperate bins
        ax.axhline(bin_vertical_offset, color='black', linestyle='-', linewidth=0.5)
        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)

    # only set x axis for the bottom plot
    ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
    ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
    ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
    ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
    ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
    ax.set_xlabel("Time (s)", fontsize=txt_size) 

    # set the axis and title
    vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
    ticks = np.linspace(0, vs, 4, dtype=int)
    ax.set_ylim(-y_margin, bin_vertical_offset+y_margin)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=txt_size, rotation=45)

    if plot_legend:
        # add legend to the top plot
        from matplotlib.lines import Line2D
        legend_elements = []
        # for i in range(len(init_p_list)): 
        for i, pid in enumerate(pid2name.keys()):
            legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=pid2name[pid]))
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.2),
        ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
            

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
    # save the figure
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if tool == "matplotlib":
            if "format" in kwargs and isinstance(kwargs["format"], list):
                fmt_list = kwargs.pop("format")
                if path_parse[-1] not in fmt_list:
                    kwargs["format"].append(path_parse[-1])
                for f in fmt_list:
                    save_path = ".".join(path_parse[:-1]) + "." + f
                    if tool == "matplotlib":
                        plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
            elif "format" not in kwargs and len(path_parse) > 1: 
                kwargs["format"] = path_parse[-1]
            else:
                kwargs["format"] = "pdf"
                save_path = save_path + ".pdf"        
                if tool == "matplotlib":
                    plt.savefig(save_path, bbox_inches='tight', **kwargs)



def get_task_layout(bin_list:List[SchedulingTableInt], init_p_list:List[ProcessInt]
                        , show:bool=False, save:bool=False, save_path:str="task_layout.pdf", **kwargs):

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    # plot timeline and task name bin by bin
    # and select color for the task automatically
    fig = plt.figure(figsize=(50, 20))
    vertical_grid_size = 1
    
    for _SchedTab in bin_list:
        tab_temp_size = len(_SchedTab.scheduling_table)
        ax = fig.add_subplot(len(bin_list), 1, len(bin_list)-_SchedTab.id)
        vertical_offset = 0
        horizen_grid = set()
        bin_pack_result = _SchedTab.index_occupy_by_id()
        ordered_k = [k for k, v in sorted(bin_pack_result.items(), key=lambda item: item[1][1])]
        for pid in ordered_k:
            alloc_info = bin_pack_result[pid]
            _p = init_p_list[pid]
            _p_name = _p.task.name
            _p_color = mcolors.XKCD_COLORS[[colors[pid%len(colors)]]]
            for s, size, l in zip(*alloc_info):
                ax.broken_barh([(s, l)], (vertical_offset*vertical_grid_size, size*vertical_grid_size), facecolors=_p_color)
                ax.text(s+l//2, (vertical_offset+size//2)*vertical_grid_size, _p_name, ha='center', va='center', color='black', fontsize=10)
                horizen_grid.add(s)
                horizen_grid.add(s+l)
            vertical_offset += size
        # ax.set_ylim(0, vertical_offset*vertical_grid_size)
        ax.set_xlim(0, tab_temp_size)
        ax.set_xlabel('Time')
        # add bin name
        ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})")
        # plot the grid
        for x in horizen_grid:
            ax.axvline(x, color='black', linestyle='-', linewidth=0.5)
    # sub-figures shares the same x-axis
    # fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    # plot the result
    if show:
        plt.show()
    # save the figure
    if "format" not in kwargs and save_path.split(".")[-1] == "pdf" and save:
        kwargs["format"] = "pdf"
    plt.savefig(save_path, **kwargs)
    print("save figure to", save_path)

def get_task_layout_sparse(bin_list:List[SchedulingTableInt], pid2name:Dict[int, str], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = hyper_p * (warmup)
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    colors=list(mcolors.XKCD_COLORS.keys())
    
    base_vertical_offset = 0
    y_margin = 0.5 
    time_grid_size = 0.004
    x_margin = time_grid_size

    # plot timeline and task name bin by bin
    # and select color for the task automatically
    if plot_legend:
        fig_size = (40, 50)
    else:
        fig_size = (60, 30)
    if tool == "matplotlib":
        fig = plt.figure(figsize=fig_size)
    

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    for bin_idx, _SchedTab in enumerate(bin_list): 
        _SchedTab: SchedulingTableInt
        # bin_temp_size = len(_SchedTab.scheduling_table)
        # bin_spatial_size = _SchedTab.scheduling_table[0].size
        bin_spatial_size = _SchedTab.num_resources

        if tool == "matplotlib":
            ax = fig.add_subplot(len(bin_list), 1, len(bin_list)-_SchedTab.id)
        
        _SchedTab.to_sparse_dict(0)
        # build a position dict
        position_dict = {}
        aval_pos = [i for i in range(bin_spatial_size)]

        while _SchedTab.sparse_idx_next:
            pre_idx, curr_cfg, slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
            if _SchedTab.sparse_idx == 0:
                rsc_map_idx = 0
                cum_pos = 0
                for k,v in curr_cfg.items():
                    position_dict[k] = [[cum_pos], [v], True] 
                    aval_pos = [i for i in aval_pos if i not in range(cum_pos, cum_pos+v)]
                    cum_pos += v
            else:
                rsc_map_idx = _SchedTab.sparse_list[_SchedTab.sparse_idx_prev][0] + _SchedTab.sparse_list[_SchedTab.sparse_idx_prev][2] 
                rsc_map = curr_cfg
                pre_rsc = _SchedTab.sparse_list[_SchedTab.sparse_idx_prev][1]

                # update the position dict
                new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
                expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
                old_pid = set(pre_rsc.keys()) - expired_pid

                # remove the expired task from the position dict
                for pid in expired_pid:
                    # release the old position
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        aval_pos += [i for i in range(s, e)]
                    position_dict.pop(pid)
                
                # check if the old task's allocation is changed
                size_plus = []
                size_minus = []
                for pid in sorted(old_pid):
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    if new_size > old_size:
                        size_plus.append(pid)
                    elif new_size < old_size:
                        size_minus.append(pid)

                for group in [size_minus, size_plus]:
                    for pid in group: 
                        old_size = pre_rsc[pid]
                        new_size = rsc_map[pid]

                        # release the old position
                        for s, size in zip(*position_dict[pid][:-1]):
                            e = s + size
                            aval_pos += [i for i in range(s, e)]
                        aval_pos.sort()
                        # get the start position of the old task
                        cum_pos = position_dict[pid][0][0]
                        # divide the available position into two parts
                        left_pos = aval_pos[:aval_pos.index(cum_pos)]
                        right_pos = aval_pos[aval_pos.index(cum_pos):]
                        # select the leftmost position from cum_pos
                        interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                        if len(interval_picked) < new_size:
                            # select the leftmost position from left_pos
                            interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(new_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, position_dict[pid][-1]]

                # pick a proper position for the new task in the available position
                for pid in new_pid:
                    p_size = rsc_map[pid]
                    # search for a gap in the available positions that can accommodate the task's new size
                    gap_start = None
                    gap_size = 0
                    for pos in aval_pos:
                        if gap_start is None:
                            gap_start = pos
                        gap_size += 1
                        if gap_size == p_size:
                            break
                        if pos + 1 not in aval_pos:
                            gap_start = None
                            gap_size = 0

                    if gap_start is not None and gap_size == p_size:
                        # allocate the task to the found gap
                        start = [gap_start]
                        size = [gap_size]
                        position_dict[pid] = [start, size, True]
                        # remove the selected positions from aval_pos
                        aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                    else:
                        # select the leftmost position
                        interval_picked = aval_pos[:p_size]
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(p_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, True]
                
            _SchedTab.idx_plus_1()

            s = pre_idx*time_step
            if pre_idx != rsc_map_idx:
                # add the vertical grid at the beginning of the bar
                if s >= plot_start and s <= plot_end:
                    add_v_grid(s, ax)

            rsc_map_idx = pre_idx + slot_num
            e = rsc_map_idx*time_step
            # sparse list contains no empty cfg
            assert len(curr_cfg) > 0 

            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end
            if s >= plot_end or e <= plot_start: 
                continue

            # plot the task layout
            for pid, size in curr_cfg.items():
                _p_name = pid2name[pid]
                _p_color = mcolors.XKCD_COLORS[colors[pid%len(colors)]]
                is_new = position_dict[pid][-1]

                for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                    # culculate the position of the bar
                    bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                    text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                    # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                    h_pos, h_size, v_pos, v_size = s, e-s, bar_vertical_offset, vertical_size*vertical_grid_size
                    add_bar(h_pos, h_size, v_pos, v_size, _p_color, ax)
                    # plot the text
                    if not plot_legend:
                        if is_new:
                            position_dict[pid][-1] = False
                            add_text((s+e)/2, text_vertical_offset, _p_name, ax)
            # the vertical grid at the end of the bar
            add_v_grid(e, ax)

            
        # set the axis and title
        vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
        ticks = np.linspace(vs, vs+bin_spatial_size-1, 4, dtype=int)
        if tool == "matplotlib":
            ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
            ax.set_ylim(bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks, fontsize=txt_size)

        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        if tool == "matplotlib":
            ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)

    # sub-figures shares the same x-axis
    # fig.subplots_adjust(hspace=0)
    if tool == "matplotlib":
        plt.setp([a.get_xticklabels() for a in fig.axes[1:]], visible=False)

    if plot_legend:
        if tool == "matplotlib":
            # add legend
            ax = fig.axes[-1]
            from matplotlib.lines import Line2D
            legend_elements = []
            # for i in range(len(init_p_list)): 
            for i, pid in enumerate(pid2name.keys()):
                legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=pid2name[pid]))
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.2),
            ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
            # reset x ticks: text size 30, rotation 45, distance time_grid_size * 2
            # ticks format: .3f
            ax = fig.axes[0]
            ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
            ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
            ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
            ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
            # remove frame
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # set x axis label as Time (s), text size 30
            ax.set_xlabel("Time (s)", fontsize=txt_size) 

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
    # save the figure
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if tool == "matplotlib" or tool == "plotly":
            if "format" in kwargs and isinstance(kwargs["format"], list):
                fmt_list = kwargs.pop("format")
                if path_parse[-1] not in fmt_list:
                    kwargs["format"].append(path_parse[-1])
                for f in fmt_list:
                    save_path = ".".join(path_parse[:-1]) + "." + f
                    if tool == "matplotlib":
                        plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
                    elif tool == "plotly":
                        fig.write_image(save_path)
            elif "format" not in kwargs and len(path_parse) > 1: 
                kwargs["format"] = path_parse[-1]
            else:
                kwargs["format"] = "pdf"
                save_path = save_path + ".pdf"        
                if tool == "matplotlib":
                    plt.savefig(save_path, bbox_inches='tight', **kwargs)

def add_bar(h_pos, h_size, v_pos, v_size, _p_color, ax, backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        ax.broken_barh([(h_pos, h_size)], (v_pos, v_size), facecolors=_p_color)
    else:
        raise NotImplementedError

def add_text(h_pos, v_pos, _p_name, ax, backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        ax.text(h_pos, v_pos, _p_name, ha='center', va='center', color='black', fontsize=10)

def add_v_grid(v_pos, ax, backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        ax.axvline(v_pos, color='black', linestyle='-', linewidth=0.5)

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

def get_sparse_flops(bin_list:List[SchedulingTableInt], glb_p_list: List[ProcessInt], timestep, event_range):

    process_dict = {_p.pid:_p for _p in glb_p_list}
    # build event list
    process_stim_dict = OrderedDict()
    for pid, _p in process_dict.items():
        _p:ProcessInt
        stimu_tab = _p.task.extract_sensor_event(event_range, verbose=False)
        l = []
        for stimu_t in stimu_tab:
            start_t, ddl_t = elim_nume_error(stimu_t+_p.task.ERT), elim_nume_error(stimu_t+_p.task.ERT+_p.task.ddl)
            # quantize the start time and ddl time
            slot_s = int(math.ceil(start_t/timestep)) 
            slot_e = int(math.floor(ddl_t/timestep))
            l.append([_p.pid, slot_s, slot_e, _p.task.name, stimu_t]) 
        process_stim_dict[pid] = l

    rsc_recoder = {}
    for _bin in bin_list:
        _bin:SchedulingTableInt
        # # [task_id] = (s, size, l)
        # bin_pack_result:Dict[int, Tuple[List[int], List[int], List[int]]] = _bin.index_occupy_by_id()
        # # merge the result to the recoder, also add a extra item of bin_id 
        # for pid, item in bin_pack_result.items():
        #     bin_id = [_bin.id] * len(item[0])
        #     if pid in rsc_recoder:
        #         s, size, l, b_id = rsc_recoder[pid]
        #         rsc_recoder[pid] = [s+item[0], size+item[1], l+item[2], b_id+bin_id]
        #     else:
        #         rsc_recoder[pid] = [item[0], item[1], item[2], bin_id]

        # update the rsc_recoder by the sparse list
        sparse_list = _bin.sparse_list
        for pre_idx, pre_rsc, slot_num in sparse_list:
            for pid, size in pre_rsc.items():
                if pid in rsc_recoder:
                    s, size_l, l, b_id = rsc_recoder[pid]
                    rsc_recoder[pid] = [s+[pre_idx], size_l+[size], l+[slot_num], b_id+[_bin.id]]
                else:
                    rsc_recoder[pid] = [[pre_idx], [size], [slot_num], [_bin.id]]


    for pid, item in rsc_recoder.items():
        s, size, l, b_id = item
        # sort the result by the start time
        # sort s and update the size and l
        s, size, l, b_id = zip(*sorted(zip(s, size, l, b_id), key=lambda x: x[0]))
        rsc_recoder[pid] = [s, size, l, b_id]
    # sort the record by pid
    rsc_recoder = OrderedDict(sorted(rsc_recoder.items(), key=lambda item: item[0]))
    
    # inner dict is indexed by slot index
    # outter dict is indexed by bin index
    sparse_flops_dict = OrderedDict()
    sparse_cores_dict = OrderedDict()
    sparse_event_dict = OrderedDict()
    # init the above dict
    for bin_idx in range(len(bin_list)):
        sparse_flops_dict[bin_idx] = {}
        sparse_cores_dict[bin_idx] = {}
        sparse_event_dict[bin_idx] = {}
    # event pattern:
    # f"start-{pid:d}_{j:d}", f"complete-{pid:d}_{j:d}", f"migrate-{pid:d}_from_{pre_bin_idx}", f"migrate-{pid:d}_to_{next_bin_idx}"
    #  start
    #  ts
    #  complete
    #  migrate
    
    # check legelty
    assert set(rsc_recoder.keys()) == set(process_stim_dict.keys())
    for pid in rsc_recoder.keys():
        alloc_item = rsc_recoder[pid]
        stim_item = process_stim_dict[pid]
        i=0
        cum_ops = 0
        ref_flops = process_dict[pid].totcpu
        s_j_istart = 0 # the start index of chunk, allocated to the jth stimulus
        for j in range(len(stim_item)):
            s_j = stim_item[j][1]
            e_j = stim_item[j][2]
            pid = stim_item[j][0]
            s_j_istart = i
            if i < len(alloc_item[0]):
                s_i = alloc_item[0][i]
                bin_idx = alloc_item[3][i]
                update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, item={f"start-{pid:d}_{j:d}"}, item_type=set)
            while i < len(alloc_item[0]):
                s_i = alloc_item[0][i]
                size_i = alloc_item[1][i]
                l_i = alloc_item[2][i]
                e_i = alloc_item[0][i] + alloc_item[2][i]
                pre_bin_idx = alloc_item[3][i-1 if i>0 else 0]
                bin_idx = alloc_item[3][i]
                next_bin_idx = alloc_item[3][i+1 if i<len(alloc_item[0])-1 else len(alloc_item[0])-1]
                if pre_bin_idx != bin_idx and i > s_j_istart:
                    update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, 
                                       item={f"migrate-{pid:d}_from_{pre_bin_idx}"}, item_type=set)
                if cum_ops >= ref_flops:
                    cum_ops = 0
                    break
                if e_i <= e_j:
                    ops = size_i * l_i * timestep * FLOPS_PER_CORE
                    ops_tbd = min(ops, ref_flops-cum_ops)
                    cum_ops += ops
                    update_sparse_dict(bin_idx, pid, s_i, sparse_flops_dict, item=ops_tbd, item_type=float)
                    update_sparse_dict(bin_idx, pid, s_i, sparse_cores_dict, item=size_i, item_type=int)
                    if ops>ops_tbd:
                        update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, item={f"complete-{pid:d}_{j:d}"}, item_type=set)
                    elif next_bin_idx != bin_idx:
                        update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, 
                                        item={f"migrate-{pid:d}_to_{next_bin_idx:d}"}, item_type=set)

                    i += 1
                else:
                    assert cum_ops >= ref_flops
                    cum_ops = 0
                    break
        # NOTE: check if all the allocation are found the corresponding stimulus
        assert i == len(alloc_item[0])
    # NOTE:check whether the sparse_cores and the sparse_flops have the same length with the sparse_list
    for bin_idx in range(len(bin_list)):
        # verify the sparse index matching
        assert set(sparse_flops_dict[bin_idx].keys()) == set(sparse_cores_dict[bin_idx].keys()) 
        # event keys is a subset of slot keys
        assert set(sparse_event_dict[bin_idx].keys()) <= set(sparse_flops_dict[bin_idx].keys())

        # sort the dict by key
        sorted_slot_n = sorted(sparse_flops_dict[bin_idx].keys())
        sparse_cores = [sparse_cores_dict[bin_idx][k] for k in sorted_slot_n]
        sparse_flops = [sparse_flops_dict[bin_idx][k] for k in sorted_slot_n]
        sparse_event = [sparse_event_dict[bin_idx][k] if k in sparse_event_dict[bin_idx] else {} for k in sorted_slot_n ]

        sparse_list = bin_list[bin_idx].sparse_list
        assert len(sparse_flops) == len(sparse_list)
        assert len(sparse_cores) == len(sparse_list)
        # check have same keys
        for i in range(len(sparse_list)): 
            # has the same slot index
            assert sparse_list[i][0] == sorted_slot_n[i]
            # has the same task id
            assert set(sparse_list[i][1].keys()) == set(sparse_flops[i].keys())
            slot_n = sparse_list[i][0]
            assert set(sparse_list[i][1].keys()) >= set(sparse_event[i].keys())
        # sort the dict by key
        sparse_flops_dict[bin_idx] = OrderedDict(zip(sorted_slot_n, sparse_flops))
        sparse_cores_dict[bin_idx] = OrderedDict(zip(sorted_slot_n, sparse_cores))
        sparse_event_dict[bin_idx] = OrderedDict(zip(sorted_slot_n, sparse_event))
        bin_list[bin_idx].sparse_flops = list(zip(sorted_slot_n, sparse_flops))
        bin_list[bin_idx].sparse_cores = list(zip(sorted_slot_n, sparse_cores))
        bin_list[bin_idx].sparse_event = list(zip(sorted_slot_n, sparse_event))
        bin_list[bin_idx].alloc_mod = "proper"
    return sparse_flops_dict, sparse_cores_dict, sparse_event_dict

# event msg parser
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
    pattern = re.compile(r"^(?P<event_type>\w+)-(?P<pid>\d+)(_(?P<ts>\d+))?(_from_(?P<from>\d+))?(_to_(?P<to>\d+))?$")
    match = pattern.match(msg)
    # {k: t(v) for k,v,t in zip(pattern_keys, match.groups(), pattern_type) if v is not None}
    group_dict = match.groupdict()
    pattern_type = {"event_type":str, "pid":int, "ts":int, "from":int, "to":int}
    result = {k: t(v) for k,v,t in zip(group_dict.keys(), group_dict.values(), pattern_type.values()) if v is not None}
    return result

def update_sparse_dict(bin_idx, pid, s_i, sparse_dict, item, item_type):
    if s_i not in sparse_dict[bin_idx]:
        sparse_dict[bin_idx][s_i] = {}
    if hasattr(item_type, "update"):
        old = sparse_dict[bin_idx][s_i].get(pid, item_type())
        old.update(item)
        sparse_dict[bin_idx][s_i].update({pid:old})
    else:
        sparse_dict[bin_idx][s_i].update({pid:item})

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
            
def load_bin_list(bin_list_save_path, min_num_bins=-1):
    bin_list = load_pickle(bin_list_save_path)
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