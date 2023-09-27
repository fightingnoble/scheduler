from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt, ProcessBase
    from typing import List, Callable, Dict, Iterator
    from model.lru import LRUCache

import numpy as np
import warnings
from global_var import *
from sched.sort_function import get_process_sort
from sched.scheduling_table import SchedulingTableInt, get_freespace_features

# =============== Bin related functions ===============

def del_bin(bin_list:List[SchedulingTableInt]):
    """
    del the bin which is empty
    """
    for _bin in bin_list:
        if _bin.is_empty():
            bin_list.remove(_bin)
            del _bin

def new_bin(spatial_size:int, temporal_size:int, id:int = 0, name:str = "bin"):
    SchedTab = SchedulingTableInt(spatial_size, temporal_size, id=id, name=name)
    return SchedTab

def bin_iter_descending_req_rsc_size(_new_bin:Callable, get_core_size:Callable, p_list:List[ProcessBase]):
    """
    generate the bin list in the descending order of the core size
    """        
    p_list = [(get_core_size(_p), _p) for _p in p_list]
    p_list.sort(key=lambda x: x[0], reverse=True)
    # TODO: bug here, name conflict
    yield from (_new_bin(bin_id,x[0],x[1].task.name) for bin_id, x in enumerate(p_list))

def bin_iter_list(_new_bin:Callable, size_list:List[int], name_list:List[str]): 
    """
    generate the bin list according to the size_list with no capacity limitation
    """ 
    assert len(size_list) == len(name_list)
    yield from (_new_bin(bin_id, size, name) for bin_id, (size, name) in enumerate(zip(size_list, name_list)))

def bin_iter_uniform_dist(_new_bin: Callable, max_core_size: int, size_list: List[int], name_list: List[str]):
    """
    generate the bin list according to the size_list until the max_core_size is reached
    """
    # index the size list whose cumsum is not larger than the max_core_size
    Cum_req_size = np.cumsum(size_list)
    if Cum_req_size[-1] > max_core_size:
        idx = (np.cumsum(size_list) > max_core_size).nonzero()[0][0]
        # distribute these spare resources to the first idx-1 bins
        if Cum_req_size[idx - 1] < max_core_size:
            size_list = []
            Cum_alloc_size = Cum_req_size * max_core_size / Cum_req_size[idx - 1]
            for _idx in range(idx):
                if _idx == 0:
                    size = int(Cum_alloc_size[_idx])
                    Cum_req_size[_idx] = size
                else:
                    size = int(Cum_alloc_size[_idx] - Cum_alloc_size[_idx - 1])
                    Cum_req_size[_idx] = size + Cum_req_size[_idx - 1]
                size_list.append(size)
        # yield from (_new_bin(bin_id, size, name) for bin_id, (size, name) in enumerate(zip(Cum_req_size[:idx], name_list[:idx])))
        return bin_iter_list(_new_bin, size_list[:idx], name_list[:idx])
    else:
        return bin_iter_list(_new_bin, size_list, name_list)

def get_size_mainPlusRDA(get_core_size:Callable, p_list:List[ProcessInt], RDA_ratio:float=1.2): 
    for _p in p_list:
        size_main = get_core_size(_p)
        task_type = _p.task.timing_flag
        # get RDA size
        if task_type == "deadline": 
            size_RDA = int(np.ceil(size_main * RDA_ratio))
        else:
            size_RDA = 0
        size = size_main + size_RDA

def sort_bin_list_EAT(_p:ProcessInt, time_slot_s, time_slot_e, timestep, _p_index_by_pid:Dict[int, ProcessInt],
                  bin_list:List[SchedulingTableInt], affinity_search_bin_id_list:List[int],
                  bin_name_list:List[str], rsc_recoder_his:Dict[int, LRUCache], get_process_sort=get_process_sort):
    """
        find the earliest bin that can provide enough resources
        feature:
        - free: slot_s, avil_unit, preemption: slot_s, avil_unit

    """
    bin_feature_list = []
    for _bin_id in affinity_search_bin_id_list:
        process_sort = get_process_sort([bin_name_list[_bin_id]], rsc_recoder_his)
        _bin = bin_list[_bin_id]
        preemptable_l = {}
        ordered_occupant_dict:Dict[int, List[int]] = _bin.index_occupy_by_id(time_slot_s, time_slot_e)
        for pid in ordered_occupant_dict:
            if process_sort(_p_index_by_pid[pid]) > process_sort(_p):
                alloc_slot_s_t, alloc_size_t, allo_slot_t = ordered_occupant_dict[pid]
                total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
                preemptable_l[pid] = [alloc_slot_s_t[0], total_alloc_unit_t]
        preemptable_units = sum([_unit for _s, _unit in preemptable_l.values()]) if preemptable_l else 0
        preemptable_start = min([_s for _s, _unit in preemptable_l.values()]) if preemptable_l else float("inf")
        
        aval_l = _bin.idx_free_by_slot(time_slot_s, time_slot_e, key=_p.pid)
        available_units = sum(aval_l)
        # index 1st non-zero element
        available_start = time_slot_s + np.nonzero(aval_l)[0][0] if available_units > 0 else float("inf")
        bin_feature_list.append([_bin_id, available_start, available_units, preemptable_start, preemptable_units])
    # sort the bin according to the feature
    itr = filter(lambda x: (x[2]+x[4])*timestep*FLOPS_PER_CORE>_p.remburst, sorted(bin_feature_list, key=lambda x: min(x[1], x[3])))
    affinity_search_bin_id_list = [x[0] for x in itr]
    return affinity_search_bin_id_list

def sort_bin_list_by_barycenter(_p:ProcessInt, time_slot_s, time_slot_e, timestep, _p_index_by_pid:Dict[int, ProcessInt],
                  bin_list:List[SchedulingTableInt], affinity_search_bin_id_list:List[int],
                  bin_name_list:List[str], rsc_recoder_his:Dict[int, LRUCache], get_process_sort=get_process_sort):
    """
        find the earliest bin that can provide enough resources
        feature:
        - free: slot_s, avil_unit, preemption: slot_s, avil_unit

    """
    bin_feature_list = []
    for _bin_id in affinity_search_bin_id_list:
        process_sort = get_process_sort([bin_name_list[_bin_id]], rsc_recoder_his)
        _bin = bin_list[_bin_id]

        rsc_avl = _bin.idx_free_by_slot(time_slot_s, time_slot_e, key=_p.pid)
        # divide the rsc_avl into intervals
        rsc_avl = np.array(rsc_avl)
        boader = (rsc_avl[0:-1] != rsc_avl[1:]).nonzero()[0] + 1
        s = [0] + boader.tolist() 
        e = boader.tolist() + [len(rsc_avl)] 
        slot_n = [rsc_avl[s[i]] for i in range(len(s))]
        chunk_s = [time_slot_s + s_i for s_i in s]
        preemptable_n = index_preeempt_num_cores_by_interval(_p, _bin, chunk_s, _p_index_by_pid, process_sort)

        tot_avl = np.array(slot_n) + np.array(preemptable_n)
        free_spaces = []
        for i in range(len(s)):
            if tot_avl[i] > 0:
                free_spaces.append([s[i], _bin.num_resources - tot_avl[i], e[i]-s[i], tot_avl[i]])

        if free_spaces == []:
            free_area = 0
            bary_x = float("inf")
            bary_y = float("inf")
            available_start = float("inf")
        else:
            free_area, bary_x, bary_y = get_freespace_features(free_spaces)
            bary_x = bary_x + time_slot_s
            # index 1st non-zero element
            available_start = free_spaces[0][0]+time_slot_s 
        bin_feature_list.append([_bin_id, available_start, free_area, bary_x, bary_y])
    # sort the bin according to the feature
    itr = filter(lambda x: x[2]*timestep*FLOPS_PER_CORE>_p.remburst, sorted(bin_feature_list, key=lambda x: (x[3], x[4]),))
    affinity_search_bin_id_list = [x[0] for x in itr]
    return affinity_search_bin_id_list

def index_preeempt_num_cores_by_interval(_p, _bin, chunk_s, _p_index_by_pid, process_sort):
    preemptable_n = [0 for i in range(len(chunk_s))]
    for i in range(len(chunk_s)):
        s_i = chunk_s[i]
        rsc_map = _bin.scheduling_table[s_i].rsc_map
        for pid in rsc_map:
            if process_sort(_p_index_by_pid[pid]) > process_sort(_p):
                preemptable_n[i] += rsc_map[pid]
    return preemptable_n

def get_initlist_and_biniter(bin_list, glb_p_list, total_cores, _new_bin, method="manual"):
    if method == "static_1_bin":
        iter_next_bin_obj,bin_name_list = static_1_bin(bin_list, glb_p_list, total_cores, _new_bin)
    elif method == "manual":
        iter_next_bin_obj,bin_name_list = manual_defined_reservation(bin_list, glb_p_list, total_cores, _new_bin)
    else:
        raise NotImplementedError
    return iter_next_bin_obj,bin_name_list

def manual_defined_reservation(bin_list, glb_p_list, total_cores, _new_bin):
    size_l = []
    name_l = []
    for _p in glb_p_list:
        if _p.task.pre_assigned_resource_flag:
            if _p.task.timing_flag == "deadline":
                size_l.append(_p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size)
            else:
                size_l.append(_p.task.pre_assigned_resource.main_size)
            name_l.append(_p.task.name)

    iter_next_bin_obj = bin_iter_uniform_dist(_new_bin, total_cores, size_l, name_l)
    bin_list.extend(list(iter_next_bin_obj)) 
    bin_name_list = [bin.name for bin in bin_list]
    return iter_next_bin_obj,bin_name_list

def static_1_bin(bin_list, glb_p_list, total_cores, _new_bin):
    size_l = [total_cores]
    name_l = ["bin"]
    iter_next_bin_obj = bin_iter_list(_new_bin, size_l, name_l)
    bin_list.extend(list(iter_next_bin_obj)) 
    bin_name_list = [bin.name for bin in bin_list]
    return iter_next_bin_obj,bin_name_list