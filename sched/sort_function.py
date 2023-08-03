from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from model.lru import LRUCache

from typing import List, Dict
import numpy as np

from global_var import *
from sched.scheduling_table import SchedulingTableInt, get_freespace_features


# how does task affinity match with the existing bins
def get_target_bin_score(_p:ProcessInt, bin_name_list:List[str], rsc_recoder_his:Dict[int, LRUCache], reverse=True): 
    """
    measure how well the affinity target matches with the existing bins
    """
    pre_alloc_flg = False
    # case 1: task is pre-assigned with the resource
    p_name = _p.task.name
    if p_name in bin_name_list: 
        pre_alloc_flg = True
    
    # lvl 1: target is pre-assigned with the resource
    # i.e., "MultiCameraFusion": ["ImageBB"]
    affinity_tgt_bin_id_list = []
    for task_n, task_id in zip(_p.task.affinity_n, _p.task.affinity): 
        if task_n in bin_name_list:
            tgt_id = bin_name_list.index(task_n)
            if tgt_id not in affinity_tgt_bin_id_list:
                affinity_tgt_bin_id_list.append(tgt_id)
            else:
                affinity_tgt_bin_id_list.append(-1)
        else:
            affinity_tgt_bin_id_list.append(-1)

    # lvl 2: target was allocated with the resource
    # i.e., "Depth_estimation": ["Lane_drivable_area_det", "Optical_Flow"]
    preferred_bin_id_list = []
    for task_n, task_id in zip(_p.task.affinity_n, _p.task.affinity): 
        # case 3: suppose the target was allocated with the resource
        if task_id in rsc_recoder_his and task_n not in bin_name_list:
            if not rsc_recoder_his[task_id].is_empty():
                preference = rsc_recoder_his[task_id].get_mru()
                if preference not in affinity_tgt_bin_id_list and preference not in preferred_bin_id_list:
                    preferred_bin_id_list.append(preference)
                else:
                    preferred_bin_id_list.append(-1)
            else:
                preferred_bin_id_list.append(-1)
        else:
            preferred_bin_id_list.append(-1)

    # score function
    weight = 1/2**(np.arange(len(_p.task.affinity))+1) # 1/2, 1/4, 1/8, ...

    score0 = 1. if pre_alloc_flg else 0.
    score1 = np.sum(weight*(np.array(affinity_tgt_bin_id_list)!=-1))
    score2 = np.sum(weight*(np.array(preferred_bin_id_list)!=-1))
    
    if reverse:
        return (1-score0, 1-score1, 1-score2)
    return (score0, score1, score2)

def get_process_sort(bin_name_list, rsc_recoder_his):
    cond_fn1 = lambda x: x.deadline
    cond_fn2 = lambda x: get_target_bin_score(x, bin_name_list, rsc_recoder_his, reverse=True)
    def sort_fn(x):
        a = cond_fn1(x)
        b,c,d = cond_fn2(x)
        return (b,c,a,d,)
    return sort_fn


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
        preemptable_n = [0 for i in range(len(s))]
        # for s_i, e_i, slot_n_i in zip(s, e, slot_n):
        for i in range(len(s)):
            s_i, e_i, slot_n_i = s[i], e[i], slot_n[i]
            rsc_map = _bin.scheduling_table[time_slot_s + s_i].rsc_map
            for pid in rsc_map:
                if process_sort(_p_index_by_pid[pid]) > process_sort(_p):
                    preemptable_n[i] += rsc_map[pid]

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

