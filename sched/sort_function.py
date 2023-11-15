from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from model.lru import LRUCache

from typing import List, Dict
import numpy as np
from global_var import *


# how does task affinity match with the existing bins
def get_target_bin_score(_p:ProcessInt, bin_name_list:List[str], rsc_recoder_his:Dict[int, LRUCache], reverse=True): 
    """
    measure how well the affinity target matches with the existing bins, 
    reverse: 
        True, lower score means higher priority
        False, lower scores means lower priority
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

def get_process_sort(bin_name_list, rsc_recoder_his, ex_fn=None):
    """
        lower score means higher priority
    """
    cond_fn1 = lambda x: round(x.deadline, numerical_tol_bit)
    cond_fn2 = lambda x: get_target_bin_score(x, bin_name_list, rsc_recoder_his, reverse=True)
    def sort_fn(x):
        a = cond_fn1(x)
        b,c,d = cond_fn2(x)
        if ex_fn is not None: 
            return (b,c,a,d, *ex_fn(x))
        else:
            return (b,c,a,d,)
    return sort_fn

