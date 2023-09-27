from typing import List, Dict
from sched.scheduling_table import SchedulingTableInt
from sched.scheduling_table import Resource_model_int

class Monitor(object):

    def __init__(self, 
                 num_resources: int, 
                 hyper_period: int, 
                 id: int = None, name: str = None, 
                 ) -> None:
        self.trace_recoder = SchedulingTableInt(num_resources, 0, id, name, hyper_period)

    def get_trace(self) -> SchedulingTableInt:
        return self.trace_recoder
    
    def add_a_record(self, _record: Resource_model_int) -> None:
        self.trace_recoder.append(_record)

    def add_a_placehold_record(self, ) -> None:
        self.trace_recoder.append(Resource_model_int(self.trace_recoder.num_resources))

def get_target_bin_id(_p, bin_name_list, rsc_recoder_his):
    # find the target bin of the affinity target
    affinity_tgt_bin_id_list = []
    for task_n in _p.task.affinity_n:
        # suppose the target is pre-assigned with the resource but is not allocated
        if task_n in bin_name_list:
            affinity_tgt_bin_id_list.append(bin_name_list.index(task_n))
    # suppose the target was allocated with the resource
    for _pid in _p.task.affinity:
        if _pid in rsc_recoder_his:
            if not rsc_recoder_his[_pid].is_empty():
                preference = rsc_recoder_his[_pid].get_mru()
                if preference not in affinity_tgt_bin_id_list:
                    affinity_tgt_bin_id_list.append(preference)
    return affinity_tgt_bin_id_list

def get_rsc_2b_released(rsc_recoder, n_slot, _p):
    alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[_p.pid]
    if isinstance(alloc_slot_s_t, list):
        alloc_slot_s, alloc_size, allo_slot = [], [], []
        for i in range(len(alloc_slot_s_t)):
            if alloc_slot_s_t[i]+allo_slot_t[i] >= n_slot: 
                alloc_slot_s.append(alloc_slot_s_t[i] if alloc_slot_s_t[i] > n_slot else n_slot )
                alloc_size.append(alloc_size_t[i] )
                allo_slot.append(allo_slot_t[i] if alloc_slot_s_t[i] > n_slot else allo_slot_t[i]-(n_slot - alloc_slot_s_t[i]) )
    else:
        alloc_slot_s = alloc_slot_s_t if alloc_slot_s_t > n_slot else n_slot
        alloc_size = alloc_size_t
        allo_slot = allo_slot_t if alloc_slot_s_t > n_slot else allo_slot_t-(n_slot - alloc_slot_s_t)
    return bin_id_t,alloc_slot_s,alloc_size,allo_slot
