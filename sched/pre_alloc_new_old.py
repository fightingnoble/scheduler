"""pre_alloc_new_old.py — moved dead code from pre_alloc_new.py (B3-MOVE-006/008 (reclassified unused→old)).
Contents: bin_select_new (dead wrapper of bin_sel, 0 callers) + commented test block (refs pre_alloc.py old fns).
Moved, not rewritten. Archive only — not importable/runnable. Recover: git checkout archive/test_pipeline-20260612 -- sched/pre_alloc_new.py
"""

from __future__ import annotations

from typing import List, Dict, Iterator, Callable
from collections import OrderedDict
import math
import numpy as np
import warnings

from global_var import *
from task.task_agent import ProcessInt
from task.task_agent import TaskInt
from model.lru import LRUCache
from model.task_queue_agent import TaskQueue
from sched.scheduling_table import SchedulingTableInt
from sched.sort_function import get_process_sort
from sched.bin_ops import sort_bin_list_EAT, sort_bin_list_by_barycenter
from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id
from sched.ref_alloc_search import TaskConstraints



# === B3-MOVE-006: bin_select_new (dead) ===
def bin_select_new(
        # request parameters
        _p:ProcessInt, n_slot:int, time_slot_s, time_slot_e, req_rsc_size,
        # sched components
        process_dict:Dict[int, ProcessInt], rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
        iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], 

        # sched parameters
        timestep, quantumSize, 
        binpack_cfg:Dict,
        preemption_list=list(),
        verbose:bool = False, DEBUG=False
        ):

    # --- 入口参数检查与读取 ---
    bin_sel_mod = binpack_cfg.get("bin_sel_mod", "search")
    if bin_sel_mod == "pre_defined":
        if 'mapping' not in binpack_cfg or binpack_cfg['mapping'] is None:
            raise KeyError("binpack_cfg['mapping'] must be provided when bin_sel_mod is 'pre_defined'")
        pid2bin_id:Dict[int, int] = binpack_cfg['mapping']
    # -----------------------

    # strategy: 
    # 1. the resource constraint should be respected
    # 2. the pre-defined resource preservation should be respected 
    # 3. the affinity settings of all the tasks should be respected 
    # 4. all tasks should be allocated with the resource
    # 5. tasks is expected to migrate as less as possible
    # 6. the resource should be allocated as compact as possible
    # 7. the resource should be allocated as balanced as possible

    if bin_sel_mod == "pre_defined":
        bin_id = pid2bin_id[_p.pid]
        affinity_tgt_bin_id_list = [bin_id,]
        affinity_search_bin_id_list = []
    else:
        affinity_tgt_bin_id_list, affinity_search_bin_id_list = bin_sel(_p, time_slot_s, time_slot_e, req_rsc_size, rsc_recoder_his, 
                                                                    bin_list, bin_name_list, timestep, binpack_cfg, process_dict)


    return affinity_tgt_bin_id_list, affinity_search_bin_id_list

# === B3-MOVE-008: commented test block (dead, references pre_alloc.py old fns) ===
# test code
# if __name__ == "__main__":
#     import argparse
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('case', type=str, help='case name')
#     args = argparser.parse_args()

#     # create a scheduling table
#     scheduling_table = SchedulingTableInt(30, 20, 0, "test")

#     # create a task set
#     # first branch: have free cores and free slots at beginning
#     # t1 [15, 19] 25 rsc and 2 slot
#     # t2 [3, 12] 24 rsc and 7 slot
#     # second branch: select a interval with enough free cores and free slots
#     # t3 [0, 20] 10 rsc and 4 slot
#     # thrird branch: evently distribute the resources in the expected interval
#     # t4 [0, 7] 10 rsc and 4 slot
#     # last branch: As soon as possible
#     # t5 [0, 16] 10 rsc and 8 slot
#     t1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
#                 ERT=15, ddl=19, period=30, exp_comp_t=2, 
#                 i_offset=0, jitter_max=0,
#                 flops=25*2*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
#     t2 = TaskInt(task_name="task2", task_id=2, task_flag="moveable", timing_flag="deadline",
#                 ERT=3, ddl=12, period=30, exp_comp_t=7,
#                 i_offset=0, jitter_max=0,
#                 flops=24*7*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
#     t3 = TaskInt(task_name="task3", task_id=3, task_flag="moveable", timing_flag="deadline", 
#                 ERT=0, ddl=20, period=30, exp_comp_t=4,
#                 i_offset=0, jitter_max=0,
#                 flops=10*4*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
#     t4 = TaskInt(task_name="task4", task_id=4, task_flag="moveable", timing_flag="deadline",
#                 ERT=0, ddl=7, period=30, exp_comp_t=4,
#                 i_offset=0, jitter_max=0,
#                 flops=10*4*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
#     t5 = TaskInt(task_name="task5", task_id=5, task_flag="moveable", timing_flag="deadline",
#                 ERT=0, ddl=16, period=30, exp_comp_t=8,
#                 i_offset=0, jitter_max=0,
#                 flops=10*8*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)

#     task_list:List[TaskInt] = [t1, t2, t3, t4, t5]

#     init_p_list = []
#     pid = 0
#     for _p in task_list[0:5]: 
#         # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
#         r = _p.get_release_time()
#         d = _p.get_deadline_time()
#         p = _p.make_process(r, d, pid)
#         p.remburst = p.task.flops
#         pid += 1
#         init_p_list.append(p)

#     # allocate resources
#     alloc_info = [None for i in range(10)]
#     require_rsc_size = [0 for i in range(10)]
#     require_rsc_size[0:5] = [25, 24, 10, 10, 10]

#     for i in range(4):
#         # print(f"task {i} allocation\n")
#         alloc_info[i] = scheduling_table.insert_task(init_p_list[i], require_rsc_size[i], 
#                                                      init_p_list[i].release_time, init_p_list[i].deadline, 
#                                                      init_p_list[i].exp_comp_t, verbose=False)
    
#     print("occupy by id\n:")
#     # scheduling_table.print_alloc_detail({_p.pid:_p.task.name for _p in init_p_list}, 1)
#     scheduling_table.print_scheduling_table({_p.pid:_p.task.name for _p in init_p_list}, 1)
    
#     # create a new process

#     timestep = 1
#     quantumSize = 1
#     rsc_recoder = {pid: (*info[1:], 0) for pid, info in zip(range(4), alloc_info)}
#     rsc_recoder_his = {pid: LRUCache() for pid in range(4)}
#     for lru in rsc_recoder_his.values():
#         lru.put(0)
    
#     quantum_check_en = False
#     return_all_occupant = False
#     key = lambda _p: _p.deadline
#     _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
#     n_slot = 13
#     strategy = "first_fit"
#     iter_next_bin_obj, bin_list, bin_name_list = iter([scheduling_table]), [scheduling_table], ["test"]
    
#     if args.case == "get_preempt_candi":
#         # test get_preempt_candi
#         time_slot_s, time_slot_e = 13, init_p_list[4].deadline
#         fail_info = get_preempt_candi(init_p_list[4], scheduling_table, 
#                     time_slot_s, time_slot_e, 
#                     timestep, FLOPS_PER_CORE, quantumSize, 
#                     rsc_recoder, _p_index_by_pid, key=key, quantum_check_en=quantum_check_en, 
#                     return_all_occupant=return_all_occupant) 
#         print(fail_info)
    
#     elif args.case == "check_and_alloc_at_queue":
#         # test check_and_alloc_at_queue
#         # expected slot number    
#         _p4 = init_p_list[4]
#         time_slot_s, time_slot_e, req_rsc_size = _p4.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)
#         expected_slot_num = time_slot_e-time_slot_s 
#         state, succ_info, fail_info = check_and_preemt_at_queue(_p4, scheduling_table, timestep, FLOPS_PER_CORE, 
#                                                 quantum_check_en, quantumSize, rsc_recoder, 
#                                                 time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
#                                                 _p_index_by_pid, key, return_all_occupant=strategy=="best_fit")
#         print(fail_info)
    
#     elif args.case == "bin_select": 
#         _p4 = init_p_list[4]
#         # expected rsc_size and slot number
#         time_slot_s, time_slot_e, req_rsc_size = _p4.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)

#         state, bin_id, succ_info, fail_info = bin_select(_p4, time_slot_s, time_slot_e, req_rsc_size, 
#                 init_p_list, 
#                 timestep, FLOPS_PER_CORE, 
#                 quantum_check_en, quantumSize, 
#                 rsc_recoder, rsc_recoder_his, 
#                 iter_next_bin_obj, bin_list, bin_name_list, 
#                 strategy,
#                 key)
#         print(fail_info)
    
