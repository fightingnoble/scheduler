from __future__ import annotations

from typing import Union, List, Dict, Iterator, Callable, Union
from collections import OrderedDict
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

from global_var import *
from model.lru import LRUCache
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from task_agent import TaskInt
from model.task_queue_agent import TaskQueue 
from task_agent import ProcessInt, ProcessBase
from sched.monitor_agent import Monitor
import warnings

def glb_alloc_new(init_p_list, quantum_check_en, quantumSize, timestep, ready_queue, running_queue, rsc_recoder, 
                  rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, bin_list:TaskQueue, bin_name_list, n_slot, curr_t):
    # =================================================
    # push the ready task into the idle slot
    # input: 
    #   deadline (hard or expected), release time(real or expected)
    # output: 
    #   resource size, start time, end time
    # action: 
    #   try to allocate the enough resources for the task to finish before the deadline
    # strategy:
    #   timing constraint first: use just enough resources to finish the task before the deadline
    #   estimate the resource size according to the remaining operation and the relative deadline
    #   allow the task to execute even though the resource is not enough
    #   allow the lateness

    # A. rearange the task in the ready queue
    #   1. sort the task according to the deadline
    #   2. sort the task according to the affinity to the existing bins                        
    # B. calculate the affinity preference core
    # C. priorize the task that has firm affinity with the existing bins

    # rearange the task in the ready queue
    cond_fn1 = lambda x: x.deadline
    cond_fn2 = lambda x: get_target_bin_score(x, bin_name_list=bin_name_list, rsc_recoder_his=rsc_recoder_his, reverse=True)
    # sorted_ready_queue = sorted(ready_queue.queue, key=lambda x: (cond_fn1(x), cond_fn2(x)))
    # sort_fn = lambda x: (cond_fn2(x), cond_fn1(x))
    def sort_fn(x):
        a = cond_fn1(x)
        b,c,d = cond_fn2(x)
        return (b,c,a,d,)
    sorted_ready_queue = sorted(ready_queue.queue + running_queue.queue + issue_list.queue, key=sort_fn)

    for _p in sorted_ready_queue: 
        # check if the task is preempted
        if _p not in preempt_list and _p not in ready_queue.queue:
            assert _p in running_queue.queue or _p in issue_list.queue
            continue

        # TODO: Add the logic to ensure the task have allocated enough resource, otherwise, 
        #         we should not issue the task or compensate the resource latter. 

        # issue the task
        allocate_rsc_4_process_new(_p, n_slot, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list, 
                quantum_check_en, strategy='first_fit', glb_key=sort_fn, verbose=False, DEBUG=False)

    # update the running task
    if len(preempt_list):
        for _p in preempt_list:
            if _p in running_queue.queue:
                running_queue.remove(_p)
                ready_queue.put(_p)
        preempt_list.clear()

def allocate_rsc_4_process_new(_p:ProcessInt, n_slot:int, 
                init_p_list:List[ProcessInt], 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
                ready_queue:TaskQueue, running_queue:TaskQueue, 
                issue_list:TaskQueue, preempt_list:List[ProcessInt], 
                iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], 
                quantum_check_en:bool, strategy:str, glb_key:callable[[ProcessInt], int]=None,
                verbose:bool=False, DEBUG:bool=False):

    # expected rsc_size and slot number
    time_slot_s, time_slot_e, req_rsc_size = _p.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)
    expected_slot_num = time_slot_e - time_slot_s

    # try to push the task into the bins in the bin_list
    state, bin_id, succ_info, fail_info = bin_select(_p, time_slot_s, time_slot_e, req_rsc_size, 
                init_p_list, 
                timestep, FLOPS_PER_CORE, 
                quantum_check_en, quantumSize, 
                rsc_recoder, rsc_recoder_his, 
                iter_next_bin_obj, bin_list, bin_name_list, 
                strategy,
                glb_key)
    
    # case 1: is pre-assigned with the resource, 
    # - state is False and fail_info is None, but fail_info is not None
    # - state is True 
    # case 2: allocateable on free resources, state is True
    # case 3: allocateable by preempting some tasks, state is False, but fail_info is not None
    # case 4: create a new bin, state is False and fail_info is None, but bin_id is not -1
    # case 5: no resource is available
    if bin_id != -1:
        bin = bin_list[bin_id]
        if not state: 
            if fail_info is not None:
                # case 1, 3: allocateable by preempting some tasks
                flops_2b_preempt, ordered_occupant_dict, p_2b_realloc, total_FLOPS_occupied = fail_info
                
                # preempt the conflict tasks
                for _p_2b_preempt in p_2b_realloc:
                    pid = _p_2b_preempt.pid
                    # remove the task from the ready_queue.queue, preemptable_list, issue_list.queue
                    if _p_2b_preempt in issue_list:
                        issue_list.remove(_p_2b_preempt)
                        # update the rsc_recoder_his
                        rsc_recoder_his[pid].withdraw()
                        if len(rsc_recoder_his[pid].bk) == 0:
                            rsc_recoder_his.pop(pid)
                    elif _p_2b_preempt in ready_queue:
                        ready_queue.remove(_p_2b_preempt)
                    elif _p_2b_preempt in running_queue:
                        running_queue.remove(_p_2b_preempt)
                        if _p_2b_preempt.currentburst == 0: 
                             raise ValueError("A unexpected situation happens, task is not executed but in the running queue")
                    # mark for reallocation in current time slot
                    preempt_list.append(_p_2b_preempt)
                    
                    # pop the task from the bin
                    print(f"pop the task {_p_2b_preempt.task.id}:{_p_2b_preempt.task.name}({pid})from the bin {bin_id}")
                    # release all the tasks in the p_2b_preempt
                    alloc_slot_s_t, alloc_size_t, allo_slot_t = ordered_occupant_dict[pid]
                    bin_id_t, alloc_slot_s_t, alloc_size_t, allo_slot_t = get_rsc_2b_released(rsc_recoder, n_slot, _p_2b_preempt) 
                    bin.release(_p_2b_preempt, alloc_slot_s_t, alloc_size_t, allo_slot_t, verbose=False)
                    # update the rsc_recoder
                    rsc_recoder.pop(_p_2b_preempt.pid)

                    if _p_2b_preempt.currentburst != 0:
                        # task is in the running queue
                        # update the task status
                        _p_2b_preempt.task.preemption_count += 1
                        _p_2b_preempt.currentburst = 0 
                    
                    if strategy == "best_fit":
                        # alloc_slot_s_t, alloc_size_t, allo_slot_t = ordered_occupant_dict[pid]
                        total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
                        total_FLOPS_alloc_t = total_alloc_unit_t * timestep * FLOPS_PER_CORE
                        flops_2b_preempt -= total_FLOPS_alloc_t
                        if flops_2b_preempt <= 1e-2*timestep*FLOPS_PER_CORE:
                            break

            # case 1, 3, 4
            # allocate the current task
            state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc = \
                try_to_allocate(_p, bin, timestep, FLOPS_PER_CORE, 
                                time_slot_s, time_slot_e, 
                                req_rsc_size, expected_slot_num, )
            if not state:
                state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
                # bin.add_lock(_p, time_slot_s, time_slot_e)
                # Warning(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.pid}) is not allocated successfully after preemption or create a new bin")
                # raise ValueError(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.pid}) is not allocated successfully after preemption in its own bin")
        else:
            # case 1, 2
            alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc = succ_info
    else:
        state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None

        
    # print the allocation result
    if state:
        print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
        print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
        if not (isinstance(alloc_slot_s, list) and isinstance(alloc_size, list) and isinstance(allo_slot, list)):
            print(f"\tgot {total_alloc_unit:d} ({alloc_size:d} cores x {allo_slot:d} slots @ {alloc_slot_s:d}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
            _p.exp_comp_t = allo_slot * timestep
        elif len(alloc_slot_s) == len(alloc_size) == len(allo_slot) == 1:
            print(f"\tgot {total_alloc_unit:d} ({alloc_size[0]:d} cores x {allo_slot[0]:d} slots @ {alloc_slot_s[0]:d}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
            _p.exp_comp_t = allo_slot[0] * timestep
        else:
            alloc_slot_s_str = (r"{},"*len(alloc_slot_s)).format(*alloc_slot_s)
            alloc_size_str = (r"{},"*len(alloc_size)).format(*alloc_size)
            allo_slot_str = (r"{},"*len(allo_slot)).format(*allo_slot)
            print(f"\tgot {total_alloc_unit:d} ({alloc_size_str:s} cores x {allo_slot_str:s} slots @ {alloc_slot_s_str:s}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
            _p.exp_comp_t = np.sum(np.array(allo_slot)) * timestep
    elif DEBUG:
            print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
            print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
            print("\t[{:s}]\n".format("FAILED" if not state else "SUCCESS"))
    else:
        warnings.warn(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
        warnings.warn(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
        warnings.warn("\t[{:s}]\n".format("FAILED" if not state else "SUCCESS"))

    # record the allocation result and prepare the issue list
    if state:
        rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot, bin_id]
        issue_list.put(_p)
        if _p in preempt_list:
            preempt_list.remove(_p)
        if _p.pid in rsc_recoder_his:
            rsc_recoder_his[_p.pid].put(bin_id)
        else:
            rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)
    else:
        Warning("TASK {:d}:{:s}({:d}) IS DELAY ISSUED!!".format(_p.task.id, _p.task.name, _p.pid))

def bin_select(_p:ProcessInt, time_slot_s, time_slot_e, req_rsc_size,
                init_p_list:List[ProcessInt], 
                timestep, FLOPS_PER_CORE, 
                quantum_check_en, quantumSize, 
                rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
                iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], 
                strategy:str,
                glb_key:Callable[[ProcessInt], float]=None,):
 
    # strategy: 
    # 1. the resource constraint should be respected
    # 2. the pre-defined resource preservation should be respected 
    # 3. the affinity settings of all the tasks should be respected 
    # 4. all tasks should be allocated with the resource
    # 5. tasks is expected to migrate as less as possible
    # 6. the resource should be allocated as compact as possible
    # 7. the resource should be allocated as balanced as possible

    # initialize the resource request parameters
    p_name = _p.task.name
    expected_slot_num = time_slot_e-time_slot_s 
    _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
    # TODO: arange the bin_list according to the affinity of the class

    state, bin_id, succ_info, fail_info = False, -1, None, None

    # 2. the pre-defined resource preservation should be respected 
    #   For the task that is pre-assigned with the resource, the affinity is set to be itself
    if p_name in bin_name_list:
        bin_id = bin_name_list.index(p_name)
        state, succ_info, fail_info = check_and_alloc_at_queue(_p, bin_list[bin_id], timestep, FLOPS_PER_CORE, 
                                    quantum_check_en, quantumSize, rsc_recoder, 
                                    time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
                                    _p_index_by_pid, glb_key, return_all_occupant=False)

    else: 

        # index free resources in each bins
        # rsc_avl_list = [sum(_bin.idx_free_by_slot(time_slot_s, time_slot_e, key=_p.pid)) for _bin in bin_list]

        affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)

        # mark other bins as the targets of the search
        affinity_search_bin_id_list = [n for n in range(len(bin_list)) if n not in affinity_tgt_bin_id_list] 
        # arrange the targets of the search in the order of the fitness of the size
        affinity_search_bin_id_list.sort(key=lambda x: abs(bin_list[x].num_resources - req_rsc_size))

        # 3. the affinity settings of all the tasks should be respected 
        # try to find bin to fit the task
        # strategy:
        # - first fit
        # - best fit (TODO)
        fail_info_list = []
        for bin_id in affinity_tgt_bin_id_list + affinity_search_bin_id_list: 
            # rearange the task in the ready queue
            cond_fn1 = lambda x: x.deadline
            cond_fn2 = lambda x: get_target_bin_score(x, bin_name_list=[bin_name_list[bin_id]], rsc_recoder_his=rsc_recoder_his, reverse=True)
            # sorted_ready_queue = sorted(ready_queue.queue, key=lambda x: (cond_fn1(x), cond_fn2(x)))
            # sort_fn = lambda x: (cond_fn2(x), cond_fn1(x))
            def sort_fn(x):
                a = cond_fn1(x)
                b,c,d = cond_fn2(x)
                return (b,c,a,d,)
                # return (b,a,c,d,)
                # return (b,a, )

            state, succ_info, fail_info = check_and_alloc_at_queue(_p, bin_list[bin_id], timestep, FLOPS_PER_CORE, 
                                        quantum_check_en, quantumSize, rsc_recoder, 
                                        time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
                                        _p_index_by_pid, key=sort_fn, return_all_occupant=strategy=="best_fit")
            if state: 
                break
            elif fail_info is not None:
                if strategy == "first_fit": 
                    break
                elif strategy == "best_fit":
                    fail_info_list.append(fail_info)
        
        if strategy == "first_fit" and not state and fail_info is None:
            bin_id = -1
            fail_info = None
        
        # TODO: implement the best fit strategy
        if strategy == "best_fit": 
            if len(fail_info) > 0:
                # min flops_2b_preempt-total_FLOPS_occupied
                fail_info.sort(key=lambda x: abs(x[0]-x[4]))
                raise NotImplementedError("The best fit strategy is not implemented yet")
            else:
                bin_id = -1
                
        # check the state; if the task cannot be pushed into any bin, create a new bin
        if bin_id == -1:
            try:
                bin = next(iter_next_bin_obj)
                bin_id = bin.id
            except StopIteration:
                warnings.warn("No more bin can be created")
    return state, bin_id, succ_info, fail_info

def check_and_alloc_at_queue(_p, bin:SchedulingTableInt, timestep, FLOPS_PER_CORE, 
                             quantum_check_en, quantumSize, rsc_recoder, 
                             time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
                             _p_index_by_pid, key, return_all_occupant, 
                             partial_alloc_en=False, partial_preempt_en=False,
                             verbose=False):
    # try to allocate the resource on the bin
    # check for free resources
    state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc = \
        try_to_allocate(_p, bin, timestep, FLOPS_PER_CORE, 
                        time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
                        partial_alloc_en=partial_alloc_en, verbose=verbose)
    # check for allocated resources
    if not state:
        # reset the state
        state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
        total_alloc_unit, total_FLOPS_alloc = 0, 0
        fail_info = get_preempt_candi(_p, bin, 
                time_slot_s, time_slot_e, 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder, _p_index_by_pid, key=key, 
                quantum_check_en=quantum_check_en, 
                return_all_occupant=return_all_occupant) 
        flops_2b_preempt, ordered_occupant_dict, p_2b_realloc, total_FLOPS_occupied = fail_info
        # check if the preempted candidates can provide enough resources
        if flops_2b_preempt-total_FLOPS_occupied > 1e-2*timestep*FLOPS_PER_CORE: 
            # if partial allocation is allowed and the preempted candidates can provide more than
            # 1 unit of the resource, i.e. timestep*FLOPS_PER_CORE, then the task can be allocated
            if not partial_preempt_en and timestep*FLOPS_PER_CORE - total_FLOPS_occupied > 1e-2*timestep*FLOPS_PER_CORE:
                fail_info = None
    else:
        fail_info = None
    
    succ_info = (alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc)
    return state, succ_info, fail_info

def try_to_allocate(_p, bin: SchedulingTableInt, timestep, FLOPS_PER_CORE,
                    time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
                    partial_alloc_en:bool = False, verbose:bool = False):
    state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
    total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
    total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
    # check if the task is allocated successfully
    if state and (total_FLOPS_alloc >= _p.remburst): 
        print(f"task {_p.task.id}:{_p.task.name}({_p.pid}) is allocated successfully in the bin {bin.id}")
        if bin.locker == _p.pid:
            bin.release_lock(_p, time_slot_s, time_slot_e)
    # check tatal FLOPS allocated if partial_alloc_en is disabled
    elif (0 < total_FLOPS_alloc < _p.remburst) and partial_alloc_en:
        Warning("The allocated FLOPS is not enough for the task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
    else: 
        # In this condition, some tasks are not preempted successfully, and the current task is blocked by these tasks, 
        # the current task will be delayed to issue
        # release the resource
        bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
        # reset the state
        state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
    return state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc

def get_preempt_candi(_p:ProcessInt, bin:SchedulingTableInt, 
        time_slot_s:int, time_slot_e:int, 
        timestep, FLOPS_PER_CORE, quantumSize, 
        rsc_recoder:dict, _p_index_by_pid:Dict[int, ProcessInt], 
        key:Callable[[ProcessInt], int]=lambda _p: _p.deadline, 
        quantum_check_en:bool = True, return_all_occupant:bool = False):
    """
    Preempt the conflict tasks and reallocate the resources to the current task
    quantum_check_en:
        disable the quantum check in the stage of preallocation
    """
    p_2b_realloc:List[ProcessInt] = []
    bin_id = bin.id
    # find out the conflict tasks
    ordered_occupant_dict:Dict[int, List[int]] = bin.index_occupy_by_id(time_slot_s, time_slot_e)

    # decide which task to preempt
    #   note that here is an assumption that if the current bin is designed for the task, 
    #   all the candidate tasks are preempted
    if key is not None:
        # sort the conflict tasks by their priority, in accending order
        ordered_occupant_dict = OrderedDict(sorted(ordered_occupant_dict.items(), key=lambda item: key(_p_index_by_pid[item[0]]), reverse=True))
        # filter the tasks with lower priority
        ordered_occupant_dict = OrderedDict(filter(lambda item: key(_p_index_by_pid[item[0]]) > key(_p), ordered_occupant_dict.items()))
    occupation_candi = list(ordered_occupant_dict.keys())

    aval_flops = sum(bin.idx_free_by_slot(time_slot_s, time_slot_e, _p.pid)) * timestep * FLOPS_PER_CORE
    # flops_2b_preempt = (time_slot_e - time_slot_s) * bin.scheduling_table[0].size * timestep * FLOPS_PER_CORE - aval_flops
    flops_2b_preempt = _p.remburst - aval_flops
    
    # evaluate the preemption possibility
    # if the preemptable conflict task can provide enough FLOPS to the current task
    total_FLOPS_occupied = 0
    for pid in occupation_candi:
        _p_2b_preempt = _p_index_by_pid[pid]
        _, _, _, bin_id_t = rsc_recoder[pid]
        assert bin_id == bin_id_t
        # task is runnning but has executed for an integer multiples of the quantum size (control the pre-emption grain)
        if quantum_check_en:
            cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
            reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
            if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                continue
        else:
            reach_preempt_grain = True

        alloc_slot_s_t, alloc_size_t, allo_slot_t = ordered_occupant_dict[pid]
        total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
        total_FLOPS_alloc_t = total_alloc_unit_t * timestep * FLOPS_PER_CORE
        total_FLOPS_occupied += total_FLOPS_alloc_t
        p_2b_realloc.append(_p_2b_preempt)
        if not return_all_occupant:
            if flops_2b_preempt-total_FLOPS_occupied <= 1e-2*timestep*FLOPS_PER_CORE: 
                break
    return flops_2b_preempt, ordered_occupant_dict, p_2b_realloc, total_FLOPS_occupied, 

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
            preference = rsc_recoder_his[_pid].get_mru()
            if preference not in affinity_tgt_bin_id_list:
                affinity_tgt_bin_id_list.append(preference)
    return affinity_tgt_bin_id_list

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
            preference = rsc_recoder_his[task_id].get_mru()
            if preference not in affinity_tgt_bin_id_list and preference not in preferred_bin_id_list:
                preferred_bin_id_list.append(preference)
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

def get_rsc_2b_released(rsc_recoder, n_slot, _p):
    alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[_p.pid]
    if isinstance(alloc_slot_s_t, list):
        alloc_slot_s, alloc_size, allo_slot = [], [], []
        for i in range(len(alloc_slot_s_t)):
            if alloc_slot_s_t[i]+allo_slot_t[i] > n_slot: 
                alloc_slot_s.append(alloc_slot_s_t[i] if alloc_slot_s_t[i] > n_slot else n_slot )
                alloc_size.append(alloc_size_t[i] )
                allo_slot.append(allo_slot_t[i] if alloc_slot_s_t[i] > n_slot else allo_slot_t[i]-(n_slot - alloc_slot_s_t[i]) )
    else:
        alloc_slot_s = alloc_slot_s_t if alloc_slot_s_t > n_slot else n_slot
        alloc_size = alloc_size_t
        allo_slot = allo_slot_t if alloc_slot_s_t > n_slot else allo_slot_t-(n_slot - alloc_slot_s_t)
    return bin_id_t,alloc_slot_s,alloc_size,allo_slot

# test code
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('case', type=str, help='case name')
    args = argparser.parse_args()

    # create a scheduling table
    scheduling_table = SchedulingTableInt(30, 20, 0, "test")

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
    t1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                ERT=15, ddl=19, period=30, exp_comp_t=2, 
                i_offset=0, jitter_max=0,
                flops=25*2*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t2 = TaskInt(task_name="task2", task_id=2, task_flag="moveable", timing_flag="deadline",
                ERT=3, ddl=12, period=30, exp_comp_t=7,
                i_offset=0, jitter_max=0,
                flops=24*7*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t3 = TaskInt(task_name="task3", task_id=3, task_flag="moveable", timing_flag="deadline", 
                ERT=0, ddl=20, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=10*4*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t4 = TaskInt(task_name="task4", task_id=4, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=7, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=10*4*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t5 = TaskInt(task_name="task5", task_id=5, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=16, period=30, exp_comp_t=8,
                i_offset=0, jitter_max=0,
                flops=10*8*FLOPS_PER_CORE, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)

    task_list:List[TaskInt] = [t1, t2, t3, t4, t5]

    init_p_list = []
    pid = 0
    for task in task_list[0:5]: 
        # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
        r = task.get_release_time()
        d = task.get_deadline_time()
        p = task.make_process(r, d, pid)
        p.remburst = p.task.flops
        pid += 1
        init_p_list.append(p)

    # allocate resources
    alloc_info = [None for i in range(10)]
    require_rsc_size = [0 for i in range(10)]
    require_rsc_size[0:5] = [25, 24, 10, 10, 10]

    for i in range(4):
        # print(f"task {i} allocation\n")
        alloc_info[i] = scheduling_table.insert_task(init_p_list[i], require_rsc_size[i], 
                                                     init_p_list[i].release_time, init_p_list[i].deadline, 
                                                     init_p_list[i].exp_comp_t, verbose=False)
    
    print("occupy by id\n:")
    # scheduling_table.print_alloc_detail({_p.pid:_p.task.name for _p in init_p_list}, 1)
    scheduling_table.print_scheduling_table({_p.pid:_p.task.name for _p in init_p_list}, 1)
    
    # create a new process

    timestep = 1
    quantumSize = 1
    rsc_recoder = {pid: (*info[1:], 0) for pid, info in zip(range(4), alloc_info)}
    rsc_recoder_his = {pid: LRUCache() for pid in range(4)}
    for lru in rsc_recoder_his.values():
        lru.put(0)
    
    quantum_check_en = False
    return_all_occupant = False
    key = lambda _p: _p.deadline
    _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
    n_slot = 13
    strategy = "first_fit"
    iter_next_bin_obj, bin_list, bin_name_list = iter([scheduling_table]), [scheduling_table], ["test"]
    
    if args.case == "get_preempt_candi":
        # test get_preempt_candi
        time_slot_s, time_slot_e = 13, init_p_list[4].deadline
        fail_info = get_preempt_candi(init_p_list[4], scheduling_table, 
                    time_slot_s, time_slot_e, 
                    timestep, FLOPS_PER_CORE, quantumSize, 
                    rsc_recoder, _p_index_by_pid, key=key, quantum_check_en=quantum_check_en, 
                    return_all_occupant=return_all_occupant) 
        print(fail_info)
    
    elif args.case == "check_and_alloc_at_queue":
        # test check_and_alloc_at_queue
        # expected slot number    
        _p4 = init_p_list[4]
        time_slot_s, time_slot_e, req_rsc_size = _p4.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)
        expected_slot_num = time_slot_e-time_slot_s 
        state, succ_info, fail_info = check_and_alloc_at_queue(_p4, scheduling_table, timestep, FLOPS_PER_CORE, 
                                                quantum_check_en, quantumSize, rsc_recoder, 
                                                time_slot_s, time_slot_e, req_rsc_size, expected_slot_num, 
                                                _p_index_by_pid, key, return_all_occupant=strategy=="best_fit")
        print(fail_info)
    
    elif args.case == "bin_select": 
        _p4 = init_p_list[4]
        # expected rsc_size and slot number
        time_slot_s, time_slot_e, req_rsc_size = _p4.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)

        state, bin_id, succ_info, fail_info = bin_select(_p4, time_slot_s, time_slot_e, req_rsc_size, 
                init_p_list, 
                timestep, FLOPS_PER_CORE, 
                quantum_check_en, quantumSize, 
                rsc_recoder, rsc_recoder_his, 
                iter_next_bin_obj, bin_list, bin_name_list, 
                strategy,
                key)
        print(fail_info)
    