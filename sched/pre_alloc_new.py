from __future__ import annotations

from typing import List, Dict, Iterator, Callable
from collections import OrderedDict
import math
import numpy as np
import warnings, copy

from utils import Found
from global_var import *
from task.task_agent import ProcessInt
from task.task_agent import TaskInt
from model.lru import LRUCache
from model.task_queue_agent import TaskQueue 
from sched.scheduling_table import SchedulingTableInt
from sched.sort_function import get_process_sort
from sched.bin_ops import sort_bin_list_EAT, sort_bin_list_by_barycenter, index_preeempt_num_cores_by_interval
from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id
def glb_alloc_new2(process_dict, quantumSize, timestep, 
                    ready_queue, running_queue, rsc_recoder, 
                    rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, 
                    bin_list:TaskQueue, bin_name_list, n_slot, curr_t, 
                    binpack_cfg:Dict,
                    show_warnings=True, 
                    verbose:bool=False, DEBUG_FG:bool=False,
                  ):
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
    process_sort = get_process_sort(bin_name_list, rsc_recoder_his)
    sorted_ready_l = ready_queue.queue + running_queue.queue + issue_list.queue
    sorted_ready_queue = TaskQueue(sorted_ready_l, descending=False, sort_f=process_sort)
    del sorted_ready_l

    # mechanism: 
    #   Partial allocation is not allowed, i.e., the task is allocated to the whole cores or none.
    #   Each task only try once; the tasks already allocated are skipped;
    #   the tasks are preempted are given an extra opportunities.
    # TODO: Allow partial allocation and add the logic to ensure the task have allocated enough resource, otherwise, 
    #         we should not issue the task or compensate the resource latter. 
    # for _p in sorted_ready_queue: 
    while len(sorted_ready_queue):
        # Each task only try once;
        _p = sorted_ready_queue.queue.pop(0)
        # the tasks already allocated are skipped;
        if _p not in ready_queue.queue:
            assert _p in running_queue.queue or _p in issue_list.queue
            continue

        # try to allocate and preempt
        state = allocate_rsc_4_process_new2(
            # request parameters
            _p, n_slot,
            # sched components
            process_dict, rsc_recoder, rsc_recoder_his,
            iter_next_bin_obj, bin_list, bin_name_list,
            # sched parameters
            timestep, quantumSize,
            binpack_cfg,
            preempt_list,
            show_warnings=show_warnings,
            verbose=verbose, DEBUG=DEBUG_FG,
        )
        # the tasks are preempted are given an extra opportunities
        for _p_2b_preempt in preempt_list:
            pid = _p_2b_preempt.pid
            # remove the task from the ready_queue.queue, preemptable_list, issue_list.queue
            if _p_2b_preempt in issue_list:
                issue_list.remove(_p_2b_preempt)
                # update the rsc_recoder_his
                rsc_recoder_his[pid].withdraw()
                if len(rsc_recoder_his[pid].bk) == 0:
                    rsc_recoder_his.pop(pid)
            elif _p_2b_preempt in running_queue:
                running_queue.remove(_p_2b_preempt)
                if _p_2b_preempt.currentburst == 0: 
                    raise ValueError(f"A unexpected situation happens, task {_p_2b_preempt.task.name} is not executed but in the running queue")
            else:
                raise ValueError("A unexpected situation happens, preempted task is not in the running queue or issue list")

            # update the rsc_recoder
            rsc_recoder.pop(_p_2b_preempt.pid)
            # rsc_recoder_his[_p_2b_preempt.pid].put_neg(bin_id, rsc_recoder[_p_2b_preempt.pid])

            if _p_2b_preempt.currentburst != 0:
                # task is in the running queue
                # update the task status
                _p_2b_preempt.task.preemption_count += 1
                _p_2b_preempt.currentburst = 0 

            ready_queue.put(_p_2b_preempt)
            if _p_2b_preempt not in sorted_ready_queue.queue:
                sorted_ready_queue.put(_p_2b_preempt)
        preempt_list.clear()

        # issue the task allocated successfully
        if state:
            issue_list.put(_p)
            assert _p in ready_queue.queue
            ready_queue.queue.remove(_p)

def allocate_rsc_4_process_new2(
        # request parameters
        _p:ProcessInt, n_slot:int, 
        # sched components
        process_dict:Dict[int, ProcessInt], rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
        iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], 

        # sched parameters
        timestep, quantumSize, 
        binpack_cfg:Dict,
        preemption_list:List[ProcessInt],
        show_warnings=True, 
        verbose:bool=False, DEBUG:bool=False,
        ):

    # expected rsc_size and slot number
    time_slot_s, time_slot_e, req_rsc_size = _p.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)
    if time_slot_s >= time_slot_e:
        return False
    expected_slot_num = time_slot_e - time_slot_s

    # try to push the task into the bins in the bin_list
    state, bin_id, succ_info = bin_select_new(
        _p, n_slot, time_slot_s, time_slot_e, req_rsc_size, 
        process_dict, rsc_recoder, rsc_recoder_his, 
        iter_next_bin_obj, bin_list, bin_name_list, 
        timestep, quantumSize, 
        binpack_cfg,
        preemption_list, 
        verbose=verbose, DEBUG=DEBUG
        )


    if state:
        alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc = succ_info
    
        # record the allocation result and prepare the issue list
        rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot, bin_id]
        if _p.pid not in rsc_recoder_his:
            rsc_recoder_his[_p.pid] = LRUCache(3)
        rsc_recoder_his[_p.pid].put(bin_id, [alloc_slot_s, alloc_size, allo_slot])

        # print the allocation result
        if verbose:
            print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
            print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
            # alloc_slot_s_str = (r"{},"*len(alloc_slot_s)).format(*alloc_slot_s)
            # alloc_size_str = (r"{},"*len(alloc_size)).format(*alloc_size)
            # allo_slot_str = (r"{},"*len(allo_slot)).format(*allo_slot)
            alloc_slot_s_str = ",".join([f"{s:d}" for s in alloc_slot_s])
            alloc_size_str = ",".join([f"{s:d}" for s in alloc_size])
            allo_slot_str = ",".join([f"{s:d}" for s in allo_slot])
            print(f"\tgot {total_alloc_unit:d} ({alloc_size_str:s} cores x {allo_slot_str:s} slots @ {alloc_slot_s_str:s}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
    else:
        if DEBUG:
            print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
            print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
            print("\t[{:s}]\n".format("FAILED" if not state else "SUCCESS"))
        if show_warnings:
            Warning("TASK {:d}:{:s}({:d}) IS DELAY ISSUED!!".format(_p.task.id, _p.task.name, _p.pid))
    return state

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
    # _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
    _p_index_by_pid = process_dict
    # TODO: arange the bin_list according to the affinity of the class

    state, bin_id, succ_info, fail_info = False, -1, None, None

    # 2. the pre-defined resource preservation should be respected 
    #   For the task that is pre-assigned with the resource, the affinity is set to be itself
    if p_name in bin_name_list:
        bin_id = bin_name_list.index(p_name)
        process_sort = get_process_sort([p_name], rsc_recoder_his)
        state, succ_info = check_and_preemt_alloc(_p, n_slot, bin_list[bin_id],
                                                time_slot_s, time_slot_e, timestep,  
                                                rsc_recoder, _p_index_by_pid,
                                                req_rsc_size, expected_slot_num, 
                                                quantumSize,
                                                binpack_cfg,
                                                process_sort,
                                                preemption_list,
                                                verbose, DEBUG
        )

    else: 

        # index free resources in each bins
        # rsc_avl_list = [sum(_bin.idx_free_by_slot(time_slot_s, time_slot_e, key=_p.pid)) for _bin in bin_list]

        affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)
        # if _p.pid in rsc_recoder_his:
        #     hate_bin_id_list = list(rsc_recoder_his[_p.pid].dict_neg.keys())
        # else:
        #     hate_bin_id_list = []

        # mark other bins as the targets of the search
        affinity_search_bin_id_list = [n for n in range(len(bin_list)) if n not in affinity_tgt_bin_id_list] 
        
        # sort the bin according to the feature
        # - free: slot_s, avil_unit, preemption: slot_s, avil_unit
        bin_sort = binpack_cfg.get("sort", "EAT")
        if bin_sort == "EAT":
            bin_sort_fn = sort_bin_list_EAT
            affinity_search_bin_id_list = bin_sort_fn(_p, time_slot_s, time_slot_e, timestep, _p_index_by_pid,
                                                        bin_list, affinity_search_bin_id_list,
                                                        bin_name_list, rsc_recoder_his)
        elif bin_sort == "barycenter":
            bin_sort_fn = sort_bin_list_by_barycenter
            affinity_search_bin_id_list = bin_sort_fn(_p, time_slot_s, time_slot_e, timestep, _p_index_by_pid,
                                                        bin_list, affinity_search_bin_id_list,
                                                        bin_name_list, rsc_recoder_his)
        elif bin_sort == "bf":
            # arrange the targets of the search in the order of the fitness of the size
            affinity_search_bin_id_list.sort(key=lambda x: abs(bin_list[x].num_resources - req_rsc_size))
        else:
            # arrange the targets of the search in the order of the reverse fitness of the size
            affinity_search_bin_id_list.sort(key=lambda x: abs(bin_list[x].num_resources - req_rsc_size), reverse=True)


        # 3. the affinity settings of all the tasks should be respected 
        # try to find bin to fit the task
        for bin_id in affinity_tgt_bin_id_list + affinity_search_bin_id_list: 
            # rearange the task in the ready queue
            process_sort = get_process_sort([bin_name_list[bin_id]], rsc_recoder_his)
            state, succ_info = check_and_preemt_alloc(_p, n_slot, bin_list[bin_id],
                                                    time_slot_s, time_slot_e, timestep,  
                                                    rsc_recoder, _p_index_by_pid,
                                                    req_rsc_size, expected_slot_num, 
                                                    quantumSize,
                                                    binpack_cfg,
                                                    process_sort,
                                                    preemption_list,
                                                    verbose, DEBUG
            )
            if state: 
                break
            else:
                bin_id = -1
        
        if not state:
            try:
                bin = next(iter_next_bin_obj)
                bin_id = bin.id
            except StopIteration:
                warnings.warn("No more bin can be created")
    return state, bin_id, succ_info

def push_into_bin(_p, bin:SchedulingTableInt, 
                  time_slot_s, req_rsc_size, expected_slot_num, preempt_en,
                  mode, rsc_avl, s, e, size, expected_req_rsc_size, 
                  verbose, DEBUG):
    policy = 'N/A'
    state, chunk_s, chunk_len = bin.block_insert(rsc_avl, expected_req_rsc_size, req_rsc_size, 
                                                                expected_slot_num, preempt_en, "asap", verbose, DEBUG)
    if state:
        policy = 'block' 
        chunk_s = [s+time_slot_s for s in chunk_s]
        curr_alloc = [req_rsc_size for _ in range(len(chunk_s))]
    elif mode != "block":
            # current allocation (C)
        curr_alloc = np.zeros(len(s), dtype=int)
        chunk_len = np.zeros(len(s), dtype=int)
        if rsc_avl[:expected_slot_num].sum() < expected_req_rsc_size: 
                # as soon as possible
            policy = 'asap'
            bin.asap_insert(_p, DEBUG, s, e, size, curr_alloc, chunk_len, expected_req_rsc_size)
        else: 
            if DEBUG:
                print("Enough resources: as evenly as possible")
            policy = 'aeap'
            bin.aeap_insert(_p, expected_slot_num, DEBUG, s, e, size, curr_alloc, chunk_len, expected_req_rsc_size)
            # allocate resources
        idx = chunk_len.nonzero()[0]
            # return True, np.array(s)[idx].tolist(), curr_slot[idx].tolist()
        chunk_s = (time_slot_s+np.array(s)[idx]).tolist()
        curr_alloc = curr_alloc[idx].tolist()
        chunk_len = chunk_len[idx].tolist()
    else:
        policy, chunk_s, curr_alloc, chunk_len = 'N/A', [], [], []
    return policy,chunk_s,curr_alloc,chunk_len

def check_and_preemt_alloc(_p:ProcessInt, n_slot:int, bin:SchedulingTableInt,
        time_slot_s:int, time_slot_e:int, timestep,  
        rsc_recoder:dict, _p_index_by_pid:Dict[int, ProcessInt],
        req_rsc_size, expected_slot_num, 
        quantumSize,
        binpack_cfg:Dict,
        key:Callable[[ProcessInt], int]=lambda _p: _p.deadline,
        preemption_list=list(),
        verbose:bool = False, DEBUG=False):

    mode = binpack_cfg["mode"]
    quantum_check_en = binpack_cfg.get("quantum_check_en", False)
    partial_alloc_en = binpack_cfg.get("partial_alloc_en", False)
    release_temp_rda = binpack_cfg.get("release_temp_rda", True)
    preempt_en = binpack_cfg.get("preempt_en", True)
    assert mode in ["non-block", "block"]
    rsc_avl = bin.idx_free_by_slot(time_slot_s, time_slot_e, key=_p.pid)
    rsc_avl = np.array(rsc_avl)

    # allow sharing or not
    if mode == "block":
        # not allow sharing, the slot in used (i.e., < bin.num_resources) is not available
        rsc_avl[rsc_avl < bin.num_resources] = 0

    
    s, e, size = bin.interval_sparsifier(rsc_avl)
    
    chunk_s = [time_slot_s + s_i for s_i in s]
    chunk_e = [time_slot_s + e_i for e_i in e]

    # get preemptable_map: {pid: [chunk_s, size, chunk_len]}, resource map of the preemptable candidates
    # get preemptable_n: the number of the cores in each chunk
    preemptable_map, preemptable_n = index_occupy_by_id_chunk_ver(_p, bin, chunk_s, chunk_e, _p_index_by_pid, 
                                                                  key, quantum_check_en, quantumSize, rsc_recoder)
    # get the total size occupied by the preemptable tasks
    tot_avl = np.array(size) + np.array(preemptable_n)
    free_spaces = []
    for i in range(len(s)):
        if tot_avl[i] > 0:
            free_spaces.append([s[i], bin.num_resources - tot_avl[i], e[i]-s[i], tot_avl[i]])
    free_area = sum(w*h for x,y,w,h in free_spaces)

    # required (R)
    expected_req_rsc_size = req_rsc_size * expected_slot_num
    remburst_req_rsc_size = _p.remburst / timestep / FLOPS_PER_CORE
    # policy: block, asap, all, aeap, N/A
    policy = 'N/A'

    if free_area == 0:
        state, alloc_s, alloc_size, alloc_len = False, [], [], []
    elif 0 < free_area < expected_req_rsc_size and mode != "block":
        if remburst_req_rsc_size <= free_area or partial_alloc_en:
            policy = 'all'
            idx = np.where(size)[0]
            # (time_slot_s+np.array(s)[idx]).tolist(), slot_n[idx].tolist(), (e[idx]-s[idx]).tolist()
            alloc_s = [time_slot_s+s[i] for i in idx]
            alloc_size = [size[i] for i in idx]
            alloc_len = [(e[i]-s[i]) for i in idx]
            Warning("The allocated FLOPS is not enough for the task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
    else:
        rsc_tot = np.zeros_like(rsc_avl)
        for i in range(len(s)):
            rsc_tot[s[i]:e[i]] = tot_avl[i]
        policy, alloc_s, alloc_size, alloc_len = push_into_bin(
            _p, bin, 
            time_slot_s, req_rsc_size, expected_slot_num, preempt_en,
            mode, rsc_tot, s, e, size, expected_req_rsc_size, 
            verbose, DEBUG
            )
        # iterative preempt the tasks until tot_avl can cover [chunk_s, curr_alloc, chunk_len]
        # compare three elements:
        #   1. chunk of the current allocation
        #   2. chunk of the preemptable tasks
        #   3. chunk of the free spaces

        p_2b_realloc:List[ProcessInt] = []
        bin_id = bin.id
        rsc_alloc = np.zeros_like(rsc_avl)
        for i in range(len(alloc_s)):
            rsc_alloc[alloc_s[i]-time_slot_s:alloc_s[i]-time_slot_s+alloc_len[i]] = alloc_size[i]
        # find out the conflict tasks
        conflict_slot = rsc_alloc > rsc_avl
        while np.any(conflict_slot):

            # find out the actual conflict slot
            conflict_idx = np.nonzero(conflict_slot)[0] + time_slot_s
            # find out the conflict tasks: contain the conflict slot in its allocation
            conflict_pid = []
            for pid in preemptable_map:
                s, size, length = preemptable_map[pid]
                for i in range(len(s)):
                    if s[i] <= conflict_idx[0] < s[i]+length[i]:
                        conflict_pid.append(pid)
                        break
            # perform the preemption: 
            # mechanism: to preempt the task with the highest priority, i.e., the 1st one
            # TODO: add an extra priority level for the tasks with the same priority
            _pid_2b_preempt = conflict_pid.pop(0)
            _p_2b_preempt = _p_index_by_pid[_pid_2b_preempt]
            preemption_list.append(_p_2b_preempt)
            # pop the task from the bin
            print(f"pop the task {_p_2b_preempt.task.id}:{_p_2b_preempt.task.name}({_pid_2b_preempt})from the bin {bin_id}")
            # resource to be released
            if not release_temp_rda:
                # A: the resource occupied from time_slot_s to time_slot_e
                alloc_s_t, alloc_size_t, alloc_len_t = preemptable_map[_pid_2b_preempt]
            else:
                # B: the resource occupied from n_slot to future
                bin_id_t, alloc_s_t, alloc_size_t, alloc_len_t = get_rsc_2b_released(rsc_recoder, n_slot, _p_2b_preempt) 
            # release all resources in this region
            bin.release(_p_2b_preempt, alloc_s_t, alloc_size_t, alloc_len_t)
            # update the resource map
            for i in range(len(alloc_s_t)):
                rsc_avl[alloc_s_t[i]-time_slot_s:alloc_s_t[i]-time_slot_s+alloc_len_t[i]] += alloc_size_t[i]
            # update the conflict slot
            conflict_slot = rsc_alloc > rsc_avl
            # update the preemptable_map
            preemptable_map.pop(_pid_2b_preempt)
            
        policy, alloc_s, alloc_size, alloc_len = push_into_bin(
            _p, bin, 
            time_slot_s, req_rsc_size, expected_slot_num, preempt_en,
            mode, rsc_avl, s, e, size, expected_req_rsc_size, 
            verbose, DEBUG
        )
        idx = np.where(alloc_size)[0]
        # (time_slot_s+np.array(s)[idx]).tolist(), slot_n[idx].tolist(), (e[idx]-s[idx]).tolist()
        alloc_s = [alloc_s[i] for i in idx]
        alloc_size = [alloc_size[i] for i in idx]
        alloc_len = [alloc_len[i] for i in idx]

    if policy != 'N/A':
        bin.allocate(_p.pid, alloc_s, alloc_size, alloc_len, verbose=DEBUG)
        total_alloc_unit = np.sum(np.array(alloc_size) * np.array(alloc_len))
        total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
        print(f"task {_p.task.id}:{_p.task.name}({_p.pid}) is allocated successfully in the bin {bin.id} (policy: {policy})")
        if bin.locker == _p.pid:
            bin.release_lock(_p, time_slot_s, time_slot_e)
        return True, (alloc_s, alloc_size, alloc_len, total_alloc_unit, total_FLOPS_alloc)
    else: 
        # Resource is blocked by other tasks, the current task will be delayed to issue
        return False, (None, None, None, 0, 0)
            
def index_occupy_by_id_chunk_ver(_p, bin, chunk_s, chunk_e, _p_index_by_pid, process_sort, 
                                 quantum_check_en, quantumSize, rsc_recoder):
    """
    index and sort the conflict tasks by their priority, in decending order
    """
    preemptable_n = [0 for i in range(len(chunk_s))]
    preemptable_map = OrderedDict() 
    for i in range(len(chunk_s)):
        s_i, e_i = chunk_s[i], chunk_e[i]
        rsc_map = bin.scheduling_table[s_i].rsc_map
        for pid in rsc_map:
            # check priority
            if process_sort(_p_index_by_pid[pid]) <= process_sort(_p):
                continue

            # check if the task reach the preemption grain
            if quantum_check_en:
                _p_2b_preempt = _p_index_by_pid[pid]
                _, _, _, bin_id_t = rsc_recoder[pid]
                assert bin.id == bin_id_t
                # task is runnning but has executed for an integer multiples of the quantum size (control the pre-emption grain)
                cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
                if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain:
                    continue
            
            preemptable_n[i] += rsc_map[pid]

            if pid not in preemptable_map:
                preemptable_map[pid] = [[s_i], [rsc_map[pid]], [e_i-s_i]]
            else:
                s,size, length = preemptable_map[pid]
                if rsc_map[pid] == size[-1] and s_i == s[-1] + length[-1]:
                    length[-1] += e_i-s_i
                else:
                    s.append(s_i)
                    size.append(rsc_map[pid])
                    length.append(e_i-s_i)
                    preemptable_map[pid] = [s, size, length] 

    # sort the conflict tasks by their priority, in decending order
    preemptable_map = OrderedDict(sorted(preemptable_map.items(), key=lambda item: process_sort(_p_index_by_pid[item[0]]), reverse=True))
    return preemptable_map, preemptable_n

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
    