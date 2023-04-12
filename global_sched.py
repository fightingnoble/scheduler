from __future__ import annotations

from typing import Union, List, Dict, Iterator, Callable, Union
from collections import OrderedDict
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

from global_var import *
from lru import LRUCache
from scheduling_table import SchedulingTableInt
from resource_agent import Resource_model_int
from task_agent import TaskInt
from task_queue_agent import TaskQueue 
from task_agent import ProcessInt, ProcessBase

import warnings
from pre_alloc import get_target_bin_score, glb_alloc_new, get_rsc_2b_released
# ==================== top-level scheduling procedure ====================
def push_task_into_bins(init_p_list: List[TaskInt], affinity, 
                        total_cores:int, quantum_check_en, quantumSize, 
                        timestep, hyper_p, n_p=1, verbose=False, *, 
                        animation=False, warmup=False, drain=False,):

    """
    implement a naive 2d bin-packing algorithm
    input: task_list, which is already arranged in the topological order
    output: a list of bins, each bin is a list of tasks
    """
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(sim_range/timestep)
    tab_spatial_size = total_cores

    pid_idx = {_p.task.name:_p.pid for _p in init_p_list}
    pid_max = max(pid_idx.values())
    
    # monitor the wake-up time: (ascending)
    # activate by the new period or the arrival of the blocked io data
    # TODO: add a queue update logic
    wait_queue:TaskQueue = TaskQueue(init_p_list, sort_f=lambda x: x.release_time, descending=False)
    # monitor the deadline: (ascending)
    ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, descending=False)
    # monitor the deadline for pre-emption: (descending)
    running_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline)
    
    rsc_recoder = {}
    rsc_recoder_his = {}

    expired_list:List[ProcessInt] = []
    def issue_sort_fn(x:ProcessInt):
        alloc_slot_s, alloc_size, allo_slot, bin_id = rsc_recoder[x.pid]
        if isinstance(alloc_slot_s, int):
            return alloc_slot_s
        else:
            return alloc_slot_s[0]
    # monitor the issue time: (ascending)
    issue_list:TaskQueue = TaskQueue(sort_f=issue_sort_fn, descending=False)
    completed_list:List[ProcessInt] = []
    miss_list:List[ProcessInt] = []
    preempt_list:List[ProcessInt] = []
    curr_cfg:Resource_model_int
 
    # try to push the task into the bins in the bin_list
    # if the task cannot be pushed into any bin, create a new bin
    # _new_bins = lambda id: new_bins(total_cores, int(sim_range/timestep), id=id, name="bin"+str(id))

    # define _new
    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, tab_temp_size, id=id, name=name)

    def get_core_size(_p):
        _, _, req_rsc_size = _p.rsc_req_estm(0, timestep, FLOPS_PER_CORE)
        return req_rsc_size

    size_l = []
    name_l = []
    for _p in init_p_list:
        if _p.task.pre_assigned_resource_flag:
            size_l.append(_p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size)
            name_l.append(_p.task.name)

    # iter_next_bin_obj = bin_iter_list(_new_bin, size_l, name_l)
    iter_next_bin_obj = bin_iter_uniform_dist(_new_bin, total_cores, size_l, name_l)
    # iter_next_bin_obj = _next_bin_obj_1(max_core_size=256, size_list=size_l, name_list=name_l)
    bin_list:List[SchedulingTableInt] = list(iter_next_bin_obj)
    bin_name_list = [bin.name for bin in bin_list]
    # bin_list:List[SchedulingTableInt] = [next(iter_next_bin_obj)]
    # bin_name_list = [bin_list[0].name]

    if animation:
        # for animation generation
        frame_list = []
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        plot_window = tab_spatial_size

    for n_slot in range(tab_temp_size):
        curr_t = n_slot * timestep

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")
        
        # enqueue the process that is released in this slot
        lt = []
        for _p in wait_queue:
            if _p.release_time <= n_slot * timestep and _p.release_time < event_range:
                lt.append(_p)
        for _p in lt:
            print("TASK {:d}:{:s}({:d}) RELEASEED AT {}!!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
            ready_queue.put(_p)
            wait_queue.remove(_p)
        lt.clear()

        # ========================================================
        # scan the task list: check finish
        if len(running_queue):
            for _p in running_queue:
                # judge if task complete: completion_count += 1, cum_trunAroundTime += (time + 1.0 - a_time), 
                # update arrival time, deadline, clear current execution unit
                if (_p.totburst >= _p.totcpu):
                    completed_list.append(_p)
                    _p.set_state("suspend")
        
        if completed_list:
            for _p in completed_list:
                # release the resource and move to the wait list
                bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                
                _SchedTab = bin_list[bin_id_t]
                _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                print("TASK {:d}:{:s}({:d}) COMPLETED!!".format(_p.task.id, _p.task.name, _p.pid))
                # update statistics
                # TODO: add lock 
                _p.task.completion_count += 1
                _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
                
                _p.release_time += _p.task.period
                _p.deadline += _p.task.period

                _p.end_time = curr_t * timestep
                _p.currentburst = 0
                _p.burst = 0
                _p.totburst = 0
                _p.remburst = _p.task.flops
                _p.cumulative_executed_time = 0
                _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
                rsc_recoder.pop(_p.pid)
                running_queue.remove(_p)
                wait_queue.put(_p)

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            completed_list.clear()
        # ========================================================

        # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
        for _p in (ready_queue.queue + running_queue.queue):
            if(_p.deadline < curr_t):
                miss_list.append(_p)
                _p.set_state("suspend")
        
        if miss_list:
            for _p in miss_list:
                # release the resource and move to the wait list
                if _p in ready_queue.queue:
                    ready_queue.remove(_p)
                elif _p in running_queue.queue:
                    bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                    
                    _SchedTab = bin_list[bin_id_t]
                    _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                    rsc_recoder.pop(_p.pid)
                    running_queue.remove(_p)
                # print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid));
                print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid)+\
                        "released @ {} start @ {}:\n".format(_p.release_time, _p.start_time) +\
                        "is expected to finish {}T OPs before {} is expired, ".format(_p.totcpu, _p.deadline) +\
                        "but only executed {}T OPs in {}s time !!".format(_p.totburst, _p.cumulative_executed_time))
                _p.task.missed_deadline_count += 1
                _p.release_time += _p.task.period
                _p.deadline += _p.task.period

                _p.burst = 0
                _p.totburst = 0
                _p.remburst += _p.task.flops
                _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
                _p.cumulative_executed_time = 0
                _p.currentburst = 0

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            miss_list.clear()

        # glb_alloc(init_p_list, affinity, quantumSize, timestep, pid_idx, ready_queue, running_queue, rsc_recoder, rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list, n_slot, curr_t)        
        glb_alloc_new(init_p_list, quantum_check_en, quantumSize, timestep, ready_queue, running_queue, rsc_recoder, 
                  rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list, n_slot, curr_t)

        # issue the task
        # if the task of the queue equals to the current slot, then issue the task
        if len(issue_list):
            for _p in issue_list:
                if _p in ready_queue:
                    ready_queue.remove(_p)
            while len(issue_list):
                _p = issue_list[0]
                if issue_sort_fn(_p) == n_slot: 
                    running_queue.put(_p)
                    if _p.totburst == 0:
                        _p.start_time = curr_t
                    _p.waitTime = 0
                    issue_list.get()
                else:
                    break
            # issue_list.clear()
            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            # if animation:
            #     frm_arr = []
            #     for _SchedTab in bin_list:
            #         frm_arr.append(_SchedTab.get_plot_frame(n_slot, n_slot+plot_window))
            #     frm_arr = np.concatenate(frm_arr, axis=0)
            #     frm = ax.imshow(frm_arr, animated=True, vmax=pid_max)
            #     frame_list.append([frm])

        # =================================================
        _p_dict = {p.pid:p for p in running_queue}
        for _SchedTab in bin_list:
            curr_cfg = _SchedTab.scheduling_table[n_slot]
            if curr_cfg.rsc_map:
                # update the running task
                # _p_dict_n = {p.task.name:p for p in running_queue}
                # if "MultiCameraFusion_0" in _p_dict_n.keys():
                #     if not _p_dict_n["MultiCameraFusion_0"].pid in curr_cfg.rsc_map.keys():
                #         print("MultiCameraFusion_0 is not in the current configuration")
                for pid in curr_cfg.rsc_map.keys():
                    _p = _p_dict[pid]
                    _p.currentburst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.burst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.totburst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.remburst -= curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.cumulative_executed_time += timestep

        # SchedTab.step()
        if animation: 
            if n_slot % 8 == 0 and n_slot+plot_window < tab_temp_size:
                frm_arr = []
                for _SchedTab in bin_list:
                    frm_arr.append(_SchedTab.get_plot_frame(n_slot, n_slot+plot_window))
                frm_arr = np.concatenate(frm_arr, axis=0)
                frm = ax.imshow(frm_arr, animated=True, vmax=pid_max)
                frame_list.append([frm])
    if animation:
        ani = animation.ArtistAnimation(fig, frame_list, interval=20, blit=False,
                                    repeat_delay=1000)
        writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("movie.mp4", writer=writer)

    pid2name = {pid:p_name for p_name, pid in pid_idx.items()}
    for _SchedTab in bin_list:
        print("=====================================\n")
        print(f"Scheduling Table of {_SchedTab.name}({_SchedTab.id}):")
        _SchedTab.print_scheduling_table(pid2name, timestep)
        print("=====================================\n")
    
    print("=====================================\n")
    print("bin_pack_result:")
    print("=====================================\n")
    for _SchedTab in bin_list:
        _SchedTab.print_alloc_detail(pid2name, timestep)
    
    return bin_list, init_p_list


#
def glb_alloc(init_p_list, affinity, quantumSize, timestep, pid_idx, ready_queue, running_queue, rsc_recoder, rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list, n_slot, curr_t):
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
    # cond_fn1 = lambda x: (x.task.deadline - curr_t)/x.exp_comp_t
    cond_fn1 = lambda x: x.deadline
    cond_fn2 = lambda x: get_target_bin_score(x, bin_name_list=bin_name_list, rsc_recoder_his=rsc_recoder_his, reverse=True)
    sorted_ready_queue = sorted(ready_queue.queue, key=lambda x: (cond_fn1(x), cond_fn2(x)))
        
    for _p in sorted_ready_queue: 
            # issue the task
        allocate_rsc_4_process(_p, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)

        # update the running task
    if len(preempt_list):
        for _p in preempt_list:
            if _p in running_queue.queue:
                running_queue.remove(_p)
                ready_queue.put(_p)
        preempt_list.clear()


def allocate_rsc_4_process(_p:ProcessInt, n_slot:int, 
                affinity:Dict[str, List[str]], pid_idx:dict, init_p_list:List[ProcessInt], 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
                ready_queue:TaskQueue, running_queue:TaskQueue, 
                issue_list:TaskQueue, preempt_list:List[ProcessInt], 
                iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], ):
    # initialize the resource request parameters
    p_name = _p.task.name
    time_slot_s, time_slot_e, req_rsc_size = _p.rsc_req_estm(n_slot, timestep, FLOPS_PER_CORE)

    # expected slot number
    expected_slot_num = time_slot_e-time_slot_s # int(np.ceil(_p.exp_comp_t/timestep))
    # TODO: Add the logic to ensure the task have allocated enough resource, otherwise, 
    #         we should not issue the task or compensate the resource latter. 

    # try to push the task into the bins in the bin_list
    state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
    # pre-allocation strategy: 
    # 1. the resource constraint should be respected
    # 2. the pre-defined resource preservation should be respected 
    # 3. the affinity settings of all the tasks should be respected 
    # 4. all tasks should be allocated with the resource
    # 5. tasks is expected to migrate as less as possible
    # 6. the resource should be allocated as compact as possible
    # 7. the resource should be allocated as balanced as possible

    _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
    # TODO: arange the bin_list according to the affinity of the class

    # 2. the pre-defined resource preservation should be respected 
    #   For the task that is pre-assigned with the resource, the affinity is set to be itself
    if p_name in bin_name_list:
        bin_id = bin_name_list.index(p_name)
        bin = bin_list[bin_id]
        state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
        total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
        total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
        # check if the task is allocated successfully
        # if fail collect the tasks in the target bin and preempt the tasks
        if state and (total_FLOPS_alloc >= _p.remburst):
            pass
            print("Task {} is allocated successfully".format(_p.pid))
        else: 
            # release the resource
            bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
            # reset the state
            state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
            p_2b_realloc = []
            state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc = preempt_the_conflicts(_p, bin, 
                req_rsc_size, expected_slot_num, 
                time_slot_s, time_slot_e, n_slot, 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder, rsc_recoder_his, 
                running_queue, 
                issue_list, preempt_list, p_2b_realloc, 
                _p_index_by_pid, quantum_check_en=False) 

            # reallocate the tasks in the P_2b_realloc
            for _p_2b_realloc in p_2b_realloc:
                allocate_rsc_4_process(_p_2b_realloc, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                    rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)
    else: 
        # find the target bin of the affinity target
        def get_target_bin_id(_p=_p, bin_name_list=bin_name_list, rsc_recoder_his=rsc_recoder_his):
            affinity_tgt_bin_id_list = []
            for task_n in _p.task.affinity_n:
                # suppose the target is pre-assigned with the resource but is not allocated
                if task_n in bin_name_list:
                    affinity_tgt_bin_id_list.append(bin_name_list.index(task_n))
            # suppose the target was allocated with the resource
            for _pid in _p.task.affinity:
                if _pid in rsc_recoder_his:
                    affinity_tgt_bin_id_list.append(rsc_recoder_his[_pid].get_mru())
            return affinity_tgt_bin_id_list

        affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)

        # # remove the thread number at the end of the name
        # # name parse
        # thread_n = _p.task.name.split('_')[-1]
        # p_name = _p.task.name
        # task_base_name = p_name.replace("_"+thread_n, "")
        # thread_n = int(thread_n)
        # affinity_tgt_n_list = affinity[task_base_name]        
        # affinity_tgt_n_list = [n+'_'+str(thread_n) for n in affinity_tgt_n_list]
        # # get affinity target id
        # # TODO: bug here， key error when the affinity target is not in the pid_idx
        # affinity_tgt_id_list = [pid_idx[n] for n in affinity_tgt_n_list if n in pid_idx]
        # # find the target bin of the affinity target
        # affinity_tgt_bin_id_list = [rsc_recoder_his[n].get_mru() for n in affinity_tgt_id_list if n in rsc_recoder_his] 

        # mark other bins as the targets of the search
        affinity_search_bin_id_list = [n for n in range(len(bin_list)) if n not in affinity_tgt_bin_id_list] 
        # arrange the targets of the search in the order of the fitness of the size
        affinity_search_bin_id_list.sort(key=lambda x: abs(bin_list[x].scheduling_table[0].size - req_rsc_size))

        # 3. the affinity settings of all the tasks should be respected 
        # search the bin in the affinity target bin list to find bin with best affinity
        for bin_id in affinity_tgt_bin_id_list: 
            bin = bin_list[bin_id]
            state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
            total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
            total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
            # check if the task is allocated successfully
            # if fail collect the tasks in the target bin and preempt the tasks
            if state and (total_FLOPS_alloc >= _p.remburst):
                print("Task {} is allocated successfully".format(_p.pid))
                break
            else:
                # release the resource
                # TODO: merge this function into the scheduling table class
                # TODO: distinguish the state of risking the lack of resources and the state of the task cannot be pushed into the bin
                bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
                # reset the state
                state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
                # p_2b_realloc = []
                # state, alloc_slot_s, alloc_size, allo_slot, \
                #     total_alloc_unit, total_FLOPS_alloc = preempt_the_conflicts(_p, bin, 
                #     req_rsc_size, expected_slot_num, 
                #     time_slot_s, time_slot_e, n_slot, 
                #     timestep, FLOPS_PER_CORE, quantumSize, 
                #     rsc_recoder, rsc_recoder_his, 
                #     running_queue, 
                #     issue_list, preempt_list, p_2b_realloc, 
                #     _p_index_by_pid, key=lambda x: affinity_fn(x, bin, rsc_recoder_his)) 
                # # reallocate the tasks in the P_2b_realloc
                # for _p_2b_realloc in p_2b_realloc:
                #     allocate_rsc_4_process(_p_2b_realloc, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                #         rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)


        if not state:
            # search the bin in the affinity search bin list
            for bin_id in affinity_search_bin_id_list: 
                bin = bin_list[bin_id]
                state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
                total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
                total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
                if state and (total_FLOPS_alloc >= _p.remburst):
                    print("Task {} is allocated successfully".format(_p.pid))
                    break
                else:
                    # release the resources if the task cannot be pushed into the bin
                    # TODO: merge this function into the scheduling table class
                    # TODO: distinguish the state of risking the lack of resources and the state of the task cannot be pushed into the bin
                    bin.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                    state = False
                    
        # check the state; if the task cannot be pushed into any bin, create a new bin
        if not state:
            try:
                bin = next(iter_next_bin_obj)
                bin_id = bin.id
                state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False) 
                bin_list.append(bin)
                bin_name_list.append(bin.name)
                total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
                total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
            except StopIteration:
                warnings.warn("No more bin can be created")

    # if the task is allowed to execute under insufficient resources
    if state and total_FLOPS_alloc < _p.remburst:
        Warning("The allocated FLOPS is not enough for the task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        
    # print the allocation result
    print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
    print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
    if state:
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
    else:
        print("\t[{:s}]\n".format("FAILED" if not state else "SUCCESS"))

    # record the allocation result and prepare the issue list
    if state:
        rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot, bin_id]
        issue_list.put(_p)
        if _p.pid in rsc_recoder_his:
            rsc_recoder_his[_p.pid].put(bin_id)
        else:
            rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)
    else:
        Warning("TASK {:d}:{:s}({:d}) IS DELAY ISSUED!!".format(_p.task.id, _p.task.name, _p.pid))


def preempt_the_conflicts(_p:ProcessInt, bin:SchedulingTableInt, 
        req_rsc_size:int, expected_slot_num:int, 
        time_slot_s:int, time_slot_e:int, n_slot:int,
        timestep, FLOPS_PER_CORE, quantumSize, 
        rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
        running_queue:TaskQueue, 
        issue_list:TaskQueue, preempt_list:List[ProcessInt], p_2b_realloc:List[ProcessInt],
        _p_index_by_pid:Dict[int, ProcessInt], key:Callable[[ProcessInt], int]=None, quantum_check_en:bool = True):
    """
    Preempt the conflict tasks and reallocate the resources to the current task
    quantum_check_en:
        disable the quantum check in the stage of preallocation
    """
    
    bin_id = bin.id
    # find out the conflict tasks
    occupation_candi_dict:Dict[int, List[int]] = bin.index_occupy_by_id(time_slot_s, time_slot_e)
    # filter the confict tasks that have lower affinity with the current bin compared with the current task
    if key is not None:
        occupation_candi_dict = {k:v for k,v in occupation_candi_dict.items() if key(_p) > key(_p_index_by_pid[k])}
    # sort the conflict tasks by their priority
    occupation_candi = list(occupation_candi_dict.keys())
    # select the task with the latest deadline
    # TODO: evaluate more strategies
    occupation_candi.sort(key=lambda x: _p_index_by_pid[x].deadline, reverse=True)


    aval_flops = sum(bin.idx_free_by_slot(time_slot_s, time_slot_e, _p.pid)) * timestep * FLOPS_PER_CORE
    # note that here is an assumption that current bin is designed for the task, all the candidate tasks are preempted
    # flops_2b_preempt = (time_slot_e - time_slot_s) * bin.scheduling_table[0].size * timestep * FLOPS_PER_CORE - aval_flops
    flops_2b_preempt = _p.remburst - aval_flops
    
    # evaluate the preemption possibility
    # if the preemptable conflict task can provide enough FLOPS to the current task
    total_FLOPS_occupied = 0
    for pid in occupation_candi:
        alloc_slot_s_t, alloc_size_t, allo_slot_t = occupation_candi_dict[pid]
        total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
        total_FLOPS_alloc_t = total_alloc_unit_t * timestep * FLOPS_PER_CORE
        total_FLOPS_occupied += total_FLOPS_alloc_t
    if flops_2b_preempt-total_FLOPS_occupied > 1e-2*timestep*FLOPS_PER_CORE: 
        state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
        total_alloc_unit, total_FLOPS_alloc = 0, 0
        return state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc
    
    # decide which task to preempt
    for pid in occupation_candi:
        # load allocation history
        _p_2b_preempt = _p_index_by_pid[pid]
        alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[pid]
        assert bin_id == bin_id_t
        
        # task is in the ready queue and issue list
        # judge if preemptable: 

        # task is runnning but has executed for an integer multiples of the quantum size (control the pre-emption grain)
        if quantum_check_en:
            cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
            reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
            if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                continue
        else:
            reach_preempt_grain = True
        
        # calculate the interval intersection of the two tasks
        # [time_slot_s, time_slot_e]
        # [alloc_slot_s_t, alloc_slot_s_t + len(alloc_size_t)]
        # total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
        alloc_slot_s_t, alloc_size_t, allo_slot_t = occupation_candi_dict[pid]
        total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
        total_FLOPS_alloc_t = total_alloc_unit_t * timestep * FLOPS_PER_CORE
        flops_2b_preempt -= total_FLOPS_alloc_t

        # pop the task from the bin
        print(f"pop the task {_p_2b_preempt.task.id}:{_p_2b_preempt.task.name}({pid})from the bin {bin_id}")
        # release all the tasks in the p_2b_preempt
        # Faulty: get same results as occupation_candi_dict[pid] 
        bin_id_t, alloc_slot_s_t, alloc_size_t, allo_slot_t = get_rsc_2b_released(rsc_recoder, n_slot, _p_2b_preempt) 
        bin.release(_p_2b_preempt, alloc_slot_s_t, alloc_size_t, allo_slot_t, verbose=False)
        # update the rsc_recoder
        rsc_recoder.pop(_p_2b_preempt.pid)
        
        if _p_2b_preempt.currentburst == 0: 
            # task is in the ready queue and issue list
            issue_list.remove(_p_2b_preempt)
            # update the rsc_recoder_his
            rsc_recoder_his[pid].withdraw()
            # update the running_queue
            if _p_2b_preempt in running_queue:
                raise ValueError("A unexpected situation happens, task is not executed but in the running queue")
                # maybe blocked by the I/O
                # TODO: add the logic to handle the blocked task
                # put into waiting list


        elif _p_2b_preempt.currentburst != 0 and reach_preempt_grain:
            # task is in the running queue
            # update the task status
            _p_2b_preempt.task.preemption_count += 1
            _p_2b_preempt.currentburst = 0 

            # update the running_queue
            preempt_list.append(_p_2b_preempt)
            print(f"task {_p_2b_preempt.task.id}:{_p_2b_preempt.task.name}({pid}) is preempted and put into the ready queue")
        
        # mark for reallocation in current time slot
        p_2b_realloc.append(_p_2b_preempt)
        if flops_2b_preempt <= 1e-2*timestep*FLOPS_PER_CORE:
            break

    # reallocate the current task
    state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
    total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
    total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
    # check if the task is allocated successfully
    # if fail collect the tasks in the target bin and preempt the tasks
    if state and (total_FLOPS_alloc >= _p.remburst): 
        print(f"task {_p.task.id}:{_p.task.name}({_p.pid}) is allocated successfully in the bin {bin_id}")
        if bin.locker == _p.pid:
            bin.release_lock(_p, time_slot_s, time_slot_e)
    else: 
        # In this condition, some tasks are not preempted successfully, and the current task is blocked by these tasks, 
        # the current task will be delayed to issue
        # release the resource
        bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
        # reset the state
        state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
        bin.add_lock(_p, time_slot_s, time_slot_e)
        Warning(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.pid}) is not allocated successfully after preemption in its expected bin")
        # raise ValueError(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.pid}) is not allocated successfully after preemption in its own bin")
    return state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc



# =============== Bin related functions ===============

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

