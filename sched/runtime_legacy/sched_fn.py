from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sched.scheduler_agent import Scheduler
    
import numpy as np
import math, warnings
from collections import OrderedDict
from copy import deepcopy
from typing import List
from queue import Queue
from global_var import *
from utils import core_distr
from functools import reduce

from model.buffer import Buffer, EventCache, TriggerCache
from model.buffer import Buffer, Data
from model.resource_agent import Resource_model_int
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe
from model.streaming_processing.wartermark_strategy import WatermarkStrategy
from model.performance import slack_comp

from task.task_agent import ProcessInt
from sched.scheduling_table import SchedulingTableInt, process_tab_event, parse_event_msg
from sched.monitor_agent import Monitor
from sched.runtime_legacy.sched_utils import *
from sched.runtime_legacy.state_trans import *
from sched.placement import update_phy_posi


def scheduler_step(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()

    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    # clear the marginal budget
    for pid in list(budget_recoder.keys()):
        _p = process_dict[pid]
        rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
        if rem_flop_budget < numerical_error_tol_abs and _p.pid in budget_recoder:
            # Each partition has splited budget recoder 
            budget_recoder.pop(pid)
            _p.rem_flop_budget[bin_id] = 0

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
                            res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    bin_event_flg = check_throttle(sched, None, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, bin_event_flg, bin_name)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)
    # buffer.recyle_no_ref("output", True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)


    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        next_cfg = curr_cfg.action_at_end_cfg(timestep, _SchedTab, tab_temp_size, tab_pointer, hyper_p_n)
        
        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))

    # At the beginning of each cfg chunk
    if curr_cfg.slot_s == n_slot:
        # load the budget at the beginning of each cfg chunk rather than the end, 
        # which is important as the cfg may be not consecutive, 
        # loading at the end may lead to launch some kernels too early
        curr_cfg.action_at_start_cfg(budget_recoder, process_dict, bin_id)

        # Policy 2: see release_rsc
        # when a task is restarted in a new bin, it needs to migrate budget from the previous bin, 
        # to ensure that the tasks have enough budget to run.
        for event_group_t, event_group in curr_cfg.event_list:
            if event_group_t == n_slot:
                process_tab_event(sched, curr_t, ready_queue, running_queue, throttle_list, 
                                event_group, _SchedTab, process_dict, bin_id,
                                msg_filter={"event_type": "migrate_from"})
        
        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    if sched.progress_aware:
        # check data availability: some tasks may be prefetched
        # TODO: model the runtime weight and feature map transfering 
        pendingToReady_cbs(sched, buffer, budget_recoder, active_list, ready_queue, throttle_list, curr_t, glb_name_p_dict, event_cache, bin_name, process_dict, show_warnings=show_warnings)
        # move the task to the ready queue
        bin_event_flg = throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg)
    else:
        pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 


    # for _p in ready_queue.queue:
    #     # NOTE:cross cancelation
    #     _p:ProcessInt
    #     rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
    #     if len(rem_flop_budget) > 1:
    #         planned_flops = sum(rem_flop_budget.values())
    #         assert len(rem_flop_budget) == 2
    #         # move the remained budget to the current partition
    #         previous_partition_id = (set(rem_flop_budget.keys()) - {bin_id,}).pop()
    #         print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is cancled from bin {previous_partition_id:d} \
    #                 and moved to bin {bin_id:d} with budget {planned_flops:.6f}")
    #         _p.rem_flop_budget[previous_partition_id] = 0
    #         _p.rem_flop_budget[bin_id] = planned_flops

    #         # judge whether current slot is executed on other patition
    #         # if so, pass the current slot
    #         if previous_partition_id < bin_id:
    #             skiped_tasks.append(_p)

    if sched.drain_flg:
        sched.check_draining_state()

    # check the running tasks
    for _p in running_queue.queue:
        # detect the lateness of the tasks
        # if _p.pid not in curr_cfg.rsc_map: 
        if budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2] < n_slot:
            if show_warnings:
                warnings.warn("Execution lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

    # build the local running configuration
    # Try to allocate the resource to the ready tasks
        # combine the ready tasks, running tasks, budget, and deadline together
        # co-operate the designed properties with EDF algorithm
            # stationary/movable
            # realtime/deadline
            # w/ or w/o need to be scaled
        # first stationary, then movable
        # tackle 3 cases:
            # 1. arrival lateness
            # 2. tighter deadline
            # 3. workload scaling
    # Two components:
        # 1. when to trigger
        # 2. compenstation algorithm
    
    # intution:
        # 1. as soon as possible (ASAP)
        # 2. as evenly as possible (AEAP)        
    # here, we adapt the ASAP to simplify the scheduling process

    # trigger condition:
        # detect risk of current block execution timeout

    # calculate the criticity of the task
    # the most critical one is the first one, with the smallest value, use the ascending order

    is_not_hard = lambda x: x.task.criticality != "hard"

    fn_crit = lambda x: x.deadline
    # fn_crit = lambda _p: budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2]

    fn_task_flag = lambda x: 0 if x.task.task_flag=="stationary" else 1

    # =============== cheduler trigger condition ===============

    # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
    # Positive cases that trigger the scheduler: 
    # A. running queue status changes: some task finish/miss/throttled & release cores
    # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
    #   idle resources, or preempt others
    # C. cfg of running tasks changes: want to extend/release some cores
    # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
    # E. resource ++: some task is waiting to get cores
    # F. some running task is starving 

    # Negative cases that not trigger the scheduler:
    # N.A: some free resources but all other thing not changes, 
    #   which means scheduler will get the same allocation for a new calling 

    # If no execution lateness in previous cfg, 
    # then the running queue is empty all the tasks chunks shares the same deadline; 
    # without spec changes, 
    # the execution sequence not matter the schedulibility. 
    # Otherwise, the task is sorted by the 
    # deadline, host or guest, running or not

    pre_rsc = res_cfg.rsc_map
    task_queue = running_queue.queue + ready_queue.queue
    # ready queue status
    trigger_condA = sched.new_ready_flg
    # running queue status && resource ++ 
    trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
    # budget indicator
    trigger_condC = sum([budget_recoder[_p.pid][3] for _p in task_queue])


    if trigger_condA or trigger_condB or trigger_condC:

        # =============== build the scehduling candidate list ===============
        preemptable_list, curr_aval_rsc = check_prempt(res_cfg, running_queue, quantumSize, quantum_check_en, preemption_en)

        if sched.drain_flg:
            score_fn = lambda x: (x.get_timestamp()>=sched.switch_border, fn_crit(x), fn_task_flag(x), x not in preemptable_list)
        else:
            score_fn = lambda x: (fn_crit(x), fn_task_flag(x), x not in preemptable_list)
        # if len(preemptable_list) > 0:
        #     threshold_item = min(preemptable_list, key=lambda x: score_fn(x))
        #     threshold_score = score_fn(threshold_item)
        # else:
        #     threshold_score = (np.inf, 1)
        threshold_score = (np.inf, 1, True)
        filtered_ready_queue = [_p for _p in ready_queue.queue if score_fn(_p) < threshold_score]
        sorted_queue = sorted(filtered_ready_queue+preemptable_list, key=score_fn,)
        # ===================================================================

        rsc_map = OrderedDict()

        while len(sorted_queue) > 0 and curr_aval_rsc > 0:
            _p:ProcessInt = sorted_queue[0]
            _p.is_starving = False

            # =============== calculate the required resource size ===============
            # 
            # get the newest assigned budget
            # planned_rsc_size = budget_recoder[_p.pid][1] # curr_cfg.rsc_map[_p.pid]
            # planned_slot_num = budget_recoder[_p.pid][2] # curr_cfg.slot_num
            
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            chunk_flops = chunk_alloc * chunk_slot_num * timestep * FLOPS_PER_CORE # _p.rem_flop_budget[bin_id]
            chunk_e = chunk_s + chunk_slot_num
            # late_slot_num = fn_trig(_p) 
            planned_flops = sum([v for v in _p.rem_flop_budget.values() if v > numerical_error_tol_abs])

            # - We discuss this issue in two scenarios:
            #     1. with data arriving on time: allocate the resources according to the budget
            #     2. with data arriving late: allocate the resources following the "EDF", and estimate the resources at runtime

            # lateness detection mechanism:
            #  both task-level and chunk-level
            # case 1: release late, i.e., the task is not released at the beginning of the current configuration
            # case 2: previous chunk is late, i.e., the task is not finished at the end of the previous configuration
            # case 3: current chunk is late, i.e., the task is not resumed at the beginning of the current configuration

            assert chunk_s <= n_slot, "chunk_s {:d} > n_slot {:d}".format(chunk_s, n_slot)
            if sched.drain_flg and _p.get_timestamp() < sched.switch_border:
                req_rsc_size = curr_aval_rsc
            elif chunk_s < n_slot < chunk_e:
                # case 1: release late !!! the running task that is identified as preemptable [chunk_s, chunk_e] 
                assert chunk_e == curr_cfg.slot_e + 1
                # req_rsc_size = math.ceil(planned_flops/(chunk_e - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate)) 
                slack = slack_comp((chunk_e - n_slot)*timestep, 0, sched.overprovision_rate)
                req_rsc_size = math.ceil(planned_flops/slack/FLOPS_PER_CORE) 
            elif n_slot >= chunk_e:
                # case 3: current chunk is late
                #   newest assigned budget is skipped
                req_rsc_size = curr_aval_rsc
            else:
                if round(planned_flops, flop1u_error_tol_bit) > round(chunk_flops, flop1u_error_tol_bit):
                    # case 2: previous chunk is late
                    #   newest assigned budget is still available but not enough
                    # req_rsc_size = math.ceil(planned_flops/(chunk_e + 1 - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate))
                    # req_rsc_size = math.ceil(planned_flops/(chunk_e - n_slot)/timestep /FLOPS_PER_CORE/(1-sched.overprovision_rate)) 
                    slack = slack_comp((chunk_e - n_slot)*timestep, 0, sched.overprovision_rate)
                    req_rsc_size = math.ceil(planned_flops/slack/FLOPS_PER_CORE) 
                else:
                    # newest assigned budget is still available                    
                    # tries to finish the remaining work assigned by the configuration chunk until the now
                    req_rsc_size = chunk_alloc 

            # **************************************************************
            # check the rsc_size is valid
            # **************************************************************
            req_rsc_size, constr = _p.get_available_cfg(req_rsc_size, curr_aval_rsc, True)
            if constr == "partial":
                _p.is_starving = True

            if req_rsc_size == 0 or constr == "N/A":
                if o3_boost_util_en:
                    skiped_tasks.append(_p)
                    sorted_queue.pop(0)
                    continue
                else:
                    break
            assert req_rsc_size > 0
            assert isinstance(req_rsc_size, (int, np.integer)), "req_rsc_size is not integer"

            if _p.totburst == 0 and chunk_s < n_slot:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is deteted a lateness of {(n_slot-chunk_s):d} slots")

            if _p.load_var == 0:
                req_rsc_size = 0
            # **************************************************************
            
            # TODO: if the req_rsc_size is larger than the aval_rsc, then add a flag to indicate the task is late                
            sorted_queue.pop(0)
            # if curr_aval_rsc >= req_rsc_size: # and req_rsc_size > 0:
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            _p.required_resource_size = req_rsc_size

        sched.new_ready_flg = False

        to_assert_flag = False
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # remove the expired task from the position dict
            for pid in expired_pid:                
                preempt_list.append(process_dict[pid])

            for pid in old_pid:
                old_size = pre_rsc[pid]
                new_size = rsc_map[pid]
                if old_size != new_size:
                    ctx_switch_list.append(process_dict[pid])
            
            for pid in new_pid:
                issue_list.append(process_dict[pid])
            
            # update the position dict
            update_phy_posi(sched, position_dict, pre_rsc, rsc_map, expired_pid, old_pid, new_pid)
            to_assert_flag, off_discount, on_discount = plan_switching(preempt_list, issue_list, ctx_switch_list, budget_recoder, pre_rsc, rsc_map)

            if len(preempt_list) > 0:
                sched.new_ready_flg = True

            # update the resource configuration by preempt_list, issue_list, ctx_switch_list
            sched.handle_taskqueue(curr_t, bin_event_flg, pre_rsc, rsc_map)

        for pid in budget_recoder:
            budget_recoder[pid][3] = False
        
        if to_assert_flag:
            # assert a barrier, sim data movement: 
            if sched.barrier_en:
                off_size = (sum(pre_rsc.values())-off_discount) * 1/3
                on_size = (sum(rsc_map.values()) - on_discount) * 2/3
                x = (on_size+off_size) 
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    if sched.drain_flg:
        sched.drain_old_cores()
        
        
    # =============== Execution stage ===============
    # execute the task status in running list
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue, True, bin_id, skiped_tasks=skiped_tasks) 

    # ===== Prepare the next slot configuration =====
    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()

    if curr_cfg.event_list:
        curr_event_group_t, curr_event_group = curr_cfg.event_list[0]
        if n_slot == curr_event_group_t:        
            # policy 1: see release_rsc
            # process migration to at the end of the slot when the task get budget in the new bin.
            process_tab_event(sched, curr_t, ready_queue, running_queue, throttle_list, curr_event_group, _SchedTab, process_dict, bin_id, 
                                msg_filter={"event_type": "migrate_to"})
            curr_cfg.event_list.pop(0)

    # just for verification
    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")

            # if curr_cfg.slot_s == n_slot:
            # if curr_cfg.slot_e < n_slot or n_slot == 0: 
            # ensure 
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

def check_prempt(res_cfg, running_queue, quantumSize, quantum_check_en=False, preemption_en=True):
    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    preemptable_list = []
    if preemption_en:
        if quantum_check_en: 
            assert quantumSize is not None
            for _p_2b_preempt in running_queue.queue:
                cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum), abs_tol=1e-2)
                if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                    continue
                else:
                    preemptable_list.append(_p_2b_preempt)
            curr_aval_rsc = aval_rsc + sum([_p.required_resource_size for _p in preemptable_list])
        else:
            curr_aval_rsc = res_cfg.size
            preemptable_list = running_queue.queue
    else:
        curr_aval_rsc = aval_rsc
    return preemptable_list,curr_aval_rsc

def scheduler_step_cyclic(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)


    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.next_item()
        curr_cfg.update(next_cfg)
        # update the deadline
        if cfg_slot_s < tab_pointer: 
            curr_cfg.slot_s = (hyper_p_n + 1) * tab_temp_size + cfg_slot_s
        else:
            curr_cfg.slot_s = hyper_p_n * tab_temp_size + cfg_slot_s
        curr_cfg.slot_e = curr_cfg.slot_s + cfg_slot_num - 1 
        curr_cfg.slot_num = cfg_slot_num
        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))

    # At the beginning of each cfg chunk
    if curr_cfg.slot_s == n_slot:
        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 

    # free resource index
    aval_rsc = res_cfg.get_available_rsc()
    assert isinstance(aval_rsc, int) or isinstance(aval_rsc, np.integer)

    # Scheduler is triggered when:
    # either the aval_rsc or the candidate changes, i.e.,
    # A. task release
    # 1. new tasks join the ready queue, preemption may happen
    # 2. Budget updated

    curr_aval_rsc = res_cfg.size
    task_queue = running_queue.queue + ready_queue.queue
    # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
    # Positive cases that trigger the scheduler: 
    # A. running queue status changes: some task finish/miss/throttled & release cores
    # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
    #   idle resources, or preempt others
    # C. cfg of running tasks changes: want to extend/release some cores
    # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
    # E. resource ++: some task is waiting to get cores
    # F. some running task is starving 

    # Negative cases that not trigger the scheduler:
    # N.A: some free resources but all other thing not changes, 
    #   which means scheduler will get the same allocation for a new calling 

    # ready queue status
    trigger_condA = sched.new_ready_flg
    # running queue status && resource ++ 
    # trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
    # budget indicator
    trigger_condC = (curr_cfg.slot_s == n_slot) 

    # 20240731: remove condition B, because in this scheduler, we never change the resource bandwidth, 
    # so no starving flage will be set. 
    if trigger_condA or trigger_condC:

        while len(task_queue) > 0:
            _p:ProcessInt = task_queue.pop(0)
            if _p.pid in curr_cfg.rsc_map.keys():
                issue_list.append(_p)
                req_rsc_size = curr_cfg.rsc_map[_p.pid]
                assert req_rsc_size > 0
                _p.required_resource_size = req_rsc_size
                curr_aval_rsc -= req_rsc_size
                assert curr_aval_rsc >= 0 

        # new_pid = set(curr_cfg.rsc_map.keys()) - set(res_cfg.rsc_map.keys())
        expired_pid = set(res_cfg.rsc_map.keys()) - set(curr_cfg.rsc_map.keys())
        old_pid = set(res_cfg.rsc_map.keys()) - expired_pid
        
        sched.new_ready_flg = False
        running_queue.queue.clear()
        res_cfg.clear()
        
        # move the expired tasks into ready queue
        for pid in expired_pid: 
            print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
            ready_queue.put(process_dict[pid])

        running_queue.queue.clear()
        # if issue the task to runnning list
        for _p in issue_list:
            assert _p.get_state() != "running"
            running_queue.put(_p)
            if _p in ready_queue.queue:
                ready_queue.queue.remove(_p)
            _p.set_state("running")
            res_cfg.allocate(_p.pid, _p.required_resource_size)
            _p.waitTime = 0 
            _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) issued "
            if _p.totburst==0:
                _p.start_time = curr_t
                _str += f"and start at {curr_t:.6f}; "
            elif _p.pid not in old_pid:
                _str += f"and resume at {curr_t:.6f}; "
            _p.curr_start_time = curr_t
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(_str)
        issue_list.clear()
    
    
    # execute the task in running list
    # update the running task
    res_cfg.updateRunningQueue(timestep, running_queue) 

    monitor.add_a_record(res_cfg)

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

def scheduler_step_fifo(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                    n_slot, timestep, 
                    event_range, sim_slot_num, curr_t, 
                    glb_name_p_dict, res_cfg, msg_queue, a_msg_queue, sensor_msg_queue, 
                    monitor:Monitor, DEBUG_FG, 
                    quantum_check_en:bool = False, quantumSize=None, preemption_en:bool=True,
                    o3_boost_util_en:bool=False,show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()

    # the tasks is not executed in this slot, including the task that gets illegel allocation
    # ```
    #     skiped_task.append(_p)
    #     sorted_queue.pop(0)
    #     continue
    # ```
    # also the tasks that is allocated but is passed in this slot
    # ```
    #     skiped_task.append(_p)
    # ```
    skiped_tasks = []

    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    tab_temp_size = len(_SchedTab.scheduling_table)
    tab_pointer = n_slot % tab_temp_size
    hyper_p_n = int(n_slot/tab_temp_size)
    curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources
    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    # clear the marginal budget
    for pid in list(budget_recoder.keys()):
        _p = process_dict[pid]
        rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
        if rem_flop_budget < numerical_error_tol_abs and _p.pid in budget_recoder:
            # Each partition has splited budget recoder 
            budget_recoder.pop(pid)
            _p.rem_flop_budget[bin_id] = 0

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    bin_event_flg = check_throttle(sched, None, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, bin_event_flg, bin_name)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)
    # buffer.recyle_no_ref("output", True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)

    # # instruction prefetching
    # cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]         
    
    # # if the pointer reaches the start of the next cfg, update the cfg
    # if tab_pointer == cfg_slot_s:
    #     # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
    #     curr_cfg.update(cached_cfg)
    #     # move the pointer to the next cfg
    #     _SchedTab.idx_plus_1()

    # At the end of each cfg chunk
    # logic for updating the cfg and replenish the budget
    if curr_cfg.slot_e < n_slot or n_slot == 0: 
        # print(f"		cfg of bin {bin_name:s} is updated @ {curr_t:.6f}")
        next_cfg = curr_cfg.action_at_end_cfg(timestep, _SchedTab, tab_temp_size, tab_pointer, hyper_p_n)
        
        # process the events in scheduling table
        process_tab_event(sched, curr_t, ready_queue, running_queue, throttle_list, curr_cfg.event_list, _SchedTab, process_dict, bin_id, 
                          msg_filter={"event_type": "migrate_to"})

        # print cfg info
        if DEBUG_FG:
            if bin_name and not bin_event_flg:
                bin_event_flg = True 
                print(f"({bin_name})")
            print(f"bin {bin_name:s} {curr_cfg.slot_s*timestep:.6f}~{curr_cfg.slot_e*timestep:.6f}")
            print(str(next_cfg))


    # At the beginning of each cfg chunk
    if curr_cfg.slot_s == n_slot:
        # cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
        curr_cfg.action_at_start_cfg(budget_recoder, process_dict, bin_id)

        # process the events in scheduling table
        process_tab_event(sched, curr_t, ready_queue, running_queue, throttle_list, curr_cfg.event_list, _SchedTab, process_dict, bin_id,
                          msg_filter={"event_type": "migrate_from"})
        
        # instruction prefetching
        cfg_slot_s, cached_cfg, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: Queue for the weight prefetching
        # TODO: how to represent the tile prefetching: when to start, when to check
        data_prefetching(sched, process_dict, w_data_pipe, curr_t, bin_id, cached_cfg=cached_cfg)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady_cbs(sched, buffer, budget_recoder, active_list, ready_queue, throttle_list, curr_t, glb_name_p_dict, event_cache, bin_name, process_dict, show_warnings=show_warnings)
    # move the task to the ready queue
    bin_event_flg = throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name, bin_event_flg)

    # for _p in ready_queue.queue:
    #     # NOTE:cross cancelation
    #     _p:ProcessInt
    #     rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
    #     if len(rem_flop_budget) > 1:
    #         planned_flops = sum(rem_flop_budget.values())
    #         assert len(rem_flop_budget) == 2
    #         # move the remained budget to the current partition
    #         previous_partition_id = (set(rem_flop_budget.keys()) - {bin_id,}).pop()
    #         print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is cancled from bin {previous_partition_id:d} \
    #                 and moved to bin {bin_id:d} with budget {planned_flops:.6f}")
    #         _p.rem_flop_budget[previous_partition_id] = 0
    #         _p.rem_flop_budget[bin_id] = planned_flops

    #         # judge whether current slot is executed on other patition
    #         # if so, pass the current slot
    #         if previous_partition_id < bin_id:
    #             skiped_tasks.append(_p)

    if sched.drain_flg:
        sched.check_draining_state()

    # check the running tasks
    for _p in running_queue.queue:
        # detect the lateness of the tasks
        # if _p.pid not in curr_cfg.rsc_map: 
        if budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2] < n_slot:
            if show_warnings:
                warnings.warn("Execution lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

    # build the local running configuration
    # Try to allocate the resource to the ready tasks
        # combine the ready tasks, running tasks, budget, and deadline together
        # co-operate the designed properties with EDF algorithm
            # stationary/movable
            # realtime/deadline
            # w/ or w/o need to be scaled
        # first stationary, then movable
        # tackle 3 cases:
            # 1. arrival lateness
            # 2. tighter deadline
            # 3. workload scaling
    # Two components:
        # 1. when to trigger
        # 2. compenstation algorithm
    
    # intution:
        # 1. as soon as possible (ASAP)
        # 2. as evenly as possible (AEAP)        
    # here, we adapt the ASAP to simplify the scheduling process

    # trigger condition:
        # detect risk of current block execution timeout

    # calculate the criticity of the task
    # the most critical one is the first one, with the smallest value, use the ascending order

    is_not_hard = lambda x: x.task.criticality != "hard"

    fn_crit = lambda x: x.deadline
    # fn_crit = lambda _p: budget_recoder[_p.pid][0] + budget_recoder[_p.pid][2]

    fn_task_flag = lambda x: 0 if x.task.task_flag=="stationary" else 1

    # =============== cheduler trigger condition ===============

    # monitored item: (budget, status) x {ready queue, running queue, No change} x (resource)
    # Positive cases that trigger the scheduler: 
    # A. running queue status changes: some task finish/miss/throttled & release cores
    # B. ready queue status changes: some task release/changes priority, these tasks may be issued to 
    #   idle resources, or preempt others
    # C. cfg of running tasks changes: want to extend/release some cores
    # D. cfg of ready tasks changes: the tasks was rejected with old cfg, but can be accepted with current one
    # E. resource ++: some task is waiting to get cores
    # F. some running task is starving 

    # Negative cases that not trigger the scheduler:
    # N.A: some free resources but all other thing not changes, 
    #   which means scheduler will get the same allocation for a new calling 

    # If no execution lateness in previous cfg, 
    # then the running queue is empty all the tasks chunks shares the same deadline; 
    # without spec changes, 
    # the execution sequence not matter the schedulibility. 
    # Otherwise, the task is sorted by the 
    # deadline, host or guest, running or not

    pre_rsc = res_cfg.rsc_map
    task_queue = running_queue.queue + ready_queue.queue
    # ready queue status
    trigger_condA = sched.new_ready_flg
    # running queue status && resource ++ 
    trigger_condB = (set(pre_rsc.keys()) != set(pre_rsc_bk.keys())) and (sum([_p.is_starving for _p in running_queue.queue]) > 0 or len(ready_queue) > 0)
    # budget indicator
    trigger_condC = sum([budget_recoder[_p.pid][3] for _p in task_queue])


    # Compared with the original scheduler, this scheduler remove the ability to 
    # adjust the resource allocation based on the workload. 
    # Consequntly, the scheduler only try to allocate the resource as planned, 
    # if there are enough resources, otherwise, the task is starving. 
    if trigger_condA or trigger_condB or trigger_condC:

        # =============== build the scehduling candidate list ===============
        preemptable_list, curr_aval_rsc = check_prempt(res_cfg, running_queue, quantumSize, quantum_check_en, preemption_en)

        # =============== no need to filter the tasks ===============
        score_fn = lambda x: (fn_crit(x), x not in preemptable_list)
        # ===================================================================
        threshold_score = (np.inf, 1, True)
        filtered_ready_queue = [_p for _p in ready_queue.queue if score_fn(_p) < threshold_score]
        sorted_queue = sorted(filtered_ready_queue+preemptable_list, key=score_fn,)

        rsc_map = OrderedDict()

        while len(sorted_queue) > 0 and curr_aval_rsc > 0:
            _p:ProcessInt = sorted_queue[0]
            _p.is_starving = False

            # =============== calculate the required resource size ===============
            # 
            # get the newest assigned budget
            # planned_rsc_size = budget_recoder[_p.pid][1] # curr_cfg.rsc_map[_p.pid]
            # planned_slot_num = budget_recoder[_p.pid][2] # curr_cfg.slot_num
            
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            # chunk_flops = chunk_alloc * chunk_slot_num * timestep * FLOPS_PER_CORE # _p.rem_flop_budget[bin_id]
            chunk_e = chunk_s + chunk_slot_num
            # late_slot_num = fn_trig(_p) 
            # planned_flops = sum([v for v in _p.rem_flop_budget.values() if v > numerical_error_tol_abs])

            # - We discuss this issue in two scenarios:
            #     1. with data arriving on time: allocate the resources according to the budget
            #     2. with data arriving late: allocate the resources following the "EDF", and estimate the resources at runtime

            # lateness detection mechanism:
            #  both task-level and chunk-level
            # case 1: release late, i.e., the task is not released at the beginning of the current configuration
            # case 2: previous chunk is late, i.e., the task is not finished at the end of the previous configuration
            # case 3: current chunk is late, i.e., the task is not resumed at the beginning of the current configuration

            req_rsc_size = chunk_alloc 

            # **************************************************************
            # check the rsc_size is valid
            # compare with the core_max, core_min, core_list, parallel_mode
            # **************************************************************

            if req_rsc_size > curr_aval_rsc:
                if show_warnings: 
                    warnings.warn(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is starving {req_rsc_size-curr_aval_rsc:d} cores")
                req_rsc_size = curr_aval_rsc
                _p.is_starving = True

            if _p.parallel_mode in ["upb","range"]:
                if req_rsc_size > _p.core_max:
                    req_rsc_size = _p.core_max
            elif _p.parallel_mode in ["lwb", "range"]:
                if req_rsc_size < _p.core_min:
                    if _p.core_min > curr_aval_rsc:
                        # no available solution
                        if o3_boost_util_en:
                            skiped_tasks.append(_p)
                            sorted_queue.pop(0)
                            continue
                        else:
                            break
                    else:
                        req_rsc_size = _p.core_min
            elif _p.parallel_mode == "list":
                # select the nearest one
                # filter the core_list by the current available resource
                core_list = [x for x in _p.core_list if x <= curr_aval_rsc]
                if len(core_list) == 0:
                    # no available solution
                    if o3_boost_util_en:
                        skiped_tasks.append(_p)
                        sorted_queue.pop(0)
                        continue
                    else:
                        break
                req_rsc_size = min(core_list, key=lambda x:abs(x-req_rsc_size))

            if _p.totburst == 0 and chunk_s < n_slot:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is deteted a lateness of {(n_slot-chunk_s):d} slots")

            assert isinstance(req_rsc_size, (int, np.integer)), "req_rsc_size is not integer"
            if req_rsc_size == 0:
                if o3_boost_util_en:
                    skiped_tasks.append(_p)
                    sorted_queue.pop(0)
                    continue
                else:
                    break
            assert req_rsc_size > 0

            if _p.load_var == 0:
                req_rsc_size = 0
            
            # TODO: if the req_rsc_size is larger than the aval_rsc, then add a flag to indicate the task is late                
            sorted_queue.pop(0)
            # if curr_aval_rsc >= req_rsc_size: # and req_rsc_size > 0:
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            _p.required_resource_size = req_rsc_size

        sched.new_ready_flg = False

        to_assert_flag = False
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # remove the expired task from the position dict
            for pid in expired_pid:                
                preempt_list.append(process_dict[pid])

            for pid in old_pid:
                old_size = pre_rsc[pid]
                new_size = rsc_map[pid]
                if old_size != new_size:
                    ctx_switch_list.append(process_dict[pid])
            
            for pid in new_pid:
                issue_list.append(process_dict[pid])
            
            # update the position dict
            update_phy_posi(sched, position_dict, pre_rsc, rsc_map, expired_pid, old_pid, new_pid)
            to_assert_flag, off_discount, on_discount = plan_switching(preempt_list, issue_list, ctx_switch_list, budget_recoder, pre_rsc, rsc_map)

            # update the resource configuration
            for _p in preempt_list:
                print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) preempted at {curr_t:.6f};")
                running_queue.remove(_p)
                ready_queue.put(_p)
                sched.new_ready_flg = True
                sched.res_release(_p.pid, False)
            preempt_list.clear()
            
            sched.handle_taskqueue(curr_t, bin_event_flg, pre_rsc, rsc_map)

        for pid in budget_recoder:
            budget_recoder[pid][3] = False
        
        if to_assert_flag:
            # assert a barrier, sim data movement: 
            if sched.barrier_en:
                off_size = (sum(pre_rsc.values())-off_discount) * 1/3
                on_size = (sum(rsc_map.values()) - on_discount) * 2/3
                x = (on_size+off_size) 
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    if sched.drain_flg:
        sched.drain_old_cores()
    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue, True, bin_id, skiped_tasks=skiped_tasks) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()

    if n_slot < sim_slot_num-1:
        next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
        if DEBUG_FG:
            if curr_cfg_ref != next_cfg:
                print(f"		cfg of bin {bin_name:s} will be updated @ {curr_t+timestep:.6f},")
            if np.logical_xor(curr_cfg_ref != next_cfg, curr_cfg.slot_s == n_slot+1 or curr_cfg.slot_e == n_slot):
                print("ERROR: cfg not match")

def scheduler_step_pglb(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                           n_slot:int, timestep:float, 
                           event_range:float, sim_slot_num:int, curr_t:float, 
                           glb_name_p_dict, res_cfg:Resource_model_int, msg_queue:Queue, a_msg_queue, sensor_msg_queue, 
                           monitor:Monitor, DEBUG_FG=False, 
                    quantum_check_en:bool = False, quantumSize=None, 
		    preemption_en:bool=True,
                    o3_boost_util_en:bool=False,
		    show_warnings=True):
    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _SchedTab, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    _SchedTab:SchedulingTableInt
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
    event_cache:EventCache = sched.event_cache
    trigger_cache:TriggerCache = sched.trigger_cache

    # extract the scheduling table
    # tab_temp_size = len(_SchedTab.scheduling_table)
    # tab_pointer = n_slot % tab_temp_size
    # hyper_p_n = int(n_slot/tab_temp_size)
    # curr_cfg_ref = _SchedTab.scheduling_table[tab_pointer] 
    bin_name = _SchedTab.name
    bin_id = _SchedTab.id
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[bin_id]
    w_msg_queue = w_data_pipe.queues[bin_id]
    bin_spatial_size = _SchedTab.num_resources

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    pre_rsc_bk = deepcopy(res_cfg.rsc_map)

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    read_msg_queue(sched, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id)

    bin_event_flg = check_miss(sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    a_data_pipe.data_tranfer_sim(curr_t)
    # out of order originated from data transfering 
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue, event_cache)

    trigger_read(inactive_list, sensor_msg_queue, trigger_cache, process_dict, timestep, curr_t, True)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, 
                                                  event_cache, trigger_cache,
                                                  bin_event_flg, bin_name)

    # TODO: simulate the congestion and the latency of the network
    w_data_pipe.data_tranfer_sim(curr_t)
    # read out all message and clear the message pipe
    for data in w_msg_queue:
        buffer.put(data)
    w_msg_queue.clear()

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 


    # Scheduler is triggered when:
    # either the aval_rsc or the candidate changes, i.e.,
        # 1. new tasks join the ready queue, preemption may happen
        # 2. some tasks leave the running queue, replenishment and curveup may happen
        # 1+2. both 1 and 2 happen: ressignment-in-turn, preemption, curveup, and replenishment may happen

    # compare the new cfg with the old one to decide the preemption
    pre_rsc = res_cfg.rsc_map
    trigger_condA = sched.new_ready_flg
    trigger_condB = set(pre_rsc.keys()) != set(pre_rsc_bk.keys()) 

    if trigger_condA or trigger_condB: 
        preemptable_list, curr_aval_rsc = check_prempt(res_cfg, running_queue, quantumSize, quantum_check_en, True)
        # sort the tasks in the ready queue and the running queue
        sort_fn = lambda x: x.deadline
        sorted_queue = sorted(ready_queue.queue + preemptable_list, key=sort_fn)

        rsc_map = OrderedDict() # record the resource allocation
        score_dict = OrderedDict() # record the process allocated with resources and unbouned by constraints
        constr_dict = OrderedDict() # record the resource constraint applied to current allocation
        slack_dict = OrderedDict()
       
        # build score dict refer to remaining slack
        for _p in sorted_queue:
            time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep) 
            if time_slot_s >= time_slot_e:
                slack_dict[_p.pid] = 0
            else:
                slack_dict[_p.pid] = time_slot_e - time_slot_s
        
        while len(sorted_queue) > 0:
            _p:ProcessInt = sorted_queue.pop(0)
            assert _p.remburst > 0
            # alloc resource depending to the number of resources
            # if curr_t < _p.deadline:
            if slack_dict[_p.pid] != 0:
                # estimate the runtime and the resource requirement
                req_rsc_size, got_latency, constr = EstimCoreNums4Process(_p, _p.remburst, slack_dict[_p.pid]*timestep, 
                                                                            "ceil", curr_aval_rsc)
            else:
                req_rsc_size, constr = _p.get_available_cfg(curr_aval_rsc, curr_aval_rsc)
                got_latency = _p.remburst/req_rsc_size/FLOPS_PER_CORE
            # check if process has no solution
            if req_rsc_size == 0 and constr == "N/A":
                # the process (we call it as failure) get no cores due to the resource constraint
                # but it is the process with higher priority rather the lower one blocking it
                # so we allow the search to continue, 
                # the process with lower priority is allowed to get the resource. 
                # Once the higher priority process release the resource,
                # and failures can get enough resource to run, 
                # failures can preempt the lower priority process anytime.
                continue
            assert req_rsc_size > 0
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            constr_dict[_p.pid] = constr
            score_dict[_p.pid] = 1/slack_dict[_p.pid] if slack_dict[_p.pid] != 0 else float("inf")
            if curr_aval_rsc <= 0:
                break
        
        if curr_aval_rsc > 0:
            # if there are still resources left, 
            # it means no late process is waiting for resources
            assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
            # also, there is no process waiting for resources in the ready queue
            assert len(sorted_queue) == 0

            while curr_aval_rsc > 0 and len(score_dict) > 0:
                for pid in list(score_dict.keys()):
                    if constr_dict[pid] == "upb":
                        score_dict.pop(pid)
                
                if len(score_dict) > 0:
                    # allocate the remaining resources proportionally to the score
                    core_distr(rsc_map, score_dict, curr_aval_rsc)
                    curr_aval_rsc = 0
                # check the rsc_size is valid
                # compare with the core_max, core_min, core_list, parallel_mode
                for pid in score_dict:
                    _p = process_dict[pid]
                    aval_size, constr = _p.get_available_cfg(rsc_map[pid], rsc_map[pid])
                    assert constr in ["upb", "none"]
                    curr_aval_rsc += rsc_map[pid] - aval_size
                    constr_dict[pid] = constr
                    rsc_map[pid] = aval_size

        sched.new_ready_flg = False
        
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # remove the expired task from the position dict
            for pid in expired_pid:                
                preempt_list.append(process_dict[pid])

            for pid in old_pid:
                ctx_switch_list.append(process_dict[pid])
            
            for pid in new_pid:
                issue_list.append(process_dict[pid])
            
            # update the position dict
            update_phy_posi(sched, position_dict, pre_rsc, rsc_map, expired_pid, old_pid, new_pid)

            # update the resource configuration
            sched.handle_taskqueue(curr_t, bin_event_flg, pre_rsc, rsc_map)

            # assert a barrier
            # data movement: 
            # size: 40MB
            # bandwidth: 100GB/s
            # direction: off-chip -> on-chip, on-chip -> off-chip
            # latency: 100ns
            if sched.barrier_en:
                off_size = sum(pre_rsc.values()) * 1/3
                on_size = sum(rsc_map.values()) * 2/3
                x = (on_size+off_size)
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()

def glb_dynamic_sched_step(sched:Scheduler, msg_dispatcher:MsgDispatcher, a_data_pipe:DataPipe, w_data_pipe:DataPipe, 
                           n_slot:int, timestep:float, 
                           event_range:float, sim_slot_num:int, curr_t:float, 
                           glb_name_p_dict, res_cfg:Resource_model_int, msg_queue:Queue, 
                           monitor:Monitor, DEBUG_FG=False, quantum_check_en:bool = False, quantumSize=None,
                           show_warnings=True):

    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, issue_list, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _, budget_recoder, rsc_recoder_his, process_dict = sched.get_state()
    buffer:Buffer = sched.get_buffer()
    res_cfg:Resource_model_int = sched.res_cfg
        
    # extract the scheduling table
    bin_name = ""
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[0]
    _SchedTab = sched._SchedTab
    bin_spatial_size = _SchedTab.num_resources

    if sched.assert_barrier:
        barrier_state = barrier.update(timestep)
        if not barrier_state:
            print(f"		Barrier is satisfied at {curr_t:.6f}")
            sched.assert_barrier = False
        barrier.cumulative_time += timestep

    pre_rsc_bk = deepcopy(res_cfg.rsc_map)
    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(
        sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, 
        res_cfg, running_queue, completed_list, 
        inactive_list, buffer, bin_event_flg, bin_name, process_dict=process_dict
        )

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    bin_event_flg = check_miss(sched, None, timestep, None, a_data_pipe, curr_t, res_cfg, weight_wait_queue, ready_queue, running_queue, miss_list, 
                            throttle_list, active_list, inactive_list, buffer, bin_event_flg, bin_name, show_warnings=show_warnings)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)
    # buffer.recyle_no_ref("output", True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    # a_data_pipe.data_tranfer_sim(curr_t)
    # cache all the src and weight data
    while a_data_pipe.buffer.queue:
        data:Data
        mode, data, dest = a_data_pipe.buffer.queue[0]
        a_data_pipe.remain_cap += data.size
        data.valid = True
        data.update_receive_time(curr_t)
        a_data_pipe.buffer.get()
        a_data_pipe.broadcast_message(data, prefix="  ")
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list)

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 

    # sort the tasks in the ready queue and the running queue
    sort_fn = lambda x: x.deadline

    # Scheduler is triggered when:
    # either the aval_rsc or the candidate changes, i.e.,
        # 1. new tasks join the ready queue, preemption may happen
        # 2. some tasks leave the running queue, replenishment and curveup may happen
        # 1+2. both 1 and 2 happen: ressignment-in-turn, preemption, curveup, and replenishment may happen

    # compare the new cfg with the old one to decide the preemption
    pre_rsc = res_cfg.rsc_map
    trigger_condA = sched.new_ready_flg
    trigger_condB = set(pre_rsc.keys()) != set(pre_rsc_bk.keys()) 

    if trigger_condA or trigger_condB: 
        # =============== build the scehduling candidate list ===============
        preemptable_list, curr_aval_rsc = check_prempt(res_cfg, running_queue, quantumSize, quantum_check_en, True)
        sorted_queue = sorted(ready_queue.queue + preemptable_list, key=sort_fn)

        rsc_map = OrderedDict() # record the resource allocation
        score_dict = OrderedDict() # record the process allocated with resources and unbouned by constraints
        constr_dict = OrderedDict() # record the resource constraint applied to current allocation
        slack_dict = OrderedDict()
       
        # build score dict refer to remaining slack
        for _p in sorted_queue:
            time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep) 
            if time_slot_s >= time_slot_e:
                slack_dict[_p.pid] = 0
            else:
                slack_dict[_p.pid] = time_slot_e - time_slot_s
        
        while len(sorted_queue) > 0:
            _p:ProcessInt = sorted_queue.pop(0)
            assert _p.remburst > 0
            # alloc resource depending to the number of resources
            # if curr_t < _p.deadline:
            if slack_dict[_p.pid] != 0:
                # estimate the runtime and the resource requirement
                req_rsc_size, got_latency, constr = EstimCoreNums4Process(_p, _p.remburst, slack_dict[_p.pid]*timestep, 
                                                                            "ceil", curr_aval_rsc)
            else:
                req_rsc_size, constr = _p.get_available_cfg(curr_aval_rsc, curr_aval_rsc)
                got_latency = _p.remburst/req_rsc_size/FLOPS_PER_CORE
            # check if process has no solution
            if req_rsc_size == 0 and constr == "N/A":
                # the process (we call it as failure) get no cores due to the resource constraint
                # but it is the process with higher priority rather the lower one blocking it
                # so we allow the search to continue, 
                # the process with lower priority is allowed to get the resource. 
                # Once the higher priority process release the resource,
                # and failures can get enough resource to run, 
                # failures can preempt the lower priority process anytime.
                continue
            assert req_rsc_size > 0
            curr_aval_rsc -= req_rsc_size
            rsc_map[_p.pid] = req_rsc_size
            constr_dict[_p.pid] = constr
            score_dict[_p.pid] = 1/slack_dict[_p.pid] if slack_dict[_p.pid] != 0 else float("inf")
            if curr_aval_rsc <= 0:
                break
        
        if curr_aval_rsc > 0:
            # if there are still resources left, 
            # it means no late process is waiting for resources
            assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
            # also, there is no process waiting for resources in the ready queue
            assert len(sorted_queue) == 0

            while curr_aval_rsc > 0 and len(score_dict) > 0:
                for pid in list(score_dict.keys()):
                    if constr_dict[pid] == "upb":
                        score_dict.pop(pid)
                
                if len(score_dict) > 0:
                    # allocate the remaining resources proportionally to the score
                    core_distr(rsc_map, score_dict, curr_aval_rsc)
                    curr_aval_rsc = 0
                # check the rsc_size is valid
                # compare with the core_max, core_min, core_list, parallel_mode
                for pid in score_dict:
                    _p = process_dict[pid]
                    aval_size, constr = _p.get_available_cfg(rsc_map[pid], rsc_map[pid])
                    assert constr in ["upb", "none"]
                    curr_aval_rsc += rsc_map[pid] - aval_size
                    constr_dict[pid] = constr
                    rsc_map[pid] = aval_size

        sched.new_ready_flg = False
        
        if rsc_map != pre_rsc:
            # layout strategy: 
            #   Currently, we only consider 1D layout, with a huristic algorithm: 
            #   reallocating the position from the original base position, i.e., cum_pos, 
            #   looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
            #   the task decrease the size is handled at first. 
            new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
            expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
            old_pid = set(pre_rsc.keys()) - expired_pid

            # remove the expired task from the position dict
            for pid in expired_pid:                
                preempt_list.append(process_dict[pid])

            for pid in old_pid:
                ctx_switch_list.append(process_dict[pid])
            
            for pid in new_pid:
                issue_list.append(process_dict[pid])
            
            # update the position dict
            # update_phy_posi(sched, position_dict, pre_rsc, rsc_map, expired_pid, old_pid, new_pid)

            # update the resource configuration
            sched.handle_taskqueue(curr_t, bin_event_flg, pre_rsc, rsc_map)

            # assert a barrier
            # data movement: 
            # size: 40MB
            # bandwidth: 100GB/s
            # direction: off-chip -> on-chip, on-chip -> off-chip
            # latency: 100ns
            if sched.barrier_en:
                off_size = sum(pre_rsc.values()) * 1/3
                on_size = sum(rsc_map.values()) * 2/3
                x = (on_size+off_size)
                barrier.assert_barrier(sched.get_ctx_lat(x))
                print(f"		Barrier asserted at {curr_t:.6f}; Counter {barrier.reset_time}s")
                sched.assert_barrier = True

    # execute the task in running list
    # update the running task
    if not barrier.state():
        res_cfg.updateRunningQueue(timestep, running_queue) 

    if not sched.assert_barrier:
        monitor.add_a_record(res_cfg)
    else:
        monitor.add_a_placehold_record()


# === B6-SPLIT-003 (2026-06-29): relocated from slack_estim.py (sole consumer = sched_fn) ===
# deps: math (top-level), Dict (typing). Self-contained, no slack_estim internal deps.
def EstimCoreNums4Process(_p:ProcessInt, flops, expected_slack, 
                          round_mode="round", curr_aval_rsc:int=None):
    if round_mode == "ceil":
        round_func = math.ceil
    elif round_mode == "floor":
        round_func = math.floor
    else:
        round_func = round
    req_rsc_size = flops / expected_slack / FLOPS_PER_CORE

    constr = None
    if _p.parallel_mode in ["upb","range"]:
        req_rsc_size = min(round_func(req_rsc_size), _p.core_max)
        if req_rsc_size==_p.core_max: 
            constr = "upb"
    elif _p.parallel_mode in ["lwb", "range"]:
        if curr_aval_rsc is not None:
            if _p.core_min > curr_aval_rsc:
                # no available solution
                return 0, "N/A"
        req_rsc_size = max(round_func(req_rsc_size), _p.core_min)
        if req_rsc_size==_p.core_min:
            constr = "lwb"
    elif _p.parallel_mode == "list":
        # select the nearest one
        # filter the core_list by the current available resource
        if curr_aval_rsc is not None:
            core_list = [x for x in _p.core_list if 0 < x <= curr_aval_rsc] 
            if len(core_list) == 0:
                # no available solution
                return 0, "N/A"
        else:
            core_list = _p.core_list
        req_rsc_size = min(_p.core_list, key=lambda x:abs(x-req_rsc_size))
        if req_rsc_size==max(_p.core_list):
            constr = "upb"
        elif req_rsc_size==min(_p.core_list):
            constr = "lwb"
    else:
        req_rsc_size = max(round_func(req_rsc_size), 1)
    if curr_aval_rsc is not None:
        req_rsc_size = min(req_rsc_size, curr_aval_rsc)
    got_latency = flops / req_rsc_size / FLOPS_PER_CORE
    return req_rsc_size, got_latency, constr

def EstimCoreNums4Task(task_dict:Dict[str, TaskBase], flops_dict, node, expected_slack, round_mode="round"):
    if round_mode == "ceil":
        round_func = math.ceil
    elif round_mode == "floor":
        round_func = math.floor
    else:
        round_func = round
    req_rsc_size = flops_dict[node] / expected_slack / FLOPS_PER_CORE

    constr = None
    _task = task_dict[node]
    if _task.parallel_mode in ["upb","range"]:
        req_rsc_size = min(round_func(req_rsc_size), _task.core_max_compile)
        if req_rsc_size==_task.core_max_compile: 
            constr = "upb"
    elif _task.parallel_mode in ["lwb", "range"]:
        req_rsc_size = max(round_func(req_rsc_size), _task.core_min_compile)
        if req_rsc_size==_task.core_min_compile:
            constr = "lwb"
    elif _task.parallel_mode == "list":
        # select the nearest one
        # filter the core_list by the current available resource
        req_rsc_size = min(_task.core_list_compile, key=lambda x:abs(x-req_rsc_size))
        if req_rsc_size==max(_task.core_list_compile):
            constr = "upb"
        elif req_rsc_size==min(_task.core_list_compile):
            constr = "lwb"
    else:
        req_rsc_size = max(round_func(req_rsc_size), 1)
    got_latency = elim_nume_error(flops_dict[node] / req_rsc_size / FLOPS_PER_CORE)
    return req_rsc_size, got_latency, constr
