from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sched.scheduler_agent import Scheduler
    
import numpy as np
import math, re
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List
from queue import Queue
from global_var import *
from utils import load_pickle, core_distr

from model.buffer import Buffer, EventCache, TriggerCache
from model.buffer import Buffer, Data
from model.resource_agent import Resource_model_int
from model.message.msg_dispatcher import MsgDispatcher
from model.message.Context_message import ContextMsg
from model.message.data_pipe import DataPipe
from model.streaming_processing.wartermark_strategy import WatermarkStrategy

from task.task_agent import ProcessInt
from sched.monitor_agent import Monitor
from sched.slack_estim import EstimCoreNums4Process
from sched.placement import core_mapping_1d

def read_msg_queue(sched:Scheduler, curr_t, msg_queue, ready_queue, throttle_list, inactive_list, active_list, 
                   running_queue, process_dict, bin_name, bin_id, res_cfg=None, budget_recoder=None,):
    # 定义正则表达式模式
    cancel_pattern = r'TASK (\d+):([\w_]+)\((\d+)\) (?:COMPLETED|MISSED DEADLINE) @ ([\d.]+)/([\d.]+)\(([\w_]+)\)!!'
    core_release_pattern = r'CORE ([\d]+) is set free'
    msg_list = []
    # read out all message and clear the message pipe
    while not msg_queue.empty():
        msg_list.append(msg_queue.get())
    if msg_list:
        for msg in msg_list:
        # 使用正则表达式进行匹配
            if match:= re.match(cancel_pattern, msg):
                # 提取匹配结果
                pid = int(match.group(3))
                name = match.group(2)
                event_time = float(match.group(5))
                bin_name_t = match.group(6)
                if pid not in process_dict or bin_name_t == bin_name:
                    continue
                _p = process_dict[pid]

                # pop the task from ready queue, active list, throttle list and running queue
                if _p in active_list:
                    active_list.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from active list @ {bin_name:s} {curr_t:.6f}")
                    inactive_list.append(_p)
                if _p in ready_queue:
                    ready_queue.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from ready queue @ {bin_name:s} {curr_t:.6f}")
                    inactive_list.append(_p)
                if _p in throttle_list:
                    throttle_list.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from throttle list @ {bin_name:s} {curr_t:.6f}")
                    inactive_list.append(_p)
                if _p in running_queue:
                    running_queue.remove(_p)
                    print(f"		{_p.task.name:s}({pid:d}) is removed from running queue @ {bin_name:s} {curr_t:.6f}")
                    sched.res_release(_p.pid)
                    inactive_list.append(_p)
                    # if budget_recoder is not None:
                    #     budget_recoder.pop(_p.pid)
                    # _p.rem_flop_budget.pop(bin_id)
            # elif match := re.match(core_release_pattern, msg):
            #     core_id = int(match.group(1))
            #     if core_id in sched.core_map_L0:
            #         sched.core_map_L0[core_id] = True
            #         if sched.drain_flg:
            #             sched.core_map.append(core_id)
            #             sched.core_map.sort()
            #             sched.core_map_L0.pop(core_id)
                    
def record_comp_bw_slot_by_slot(rsc_recoder, n_slot, curr_cfg, pid):
    if pid in rsc_recoder:
        alloc_slot_s:List[int]
        alloc_size:List[int]
        allo_slot:List[int]
        alloc_slot_s, alloc_size, allo_slot = rsc_recoder[pid]
        # merge the allocation
        if alloc_slot_s[-1] + allo_slot[-1] == n_slot and curr_cfg.rsc_map[pid] == alloc_size[-1]:
            allo_slot[-1] += 1
            rsc_recoder[pid] = [alloc_slot_s, alloc_size, allo_slot]
        else:
            alloc_slot_s.append(n_slot)
            alloc_size.append(curr_cfg.rsc_map[pid])
            allo_slot.append(1)
            rsc_recoder[pid] = [alloc_slot_s, alloc_size, allo_slot]
    else:
        rsc_recoder[pid] = [[n_slot,], [curr_cfg.rsc_map[pid],], [1,]]

def trigger_read(inactive_list:List[ProcessInt], sensor_msg_queue:List, 
                trigger_cache:TriggerCache, process_dict:Dict,
                timestep, curr_t, DEBUG_FG):
    """
        1. Read the sensor message queue and update the trigger cache
        2. Update the ctrl dependency flags
    """

    # for pid, next_ingestion_time, next_event_time in sensor_msg_queue:
    for pid, msg in sensor_msg_queue:
        if pid in process_dict:
            trigger_cache.sensor_cache[pid].append(msg)
    sensor_msg_queue.clear()

    # read the trigger cache
    for _p in inactive_list:
        if trigger_cache is None:
            pred_ctrl = _p.pred_ctrl
            event_triggers = _p.event_triggers
        else:
            pred_ctrl = trigger_cache[_p.pid]
            event_triggers = trigger_cache.sensor_cache[_p.pid]
        
        _p.sim_trigger(curr_t, timestep, pred_ctrl, event_triggers)

def data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue: List[Data], 
                   event_cache:EventCache=None):
    msg_dict:Dict[int, list[Data]] = {}
    # read out all message and clear the message pipe
    for data in a_msg_queue:
        tgt_p_name_l = data.track_downstream()
        for key in tgt_p_name_l:
            tgt_pid = glb_name_p_dict[key].pid
            if tgt_pid in process_dict:
                if tgt_pid not in msg_dict:
                    msg_dict[tgt_pid] = []
                msg_dict[tgt_pid].append(data)
                # add the ref_pid to the data
                data.ref_pid.append(tgt_pid)
    for data in set([data for data_l in msg_dict.values() for data in data_l]):
        buffer.put(data)
    a_msg_queue.clear()

    for tgt_pid, data_l in msg_dict.items():
        _p = process_dict[tgt_pid]
        if event_cache is None:
            pred_data = _p.pred_data
        else:
            pred_data = event_cache[tgt_pid]
        for key, attr in pred_data.items():
            for data in data_l:
                if data.pid == glb_name_p_dict[key].pid:
                    attr["event_queue"].put(data)
                    attr["valid"] = True
                    attr["time"] = curr_t
                    attr["data"] = data
                    
                    # TODO: fix the event time as the actual time
                    if bin_name and not bin_event_flg:
                        bin_event_flg = True
                        print(f"({bin_name})")
                    print(f"		{_p.task.name} received event {key:s} @ {curr_t:.6f}/{data.ctx.get_timestamp():.6f}")
    return bin_event_flg

# =================== functions related to data transfer ===================

def data_prefetching(sched:Scheduler, init_p_list, wait_queue:DataPipe, curr_t, bin_id, cached_cfg=None):    
    # infinite bandwidth, buffer size, constant latency
    for pid in cached_cfg.keys():
        _p = init_p_list[pid]
        msg:ContextMsg = ContextMsg.create_weight_ctx()
        data = Data(_p.pid, _p.io_time, (0,), "weight")
        data.ctx = msg
        data.cache_msg_transfer(curr_t)

        on_size = cached_cfg[pid] * 1/3
        x = on_size
        data.size = sched.get_ctx_lat(x)
        wait_queue.put(data, "unicast", [bin_id,])
        _p.set_state("wait")


# unused functions
def RunningQueueToWait(running_queue, wait_queue:DataPipe):
    # CPU[i]->running->burst == CPU[i]->running->cpu
    l_wait = []
    for _p in running_queue:
        exe_io_tile = _p.burst / _p.cpu_time
        exe_io_tile_r = round(exe_io_tile)
        exe_comp = math.isclose(exe_io_tile, exe_io_tile_r, abs_tol=1e-2)
        if exe_comp:
            l_wait.append(_p)
    for _p in l_wait:
        _p.burst = 0
        _p.ready = False
        l_wait.remove(_p)
        running_queue.remove(_p)
        wait_queue.put(_p)
        print("		TASK {:d}:{:s}({:d}) WAIT!!".format(_p.task.id, _p.task.name, _p.pid))

def update_depend(tasc_dict:Dict[str, ProcessInt], completed_task:List[ProcessInt]): 
    """
    update the dependency list
    input: 
        task_dict: the dictionary of tasks
        completed_task: the list of completed tasks
    action:
        update the dependency list of the successor tasks of the completed tasks
    output:
        None
    """
    for _p in completed_task:
        for _s in _p.succ_data: 
            succ_task = tasc_dict[_s]
            succ_task.pred_data[_p.task.name] = True
        for _s in _p.succ_ctrl:
            succ_task = tasc_dict[_s]
            succ_task.pred_ctrl[_p.task.name] = True

def check_depends(task_list:List[ProcessInt])->List[ProcessInt]:
    """
    check the denpendency of task in the list
    input:
        task_list: the list of tasks
    output:
        ready_task: the list of tasks that all denpendencies are satisfied
    """
    active = []
    for _p in task_list:
        if _p.check_depends():
            active.append(_p)
            print("		TASK {:d}:{:s}({:d}) is avtivated!!".format(_p.task.id, _p.task.name, _p.pid))
    return active

def load_sched_tab(num_cores, e2e_latency:float, aux_scale_factor:int, bin_path_format:str, scheduler_list:List[Scheduler]):
    bin_list_save_path = bin_path_format.format(aux_scale_factor, e2e_latency, num_cores)
    bin_list = load_pickle(bin_list_save_path)
    for _SchedTab, scheduler in zip(bin_list, scheduler_list):
        scheduler._SchedTab_L0 = _SchedTab
        assert scheduler._SchedTab_L0.id == _SchedTab.id
    old_cores = [scheduler._SchedTab.num_resources for scheduler in scheduler_list]
    new_cores = [scheduler._SchedTab_L0.num_resources for scheduler in scheduler_list]
    
    assert len(old_cores) == len(new_cores)
    old_map = core_mapping_1d(old_cores)
    new_map = core_mapping_1d(new_cores)
    for scheduler in scheduler_list:
        scheduler.core_map_L0 = dict.fromkeys(new_map[scheduler._SchedTab_L0.id])
        for k in scheduler.core_map_L0.keys():
            if k in old_map[scheduler._SchedTab.id]:
                scheduler.core_map_L0[k] = True
            
    # reverse the old map
    old_map_rev = {}
    for k, items in old_map.items():
        for item in items:
            old_map_rev[item] = k
    # reverse the new map
    new_map_rev = {}
    for k, items in new_map.items():
        for item in items:
            new_map_rev[item] = k
    # create the mapping from the new scheduler to the old one
    map_new_to_old = {}
    for k, items in new_map.items():
        new_to_old_t = {}
        for item in items:
            src_partition = old_map_rev[item]
            if src_partition not in new_to_old_t:
                new_to_old_t[src_partition] = []
            new_to_old_t[src_partition].append(item)
        map_new_to_old[k] = new_to_old_t
    return map_new_to_old, new_map_rev
    
# Not verified extracted functions
def plan_switching(preempt_list, issue_list, ctx_switch_list, budget_recoder, pre_rsc, rsc_map):
    """
    logic to judge whether the swithing is planned or decided by the scheduler at runtime
    the swithing out of plan features:
    1. some tasks release core and other tasks take the core
    2. this overtake behavior is not planned in the scheduling table
    step: 
    1. check whether some cores are released
    2. check whether this plan is in the scheduling table
    """
    off_discount = 0
    on_discount = 0
    for _p in ctx_switch_list:
        old_size = pre_rsc[_p.pid]
        new_size = rsc_map[_p.pid]
        if old_size > new_size: 
            if budget_recoder[_p.pid][3]:
                chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
                if new_size != chunk_alloc:
                    to_assert_flag = True
                else:
                    off_discount += chunk_alloc
            else:
                to_assert_flag = True
        else:
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            if new_size == chunk_alloc:
                on_discount += chunk_alloc

            
    if len(preempt_list):
        to_assert_flag = True

    if to_assert_flag:
        for _p in issue_list:
            chunk_s, chunk_alloc, chunk_slot_num, updated_flg = budget_recoder[_p.pid]
            if chunk_alloc == rsc_map[_p.pid]:
                on_discount += chunk_alloc
    return to_assert_flag,off_discount,on_discount

