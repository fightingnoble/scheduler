"""global_sched_repack.py — Step 3 repack (Intra-Bin Time Window Assignment).

Moved from global_sched.py (B6-SPLIT-001, 2026-06-29). Byte-identical relocation.
Functions: push_task_into_bins_new, push_step_new.
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from typing import List, Dict
    from sched.scheduling_table import SchedulingTableInt

import math
from collections import OrderedDict, defaultdict
import numpy as np
from global_var import *
from model.resource_agent import Resource_model_int
from model.task_queue_agent import TaskQueue 
from model.buffer import Buffer
from model.message.msg_dispatcher import MsgDispatcher
from model.message.message_handler import message_trigger_event_new
from model.streaming_processing.wartermark_strategy import WatermarkStrategy
from model.message.data_pipe import DataPipe
from model.position_table import PosTableInt
from model.performance import slack_comp

from sched.monitor_agent import Monitor
from sched.scheduler_agent import Scheduler, check_miss, check_complete
from sched.scheduler_agent import data_pipe_read, pendingToReady
from sched.pre_alloc_new import glb_alloc_new2
from sched.bin_ops import new_bin, get_initlist_and_biniter, static_1_bin
from sched.sort_function import get_process_sort
from sched.packing_solver.gurobi_MP_semi2DClst import ClusterGurobiSolverSemi2D
from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id
from sched.bin_ops import bin_iter_list
from networkx import DiGraph
from functools import reduce
from sched.slack_estim import get_chains

from sched.binpack_config import BinPackConfig

# 默认配置实例（函数签名 fallback；实际运行时由 input_parser 从 JSON 加载）
from sched.binpack_config import BinPackConfig
default_binpack_cfg = BinPackConfig()

def push_task_into_bins_new(
        
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, quantile,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
        ):

    """
    implement a naive 2d bin-packing algorithm
    input: task_list, which is already arranged in the topological order
    output: a list of bins, each bin is a list of tasks
    """
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores

    # --- 入口参数检查与读取 ---
    bin_sel_mod = binpack_cfg.get("bin_sel_mod", "search")
    reservation_policy = binpack_cfg.get("reservation_policy", "manual")
    # 注入 total_cores 到 binpack_cfg，供下层函数使用
    # 注意：这里直接修改字典，因为 binpack_cfg 通常是每次调用时新创建的
    binpack_cfg["total_cores"] = total_cores
    # -----------------------

    glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    sched, monitor,  msg_queue = scheduler_list[0], monitor_list[0], msg_dispatcher.queues[0]
    inactive_list:List[ProcessInt] = sched.inactive_list
    for _p in glb_p_list:
        if _p not in inactive_list:
            inactive_list.append(_p)
    sched.process_dict.update({p.pid:p for p in glb_p_list})
    rsc_recoder = sched.budget_recoder

    def issue_sort_fn(x:ProcessInt):
        alloc_slot_s, alloc_size, allo_slot, bin_id = rsc_recoder[x.pid]
        if isinstance(alloc_slot_s, int):
            return alloc_slot_s
        else:
            return alloc_slot_s[0]
    # monitor the issue time: (ascending)
    issue_list:TaskQueue = TaskQueue(sort_f=issue_sort_fn, descending=False)
    curr_cfg:Resource_model_int
 
    # try to push the task into the bins in the bin_list
    # if the task cannot be pushed into any bin, create a new bin

    # define _new_bins
    # _new_bins = lambda id: new_bins(total_cores, int(sim_range/timestep), id=id, name="bin"+str(id))
    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)


    if bin_sel_mod != "pre_defined":
        iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
            bin_list, glb_p_list, total_cores, 
            _new_bin, reservation_policy)
    else:
        bin_name_list = [_bin.name for _bin in bin_list]
        iter_next_bin_obj = iter([])

    for n_slot in range(sim_slot_num):
        curr_t = n_slot * timestep

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")
        
        # enqueue the process that is released in this slot
        # push_step(init_p_list, quantum_check_en, quantumSize, timestep, animation, event_range, sim_slot_num, pid_max, wait_queue, ready_queue, running_queue, rsc_recoder, rsc_recoder_his, issue_sort_fn, issue_list, completed_list, miss_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list, frame_list, ax, plot_window, n_slot, curr_t)
        message_trigger_event_new(event_iter_dict, inactive_list, glb_p_list, None, None, None, timestep, curr_t, True) 
        push_step_new(
            sched, msg_dispatcher, a_data_pipe, w_data_pipe, 
            n_slot, timestep, 
            event_range, sim_slot_num, curr_t, 
                        
            glb_name_p_dict, None, 
            issue_sort_fn, issue_list, 
            iter_next_bin_obj, 
            
            quantumSize, bin_list, bin_name_list, 
            
            binpack_cfg=binpack_cfg,
            show_warnings=show_warnings,
            verbose=verbose, DEBUG_FG=DEBUG_FG
            )
        # update the wait task
        w_data_pipe.update_wait_time(timestep)
        a_data_pipe.update_wait_time(timestep)

    for _bin in bin_list:
        _bin.alloc_mod = "compact"
    return bin_list

def push_step_new(
        sched: Scheduler, msg_dispatcher: MsgDispatcher,
        a_data_pipe: DataPipe, w_data_pipe: DataPipe,
        n_slot: int, timestep: float, 
        event_range: float, sim_slot_num: int, curr_t: float,

        glb_name_p_dict, res_cfg: Resource_model_int,
        issue_sort_fn, issue_list:TaskQueue,
        iter_next_bin_obj, 

        quantumSize, bin_list:TaskQueue, bin_name_list, 
        binpack_cfg:Dict,
        show_warnings=True, 
        verbose:bool=False, DEBUG_FG:bool=False,
        ):


    weight_wait_queue, ready_queue, running_queue, \
        miss_list, preempt_list, _, completed_list, throttle_list,\
            inactive_list, active_list = sched.get_queues()
    position_dict=sched.position_dict
    ctx_switch_list:List[ProcessInt] = sched.ctx_switch_list
    barrier = sched.barrier

    curr_cfg, _, rsc_recoder, rsc_recoder_his, process_dict = sched.get_state()
    buffer:Buffer = sched.get_buffer()
        
    # extract the scheduling table
    bin_event_flg = False
    a_msg_queue = a_data_pipe.queues[0]
    bin_name = ""
    _SchedTab = sched._SchedTab

    # (running_queue)
    # check running tasks
    bin_event_flg = check_complete(sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, None, 
                                   running_queue, completed_list, inactive_list, buffer, 
                                   bin_event_flg, bin_name, save_trace=False, 
                                   mode="future", bin_list=bin_list, 
                                   n_slot=n_slot, rsc_recoder=rsc_recoder, 
                                   detail_alloc_info=sched.detail_alloc_info)

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    bin_event_flg = check_miss(sched, None, timestep, None, a_data_pipe, curr_t, None, weight_wait_queue, ready_queue, 
                               running_queue, miss_list, throttle_list, active_list, 
                               inactive_list, buffer, bin_event_flg, bin_name, 
                               mode="future", bin_list=bin_list, n_slot=n_slot, rsc_recoder=rsc_recoder, 
                               show_warnings=show_warnings)

    # spill out the data of type "output", which is expired
    # buffer.pop_timeout("output", curr_t, True)

    # tackle the event in message pipe, set the valid flag in pred_data of each process
    # update barrier status
    # update the data status
    # if not msg_pipe.empty():

    # a_data_pipe.data_tranfer_sim(curr_t)
    # cache all the src and weight data
    # while a_data_pipe.buffer.queue:
    #     data:Data
    #     mode, data, dest = a_data_pipe.buffer.queue[0]
    #     a_data_pipe.remain_cap += data.size
    #     data.valid = True
    #     data.update_receive_time(curr_t)
    #     a_data_pipe.buffer.get()
    #     a_data_pipe.broadcast_message(data, prefix="  ")
    a_data_pipe.data_tranfer_sim(curr_t)
    bin_event_flg = data_pipe_read(curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg, a_msg_queue)

    # check release
    # check the dependencies of the tasks in inactive list
    # if the dependencies are satisfied, move the task to the wait queue
    # bin_event_flg = chk_release(sched, event_range, curr_t, inactive_list, active_list, _SchedTab, timestep, bin_event_flg, bin_name) 
    bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, )

    # check data availability: some tasks may be prefetched
    # TODO: model the runtime weight and feature map transfering 
    pendingToReady(sched, active_list, ready_queue, buffer, curr_t, glb_name_p_dict, bin_name, ) 

    # compare the new cfg with the old one to decide the preemption
    trigger_condA = sched.new_ready_flg
    trigger_condB = sched.new_drop_flg

    if trigger_condA or trigger_condB: 
        # glb_alloc_new(process_dict, quantum_check_en, quantumSize, timestep, exec_t_comp_ratioA, ready_queue, running_queue, rsc_recoder, 
        #             rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list, n_slot, curr_t, 
        #             DEBUG_FG=DEBUG_FG, show_warnings=show_warnings, binpack_cfg=binpack_cfg,)
        glb_alloc_new2(
            process_dict, quantumSize, timestep, 
            ready_queue, running_queue, rsc_recoder, 
            rsc_recoder_his, issue_list, preempt_list, iter_next_bin_obj, 
            bin_list, bin_name_list, n_slot, curr_t, 
            binpack_cfg,
            show_warnings, 
            verbose, DEBUG_FG
        )
        sched.new_ready_flg = False
        sched.new_drop_flg = False

    # issue the task
    # if the task of the queue equals to the current slot, then issue the task
    if len(issue_list):
        for _p in issue_list:
            if _p in ready_queue:
                ready_queue.remove(_p)
        while len(issue_list):
            _p = issue_list[0]
            if issue_sort_fn(_p) == n_slot: 
                # assert _p.get_state_name() != "running"
                running_queue.put(_p)
                if _p.totburst == 0:
                    _p.start_time = curr_t
                _p.waitTime = 0
                issue_list.get()
                # _p.set_state("running")
            else:
                break

    # =================================================
    for _SchedTab in bin_list:
        curr_cfg:Resource_model_int = _SchedTab.scheduling_table[n_slot]
        curr_cfg.updateRunningQueue(timestep, running_queue) 
