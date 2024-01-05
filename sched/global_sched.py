from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from typing import List, Dict
    from sched.scheduling_table import SchedulingTableInt

import math
from collections import OrderedDict
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

from sched.monitor_agent import Monitor
from sched.scheduler_agent import Scheduler, check_miss, check_complete
from sched.scheduler_agent import data_pipe_read, pendingToReady
from sched.pre_alloc import glb_alloc_new
from sched.pre_alloc_new import glb_alloc_new2
from sched.bin_ops import new_bin, get_initlist_and_biniter, static_1_bin
from sched.sort_function import get_process_sort
from sched.packing_solver.gurobi_MP_semi2DClst import ClusterGurobiSolverSemi2D
from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id
from sched.bin_ops import bin_iter_list
from networkx import DiGraph
from functools import reduce

default_binpack_cfg = {
            "sort":"EAT", "sort_reverse":True, "mode": 'non-block', "partial_alloc_en":False, 
            "quantum_check_en":False, "release_temp_rda":True, "reservation_policy": "manual", 
            "algorithm": "coalescing",
            "bin_sel_mod": "search", # "pre_defined", "search"
            "affinity_en": True, 
            "affinity_level": 2,
            "mapping": {}
}
# ==================== top-level scheduling procedure ====================
def push_task_into_bins_new(
        
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

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

    def get_core_size(_p):
        _, _, req_rsc_size = _p.rsc_req_estm(0, timestep, FLOPS_PER_CORE)
        return req_rsc_size

    bin_sel_mod = binpack_cfg.get("bin_sel_mod", "search")
    if bin_sel_mod != "pre_defined":
        iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
            bin_list, glb_p_list, total_cores, 
            _new_bin, binpack_cfg["reservation_policy"])
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
            n_slot, timestep, exec_t_comp_ratioB, 
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

    pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
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
    layout = {_bin.name:_bin.num_resources for _bin in bin_list}
    print("max_core_num:", sum(layout.values()))
    print(f"max_core_layout: {layout}")
    return bin_list

def push_step_new(
        sched: Scheduler, msg_dispatcher: MsgDispatcher,
        a_data_pipe: DataPipe, w_data_pipe: DataPipe,
        n_slot: int, timestep: float, exec_t_comp_ratioB: float, 
        event_range: float, sim_slot_num: int, curr_t: float,

        glb_name_p_dict, res_cfg: Resource_model_int,
        issue_sort_fn, issue_list,
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
    bin_spatial_size = _SchedTab.num_resources
    binpack_cfg["exec_t_comp_ratioB"] = exec_t_comp_ratioB

    # (running_queue)
    # check running tasks
    release_temp_rda = binpack_cfg.get("release_temp_rda", True)
    # bp_rls_mode = "future" if release_temp_rda else "none"
    bin_event_flg = check_complete(sched, None, timestep, msg_dispatcher, a_data_pipe, curr_t, None, 
                                   running_queue, completed_list, inactive_list, buffer, 
                                   bin_event_flg, bin_name, save_trace=False, 
                                   mode="future", bin_list=bin_list, 
                                   n_slot=n_slot, rsc_recoder=rsc_recoder, 
                                   detail_alloc_info=sched.detail_alloc_info)

    # check whether the task is miss
    # TODO: other ready tasks shoud be checked
    # TODO: cache eviction
    bin_event_flg = check_miss(sched, None, None, curr_t, None, weight_wait_queue, ready_queue, 
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
                running_queue.put(_p)
                if _p.totburst == 0:
                    _p.start_time = curr_t
                _p.waitTime = 0
                issue_list.get()
            else:
                break

    # =================================================
    for _SchedTab in bin_list:
        curr_cfg:Resource_model_int = _SchedTab.scheduling_table[n_slot]
        curr_cfg.updateRunningQueue(timestep, running_queue) 
        # curr_cfg.updateRunningQueue(timestep, running_queue, mode="verify" if release_temp_rda else "normal") 

def naive_iso(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
        ):
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(hyper_p//timestep)
    # assert math.isclose(hyper_p, tab_temp_size*timestep, abs_tol=numerical_error_tol_abs), \
    #         "hyper_p should be the multiple of timestep"
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores
    # glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    from task.task_cfg import pre_assign_priority
    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
        bin_list, glb_p_list, 0, 
        _new_bin, "all_isolation", pre_assign_priority)
    print(bin_list)
    
def test_mem_planner(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
):
    
    event_range = hyper_p * (n_p+warmup)

    from mapper.mem_planner import Block, MemMap, scan_overlap_2d, layout_plot

    # build block list
    block_list = []
    block_idx = 0 
    for _p in glb_p_list:
        _p:ProcessInt
        stimu_tab = _p.task.extract_sensor_event(event_range)
        # (task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t)
        for stimu_t in stimu_tab:
            item = (_p.task.name, _p.pid, 
              _p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size, 
              stimu_t, stimu_t+_p.task.ERT, stimu_t+_p.task.ERT+_p.task.ddl, _p.task.exp_comp_t)
            start_t, ddl_t = item[4], item[5]
            # quantize the start time and ddl time
            slot_s = elim_nume_error(int(math.ceil(start_t/timestep)) * timestep)
            slot_e = elim_nume_error(int(math.floor(ddl_t/timestep)) * timestep)
            # s:float # size
            # r:float # release time
            # c:float # deadline
            # idx:int # index
            block_list.append(Block(item[2], slot_s, slot_e, block_idx))
            block_idx += 1
            
    mapper:MemMap = MemMap()
    position_recoder, conflict_graph = mapper.prority_mapper(block_list, timestep, sort_fn=lambda x: (x.s*x.lifetime**2, x.lifetime, x.s, -x.r, x.idx))
    status = scan_overlap_2d(mapper.position_recoder, block_list)
    layout_plot(position_recoder, block_list, show=True, tick_dens=4)
    print(position_recoder)
    
    
def coleasing_alloc_1bin(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
        ):
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(hyper_p//timestep)
    # assert math.isclose(hyper_p, tab_temp_size*timestep, abs_tol=numerical_error_tol_abs), \
    #         "hyper_p should be the multiple of timestep"
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores
    # glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
        bin_list, glb_p_list, 0, 
        _new_bin, binpack_cfg["reservation_policy"])
    _bin = bin_list[0]

    # build event list
    process_block_sortby_start_slot = []    
    for _p in glb_p_list:
        _p:ProcessInt
        stimu_tab = _p.task.extract_sensor_event(event_range)
        # (task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t)
        for stimu_t in stimu_tab:
            item = (_p.task.name, _p.pid, 
              _p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size, 
              stimu_t, elim_nume_error(stimu_t+_p.task.ERT), elim_nume_error(stimu_t+_p.task.ERT+_p.task.ddl), _p.task.exp_comp_t)
            start_t, ddl_t = item[4], item[5]
            # quantize the start time and ddl time
            slot_s = int(math.ceil(start_t/timestep)) * timestep
            slot_e = int(math.floor(ddl_t/timestep)) * timestep
            process_block_sortby_start_slot.append((item[0], item[1], item[2], item[3], slot_s, slot_e, item[6]))

    # sort the event list by the start time, ddl, exp_comp_t
    process_block_sortby_start_slot.sort(key=lambda x: (x[4], x[5], x[6]))
    event_set = OrderedDict()
    for item in process_block_sortby_start_slot:
        event_set.update({item[4]:None})
        event_set.update({item[5]:None})
    event_set = sorted(event_set.keys())
    
    # convergence flag
    bin_size_ok = False
    slot_ok = np.full(len(event_set), False)
    # some cache
    curr_items = [] # cache the current items
    pending_items = [] # cache the pending items
    max_core_num = _bin.num_resources # cache the max core number
    max_core_layout = {} # start, core_size, slot_num
    position_recoder = PosTableInt(len(event_set)) # cache the current usage of the cores
    prev_t = 0
    cum_flops = {}
    process_dict = {p.pid:p for p in glb_p_list}

    # scan the event list, place, and pop the task into the bin, 
    # expanding the bin size if necessary
    for curr_t in list(event_set):
        # curr_t = n_slot * timestep
        n_slot = int(curr_t/timestep)
        tab_pointer = n_slot % tab_temp_size
        hyper_p_n = int(n_slot/tab_temp_size)

        curr_n_period = int(curr_t/hyper_p)
        prev_n_period = int(prev_t/hyper_p)
        if warmup and prev_t == 0 and curr_n_period == 0:
            print("="*20, "WARMUP", "="*20, "\n")
        elif curr_n_period > prev_n_period:
            if curr_t > event_range:
                print("="*20, "DRAIN", "="*20, "\n")
            else:
                print("="*20, "PERIOD {:d}".format(curr_n_period), "="*20, "\n")

        if prev_t < curr_t and curr_items:
            flops_per_core = (curr_t-prev_t) * FLOPS_PER_CORE 
            flops_dict = {}
            cores_dict = {}
            for item in curr_items:
                # task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = item
                req_size = item[2]
                pid = item[1]
                start_t, ddl_t = item[4], item[5]
                size_del_rda = process_dict[pid].task.flops/FLOPS_PER_CORE/(ddl_t-start_t)
                cores_dict[pid] = int(math.ceil(size_del_rda/(1-exec_t_comp_ratioB)))
                if cum_flops[pid] > 0:
                    flops = flops_per_core*math.ceil(size_del_rda)
                    flops = min(flops, cum_flops[pid])
                    flops_dict.update({pid:flops})
                    cum_flops[pid] -= flops
                else:
                    flops_dict.update({pid:0})
            _bin.sparse_flops.append([round(prev_t/timestep), flops_dict])
            _bin.sparse_cores.append([round(prev_t/timestep), cores_dict])

        # get the pending items
        while len(process_block_sortby_start_slot):
            task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = process_block_sortby_start_slot[0]
            if start_t <= curr_t:
                pending_items.append(process_block_sortby_start_slot.pop(0))
                print("task_name:", task_name, "stimu_t:", stimu_t, "start_t:", start_t, "ddl_t:", ddl_t, "exp_comp_t:", exp_comp_t)
            else:
                break
        
        # pop the items that are finished regarding the ddl
        while len(curr_items):
            if curr_items[0][5] <= curr_t:
                pid = curr_items[0][1]
                assert cum_flops[pid] == 0
                curr_items.pop(0)
                cum_flops.pop(pid)
            else:
                break

        # get the current items
        # use cache curr_items
        # ordered_occupant_dict = _bin.index_occupy_by_id(n_slot)

        # get the history placement of the current items
        if bin_size_ok and slot_ok.all():
            position_dict = position_recoder.get_record(n_slot)
            

        # vefify the legality of the placement of the history items

        # determine the placement of the pending items, as well as the bin size

        all_items = curr_items + pending_items
        # max_core_num
            # without laxity, a tasks is placed iff it is released at the current slot
            # no neet to scan the bin in the future monents
        max_core_num_tmp = sum([item[2] for item in all_items])

        # update the bin size
        if max_core_num_tmp > max_core_num:
            print(f"max_core_num: {max_core_num} -> {max_core_num_tmp}")
            _bin = bin_list[0]
            _bin.add_rsc_num(max_core_num_tmp - max_core_num)
            max_core_num = max_core_num_tmp
            max_core_layout = [n_slot, {
                item[1]:item[2] for item in all_items
            }, min({int(math.floor(item[5]/timestep)) for item in all_items}) - n_slot]

        # update the position table
        if bin_size_ok and slot_ok.all():
            position_recoder.set_record(tab_pointer, position_dict)

        # place the pending items into the bin

        for item in pending_items:
            task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = item
            slot_s = round(start_t/timestep)
            slot_e = round(ddl_t/timestep)
            slot_n = slot_e - slot_s
            _bin.allocate(pid, [slot_s,], [req_size,], [slot_n,], DEBUG_FG)
            curr_items.append(item)
            cum_flops.update({pid:process_dict[pid].task.flops})
        pending_items.clear()

        # sort the current items and pending items by the ddl
        curr_items.sort(key=lambda x: x[5])
        prev_t = curr_t
    
    # sparsify the Scheduling table
    # initialize the interval info: slot_s, slot_e, core_size, flops
    _bin.to_sparse_dict()
    _bin.alloc_mod = "max"
    
    print("max_core_layout:", max_core_layout)
    assert max_core_num == sum(max_core_layout[1].values()), "max_core_num should be equal to the sum of the core size of the current items"
    print("max_core_num:", max_core_num)
    return max_core_layout


def coleasing_alloc_split_bin_new(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
):

    max_core_layout = coleasing_alloc_1bin(
        bin_list,
        glb_p_list, affinity, event_iter_dict,
        total_cores, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list, monitor_list,
        msg_dispatcher,
        a_data_pipe, w_data_pipe,

        n_p, binpack_cfg, 
        show_warnings, 
        verbose, DEBUG_FG,
        warmup=warmup, drain=drain
        )
    # split the bin
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores
    assert len(bin_list) == 1
    _bin_tb_split = bin_list[0]
        

    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)
    planed_bin_list = []
    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
        planed_bin_list, glb_p_list, total_cores, 
        _new_bin, "manual")
    layout = {_bin.name:_bin.num_resources for _bin in bin_list}
    print("max_core_num:", sum(layout.values()))
    print(f"max_core_layout: {layout}")
    max_core_num = sum(layout.values())

    # use a gurobi solver to determine the placement of the tasks
    # for cfg_slot_s, cached_cfg, cfg_slot_num in _bin.sparse_flops:
    from sched.packing_solver.gurobi_MP_semi2DBp import BinPackingGurobiSolverSemi2D
    from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id

    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))
    sched:Scheduler = scheduler_list[0]
    rsc_recoder_his = sched.rsc_recoder_his

    M = len(planed_bin_list)
    bin_size = {_bin.id:_bin.num_resources for _bin in planed_bin_list}
    probs = [list(cfg.keys()) for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    J = len(probs)
    duation = [slot_num for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]

    # collect used items from problems
    from functools import reduce
    col_pid = set(reduce(lambda x,y: x+y, probs))

    placed_p = {}
    tbd_p = {}
    for pid, _p in process_dict.items():
        if pid not in col_pid:
            continue
        _p:ProcessInt
        p_name = _p.task.name
        size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
        if p_name in bin_name_list:
            bin_id = bin_name_list.index(p_name)
            assert size == planed_bin_list[bin_id].num_resources
            placed_p.update({pid:[size, bin_id]})
        else:
            affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)
            if affinity_tgt_bin_id_list:
                placed_p.update({pid:[size, affinity_tgt_bin_id_list[0]]})
                print(f"{p_name} -> {[bin_name_list[i] for i in affinity_tgt_bin_id_list]}")
            else:
                tbd_p.update({pid:size})
    N = len(tbd_p)
    K = len(placed_p)

    solver = BinPackingGurobiSolverSemi2D(
        M, N, K, J, bin_size, tbd_p, placed_p, probs, duation
    )
    solver.create_variables()
    solver.create_constraints()
    sol = solver.solve()
    # update the placed items to the sol
    for pid, (size, bin_id) in placed_p.items():
        assert pid not in sol
        sol.update({pid:bin_id})
    # print(sol)
    # rebuild the scheduling table list
    for cfg_slot_s, cached_cfg, cfg_slot_num in _bin_tb_split.sparse_list:
        for pid, size in cached_cfg.items():
            _bin:SchedulingTableInt = planed_bin_list[sol[pid]]
            _bin.allocate(pid, [cfg_slot_s,], [size,], [cfg_slot_num,], False)
    for _bin in planed_bin_list:
        _bin: SchedulingTableInt
        _bin.to_sparse_dict()
    bin_list.clear()
    bin_list.extend(planed_bin_list)

def coleasing_alloc_cluster(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        job_graph:DiGraph=None, src_nodes:List=None, end_nodes:List=None, n_partition:int=9999,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
):

    max_core_layout = coleasing_alloc_1bin(
        bin_list,
        glb_p_list, affinity, event_iter_dict,
        total_cores, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list, monitor_list,
        msg_dispatcher,
        a_data_pipe, w_data_pipe,

        n_p, binpack_cfg, 
        show_warnings, 
        verbose, DEBUG_FG,
        warmup=warmup, drain=drain
        )
    # split the bin
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores
    assert len(bin_list) == 1
    _bin_tb_split = bin_list[0]
    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))

    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    from sched.slack_estim import get_chains
    rt_chains, ddl_chains = get_chains(job_graph, src_nodes, end_nodes, {
        _p.task.name:_p.task.flops for pid, _p in process_dict.items()
    })

    # create bins
    # get allowed Bin names and sizes
    allowed_bin_name = []
    for _p in glb_p_list:
        if _p.task.pre_assigned_resource_flag:
            allowed_bin_name.append(_p.task.name)

    # use a gurobi solver to determine the placement of the tasks
    placed_p, bin_name_list, sol, bin_size = split_affinity_first(glb_p_list, scheduler_list, n_partition, _bin_tb_split, rt_chains, ddl_chains, allowed_bin_name)

    # update the placed items to the sol
    for pid, (size, bin_id) in placed_p.items():
        assert pid not in sol
        sol.update({pid:bin_id})
    # print(sol)

    # create the bins from the bin size and the bin name    
    iter_next_bin_obj =bin_iter_list(_new_bin, bin_size, bin_name_list)
    planed_bin_list = list(iter_next_bin_obj)

    # rebuild the scheduling table list
    for cfg_slot_s, cached_cfg, cfg_slot_num in _bin_tb_split.sparse_list:
        for pid, size in cached_cfg.items():
            _bin:SchedulingTableInt = planed_bin_list[sol[pid]]
            _bin.allocate(pid, [cfg_slot_s,], [size,], [cfg_slot_num,], False)
    for _bin in planed_bin_list:
        _bin: SchedulingTableInt
        _bin.to_sparse_dict()
    bin_list.clear()
    bin_list.extend(planed_bin_list)

    # layout format: start slot, allocation map, duation slot
    bin_size_list = {_bin.id:_bin.num_resources for _bin in bin_list}
    pid2_bin_id = sol
    print("max_core_num:", sum(bin_size_list.values()))
    print(f"max_core_layout: {bin_size_list}")
    print(f"pid2_bin_id: {pid2_bin_id}")
    max_core_num = sum(bin_size_list.values())
    return pid2_bin_id, bin_size_list

def split_affinity_first(glb_p_list, scheduler_list, n_partition, _bin_tb_split, rt_chains, ddl_chains, allowed_bin_name):
    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))
    name2pid = {p.task.name:p.pid for p in glb_p_list}
    sched:Scheduler = scheduler_list[0]

    probs = [list(cfg.keys()) for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    J = len(probs)
    duation = [slot_num for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    # collect used items from problems
    col_pid = set(reduce(lambda x,y: x+y, probs))

    placed_p = {}
    tbd_p = {}
    affinity_mode = "search" 
    affinity_dict2 = {}
    affinity_dict1 = {}
    assert affinity_mode in ["manual", "search"]

    # generate the bin name list, as wella as deduce the mapping relation from pid to bins by priority of chains
    bin_name_list = []
    for chain, tot_ops, slack in ddl_chains+rt_chains:
        # search the target bin
        tgt_bin_id = None
        for node in chain:
            pid = name2pid[node]
            if pid in placed_p: # the node has been placed in another chain, skip, also skip the partition
                continue
            # if node in bin_name_list:
            #     tgt_bin_id = bin_name_list.index(node)
            #     break
            if node in allowed_bin_name:
                tgt_bin_id = len(bin_name_list)
                bin_name_list.append(node)
                break
        # set the placement
        if tgt_bin_id is not None:
            # the placed items
            for node in chain:
                pid = name2pid[node]
                if pid not in placed_p:
                    _p:ProcessInt = process_dict[pid]
                    size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
                    placed_p.update({pid:[size, tgt_bin_id]})
                    print(f"{node} -> {bin_name_list[tgt_bin_id]}")
            print(f"Bin {tgt_bin_id}:{bin_name_list[tgt_bin_id]} is used for the chain {chain}")
            n_partition -= 1
        if n_partition == 0:
            break
    # mark others as tbd
    for pid, _p in process_dict.items():
        if pid not in col_pid:
            continue
        _p:ProcessInt
        if pid not in placed_p:
            size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
            tbd_p.update({pid:size})

    M = len(bin_name_list)
    N = len(tbd_p)
    K = len(placed_p)

    solver = ClusterGurobiSolverSemi2D(
        M, N, K, J, tbd_p, placed_p, probs, duation, 1000, affinity_dict1, affinity_dict2
    )
    solver.create_variables()
    solver.create_constraints()
    sol, bin_size = solver.solve()
    return placed_p,bin_name_list,sol,bin_size

def split_size_first(glb_p_list, scheduler_list, n_partition, _bin_tb_split, rt_chains, ddl_chains, allowed_bin_name):
    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))
    name2pid = {p.task.name:p.pid for p in glb_p_list}
    sched:Scheduler = scheduler_list[0]

    probs = [list(cfg.keys()) for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    J = len(probs)
    duation = [slot_num for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    # collect used items from problems
    col_pid = set(reduce(lambda x,y: x+y, probs))

    placed_p = {}
    tbd_p = {}
    affinity_mode = "search" 
    affinity_dict2 = {}
    affinity_dict1 = {}
    assert affinity_mode in ["manual", "search"]

    # generate the bin name list, as wella as deduce the mapping relation from pid to bins by priority of chains
    bin_name_list = []
    for chain, tot_ops, slack in ddl_chains+rt_chains:
        # search the target bin
        tgt_bin_id = None
        for node in chain:
            pid = name2pid[node]
            if pid in placed_p: # the node has been placed in another chain, skip, also skip the partition
                continue
            # if node in bin_name_list:
            #     tgt_bin_id = bin_name_list.index(node)
            #     break
            if node in allowed_bin_name:
                tgt_bin_id = len(bin_name_list)
                bin_name_list.append(node)
                break
        # set the placement
        if tgt_bin_id is not None:
            # the placed items
            for node in chain:
                pid = name2pid[node]
                if pid not in placed_p:
                    _p:ProcessInt = process_dict[pid]
                    size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
                    placed_p.update({pid:[size, tgt_bin_id]})
                    print(f"{node} -> {bin_name_list[tgt_bin_id]}")
            print(f"Bin {tgt_bin_id}:{bin_name_list[tgt_bin_id]} is used for the chain {chain}")
            n_partition -= 1
        if n_partition == 0:
            break
    # mark others as tbd
    for pid, _p in process_dict.items():
        if pid not in col_pid:
            continue
        _p:ProcessInt
        if pid not in placed_p:
            size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
            tbd_p.update({pid:size})

    M = len(bin_name_list)
    N = len(tbd_p)
    K = len(placed_p)

    solver = ClusterGurobiSolverSemi2D(
        M, N, K, J, tbd_p, placed_p, probs, duation, 1000, affinity_dict1, affinity_dict2
    )
    solver.create_variables()
    solver.create_constraints()
    sol, bin_size = solver.solve()
    return placed_p,bin_name_list,sol,bin_size


def solver_1round(planed_bin_list, bin_name_list, process_dict, rsc_recoder_his, M, probs, J, duation, col_pid, placed_p, tbd_p, affinity_mode, affinity_dict2, affinity_dict1):
    if affinity_mode == "manual":
        for pid, _p in process_dict.items():
            if pid not in col_pid:
                continue
            _p:ProcessInt
            p_name = _p.task.name
            size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
            if p_name in bin_name_list:
                bin_id = bin_name_list.index(p_name)
                assert size == planed_bin_list[bin_id].num_resources
                placed_p.update({pid:[size, bin_id]})
            else:
                affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)
                if affinity_tgt_bin_id_list:
                    placed_p.update({pid:[size, affinity_tgt_bin_id_list[0]]})
                    print(f"{p_name} -> {[bin_name_list[i] for i in affinity_tgt_bin_id_list]}")
                else:
                    tbd_p.update({pid:size})
    else:
        for pid, _p in process_dict.items():
            if pid not in col_pid:
                continue
            _p:ProcessInt
            p_name = _p.task.name
            size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
            if p_name in bin_name_list:
                bin_id = bin_name_list.index(p_name)
                assert size == planed_bin_list[bin_id].num_resources
                placed_p.update({pid:[size, bin_id]})
            else:
                affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)
                tbd_p.update({pid:size})
        for pid in tbd_p:
            _p:ProcessInt = process_dict[pid]
            print(f"{_p.task.name} -> {_p.task.affinity_n}")
            for prefer_idx, tgt_pid in enumerate(_p.task.affinity):
                if tgt_pid not in placed_p:
                    affinity_dict2[(pid, tgt_pid)] = 1/((prefer_idx+1)**2)
                else:
                    p_name = process_dict[tgt_pid].task.name
                    bin_id = bin_name_list.index(p_name)
                    affinity_dict1[(pid, bin_id)] = 1/((prefer_idx+1)**2)

    N = len(tbd_p)
    K = len(placed_p)

    solver = ClusterGurobiSolverSemi2D(
        M, N, K, J, tbd_p, placed_p, probs, duation, 1000, affinity_dict1, affinity_dict2
    )
    solver.create_variables()
    solver.create_constraints()
    sol, bin_size = solver.solve()
    return bin_size,sol


def coleasing_alloc_split_bin(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, wsc_slack_ratio, exec_t_comp_ratioB,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
        ):
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(hyper_p//timestep)
    # assert math.isclose(hyper_p, tab_temp_size*timestep, abs_tol=numerical_error_tol_abs), \
    #         "hyper_p should be the multiple of timestep"
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores
    # glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
        bin_list, glb_p_list, total_cores, 
        _new_bin, binpack_cfg["reservation_policy"])
    layout = {_bin.name:_bin.num_resources for _bin in bin_list}
    print("max_core_num:", sum(layout.values()))
    print(f"max_core_layout: {layout}")
    max_core_num = sum(layout.values())

    # build event list
    process_block_sortby_start_slot = []    
    for _p in glb_p_list:
        _p:ProcessInt
        stimu_tab = _p.task.extract_sensor_event(event_range)
        # (task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t)
        for stimu_t in stimu_tab:
            item = (_p.task.name, _p.pid, 
              _p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size, 
              stimu_t, elim_nume_error(stimu_t+_p.task.ERT), elim_nume_error(stimu_t+_p.task.ERT+_p.task.ddl), _p.task.exp_comp_t)
            start_t, ddl_t = item[4], item[5]
            # quantize the start time and ddl time
            slot_s = int(math.ceil(start_t/timestep)) * timestep
            slot_e = int(math.floor(ddl_t/timestep)) * timestep
            process_block_sortby_start_slot.append((item[0], item[1], item[2], item[3], slot_s, slot_e, item[6]))

    # sort the event list by the start time, ddl, exp_comp_t
    process_block_sortby_start_slot.sort(key=lambda x: (x[4], x[5], x[6]))
    event_set = OrderedDict()
    for item in process_block_sortby_start_slot:
        event_set.update({item[4]:None})
        event_set.update({item[5]:None})
    event_set = sorted(event_set.keys())
    
    # convergence flag
    bin_size_ok = False
    slot_ok = np.full(len(event_set), False)
    # some cache
    curr_items = [] # cache the current items
    pending_items = [] # cache the pending items
    max_core_layout = {} # start, core_size, slot_num
    position_recoder = PosTableInt(len(event_set)) # cache the current usage of the cores
    prev_t = 0
    cum_flops = {}
    process_dict = {p.pid:p for p in glb_p_list}

    sched:Scheduler = scheduler_list[0]
    rsc_recoder = sched.budget_recoder
    rsc_recoder_his = sched.rsc_recoder_his
    def tie_break(_p:ProcessInt):
        if _p.pid in rsc_recoder:
            alloc_slot_s, alloc_size, allo_slot, bin_id = rsc_recoder[_p.pid]
            for s, size, l in zip(alloc_slot_s, alloc_size, allo_slot):
                if s <= n_slot < s+l:
                    return [s, _p.pid]
        return [float("inf"), _p.pid]

    # scan the event list, place, and pop the task into the bin, 
    # expanding the bin size if necessary
    for curr_t in list(event_set):
        # curr_t = n_slot * timestep
        n_slot = int(curr_t/timestep)
        tab_pointer = n_slot % tab_temp_size
        hyper_p_n = int(n_slot/tab_temp_size)

        curr_n_period = int(curr_t/hyper_p)
        prev_n_period = int(prev_t/hyper_p)
        if warmup and prev_t == 0 and curr_n_period == 0:
            print("="*20, "WARMUP", "="*20, "\n")
        elif curr_n_period > prev_n_period:
            if curr_t > event_range:
                print("="*20, "DRAIN", "="*20, "\n")
            else:
                print("="*20, "PERIOD {:d}".format(curr_n_period), "="*20, "\n")

        # if prev_t < curr_t and curr_items:
        #     flops_per_core = (curr_t-prev_t) * FLOPS_PER_CORE 
        #     flops_dict = {}
        #     cores_dict = {}
        #     for item in curr_items:
        #         # task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = item
        #         req_size = item[2]
        #         pid = item[1]
        #         start_t, ddl_t = item[4], item[5]
        #         size_del_rda = process_dict[pid].task.flops/FLOPS_PER_CORE/(ddl_t-start_t)
        #         cores_dict[pid] = int(math.ceil(size_del_rda/(1-exec_t_comp_ratioB)))
        #         if cum_flops[pid] > 0:
        #             flops = flops_per_core*math.ceil(size_del_rda)
        #             flops = min(flops, cum_flops[pid])
        #             flops_dict.update({pid:flops})
        #             cum_flops[pid] -= flops
        #         else:
        #             flops_dict.update({pid:0})
        #     _bin.sparse_flops.append([round(prev_t/timestep), flops_dict])
        #     _bin.sparse_cores.append([round(prev_t/timestep), cores_dict])

        # get the pending items
        while len(process_block_sortby_start_slot):
            task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = process_block_sortby_start_slot[0]
            if start_t <= curr_t:
                pending_items.append(process_block_sortby_start_slot.pop(0))
                print("task_name:", task_name, "stimu_t:", stimu_t, "start_t:", start_t, "ddl_t:", ddl_t, "exp_comp_t:", exp_comp_t)
            else:
                break
        
        # pop the items that are finished regarding the ddl
        while len(curr_items):
            if curr_items[0][5] <= curr_t:
                pid = curr_items[0][1]
                # assert cum_flops[pid] == 0
                curr_items.pop(0)
                # cum_flops.pop(pid)
            else:
                break

        # get the current items
        # use cache curr_items
        # ordered_occupant_dict = _bin.index_occupy_by_id(n_slot)

        # get the history placement of the current items
        if bin_size_ok and slot_ok.all():
            position_dict = position_recoder.get_record(n_slot)
            

        # vefify the legality of the placement of the history items

        # determine the placement of the pending items, as well as the bin size

        all_items = curr_items + pending_items
        # max_core_num
            # without laxity, a tasks is placed iff it is released at the current slot
            # no neet to scan the bin in the future monents
        max_core_num_tmp = sum([item[2] for item in all_items])

        # # update the bin size
        # if max_core_num_tmp > max_core_num:
        #     print(f"max_core_num: {max_core_num} -> {max_core_num_tmp}")
        #     _bin = bin_list[0]
        #     _bin.add_rsc_num(max_core_num_tmp - max_core_num)
        #     max_core_num = max_core_num_tmp
        #     max_core_layout = [n_slot, {
        #         item[1]:item[2] for item in all_items
        #     }, min({int(math.floor(item[5]/timestep)) for item in all_items}) - n_slot]

        # update the position table
        if bin_size_ok and slot_ok.all():
            position_recoder.set_record(tab_pointer, position_dict)

        # place the pending items into the bin

        for item in pending_items:
            task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = item
            slot_s = round(start_t/timestep)
            slot_e = round(ddl_t/timestep)
            slot_n = slot_e - slot_s

            p_name = _p.task.name
            # seletc the bin
            if p_name in bin_name_list:
                bin_id = bin_name_list.index(p_name)
                process_sort = get_process_sort([p_name], rsc_recoder_his, tie_break)
                _bin = bin_list[bin_id]



            _bin.allocate(pid, [slot_s,], [req_size,], [slot_n,], DEBUG_FG)
            curr_items.append(item)
            # cum_flops.update({pid:process_dict[pid].task.flops})
        pending_items.clear()

        # sort the current items and pending items by the ddl
        curr_items.sort(key=lambda x: x[5])
        prev_t = curr_t
    
    # sparsify the Scheduling table
    # initialize the interval info: slot_s, slot_e, core_size, flops
    _bin.to_sparse_dict()
    _bin.alloc_mod = "max"
    
    print("max_core_layout:", max_core_layout)
    assert max_core_num == sum(max_core_layout[1].values()), "max_core_num should be equal to the sum of the core size of the current items"
    print("max_core_num:", max_core_num)
    return max_core_layout

