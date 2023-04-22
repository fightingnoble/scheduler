from __future__ import annotations

from typing import Union, List, Dict, Iterator, Callable, Union
import copy

import math
import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

from global_var import *
from model.lru import LRUCache
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from task.task_agent import TaskInt
from model.task_queue_agent import TaskQueue 
from task.task_agent import ProcessInt, ProcessBase
from scheduler_agent import Scheduler 
from sched.monitor_agent import Monitor
from model.msg_dispatcher import MsgDispatcher
from model.buffer import Buffer, Data


def DynRT_overall(affinity, 
                scheduler_list: List[Scheduler], monitor_list:List[Monitor],
                rsc_list:List[Resource_model_int], 
                total_cores:int, 
                glb_p_list:List[ProcessInt],
                timestep, hyper_p, n_p=1, msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
                verbose=False, *, warmup=False, drain=False,):
    
    # set/load the simulation parameters
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(sim_range/timestep)
    tab_spatial_size = total_cores
    
    # init the global queues

    # init the tasks that require external triggering
    sim_trigger_p_list = [p for p in glb_p_list if p.task.trigger_mode!='N']

    # init the global allocation preference, w.r.t. the time slot
    pre_alloc_table = {}
    for sched in scheduler_list:
        _SchedTab = sched.scheduling_table
        bin_pack_result = _SchedTab.index_occupy_by_id()

        # sort the result by the start time
        # item[1] is alloc_slot_s, alloc_size, allo_slot
        # item[1][0] is alloc_slot_s
        for k, v in bin_pack_result.items():
            if k not in pre_alloc_table.keys():
                pre_alloc_table[k] = [[v[0], v[1], v[2], _SchedTab.id]]
            else:
                pre_alloc_table[k].append([v[0], v[1], v[2], _SchedTab.id])
    
    partition_affinity = {}
    for k in list(pre_alloc_table.keys()):
        partition_affinity[k] = [[slot_n, bin_id] for slot_n, _, _, bin_id in sorted(pre_alloc_table[k], key=lambda item: item[0])]


    # suppose the task set is fixed, and all the tasks are periodic
    
    # load the task set, task graph, and scheduling table 

    # map the initial tasks to each partition
    for sched in scheduler_list:
        # extract scheudler, including queues and lists from scheduler_list
        ready_queue:TaskQueue = sched.ready_queue
        wait_queue:TaskQueue = sched.weight_wait_queue
        # inactive_list:List[ProcessInt] = sched.inactive_list
        buffer = sched.get_buffer()
        _SchedTab, curr_cfg, process_dict = sched._SchedTab, sched.curr_cfg, sched.process_dict
        init_cfg = _SchedTab.scheduling_table[0]
        task_pid_list = list(process_dict.keys())

        # init the tasks queue

        _SchedTab.to_sparse_dict(-1)
        curr_cfg.slot_e = -1 #cfg_slot_s + cfg_slot_num - 1
        curr_cfg.slot_s = -1 # cfg_slot_s
        curr_cfg.slot_num = 0 # cfg_slot_num

        # put the initial tasks into the ready queue
        print(f"Bin {_SchedTab.id:d} initial queue:")
        print("	ready tasks:")
        for pid in init_cfg.rsc_map: 
            _p = process_dict[pid]
            ready_queue.put(_p)
            _p.set_state("ready")
            _p.released = True 
            print("		TASK {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        print("")
        
        # instruction prefetching
        cfg_slot_s, cached_map, cfg_slot_num  = _SchedTab.sparse_list[_SchedTab.sparse_idx_next]

        # weight prefetching based on the scheduling table
        # TODO: how to represent the tile prefetching: when to start, when to check
        init_prefetch_obj = {k:v for k,v in cached_map.items() if k not in init_cfg.rsc_map}
        # data_prefetching(init_p_list, wait_queue, cached_cfg=init_prefetch_obj)

        print("	prefetched done:")
        for pid in init_prefetch_obj:
            _p = process_dict[pid]
            # skip data prefetching; put the data into the buffer directly
            data = Data(_p.pid, 1, (0,), "weight", _p.io_time)
            data.valid = True
            buffer.put(data)
            print("		TASK {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        print("")


    # start the simulation
    sim_slot_num = math.ceil(sim_range/timestep)
    for n_slot in range(sim_slot_num):
        curr_t = n_slot * timestep
        tab_pointer = n_slot % tab_temp_size

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")


    # Simulation: 
    # recieve the datastream, route the data source to each partition and trigger the local scheudler
    # enqueue the process that is released in this slot
    # simulate the event trigger
    for _p in sim_trigger_p_list:
        trigger_state = _p.sim_trigger(curr_t, timestep)
        if trigger_state or curr_t == 0:
            # test case: 
            # sensor data arrival time varies
            # inject noise to self.task.period, self.task.i_offset
            _p.i_offset = _p.task.i_offset + _p.task.exp_comp_t * truncnorm.rvs(-0.2, 0.2, size=1, scale=1)[0] # 0.2 # truncnorm.rvs(-0.2, 0.2, size=1, scale=1)[0]
            if _p.i_offset < 0:
                if curr_t == 0:
                    for key in _p.pred_ctrl.keys():
                        _p.pred_ctrl[key]["valid"] = True
                        _p.pred_ctrl[key]["ingestion_time"] = _p.i_offset
                        trigger_state = True
                    _p.i_offset = _p.task.i_offset + _p.task.exp_comp_t * truncnorm.rvs(-0.2, 0.2, size=1, scale=1)[0] # 0.2 # truncnorm.rvs(-0.2, 0.2, size=1, scale=1)[0]
                    if _p.i_offset < 0:
                        _p.i_offset += _p.task.period
                else:
                    _p.i_offset += _p.task.period
        # dispatch the process to the corresponding partition according to the partition affinity
        affinity_list = partition_affinity[_p.pid]
        if trigger_state:
            for i in range(len(partition_affinity[_p.pid])):
                if curr_t > affinity_list[i][0] and curr_t <= affinity_list[i+1][0]:
                    target_bin = affinity_list[i][1]
                    break
            msg_dispatcher.send_message(target_bin, f"{_p.task.name}_completed", prefix="		")
            

        if trigger_state and not bin_event_flg and curr_t >= _p.task.ERT:
            bin_event_flg = True
            print(f"({bin_name})")



    lt = []
    for _p in wait_queue:
        if _p.release_time <= n_slot * timestep and _p.release_time < event_range:
            lt.append(_p)
    for _p in lt:
        print("TASK {:d}:{:s}({:d}) RELEASEED AT {}!!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
        ready_queue.put(_p)
        wait_queue.remove(_p)
    lt.clear()


    pass
    