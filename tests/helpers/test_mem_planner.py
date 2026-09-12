"""test_mem_planner.py — external test for mapper.mem_planner (mem/block planning).

Moved out of sched/global_sched.py (B6-SPLIT-001 correction, 2026-06-29).
Original was misplaced in global_sched (zero relation to its Step2/Step3 functions).
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
"""
from typing import List, Dict
import math
from global_var import elim_nume_error
from sched.scheduling_table import SchedulingTableInt
from task.task_agent import ProcessInt
from sched.scheduler_agent import Scheduler
from sched.monitor_agent import Monitor
from sched.global_sched_alloc import default_binpack_cfg
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe

def test_mem_planner(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, exec_t_comp_ratioB,

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

    from mapper.mem_planner import Block, test_priority_mapper, \
        test_seq_mapper, CyclicBlock, test_cyclic_mapper, load_block_list_from_json

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
            # block_list.append(Block(item[2], slot_s, slot_e, block_idx))
            # pid:int # process id 
            block_list.append(CyclicBlock(item[2], slot_s, slot_e, block_idx, pid=item[1]))
            block_idx += 1

    # export the block list as json
    import json
    with open("cache/block_list.json", "w") as f:
        json.dump(list(map(lambda x: x.to_dict(), sorted(block_list, key=lambda x: x.idx))), f, indent=4)
    block_list = load_block_list_from_json("cache/block_list.json", 'cyclic_block')
    
    print(test_priority_mapper(timestep, block_list))
    print(test_seq_mapper(timestep, block_list))
    print(test_cyclic_mapper(timestep, block_list))
