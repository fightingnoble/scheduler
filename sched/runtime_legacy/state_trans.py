from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sched.scheduler_agent import Scheduler
    
import numpy as np
import warnings
from typing import Dict, List, Tuple
from global_var import *
from collections import defaultdict

from model.buffer import Buffer, EventCache, TriggerCache
from model.buffer import Buffer, Data
from model.task_queue_agent import TaskQueue 
from model.message.msg_dispatcher import MsgDispatcher
from model.message.Context_message import ContextMsg
from model.message.data_pipe import DataPipe
from model.streaming_processing.wartermark_strategy import WatermarkStrategy

from task.task_agent import ProcessInt
from sched.scheduling_table import SchedulingTableInt, init_event
from sched.monitor_agent import get_rsc_2b_released



# =================== state transition functions ===================
def throttleToReady(sched, curr_t, budget_recoder, ready_queue, throttle_list, bin_name:str="", bin_event_flg:bool=False):
    l_res_ready:List[ProcessInt] = []
    for _p in throttle_list:
        if _p.pid in budget_recoder and sched._SchedTab.id in _p.rem_flop_budget:
            rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
            if rem_flop_budget> numerical_error_tol_abs:
                l_res_ready.append(_p)
        
    if bin_name and l_res_ready and not bin_event_flg:
        bin_event_flg = True 
        print(f"({bin_name})")

    if len(l_res_ready):
        sched.new_ready_flg = True

    for _p in l_res_ready: 
        throttle_list.remove(_p)
        _p.ready_util(curr_t, ready_queue)
        print("		TASK {:d}:{:s}({:d}) THROTTLE -> READY!!".format(_p.task.id, _p.task.name, _p.pid))
    return bin_event_flg

def check_miss(sched:Scheduler, budget_recoder, timestep, 
               msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
               a_data_pipe:DataPipe,
               curr_t, res_cfg, wait_queue, ready_queue,
               running_queue, miss_list:List[ProcessInt], throttle_list, 
               active_list, inactive_list, buffer,
               bin_event_flg: bool = False,
               bin_name: str = "", mode: str = "current",
               bin_list: List[SchedulingTableInt] = None,
               n_slot: int = 0, rsc_recoder=None,
               process_dict: Dict[int, ProcessInt] = None,
               show_warnings=True):

    bin_id = sched._SchedTab.id
    for _p in sorted(active_list + ready_queue.queue + running_queue.queue, key=lambda x: x.deadline):
        _p:ProcessInt
        # !!!!!!!!!!!! Bug Here !!!!!!!!!!!!!
        # task miss: hard -> stop, soft -> warning
        # chain miss: output task hard -> stop, soft -> warning 
        inst_hard_miss = _p.deadline <= curr_t 
        chain_miss = _p.get_chain_deadline(sched) <= curr_t and _p.task.chain_criticality == "hard"
        if chain_miss:
            miss_list.append(_p)
        elif inst_hard_miss:
            if _p.task.criticality == "hard":
                miss_list.append(_p)
            elif show_warnings:
                _str = f"Task {_p.task.id}:{_p.task.name}({_p.pid}) violate timing constraint @ {_p.deadline:.6f}/{_p.get_timestamp():.6f}!!"
                warnings.warn(_str)

    if bin_name and len(miss_list) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
    
    if len(miss_list)>0:
        sched.new_drop_flg = True

    for _p in miss_list:
        # release the resource and move to the wait list
        # buffer.pop(_p.pid)
        if _p in ready_queue.queue:
            ready_queue.remove(_p)
        elif _p in active_list:
            active_list.remove(_p)
        elif _p in running_queue.queue:
            assert mode in ["future", "current", "none"]
            release_rsc(sched, _p, mode, bin_list, n_slot, rsc_recoder, timestep)
            if mode != "future":
                try:
                    _p.rem_flop_budget[bin_id] -= (_p.totcpu - _p.totburst)
                except KeyError:
                    print("20231126: CodingError, attempt to remove budget of missed tasks")
                    assert False

            running_queue.remove(_p)
        print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!")
        if sched.forbid_miss: 
            print("forbid_miss is True, exit the simulation")
            import sys; sys.exit(1)

        if msg_dispatcher is not None:
            msg_dispatcher.broadcast_message(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.get_timestamp():.6f}({bin_name})!!")

        _p.task.missed_deadline_count += 1
        # _p.release_time += _p.task.period
        # _p.deadline += _p.task.period

        _p.reset_state_vars()            

        if _p.is_fork_inst:
            _p.kill_fork(process_dict)
            del _p
        else:
            inactive_list.append(_p)
        # _p.reset_depends()

        # print("Scheduling Table:")
        # print(SchedTab.print_scheduling_table())
    miss_list.clear()
    return bin_event_flg

def check_throttle(sched:Scheduler,
                   budget_recoder, curr_t, res_cfg, wait_queue, ready_queue,
                   running_queue, miss_list, throttle_list, active_list, inactive_list,
                   bin_event_flg: bool = False,
                   bin_name: str = ""):

    # for _p in (active_list + wait_queue.queue + ready_queue.queue + running_queue.queue):
    bin_id = sched._SchedTab.id
    l_throttle = []
    for _p in (running_queue.queue+ready_queue.queue):
        budget_usedup_flg = False
        # determine if the budget is used up, considering the numerical error
        # less than 1 OPS
        if _p.rem_flop_budget[bin_id] < numerical_error_tol_abs:
            budget_usedup_flg = True
        if budget_usedup_flg and _p.cbs_en: 
            l_throttle.append(_p)

    if bin_name and len(l_throttle) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
    
    for _p in l_throttle: 
        _p:ProcessInt
        _p.throttle_util(throttle_list, curr_t)
        if _p in running_queue.queue:
            sched.res_release(_p.pid)
            # budget_recoder.pop(_p.pid)
            running_queue.remove(_p)
        elif _p in ready_queue.queue:
            ready_queue.remove(_p)
        else:
            raise ValueError("Task is not in the running queue or ready queue")
    return bin_event_flg

def check_complete(sched:Scheduler, budget_recoder, timestep, 
                   msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                   a_data_pipe:DataPipe,
                   curr_t, res_cfg, 
                    running_queue:TaskQueue, completed_list: List[ProcessInt], 
                    inactive_list:List[ProcessInt], buffer:Buffer, 
                    bin_event_flg:bool=False, 
                    bin_name:str="", 
                    save_trace:bool=True,
                    mode:str="current", 
                    bin_list:List[SchedulingTableInt]=None, 
                    n_slot:int=0, rsc_recoder=None,
                    # task_name, -> event_time -> list of allocations
                    detail_alloc_info: Dict[str, Dict[float, List[Tuple]]]=None,
                    process_dict:Dict[int, ProcessInt]=None,
                    ):
    bin_id = sched._SchedTab.id
    for _p in sorted(running_queue.queue, key=lambda x: x.is_fork_inst, reverse=True): 
        # check whether the task is completed
        """
        3. check complete:
            both `n_fork == 0` and totburst
        """
        fork_complete_flg = (_p.n_fork == 0 and _p.fork_pid_list == []) or sum([_forked_p in completed_list for _forked_p in _p.fork_p_inst]) == _p.n_fork
        if elim_nume_error(_p.totburst - _p.totcpu)>=0 and fork_complete_flg:
            completed_list.append(_p)

    if bin_name and len(completed_list) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")

    if len(completed_list)>0:
        sched.new_drop_flg = True

    for _p in completed_list:
        # update statistics
        # TODO: add lock 
        _p.task.completion_count += 1
        _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
        _p.end_time = curr_t

        # if detail_alloc_info is not None:
        #     for bin_id, _SchedTab in enumerate(bin_list):
        #         time_slot_s = int(np.ceil(_p.release_time/timestep))
        #         time_slot_e = int(_p.deadline//timestep)
        #         alloc_rsc_record = _SchedTab.index_occupy_by_id(time_slot_s, time_slot_e)
        #         if not _p.pid in alloc_rsc_record:
        #             continue
        #         _recrd = detail_alloc_info.get(_p.task.name, defaultdict(list))
        #         _recrd[(round(_p.event_time, numerical_tol_bit))].append((bin_id, *alloc_rsc_record[_p.pid]))
        #         detail_alloc_info.update({_p.task.name: _recrd})

        # reset the task
        # release the resource and move to the wait list
        assert mode in ["future", "current", "none"]
        release_rsc(sched, _p, mode, bin_list, n_slot, rsc_recoder, timestep)

        # cache processing info for ctx message and attach to the data
        msg:ContextMsg = _p.msg_cache.pop(0)
        msg.cache_processing(_p)
        _p.event_time = msg.get_timestamp()
        data = Data(_p.pid, _p.io_time, (0,), "output", curr_t, 1/_p.task.freq, _p.task.period)
        data.ctx = msg
        data.cache_data_info()

        # TODO: communication scheduling
        # cache the message sending time when the bus is allocated
        data.cache_msg_transfer(curr_t)

        off_size = _p.required_resource_size * 1/3
        x = (off_size) 
        data.size = sched.get_ctx_lat(x)

        a_data_pipe.put(data,)
        # if succ_ctrl is not empty, 
        # redirect print(data.ctx.serialize()) to the trace_file path
        if len(_p.succ_ctrl) and save_trace:
            trace_list.append(data.ctx.serialize())
            # save_chunk(sched.trace_path.replace(".pkl", ".h5"), trace_list)

            
        # detect the lateness 
        _str = f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}!!"
        if _p.deadline < curr_t:
            _str = "(lateness detected)" + _str
        else:
            _str = "		" + _str
        print(_str)
        
        # _p.release_time += _p.task.period
        # _p.deadline += _p.task.period

        # if budget_recoder is not None:
        #     per_slot_flops = FLOPS_PER_CORE*timestep*budget_recoder[_p.pid][1]
        #     # if _p.rem_flop_budget[bin_id] >= per_slot_flops-numerical_error_tol_abs:
        #     #     curr_cfg = sched.curr_cfg
        #     #     # truncate the curr_cfg
        #     #     curr_cfg.slot_num = curr_cfg.slot_num - (n_slot - curr_cfg.slot_s)
        #     #     curr_cfg.slot_s = n_slot
        #     # budget_recoder.pop(_p.pid)
        #     # _p.rem_flop_budget.pop(bin_id)
        #     # TODO: a large number of tasks are truncated, remaing budget ~= 1 slot, check why
        #     if _p.rem_flop_budget[bin_id] >= per_slot_flops+numerical_error_tol_abs and \
        #         _p.task.name.startswith("Steering_speed"):
        #         budget_recoder[_p.pid][2] -= n_slot - budget_recoder[_p.pid][0] 
        #         budget_recoder[_p.pid][0] = n_slot
        #         print(f"\t\tTruncate the curr_cfg @{n_slot}({curr_t:.6f}), \
        #                 \n\t\trem_flop_budget: {_p.rem_flop_budget[bin_id]/per_slot_flops:.6f}(x{per_slot_flops:.6f})")
        #     else:
        #         budget_recoder.pop(_p.pid)
        #         _p.rem_flop_budget.pop(bin_id)

        _p.reset_state_vars()
        running_queue.remove(_p)

        if msg_dispatcher is not None:
            msg_dispatcher.broadcast_message(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}({bin_name})!!")

        if _p.is_fork_inst:
            _p.kill_fork(process_dict)
            del _p
        else:
            inactive_list.append(_p)
        # _p.reset_depends()

    # update_depend(task_dict, completed_list)
    
    
    completed_list.clear()
    return bin_event_flg

def release_rsc(sched:SchedulingTableInt, _p:ProcessInt, mode:str, 
                # for future mode only
                bin_list:List[SchedulingTableInt], n_slot:int, rsc_recoder, 
                timestep:float):
    """
    - Allocation related:
    - 1. a task can only be executed in one bin at a time
    - 2. a task may suffer lateness, and miss its original allocated processing window in the particular bin
    - 3. a late task (i.e., 18,19,20 'Steering_speed_0_x')  is also possible to be overlapped with the allocated processing window of follwing jobs, 
        which results in 1). sum(_p.rem_flop_budget)!=0, 2). len(_p.rem_flop_budget) > 1, when performing migration.
    
    - Simulation related:
    - 1. as we do not use a real parallel simulation of multiple partitions, 
        what we do is to scan each partition and processing the scheduling events sequentially. 
    - 2. we not turely simulate inter-operator data dependencies, 
        without migrating there is a risk of two tasks are executed in two different partitions at the same time.
        
    Migration policy:    
    - 1. if a task allocated resources in a different bin, in a consecutive time slot, 
        at the end of the slot, the running task is throttled and migrated to the new bin.
        at once?? No, we trasfer just at the end of the slot when the task get budget in the new bin.
    - 2. when a task is restarted in a new bin, it needs to migrate budget from the previous bin, 
        to ensure that the tasks have enough budget to run.
        
    Migration steps:
    - 1. process the matched "migrate_to" at the end of the each slot, 
        pending the running tasks and backup the remaining budget to the 'bk' item in the rem_flop_budget
    - 2. process the matched "migrate_from" at the start of the each slot, 
        fetch the remaining budget from the 'bk' item in the rem_flop_budget. 
    
    Missing tasks:
    - 1. remove the remaining budget of the missed tasks
    - 2. as the only each task can be executed in one bin at a time, 
        we just need to subtract the remaining budget (tot_cpu - tot_burst) from the task which is not in the 
    """
    
    if mode == "future": 
                
        _recrd_single_event: List[Tuple[int]] = []
        for bin_id, _SchedTab in enumerate(bin_list):
            # locate the actual used capacity in each bin
            time_slot_s = int(np.ceil(_p.start_time/timestep))
            time_slot_e = n_slot
            alloc_rsc_record = _SchedTab.index_occupy_by_id(time_slot_s, time_slot_e)
            # if the function is involked by check_miss, the timeout task may get any resources
            if not _p.pid in alloc_rsc_record:
                continue
            _recrd_single_event.append((bin_id, *alloc_rsc_record[_p.pid]))

        # Sort the _recrd_single_event by alloc_slot_s
        _recrd_single_event = list(sorted(_recrd_single_event, key=lambda x: x[1]))

        rem_flops = _p.task.totcpu 
        bin_id_prev = -1
        pre_bin = None
        for i, (bin_id, alloc_slot_s, alloc_size, allo_slot) in enumerate(_recrd_single_event):
            _bin:SchedulingTableInt = bin_list[bin_id]
            # start event
            if i == 0:
                _bin.event_list.append((alloc_slot_s[0], _p.get_timestamp(), _p.pid, init_event(_p, alloc_slot_s[0], "start", bin_id)))
            if bin_id_prev != -1 and bin_id_prev != bin_id:
                # raise event of migrate from
                # policy 2
                _bin.event_list.append((alloc_slot_s[0], _p.get_timestamp(), _p.pid, init_event(_p, alloc_slot_s[0], "migrate_from", bin_id_prev))) 
                # policy 1
                pre_bin.event_list.append((alloc_slot_s[0]-1, _p.get_timestamp(), _p.pid, init_event(_p, alloc_slot_s[0]-1, "migrate_to", tgt=bin_id))) 
                
            for (s, size, l) in zip(alloc_slot_s, alloc_size, allo_slot): 
                chunk_exp_req_rsc_size = (size*l)
                chunk_flops = elim_nume_error(math.ceil(chunk_exp_req_rsc_size) * timestep * FLOPS_PER_CORE) 
                chunk_flops = min(rem_flops, chunk_flops)
                rem_flops = elim_nume_error(rem_flops - chunk_flops)
                _bin.flops_list.append((s, _p.pid, (l, size, chunk_flops)))
                if rem_flops < 0:
                    break
            bin_id_prev = bin_id
            pre_bin = _bin

        # release the resource
        bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)                
        _SchedTab:SchedulingTableInt = bin_list[bin_id_t]
        _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
        _SchedTab.event_list.append((n_slot, _p.get_timestamp(), _p.pid, init_event(_p, n_slot, "complete", bin_id_t)))

        # alloc_s, alloc_len, alloc_size, bin_id = rsc_recoder[_p.pid] 
        # _bin:SchedulingTableInt = bin_list[bin_id]
        # _bin.event_list.append((alloc_s[-1]+alloc_len[-1], _p.get_timestamp(), _p.pid, init_event(_p, alloc_s[0], "complete", bin_id)))
        rsc_recoder.pop(_p.pid)
    elif mode == "current":
        sched.res_release(_p.pid)
        
    else:
        pass

def pendingToReady_cbs(sched, buffer:Buffer, budget_recoder, 
                       active_list:List[ProcessInt], ready_queue, throttle_list, 
                       curr_t, glb_n_task_dict:Dict[str, ProcessInt], event_cache:EventCache=None,
                       bin_name="", process_dict:Dict[int, ProcessInt]=None,
                       show_warnings=True):
    # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
    l_ready:List[ProcessInt] = []
    for _p in active_list:
        # check data availability
        w_avail = _p.pid in buffer.buffer_w
        # in_avail = _p.check_depends_data(buffer, glb_n_task_dict=glb_n_task_dict, event_cache)

        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0: 
            matched_pair, in_avail = WatermarkStrategy.check_data_depends(_p, buffer, glb_n_task_dict, 
                                                                                      _p.get_timestamp(), event_cache)
        else:
            in_avail = True

        if w_avail and in_avail:
            l_ready.append(_p)

    for _p in l_ready:
        # cache the context of the upstream weight node and src node
        _p.update_ctx('weight', buffer=buffer)
        # _p.update_ctx('upstream', buffer=buffer, glb_n_task_dict=glb_n_task_dict)
        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0:
            _p.update_ctx('upstream', matched_pair=matched_pair)
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) "
        if bin_name:
            _str = f"({bin_name})\n" + _str
        # if _p.rem_flop_budget > 0: 
        if _p.pid in budget_recoder and sched._SchedTab.id in _p.rem_flop_budget:
            rem_flop_budget = _p.rem_flop_budget[sched._SchedTab.id]
            if rem_flop_budget> numerical_error_tol_abs:
                _p.ready_util(curr_t, ready_queue)
                _str += "READY!!"
                print(_str)
                active_list.remove(_p)
                sched.new_ready_flg = True

            _p.pre_fork(True)
            fork_list = _p.p_fork()
            process_dict.update({p.pid:p for p in fork_list})
            _p.fork_p_inst.extend(fork_list)
            for _forked_p in fork_list:
                budget_recoder[_forked_p.pid] = budget_recoder[_p.pid]
                _forked_p.ready_util(curr_t, ready_queue)
                print(f"{_str}" + f" (FORKED {_forked_p.pid:d})")
        else:
            active_list.remove(_p)
            _p.throttle_util(throttle_list, curr_t, verbose=False)
            if show_warnings:
                _str += "data ready, but throttled!!"
                warnings.warn(_str)

def pendingToReady(sched, active_list:List[ProcessInt], ready_queue, 
                   buffer:Buffer, curr_t, glb_n_task_dict:Dict[str, ProcessInt], 
                   bin_name=""):
    # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
    l_ready = []
    for _p in active_list:
        # check data availability
        # w_avail = _p.pid in buffer.buffer_w
        # in_avail = _p.check_depends_data(buffer, glb_n_task_dict=glb_n_task_dict)

        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0: 
            matched_pair, in_avail = WatermarkStrategy.check_data_depends(_p, buffer, glb_n_task_dict, _p.get_timestamp())
        else:
            in_avail = True

        # if w_avail and in_avail:
        if in_avail:
            l_ready.append(_p)
    if len(l_ready)>0:
        sched.new_ready_flg = True

    for _p in l_ready:
        _p:ProcessInt
        # cache the context of the upstream weight node and src node
        # _p.update_ctx('upstream', buffer=buffer, glb_n_task_dict=glb_n_task_dict)
        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0:
            _p.update_ctx('upstream', matched_pair=matched_pair)
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) "
        if bin_name:
            _str = f"({bin_name})\n" + _str
        active_list.remove(_p)
        _p.ready_util(curr_t, ready_queue)
        _str += "READY!!"
        print(_str)
