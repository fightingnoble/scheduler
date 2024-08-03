from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sched.scheduler_agent import Scheduler
    
import numpy as np
import warnings
from typing import Dict, List, Tuple
from global_var import *

from model.buffer import Buffer, EventCache, TriggerCache
from model.buffer import Buffer, Data
from model.task_queue_agent import TaskQueue 
from model.message.msg_dispatcher import MsgDispatcher
from model.message.Context_message import ContextMsg
from model.message.data_pipe import DataPipe
from model.streaming_processing.wartermark_strategy import WatermarkStrategy

from task.task_agent import ProcessInt
from sched.scheduling_table import SchedulingTableInt
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

def chk_release(sched, event_range, curr_t, inactive_list:List[ProcessInt], active_list, 
                _SchedTab, timestep, event_cache:EventCache=None, trigger_cache:TriggerCache=None,
                bin_event_flg:bool=False, 
                bin_name:str="", DEBUG_FG:bool=False,):
    """
    check release
        1. check the dependencies of the tasks in inactive list
        2. if the dependencies are satisfied, move the task to the wait queue
    """

    l_active = []
    # if curr_t <= event_range:
    #     # simulate the event trigger
    #     trigger_state = message_trigger_event(_SchedTab.sim_triggered_list, sched.jitter_sim_en, sched.jitter_sim_para, 
    #                                           inactive_list, timestep, curr_t, True)
    #     if bin_name and trigger_state and not bin_event_flg:
    #         bin_event_flg = True
    #         print(f"({bin_name})")

    for _p in inactive_list:
        if _p.check_depends(event_cache=event_cache, trigger_cache=trigger_cache):
            l_active.append(_p)

    if bin_name and len(l_active) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
        
    for _p in l_active:
        active_list.append(_p)
        inactive_list.remove(_p)
        _p.release_time = curr_t
        _p.released = True
        _p.remburst += _p.task.flops
        _p.set_state("active")
        # _p.release_time = curr_t 
        # _p.deadline = curr_t + _p.task.ddl
        # _p.deadline += _p.task.period
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is activated @ {curr_t:.6f}!!"
        print(_str)
    return bin_event_flg

def check_miss(sched: Scheduler,
               budget_recoder, msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
               curr_t, res_cfg, wait_queue, ready_queue,
               running_queue, miss_list:List[ProcessInt], throttle_list, active_list, inactive_list, buffer,
               bin_event_flg: bool = False,
               bin_name: str = "", mode: str = "current",
               bin_list: List[SchedulingTableInt] = None,
               n_slot: int = 0,
               rsc_recoder=None,
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
                _str = f"Task {_p.task.id}:{_p.task.name}({_p.pid}) violate timing constraint @ {_p.deadline:.6f}/{_p.msg_cache[0].get_timestamp():.6f}!!"
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
            if mode == "future":
                bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                _SchedTab:SchedulingTableInt = bin_list[bin_id_t]
                _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
            elif mode == "current":
                sched.res_release(_p.pid)
            else:
                pass
            try:
                _p.rem_flop_budget[bin_id] -= (_p.totcpu - _p.totburst)
            except KeyError:
                print("20231126: CodingError, attempt to remove budget of missed tasks")
            if rsc_recoder is not None:
                rsc_recoder.pop(_p.pid)

            running_queue.remove(_p)
        print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.msg_cache[0].get_timestamp():.6f}!!")
        if sched.forbid_miss: 
            print("forbid_miss is True, exit the simulation")
            import sys; sys.exit(1)

        if msg_dispatcher is not None:
            msg_dispatcher.broadcast_message(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) MISSED DEADLINE @ {curr_t:.6f}/{_p.msg_cache[0].get_timestamp():.6f}({bin_name})!!")

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
            _p.set_state("throttled")

    if bin_name and len(l_throttle) and not bin_event_flg:
        bin_event_flg = True
        print(f"({bin_name})")
    
    for _p in l_throttle: 
        _p:ProcessInt
        _p.throttle_util(running_queue, ready_queue, throttle_list, sched,
                      curr_t)
    return bin_event_flg

def check_complete(sched:Scheduler, budget_recoder, timestep, 
                   msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                   a_data_pipe:DataPipe,
                   curr_t, res_cfg, 
                    running_queue:TaskQueue, 
                    completed_list: List[ProcessInt], 
                    inactive_list:List[ProcessInt],
                    buffer:Buffer, 
                    bin_event_flg:bool=False, 
                    bin_name:str="", 
                    save_trace:bool=True,
                    mode:str="current", 
                    bin_list:List[SchedulingTableInt]=None, 
                    n_slot:int=0, rsc_recoder=None,
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

        if detail_alloc_info is not None:
            for bin_id, _SchedTab in enumerate(bin_list):
                time_slot_s = int(np.ceil(_p.release_time/timestep))
                time_slot_e = int(_p.deadline//timestep)
                alloc_rsc_record = _SchedTab.index_occupy_by_id(time_slot_s, time_slot_e)
                if not _p.pid in alloc_rsc_record:
                    continue
                _recrd = detail_alloc_info.get(_p.task.name, {})
                _recrd.update({round(_p.event_time, numerical_tol_bit): (bin_id, *alloc_rsc_record[_p.pid])})
                detail_alloc_info.update({_p.task.name: _recrd})

        # reset the task
        # release the resource and move to the wait list
        assert mode in ["future", "current", "none"]
        if mode == "future": 
            # release the resource and move to the wait list
            bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                
            _SchedTab:SchedulingTableInt = bin_list[bin_id_t]
            _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)

        elif mode == "current":
            sched.res_release(_p.pid)
        
        else:
            pass

        # detect the lateness 
        _str = f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) COMPLETED @ {curr_t:.6f}/{_p.event_time:.6f}!!"
        if _p.deadline < curr_t:
            _str = "(lateness detected)" + _str
        else:
            _str = "		" + _str
        print(_str)
        
        _p.release_time += _p.task.period
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

        if rsc_recoder is not None:
            rsc_recoder.pop(_p.pid)

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
                                                                                      _p.msg_cache[0].get_timestamp(), event_cache)
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
            throttle_list.append(_p)
            _p.set_state("throttled")
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
            matched_pair, in_avail = WatermarkStrategy.check_data_depends(_p, buffer, glb_n_task_dict, _p.msg_cache[0].get_timestamp())
        else:
            in_avail = True

        # if w_avail and in_avail:
        if in_avail:
            l_ready.append(_p)
    if len(l_ready)>0:
        sched.new_ready_flg = True

    for _p in l_ready:
        # cache the context of the upstream weight node and src node
        # _p.update_ctx('upstream', buffer=buffer, glb_n_task_dict=glb_n_task_dict)
        if len(_p.pred_ctrl)>0 and len(_p.pred_data)>0:
            _p.update_ctx('upstream', matched_pair=matched_pair)
        _str = f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) "
        if bin_name:
            _str = f"({bin_name})\n" + _str
        ready_queue.put(_p)
        active_list.remove(_p)
        _p.ready_time = curr_t
        _p.ready = True
        _p.set_state("ready")
        _str += "READY!!"
        print(_str)

