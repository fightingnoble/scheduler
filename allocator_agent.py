from typing import Union, List, Dict, Iterator, Callable
import warnings
import numpy as np
from task.task_agent import TaskInt
from task.spec import Spec
from model.buffer import Buffer, Data
from model.msg_dispatcher import MsgDispatcher
from model.message_pipe import MessagePipe
from multiprocessing import Queue

class AllocatorInt(object):
    def __init__(self, ):
        pass

    def release(self, task_list: List[TaskInt], host_list: List[TaskInt]):
        """
        release the resource allocated to the task
        invoke the task.release() method for each task
        read the returned massage whether the task is released successfully locally
        get the host task id and the num of the released cores
        if not, invoke the task.release() method for the task on the remote node
        """
        for task in task_list:
            msg:dict = task.release()
            status = msg["released"]
            if not status:
                task_id = msg["id"]
                rsc = msg["resource"] # (main_num, RDA_num)
                # suppose the task only get resource from one node
                assert len(rsc) == 1, \
                    "Not implemented: suppose one task only get resource from one node"
                host_id = list(rsc.keys())[0]
                num = rsc[host_id] 
                host = host_list[host_id] 
                host_record_num = host.query_rsc(task_id)
                assert host_record_num == num, \
                    f"(host:{host_id}):{host_record_num} != (guset{task_id}){num}, \nNot implemented: suppose one task only get resource from one node"
                host.release(task_id, num)

    def allocate(self, task_list: List[TaskInt], host_idx: Dict[int, TaskInt], task_idx: Dict[int, TaskInt], verbose:bool=False):
        """
        allocate resource to the task in three steps:
        1. check the affinity of the task
            1.1 if task is starionary, get free reource from the host
            1.2 if task is movable, get free resource from all the hosts
        2. select and allocate the rsc to the task
        """
        for task in task_list: 
            # 1. check the affinity of the task
            if task.is_stationary():
                # 1.1 if task is starionary, get free reource from the host
                # check the legality of the affinity
                assert len(task.affinity) == 1
                host = host_idx[task.affinity[0]]
                num = task.get_rsc()
                host.allocate(task.id, num, verbose=verbose)
            else:
                # 1.2 if task is movable, get free resource from all the hosts
                # check the legality of the affinity
                assert len(task.affinity) <= len(host_idx)
                num = task.get_rsc()
                # TODO: free resource selection based on the fdl_mask
                # TODO: push operation based on the max heap
                for cluster_id in task.affinity:
                    host = host_idx[cluster_id]
                    if self.can_execute(host, task, verbose=verbose): 
                        host.allocate(task.id, num, verbose=verbose)
                        break
    
    def can_execute(self, host:TaskInt, guest:TaskInt, verbose:bool=False) -> bool:
        """
        A task is executable if it has enough resource to execute the task before the deadline
        if the task is pre-assigned, it will always return true
        if the task is not pre-assigned, it will play a insert-based scheduling 
        if the item with size (task.rsc, task.deadline) can be inserted into the host\'s queue,
        """ 
        if guest.is_pre_assigned():
            return True
        else: 
            pass

from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
import matplotlib.pyplot as plt
from global_var import *

from model.task_queue_agent import TaskQueue 
from task.task_agent import ProcessInt
import copy
from model.lru import LRUCache
from scheduler_agent import Scheduler, glb_dynamic_sched_step
from sched.monitor_agent import Monitor

from scheduler_agent import Scheduler 
from sched.monitor_agent import Monitor
from task.spec import Spec
from model.msg_dispatcher import MsgDispatcher
from model.data_pipe import DataPipe
from model.Context_message import ContextMsg

# =================== local scheduler ===================
# def sched_step_cyclic_dense(task_spec:Spec, affinity, 
#                 bin_list:List[SchedulingTableInt], scheduler_list: List[Scheduler], monitor_list:List[Monitor],
#                 rsc_list:List[Resource_model_int], curr_cfg_list:List[Resource_model_int],
#                 rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
#                 total_cores:int, quantumSize, n_slot, init_p_list:List[ProcessInt], 
#                 timestep, hyper_p, n_p=1, verbose=False, *, animation=False, warmup=False, drain=False,):
        
#     """
#     implement a step of runtime scheduling
#     """
#     # task status

#     # completed: task is completed
#     # miss: not completed before its deadline
#     # throttled: waiting for the next slot
#     # running: task is running
#     # ----------------------------------------
#     # executeable: ready task, resource ready, but not issued yet
#     # blocked: resource is not available
#     # ready: task is ready to execute, current slot is in its liveness interval and data are ready
#     # ----------------------------------------
#     # waiting: waiting for data
#     # active: task is in its liveness interval

#     event_range = hyper_p * (n_p+warmup)
#     sim_range = hyper_p * (n_p+warmup+drain)
#     sim_slot_num = int(sim_range/timestep)
#     tab_spatial_size = total_cores

#     task_spec_bk:Spec = copy.deepcopy(task_spec)
#     curr_t = n_slot * timestep
#     task_dict = {p.task.name:p for p in init_p_list}

#     # spatial management
#     # Each partition maintains a scheduling table, a task monitor, and a scheduler. 
#     for res_cfg, _SchedTab, sched, monitor in zip(rsc_list, bin_list, scheduler_list, monitor_list):

#         # print(f"	Bin {_SchedTab.id:d}:")
#         # extract scheudler, including queues and lists from scheduler_list
#         wait_queue:TaskQueue 
#         ready_queue:TaskQueue 
#         running_queue:TaskQueue 
#         miss_list:List[ProcessInt] 
#         preempt_list:List[ProcessInt] 
#         issue_list:List[ProcessInt] 
#         completed_list:List[ProcessInt]
#         throttle_list:List[ProcessInt]
#         inactive_list:List[ProcessInt]
#         active_list:List[ProcessInt]
#         curr_cfg:Resource_model_int

#         wait_queue, ready_queue, running_queue, \
#         miss_list, preempt_list, issue_list, completed_list, throttle_list,\
#             inactive_list, active_list = sched.get_queues()
        
#         # extract the scheduling table
#         tab_temp_size = len(_SchedTab.scheduling_table)
#         tab_pointer = n_slot % tab_temp_size
#         curr_cfg = _SchedTab.scheduling_table[tab_pointer] 
#         bin_name = _SchedTab.name

#         # (running_queue)
#         # check running tasks
#         check_complete(rsc_recoder, timestep, curr_t, task_dict, res_cfg, running_queue, completed_list, inactive_list)

#         # check whether the task is miss
#         # TODO: other ready tasks shoud be checked
#         check_miss(rsc_recoder, curr_t, res_cfg, wait_queue, ready_queue, running_queue, miss_list, 
#                             throttle_list, active_list, inactive_list)

#         check_throttle(rsc_recoder, curr_t, res_cfg, wait_queue, ready_queue, running_queue, miss_list, 
#                             throttle_list, active_list, inactive_list)

#         # check release
#         # check the dependencies of the tasks in inactive list
#         # if the dependencies are satisfied, move the task to the wait queue
#         l_active = []
#         if curr_t <= event_range:
#             for _p in inactive_list:
#                 if _p.check_depends():
#                     l_active.append(_p)
        
#         for _p in l_active:
#             active_list.append(_p)
#             _p.set_state("runnable")
#             inactive_list.remove(_p)
#             # _p.release_time = curr_t 
#             # _p.deadline = curr_t + _p.task.ddl
#             # _p.deadline += _p.task.period
#             print(f"		TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) is activated @ {curr_t:.6f}!!") 

#         # TODO: simulate the congestion of the network

#         # check data availability: some tasks may be prefetched
#         pendingToReady(wait_queue, ready_queue, throttle_list, curr_t, bin_name)

#         # free resource index
#         aval_rsc = res_cfg.get_available_rsc()

#         running_queue.queue.clear()
#         # rsc_recoder.clear()
#         res_cfg.clear()
#         for pid in curr_cfg.rsc_map.keys():
#             _p = init_p_list[pid]
#             if _p.totburst==0:
#                 print(f"task {_p.task.name} start at {curr_t:.6f}")
#             running_queue.put(_p)
#             res_cfg.allocate(_p.pid, curr_cfg.rsc_map[pid])
#             record_comp_bw_slot_by_slot(rsc_recoder, n_slot, curr_cfg, pid)

#         # execute the task in running list
#         # update the running task
#         updateRunningQueue(timestep, running_queue, res_cfg) 
#         # update the wait task
#         updateWaitQueue(timestep, wait_queue)

#         # data prefetching
#         # compare the next cfg with current one
#         # TODO: add a control and corresponding parameters to number of slot of forward looking
#         if n_slot < sim_slot_num-1:
#             next_cfg = _SchedTab.scheduling_table[tab_pointer+1]
#             if curr_cfg != next_cfg: 
#                 print(f"		cfg of bin {bin_name:s} is updated @ {curr_t+timestep:.6f}")
#                 # TODO: prefetch data 
#                 # TODO: how to represent the tile prefetching: when to start, when to check
                
#                 # # ensure the next cfg is not empty
#                 # if len(next_cfg.rsc_map):
#                 #     for pid in next_cfg.rsc_map.keys():
#                 #         _p = init_p_list[pid]
#                 #         if _p in active_list:
#                 #             active_list.remove(_p)
#                 #             wait_queue.put(_p)
#                 #             _p.set_state("wait")
#                 #         else: 
#                 #             print("		Arriving lateness of task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

def sched_step(task_spec:Spec, # msg_dispatcher:MsgDispatcher,#msg_pipe:Message,
                a_data_pipe:DataPipe, 
                w_data_pipe:DataPipe,
                scheduler_list: List[Scheduler], monitor_list:List[Monitor],
                rsc_list:List[Resource_model_int], 
                total_cores:int, n_slot, 
                glb_p_list:List[ProcessInt],
                timestep, hyper_p, n_p=1, verbose=False, *, warmup=False, drain=False,):
        
    """
    implement a step of runtime scheduling
    """
    # task status

    # completed: task is completed
    # miss: not completed before its deadline
    # throttled: waiting for the next slot
    # running: task is running
    # ----------------------------------------
    # executeable: ready task, resource ready, but not issued yet
    # blocked: resource is not available
    # ready: task is ready to execute, current slot is in its liveness interval and data are ready
    # ----------------------------------------
    # waiting: waiting for data
    # active: task is in its liveness interval


    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)
    tab_spatial_size = total_cores

    task_spec_bk:Spec = copy.deepcopy(task_spec)
    curr_t = n_slot * timestep
    glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    # spatial management
    # Each partition maintains a scheduling table, a task monitor, and a scheduler. 
    for res_cfg, sched, monitor,  msg_queue, in zip(rsc_list, scheduler_list, monitor_list, a_data_pipe.queues, ):

        # print(f"	Bin {_SchedTab.id:d}:")
        # extract scheudler, including queues and lists from scheduler_list
        glb_name_p_dict: Dict[str, ProcessInt]
        msg_queue:Queue
        DEBUG_FG = False

        sched.scheduler_step(a_data_pipe, w_data_pipe, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, monitor, DEBUG_FG)
    # update the wait task
    w_data_pipe.update_wait_time(timestep)
    a_data_pipe.update_wait_time(timestep)

# =================== top global scheduler ===================
def cyclic_sched(task_spec:Spec, affinity, 
                scheduler_list: List[Scheduler], monitor_list:List[Monitor],
                rsc_list:List[Resource_model_int], 
                total_cores:int, 
                glb_p_list:List[ProcessInt],
                timestep, hyper_p, n_p=1, 
                # msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
                a_data_pipe:DataPipe=None,
                w_data_pipe:DataPipe=None, 
                verbose=False, *, warmup=False, drain=False,):
    """
    partition the scheduling table
    """
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)

    # pre_ready stage for the initial tasks
    for sched in scheduler_list:
        # extract scheudler, including queues and lists from scheduler_list
        ready_queue:TaskQueue = sched.ready_queue
        wait_queue:TaskQueue = sched.weight_wait_queue
        inactive_list:List[ProcessInt] = sched.inactive_list
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
            # _p.build_ctx()
            # _p.update_ctx("trigger")

            # simulate the prefetching of the weight
            msg:ContextMsg = ContextMsg.create_weight_ctx()
            data = Data(_p.pid, 1, (0,), "weight", _p.io_time, 0, 1/_p.task.freq)
            data.ctx = msg
            data.cache_msg_transfer(0)

            data.update_receive_time(0)
            data.valid = True
            buffer.put(data)

            # _p.update_ctx('weight', buffer=buffer)
            # _p.update_ctx('upstream', buffer=buffer, glb_n_task_dict=glb_n_task_dict)
            # ready_queue.put(_p)
            # _p.set_state("ready")
            # _p.released = True 
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
            msg:ContextMsg = ContextMsg.create_weight_ctx()
            data = Data(_p.pid, 1, (0,), "weight", _p.io_time, 0, 1/_p.task.freq)
            data.ctx = msg
            data.cache_msg_transfer(0)

            data.update_receive_time(0)
            data.valid = True
            buffer.put(data)

            print("		TASK {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        print("")

        print("	inactivated tasks:")
        for pid in task_pid_list:
            # if pid not in init_cfg.rsc_map: # and pid not in cached_map:
                _p = process_dict[pid]
                inactive_list.append(_p)
                _p.set_state("suspend")
                print("		TASK {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        print("")


    for n_slot in range(sim_slot_num):
        curr_t = n_slot * timestep

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")
        
        # TODO: detect the spec change
            # modify the exp_comp_t and deadline of the tasks

        # print(f"Slot {n_slot:d}, time {curr_t:.6f}")
        sched_step(task_spec, a_data_pipe,
                    w_data_pipe,
                    scheduler_list, monitor_list,
                    rsc_list, 
                    total_cores, n_slot, 
                    glb_p_list, 
                    timestep, hyper_p, n_p, verbose, warmup=warmup, drain=drain) 


def glb_sched(task_spec:Spec, affinity, 
                scheduler_list: List[Scheduler], monitor_list:List[Monitor],
                rsc_list:List[Resource_model_int], 
                total_cores:int, 
                glb_p_list:List[ProcessInt],
                timestep, hyper_p, n_p=1, 
                msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
                a_data_pipe:DataPipe=None,
                w_data_pipe:DataPipe=None, 
                quantum_check_en:bool = False, quantumSize=None,
                verbose=False, *, warmup=False, drain=False,):
    """
    partition the scheduling table
    """
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)

    glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    # spatial management
    # Each partition maintains a scheduling table, a task monitor, and a scheduler. 
    # for res_cfg, sched, monitor,  msg_queue, in zip(rsc_list, scheduler_list, monitor_list, msg_dispatcher.queues, ):
    res_cfg, sched, monitor,  msg_queue = rsc_list[0], scheduler_list[0], monitor_list[0], msg_dispatcher.queues[0]
    inactive_list:List[ProcessInt] = sched.inactive_list
    for _p in glb_p_list:
        if _p not in inactive_list:
            inactive_list.append(_p)
    sched.process_dict.update({p.pid:p for p in glb_p_list})

    for n_slot in range(sim_slot_num):
        curr_t = n_slot * timestep

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")
        
        curr_t = n_slot * timestep

        # extract scheudler, including queues and lists from scheduler_list
        glb_name_p_dict: Dict[str, ProcessInt]
        msg_queue:Queue
        DEBUG_FG = False

        glb_dynamic_sched_step(sched, msg_dispatcher, a_data_pipe, w_data_pipe, n_slot, timestep, event_range, sim_slot_num, curr_t, glb_name_p_dict, res_cfg, msg_queue, monitor, DEBUG_FG, quantum_check_en, quantumSize)



if __name__ == "__main__": 
    import argparse
    import numpy as np 
    import pickle

    from task.task_cfg import load_taskint, create_init_p_list
    from task.task_cfg import affinity_cfg, task_graph_srcs, task_graph_ops, task_graph_sinks
    from task.task_cfg import creat_physical_graph, creat_logical_graph, init_depen
    from global_sched import push_task_into_bins

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="all", help="task name")
    parser.add_argument("--plot", action="store_true", help="plot the task timeline")
    parser.add_argument("--test_all", default=False, help="test all the task")
    parser.add_argument("--num_cores", default=266, type=int, help="number of cores")
    parser.add_argument("--BinExtendRule", default="list", type=str, help="Rule for when and how to extend the bin")
    parser.add_argument("--preemptable", default=False, action="store_true", help="enable preemption")
    parser.add_argument("--quantum_check_en", default=False, action="store_true", help="enable quantum check")
    parser.add_argument("--quantumSize", default=2, type=int, help="quantum size, # of simulation steps")
    # parser.add_argument("--hyper_p", default=None, type=float, help="hyper period")
    # parser.add_argument("--warmup", default=False, action="store_true", help="warmup")
    # parser.add_argument("--drain", default=False, action="store_true", help="drain")
    # parser.add_argument("--sim_step", default=None, type=float, help="simulation step")
    parser.add_argument("--n_p", default=1, type=int, help="number of periods")
    parser.add_argument("--jitter_sim_en", default=False, action="store_true", help="enable jitter simulation")
    parser.add_argument("--jitter_sim_para", default={"a":-0.2, "b":0.2, "loc":0, "scale":1}, type=dict, help="jitter simulation parameters")
    parser.add_argument("--file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--i_file_suffix", default="", type=str, help="file suffix")

    args = parser.parse_args() 
    glb_n_task_dict = load_taskint(args.verbose)
    num_cores = args.num_cores

    if args.test_case == "all":
        args.test_all = True
    f_gcd = np.gcd.reduce([glb_n_task_dict[task].freq for task in glb_n_task_dict])
    f_max = max([glb_n_task_dict[task].freq for task in glb_n_task_dict])
    hyper_p = 1/f_gcd
    sim_step = min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32
    quantumSize = sim_step*args.quantumSize
    num_periods = args.n_p
    np.random.seed(0)

    if args.test_case == "bin_pack" or args.test_all:
        # push_task_into_scheduling_table_cyclic_preemption_disable(task_dict, num_cores, sim_step*1, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)
        init_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
        bin_list, init_p_list = push_task_into_bins(init_p_list, affinity_cfg, num_cores, args.quantum_check_en, quantumSize, sim_step, hyper_p, num_periods, args.verbose, warmup=True, drain=True)
        from sched.scheduling_table import get_task_layout_compact
        get_task_layout_compact(bin_list, init_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
        txt_size=40, tick_dens=2, save_path=f"plot/task_bin_pack_cyclic_{num_cores}{args.file_suffix}.pdf") 

        get_task_layout_compact(bin_list, init_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path=f"plot/task_bin_pack_full_{num_cores}{args.file_suffix}.pdf")

        # select a period to save 
        assert num_periods > 1
        bin_list2save = []
        # for _sched_tab in bin_list:


        # save the bin_list and the init_p_list
        with open(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl", "wb") as f:
            pickle.dump(bin_list, f)
        # with open(f"init_p_list_{num_cores}{args.i_file_suffix}.pkl", "wb") as f:
        #     pickle.dump(init_p_list, f)
        try:
            # load the bin_list and the init_p_list
            with open(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl", "rb") as f:
                bin_list = pickle.load(f)
            print(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl saved and loaded successfully")
            # with open(f"init_p_list_{num_cores}{args.i_file_suffix}.pkl", "rb") as f:
            #     init_p_list = pickle.load(f)
            # print(f"init_p_list_{num_cores}{args.file_suffix}.pkl saved and loaded successfully")
        except:
            print(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl or init_p_list_{num_cores}{args.i_file_suffix}.pkl not found")
            exit()

    elif args.test_case == "dynamic" or args.test_all:
        init_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
        try:
            # load the bin_list and the init_p_list
            with open(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl", "rb") as f:
                bin_list = pickle.load(f)
            # with open(f"init_p_list_{num_cores}{args.i_file_suffix}.pkl", "rb") as f:
            #     init_p_list = pickle.load(f)
        except:
            print(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl not found")
            # print(f"cache/bin_list_{num_cores}{args.i_file_suffix}.pkl or init_p_list_{num_cores}{args.i_file_suffix}.pkl not found")
            bin_list, _ = push_task_into_bins(init_p_list, affinity_cfg, num_cores, args.quantum_check_en, quantumSize, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)

        logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
        physical_graph_nx = creat_physical_graph(logical_graph_nx, int(f_gcd))
        
        init_depen(glb_n_task_dict, physical_graph_nx, verbose=args.verbose)
        # from message_agent import Message
        
        task_spec = Spec(0.1, [1 for _ in init_p_list]) 
        # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
        rsc_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        # msg_dispatcher = MsgDispatcher(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list))
        w_data_pipe = DataPipe("weight", len(bin_list))
        # filter the processes with trigger_mode is not "N"
        sim_triggered_list = [[init_p_list[pid] for pid in _bin.index_occupy_by_id().keys() if init_p_list[pid].task.trigger_mode!='N'] for _bin in bin_list]
        for _bin, _sim_triggered_list in zip(bin_list, sim_triggered_list):
            _bin.sim_triggered_list = _sim_triggered_list
            print("bin: ", _bin.id, "sim_triggered_list: ", [p.task.name for p in _sim_triggered_list])
        scheduler_list = [Scheduler(_SchedTab, init_p_list, jitter_sim_en=args.jitter_sim_en, jitter_sim_para=args.jitter_sim_para) for _SchedTab in bin_list]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]

        print("sim_step: ", sim_step)
        cyclic_sched(task_spec, affinity_cfg, 
                scheduler_list, monitor_list,
                rsc_list, 
                num_cores, 
                init_p_list,
                sim_step, hyper_p, num_periods, 
                a_data_pipe, w_data_pipe, 
                args.verbose, warmup=True, drain=True)

        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in init_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        for _SchedTab in actual_sched_record:
            _SchedTab.print_alloc_detail(pid2name, sim_step)

        from sched.scheduling_table import get_task_layout_compact
        get_task_layout_compact(actual_sched_record, init_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path=f"plot/exe_monitor_full_{num_cores}{args.file_suffix}.pdf")

        # save trace_list to trace_file
        with open(f"trace/dynamic_e2e_trace_{num_cores}{args.file_suffix}.pkl", "wb") as f:
            pickle.dump(trace_list, f)
        
        try:
            with open(f"trace/dynamic_e2e_trace_{num_cores}{args.file_suffix}.pkl", "rb") as f:
                trace_list = pickle.load(f)
            print("trace saved successfully")
        except:
            print("trace file not found")
            exit(0)            

    elif args.test_case == "glb_dynamic":
        np.random.seed(0)
        init_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
        bin_list = [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")]
        logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
        physical_graph_nx = creat_physical_graph(logical_graph_nx, int(f_gcd))
        
        init_depen(glb_n_task_dict, physical_graph_nx, verbose=args.verbose)
        # from message_agent import Message
        
        task_spec = Spec(0.1, [1 for _ in init_p_list]) 
        # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
        rsc_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list))
        w_data_pipe = DataPipe("weight", len(bin_list))
        # filter the processes with trigger_mode is not "N"
        sim_triggered_list = [[p for p in init_p_list if p.task.trigger_mode!='N'],]
        for _bin, _sim_triggered_list in zip(bin_list, sim_triggered_list):
            _bin.sim_triggered_list = _sim_triggered_list
            print("bin: ", _bin.id, "sim_triggered_list: ", [p.task.name for p in _sim_triggered_list])
        scheduler_list = [Scheduler(_SchedTab, init_p_list, jitter_sim_en=args.jitter_sim_en, jitter_sim_para=args.jitter_sim_para) for _SchedTab in bin_list]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]

        print("sim_step: ", sim_step)
        glb_sched(task_spec, affinity_cfg, 
                scheduler_list, monitor_list,
                rsc_list, 
                num_cores, 
                init_p_list,
                sim_step, hyper_p, num_periods, 
                msg_dispatcher,
                a_data_pipe, w_data_pipe,
                args.quantum_check_en, quantumSize, 
                args.verbose, warmup=True, drain=True)
        
        print("number of context switch {}".format(scheduler_list[0].barrier.number_of_asserts))
        print("cumulative context switch {}".format(scheduler_list[0].barrier.cumulative_time))

        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in init_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        for _SchedTab in actual_sched_record:
            _SchedTab.print_alloc_detail(pid2name, sim_step)

        from sched.scheduling_table import get_task_layout_compact
        get_task_layout_compact(actual_sched_record, init_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path=f"plot/dyn_glb_exe_monitor_full_{num_cores}{args.file_suffix}.pdf")

        # save trace_list to trace_file
        with open(f"trace/dyn_glb_e2e_trace_{num_cores}{args.file_suffix}.pkl", "wb") as f:
            pickle.dump(trace_list, f)
        try:
            with open(f"trace/dyn_glb_e2e_trace_{num_cores}{args.file_suffix}.pkl", "rb") as f:
                trace_list = pickle.load(f)
            print("trace saved successfully")
        except:
            print("trace file not found")
            exit(0)

        # plot with bokhe
        # get_task_layout_compact(actual_sched_record, init_p_list, save= True, time_step= sim_step,
        # hyper_p=hyper_p, n_p=1, warmup=True, drain=True, plot_legend=False, 
        # txt_size=40, tick_dens=4, plot_start=0, tool="bokeh", 
        # save_path=f"plot/dyn_glb_exe_monitor_full_bokeh_{num_cores}{args.file_suffix}")
