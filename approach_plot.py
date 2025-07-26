
from example.bm4 import *

import sys
import os
# add the parent directory to the path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from approach_util import Acc_p, Sen_p, MyGraph, PartitionConfig, acc_p_factory, GlobalEvent_t


def print_progress(curr_t, pred_t, processors):
    print(f"===Decision metadata at {curr_t}===")
    if curr_t > 0:
        print(f"\tDuration: {curr_t - pred_t}")
    else:
        print(f"\tInitial decision")
    for i, proc in enumerate(processors):
        print(f"\n\tProcessor {i}: {proc}")
        # Print sys_state if it exists
        if hasattr(proc, 'sys_state'):
            print(f"\t\tstate: {proc.sys_state}")
        # Print running if it exists
        if hasattr(proc, 'running'):
            print(f"\t\tRunning_queue: {proc.running}")
        # Print res_map if it exists
        if hasattr(proc, 'res_map'):
            print(f"\t\tRes_map: {proc.res_map}")
        # Print slack_map if it exists
        if hasattr(proc, 'slack_map'):
            print(f"\t\tSlack_map: {proc.slack_map}")

pred_t = 0
curr_t = 0

G = MyGraph(
    task_graph_srcs, task_graph_ops, task_graph_sinks, 
    task_attr, src_attr
)

# replace the manual initialization
# acc_p0 = Acc_p("acc_p0", tot_core, 1, G)

# Suppose there is only one partition
partition_cfg = PartitionConfig(
    num_partitions=1,
    cap_list=[tot_core],
    base_pwr_list=[1],
    mapped_node_list=[G.sinks+G.ops],
    TSmap_list=[None],  # 如果用cyclic策略可传入具体map
    G=G, # nodes other than sources
)

acc_p_list = acc_p_factory("glb", partition_cfg)
acc_p0 = acc_p_list[0]

event_t = GlobalEvent_t([(0, "external")])
sen_p0 = Sen_p("sen_p0", 3, 1, G, G.srcs)

# 支持多种类型处理器统一调度
processors = [sen_p0, acc_p0]

while G.nodes():
    if curr_t != 0:
        print(f"\nmove from {pred_t} to {curr_t}\n") 
    print(f"===At the beginning of {curr_t}===")
    
    # Two types of divice is considered: sen_p0 and acc_p0
    # sen_p0: sensor processor
    # acc_p0: accelerator     
    # Each processor folows update -> schedule -> execute -> update cycle. 
    
    # During the update phase, the runtime will 
    # 1. calculate remaining workloads and check finishing events. 
    # process the event (1. external, 2. finish and reallocate)
    
    # The most important difference of the accelerator processor, compared to the sensor processor, is that 
    # it has to go through a extra step of reallocation, before launching the different allocation map. 
    # The reallocation is a special case of execution, which can be seen as a "system task" that reallocate the resources, 
    # and will preempt other computational tasks until it finishes. 
    # However, I don't know how to unify two types of tasks. 
    
    # A minor difference is the scheduling policy: Sensor processor follows the first-come-first-served policy (FCFS), 
    # while accelerator follows a priority-based policy with parameterized priority defination. 

    # Task dispatching: 
    # As for our ADS scenario, we mark the periodic timer event as "external", 
    # which informs the sensor processor to access and pre-process sensor data. 
    # While the accelerator processor is responsible for the remaining DNN tasks, 
    # which are triggered to schedule incomming tasks, when "finish" events occur. 
    # Specially, "reallocate" means the completion of system reallocation, 
    # which is also need to be unified with the the concept of "finish" event. 
    
    # update the running queue, calculate remaining workloads and check finishing events
    # sen_p0.update_run(pred_t, curr_t)
    # new_comp = acc_p0.update_run(pred_t, curr_t)
    new_comp_dict = {}
    for proc in processors:
        new_comp = proc.update_run(pred_t, curr_t)
        new_comp_dict[proc] = new_comp

    # process the event, update the ready queue
    # sen_p0.update_ready(curr_t)
    # new_ready_list = acc_p0.update_ready(curr_t)
    new_ready_dict = {}
    for proc in processors:
        new_ready = proc.update_ready(curr_t)
        new_ready_dict[proc] = new_ready


    # allocation progress
    # duation_sen_p = sen_p0.sched(curr_t) 
    # duation_acc_p= acc_p0.sched(curr_t, new_comp, new_ready_list)

    # 统一 sched
    duation_dict = {}
    for proc in processors:
        # 兼容不同类型的参数
        if isinstance(proc, Acc_p):
            duation = proc.sched(curr_t, new_comp_dict[proc], new_ready_dict[proc])
        else:
            duation = proc.sched(curr_t)
        duation_dict[proc] = duation


    # calculate the next 
    # duation = min(duation_acc_p, duation_sen_p)
    # 统一事件推进
    duation = min(duation_dict.values())

    # 统一事件队列（可扩展为每个处理器独立event_t）
    next_timer_event = event_t.get_next_event_time(curr_t)

    print_progress(curr_t, pred_t, processors)
    
    # status backup
    curr_t, pred_t = min(curr_t + duation, next_timer_event[0]), curr_t
    event_type = next_timer_event[1] if curr_t == next_timer_event[0] else "finish"
    # check the event type: finish, external
    assert event_type in ["finish", "external"]
