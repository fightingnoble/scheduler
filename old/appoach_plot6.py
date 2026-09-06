
from example.bm4 import *

import sys
import os
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from old.approach_util33 import Acc_p, Sen_p, MyGraph

event_t = [] # detailed info of all events, (t, type)
event_t.extend([(0, "external"), (float("inf"), "external")])
event_t = list(set(event_t))
event_t.sort()

pred_t = 0
curr_t = 0

G = MyGraph(
    task_graph_srcs, task_graph_ops, task_graph_sinks, 
    task_attr, src_attr
)
acc_p0 = Acc_p("acc_p0", tot_core, 1, G)
sen_p0 = Sen_p("sen_p0", 3, 1, G)

while G.nodes():
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
    sen_p0.update_run(pred_t, curr_t)
    new_comp = acc_p0.update_run(pred_t, curr_t)

    # process the event, update the ready queue
    sen_p0.update_ready(curr_t)
    new_ready_list = acc_p0.update_ready(curr_t)
    
    # allocation progress
    duation_sen_p = sen_p0.sched(curr_t) 
    duation_acc_p, type_ = acc_p0.sched(curr_t, new_comp, new_ready_list)

    # calculate the next 
    duation = min(duation_acc_p, duation_sen_p)
    # pos = 0
    # while  pos < len(event_t) and event_t[pos][0] < curr_t + duation:
    #     pos += 1
    # if len(event_t) ==pos or event_t[pos][0] != curr_t + duation:
    #     event_t.insert(pos,(curr_t + duation, type_))
    if event_t[0][0] <= curr_t:
        event_t.pop(0)
    next_timer_event = min(event_t, key=lambda x:x[0])

    print(f"===Decision metadata at {curr_t}===")
    print(f"\tstate: {acc_p0.sys_state}")
    if curr_t > 0:
        print(f"\tDuation:" + str(curr_t-pred_t))
    else: 
        print(f"\tInitial decision")
    print(f"\tRunning_queue: {acc_p0.running}")
    print(f"\tRes_map: {acc_p0.res_map}")
    print(f"\tSlack_map: {acc_p0.slack_map}")
    
    # status backup
    curr_t, pred_t = min(curr_t + duation, next_timer_event[0]), curr_t
    event_type = next_timer_event[1] if curr_t == next_timer_event[0] else "finish"
    # check the event type: finish, reallocate, external
    assert event_type in ["finish", "reallocate", "external"]
    if curr_t != 0:
        print(f"\nmove from {pred_t} to {curr_t}\n") 
