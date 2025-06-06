
from example.bm4 import *

import sys
import os
# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task.task_cfg import creat_logical_graph
import networkx as nx
G:nx.DiGraph = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
from collections import OrderedDict
from utils import core_distr
import math
import copy

from queue import PriorityQueue, Queue
from heapq import heappush, heappop
event_t = [] # detailed info of all events, (t, type)
# planned_event = [] # temporal list for recording expected internal triggered events in the future: reallocate, finish

# push src event to event_t
# for src, offset in src_attr.items():
#     event_t.append((offset, "external"))
event_t.append((0, "external"))
event_t = list(set(event_t))
event_t.sort()

n_pred_map =  {node:len(list(G.predecessors(node))) for node in G.nodes() if node not in task_graph_srcs} 


sen_ready_queue = Queue(-1)
sen_running_queue = dict()
n_sen_rsc = 3

acc_ready_queue = dict()
acc_running_queue = dict()

res_map = dict()
pre_rsc_map, rsc_map_t = dict(), dict()
sys_state = "S"
sys_state_emu = ["S", "R"]
slack_map = dict()

recoder = {"slack": dict(), "res":dict(),"rem_load":dict()}

pred_t = 0
def mark_finish(G:nx.DiGraph, n_pred_map, node):
    for succ in G.successors(node):
        n_pred_map[succ] -= 1
    G.remove_edges_from([(node, succ) for succ in G.successors(node)])
    G.remove_node(node)


curr_t = 0
while G.nodes():
    print(f"===Decision metadata at {curr_t}===")
    print(f"\tstate: {sys_state}")
    if event_t[0][0] > 0:
        print(f"\tDuation:" + str(event_t[0][0] - curr_t))
    else: 
        print(f"\tInitial decision")
    print(f"\tRunning_queue: {acc_running_queue}")
    print(f"\tRes_map: {res_map}")
    print(f"\tSlack_map: {slack_map}")
    
    # status backup
    pred_rsc_map = copy.deepcopy(res_map)
    pred_t, (curr_t, event_type) = curr_t, event_t.pop(0)
    # if event_type != "external":
    #     planned_event.remove(curr_t) # remove the committed internal events
    if curr_t != 0:
        print(f"\nmove from {pred_t} to {curr_t}\n") 
    print(f"===At the beginning of {curr_t}===")
    
    # Two types of divice is considered: sen_p0 and acc_p0
    # sen_p0: sensor processor
    # acc_p0: accelerator 
    # Sensor processor follows the first-come-first-served policy (FCFS), while accelerator follows a priority-based policy with parameterized priority defination. 
    
    # Only the external events tirgger the sensor nodes, which will be scheduled on the sensor processor. 
    # While the intermidiate nodes will be triggered by the "finish" event of the previous node of its predecessors, and will be scheduled on the accelerator. 
    
    # Specially, reallocation will interrupt the execution of the current task on accelerator, which will forbid the accelerator in ether scheduling or execution. 
    
    # check the event type: 
    # finish, reallocate, external
    assert event_type in ["finish", "reallocate", "external"]

    # process the event on sensor processor
    # update the running queue
    for node in list(sen_running_queue.keys()):
        rem_t  = sen_running_queue[node] - (curr_t - pred_t)
        if rem_t <= 0:
            mark_finish(G, n_pred_map, node)
            sen_running_queue.pop(node)
            print(f"\tsrc {node} arrives at {curr_t}")
        else:
            sen_running_queue[node] = rem_t

    # update the ready queue
    if event_type == "external": 
        # trigger sensor nodes
        for node in list(src_attr.keys()):
            if curr_t == src_attr[node]["offset"]:
                load = src_attr[node]["exp_lat"]
                if load > 0:
                    sen_ready_queue.put((node, load))
                    src_attr.pop(node)
                    print(f"\tsrc {node} is triggered at {curr_t}")
                else:
                    mark_finish(G, n_pred_map, node)
                    print(f"\tsrc {node} arrives at {curr_t}")
    
    # FCFS policy: serve a new task until last task is finished
    while len(sen_running_queue) < n_sen_rsc and not sen_ready_queue.empty():            
        node, rem_t = sen_ready_queue.get()
        sen_running_queue[node] = rem_t
    
    duation_sen_p = math.ceil(min([sen_running_queue[pid] for pid in sen_running_queue])) if sen_running_queue else float("inf")
    if duation_sen_p == float("inf"):
        print(f"\tNo sensor event in future at {curr_t}")
    else:
        print(f"\tNext sensor event at {curr_t + duation_sen_p}")
    
    # allocation progress
    if event_type != "reallocate":
        # calculate remaining workloads
        # and check finishing events
        for node in list(acc_running_queue.keys()):
            rem_t  = acc_running_queue[node] - res_map[node] * (curr_t - pred_t)
            if rem_t <= 0:
                acc_running_queue.pop(node)
                mark_finish(G, n_pred_map, node)
                res_map.pop(node)
                print(f"\ttask {node} finish at {curr_t}")
            else:
                acc_running_queue[node] = rem_t

        new_ready_flag = False
        # put ready task to ready_queue
        for node, n_pred in list(n_pred_map.items()):
            if n_pred == 0:
                if node in task_graph_sinks:
                    mark_finish(G, n_pred_map, node)
                    print(f"\tsink {node} finish at {curr_t}")
                    n_pred_map.pop(node)
                elif node in task_graph_ops:
                    acc_ready_queue[node] = task_attr[node]["exp_lat"]* task_attr[node]["base_size"]
                    new_ready_flag = True
                    print(f"\ttask {node} ready at {curr_t}")
                    n_pred_map.pop(node)

        # calculate slack        
        slack_map = {node:task_attr[node]['ddl'] - curr_t-swt_lat for node in list(acc_running_queue.keys()) + list(acc_ready_queue.keys())} 
        
        # if sys_state != "R" and not new_ready_flag and set(res_map.keys()) == set(pre_rsc_map.keys()) and not len(slack_map):
        # no unreleased events: len(event_t) == 0 and len(n_pred_map) == 0
        # no running and pending tasks: len(running_queue) == 0 and len(ready_queue) == 0
        # all task are done
        if not slack_map and len(event_t) == 0 and len(n_pred_map) == 0:
            print(f"===All tasks are finished at {curr_t}===")
            break
        
        # calculate min_rsc requirement
        pre_rsc_map, rsc_map_t = rsc_map_t, {} 
        score_dict = OrderedDict()
        constr_dict = OrderedDict()
        curr_aval_rsc = tot_core
        for node in sorted(slack_map, key=slack_map.get):
            slack = slack_map[node]
            if slack <= 0:
                print(f"\t{node} is timeout at {curr_t}")
                req_rsc_size = curr_aval_rsc
            else:
                assert not (node in acc_ready_queue and node in acc_running_queue) 
                req_rsc_size = math.ceil((acc_running_queue.get(node, 0) + acc_ready_queue.get(node, 0))/slack)
                if req_rsc_size > curr_aval_rsc:
                    print(f"\t{node} is hungry at {curr_t}: lack {req_rsc_size - curr_aval_rsc} tiles") 
                    req_rsc_size = curr_aval_rsc
                
            curr_aval_rsc -= req_rsc_size
            rsc_map_t[node] = req_rsc_size
            constr_dict[node] = "N/A"
            score_dict[node] = 1/slack_map[node] if slack_map[node] >0 else float("inf")
            slack_map.pop(node)
            if curr_aval_rsc <= 0:
                break
        
        # allocate free resource
        if curr_aval_rsc > 0:
            print(f"\tMinimum resource requirement at {curr_t}: {rsc_map_t}")
            # if there are still resources left, 
            # it means no late process is waiting for resources
            assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
            # also, there is no process waiting for resources in the ready queue
            assert len(slack_map) == 0
            core_distr(rsc_map_t, score_dict, curr_aval_rsc)
    else:
        pre_rsc_map = rsc_map_t

    # event: 
    # 1. rescheduling happens, generate new reallocation event
    # 2. event arrives, trigger rescheduling, but not necessarily generate new reallocation event
    # 3. new configuration is issued, impose a finish event
    
    # interference among events
    # 1. if arriving events trigger realocation, we need to evict the planned completion time of the tasks
    
    # issue task: first endurance switching progress, before turely allocated resources
    if pre_rsc_map != rsc_map_t:
        # the tasks that alloted no resource is preempted, move them to ready queue
        for node, rem_t in list(acc_running_queue.items()):
            if not node in rsc_map_t: 
                acc_ready_queue[node] = acc_running_queue.pop(node)
        
        # the tasks that alloted resource is executed, move them to running queue
        for node, rem_t in list(acc_ready_queue.items()):
            if node in rsc_map_t: 
                acc_running_queue[node] = acc_ready_queue.pop(node)

        print(f"\tEnter reallocation progress at {curr_t}")
        sys_state = "R"
        res_map.clear()
        res_map.update(dict.fromkeys(rsc_map_t, 0))
        # delete all other planned events
        pos = 0
        # for old_event in planned_event:
        #     while pos < len(event_t) and event_t[pos][0] < old_event:
        #         pos += 1
        #     while pos < len(event_t) and event_t[pos][0] == old_event: 
        #         if event_t[pos][1] != "external":
        #             event_t.pop(pos)
        #         pos += 1
        # planned_event.clear()
        duation_acc_p = swt_lat
        type_ = "reallocate"
    else:
        print(f"\texist reallocation state at {curr_t}")   
        sys_state = "S"
        res_map = rsc_map_t

        # calculate druation to next event
        # running tasks which will first finish
        duation_acc_p = math.ceil(min([acc_running_queue[pid]/res_map[pid] for pid in acc_running_queue])) if acc_running_queue else float("inf")
        if duation_acc_p == float("inf"):
            print(f"\tNo accelerator event in future at {curr_t}")
        else:
            print(f"\tNext accelerator event at {curr_t + duation_acc_p}")
        type_ = "finish"

    duation = min(duation_acc_p, duation_sen_p)
    pos = 0
    while  pos < len(event_t) and event_t[pos][0] < curr_t + duation:
        pos += 1
    if len(event_t) ==pos or event_t[pos][0] != curr_t + duation:
        event_t.insert(pos,(curr_t + duation, type_))
    # planned_event.append(curr_t + duation)
    # planned_event.sort()

    # **************** Note for event insertion logic ****************
    # A faulty event insertion manner: 
    # the next external event is not necessarily to trigger a rescheduling, 
    # if we naively drop the internal events that latter than the next external event, 
    # there is a risk that a ture event is missed. 
    # To avoid this, we need to keep the internal events in the event_t list, 
    # and drop it only if the reallocation progress is triggered.
    # next_event = curr_t + duation 
    # if not event_t or event_t[0] > next_event:
    #     event_t.insert(0, next_event)



