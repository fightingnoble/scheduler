
from example.bm3 import *

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

from queue import PriorityQueue
from heapq import heappush, heappop
event_t = []
planned_event = []

# push src event to event_t
for src, offset in src_attr.items():
    event_t.append((offset, "external"))
event_t = list(set(event_t))
event_t.sort()

n_pred_map =  {node:len(list(G.predecessors(node))) for node in G.nodes()}
ready_queue = dict()
running_queue = dict()
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
    print(f"\tDuation:" + str(event_t[0][0] - curr_t))
    print(f"\tRunning_queue: {running_queue}")
    print(f"\tRes_map: {res_map}")
    print(f"\tSlack_map: {slack_map}")
    
    # status backup
    pred_rsc_map = copy.deepcopy(res_map)
    pred_t, (curr_t, event_type) = curr_t, event_t.pop(0)
    if event_type != "external":
        planned_event.remove(curr_t)
    print(f"\nmove from {pred_t} to {curr_t}\n") 
    print(f"===At the beginning of {curr_t}===")
    
    # check the event type: 
    # finish, reallocate, external
    if event_type == "external": 
        # check stimulate events
        for node in list(src_attr.keys()):
            if curr_t == src_attr[node]:
                n_pred_map.pop(node)
                mark_finish(G, n_pred_map, node)
                src_attr.pop(node)
                print(f"\tsrc {node} arrives at {curr_t}")             
    
    if event_type != "reallocate":
        # calculate remaining workloads
        # and check finishing events
        for node in list(running_queue.keys()):
            rem_t  = running_queue[node] - res_map[node] * (curr_t - pred_t)
            if rem_t <= 0:
                running_queue.pop(node)
                mark_finish(G, n_pred_map, node)
                res_map.pop(node)
                print(f"\ttask {node} finish at {curr_t}")
            else:
                running_queue[node] = rem_t

        new_ready_flag = False
        # put ready task to ready_queue
        for node, n_pred in list(n_pred_map.items()):
            if n_pred == 0:
                if node in task_graph_sinks:
                    mark_finish(G, n_pred_map, node)
                    print(f"\tsink {node} finish at {curr_t}")
                    n_pred_map.pop(node)
                elif node in task_graph_ops:
                    ready_queue[node] = task_attr[node]["exp_lat"]* task_attr[node]["base_size"]
                    new_ready_flag = True
                    print(f"\ttask {node} ready at {curr_t}")
                    n_pred_map.pop(node)

        # calculate slack        
        slack_map = {node:task_attr[node]['ddl'] - curr_t-swt_lat for node in list(running_queue.keys()) + list(ready_queue.keys())} 
        
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
                assert not (node in ready_queue and node in running_queue) 
                req_rsc_size = math.ceil((running_queue.get(node, 0) + ready_queue.get(node, 0))/slack)
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
        # print(f"\tReallocation is triggered at {curr_t}")
        for node, rem_t in list(running_queue.items()):
            if not node in rsc_map_t: 
                ready_queue[node] = running_queue.pop(node)
                            
        for node, rem_t in list(ready_queue.items()):
            if node in rsc_map_t: 
                running_queue[node] = ready_queue.pop(node)

        print(f"\tEnter reallocation progress at {curr_t}")
        sys_state = "R"
        res_map.clear()
        res_map.update(dict.fromkeys(rsc_map_t, 0))
        # delete all other planned events
        pos = 0
        for old_event in planned_event:
            while pos < len(event_t) and event_t[pos][0] < old_event:
                pos += 1
            while pos < len(event_t) and event_t[pos][0] == old_event: 
                if event_t[pos][1] != "external":
                    event_t.pop(pos)
                pos += 1
        planned_event.clear()
        duation = swt_lat
        type_ = "reallocate"
    else:
        print(f"\texist reallocation state at {curr_t}")   
        sys_state = "S"
        res_map = rsc_map_t

        # calculate druation to next event
        # running tasks which will first finish
        duation = math.ceil(min([running_queue[pid]/res_map[pid] for pid in running_queue]))
        type_ = "finish"
    pos = 0
    while  pos < len(event_t) and event_t[pos][0] < curr_t + duation:
        pos += 1
    if len(event_t) ==pos or event_t[pos][0] != curr_t + duation:
        event_t.insert(pos,(curr_t + duation, type_))
    planned_event.append(curr_t + duation)
    planned_event.sort()
        

    
    