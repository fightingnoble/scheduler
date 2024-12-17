task_graph_sinks={
    "O1": [],
    "O2": [],
}

S1_offset = 30
S2_offset = 20



task_graph_srcs = {
    # "Entry": ["surr_view_camera_pub", "streo_camera_pub", "LiDAR_pub"],
    "S1": ["A"],
    "S2": ["C", "G", ],
}

# S1->A->B-> O1
# S1->A->E->F -> O2
# S2->C->D->E->F -> O2
# S2->G-> O2
task_graph_ops = {   
    "A": ["B", "E"],
    "B": ["O1"],
    "C": ["D",],
    "D": ["E"], 
    "E": ["F"],
    "F": ["O2"],
    "G": ["O2"],
}

# Chain 1	S1	A	B		
# Chian 2	S2	C	D	E	F
# Chain 3		G			
					
# Slack distribution (ddl-ert)					
# Chain 1	10.0 	70.0 	50.0 		
# Chian 2	10.0 	40.0 	30.0 	35.0 	15.0 
# Chain 3		    120.0 			
					
# DDL assignment (ddl)					
# Chain 1	10.0 	70.0 	120.0 		
# Chian 2	10.0 	40.0 	70.0 	105.0 	120.0 
# Chain 3		    120.0 			

# case 0:					
# Typical latency (exp_lat)					
# Chain 1	0.0 	40.0 	33.3 	0.0 	0.0 
# Chian 2	0.0 	20.0 	20.0 	23.3 	10.0 
# Chain 3		    73.3 	0.0 	0.0 	0.0 

# Case 1					
# Chain 1	30.0 	42.0 	33.3 	0.0 	0.0 
# Chian 2	20.0 	21.0 	20.0 	23.3 	10.0 
# Chain 3		80.0 	0.0 	0.0 	0.0 

# Case 2					
# Chain 1	30.0 	42.0 	50.0 	0.0 	0.0 
# Chian 2	20.0 	21.0 	25.0 	23.3 	10.0 
# Chain 3		100.0 	0.0 	0.0 	0.0 

# record the exp_lat of ops, and offset of srcs
case0 = {
    "S1": 0,
    "S2": 0,
    "A": 40,
    "B": 33.3,
    "C": 20,
    "D": 20,
    "E": 23.3,
    "F": 10,
    "G": 73.3,
}

case1 = {
    "S1": 30,
    "S2": 20,
    "A": 42,
    "B": 33.3,
    "C": 21.0,
    "D": 20.0,
    "E": 23.3,
    "F": 10,
    "G": 80,
}
case2 = {
    "S1": 30,
    "S2": 20,
    "A": 42,
    "B": 50,
    "C": 21,
    "D": 25,
    "E": 23.3,
    "F": 10,
    "G": 100,
}

case = case2
src_attr={
    "S1": case["S1"], 
    "S2": case["S2"],
}

task_attr = {
    "A": {"ert": 0, "ddl": 70, "exp_lat": case["A"], "base_size": 20},
    "B": {"ert": 70, "ddl": 120, "exp_lat": case["B"], "base_size": 20},
    "C": {"ert": 0, "ddl": 40, "exp_lat": case["C"], "base_size": 10},
    "D": {"ert": 40, "ddl": 70, "exp_lat": case["D"], "base_size": 10},
    "E": {"ert": 70, "ddl": 105, "exp_lat": case["E"], "base_size": 10},
    "F": {"ert": 105, "ddl": 120, "exp_lat": case["F"], "base_size": 10},
    "G": {"ert": 0, "ddl": 120, "exp_lat": case["G"], "base_size": 10},
}

tot_core = 40 

from task.task_cfg import creat_logical_graph
import networkx as nx
G:nx.DiGraph = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
from collections import OrderedDict
from utils import core_distr
import math
import copy

event_t = []

# push src event to event_t
for src, offset in src_attr.items():
    event_t.append(offset)
event_t.sort()

n_pred_map =  {node:len(list(G.predecessors(node))) for node in G.nodes()}
ready_queue = {}
running_queue = {}
res_map = {}
pre_rsc_map = {}
swt_lat = 2
sys_state = "S"
sys_state_emu = ["S", "R"]


recoder = {"slack": {}, "res":{},"rem_load":{}}

pred_t = 0
def mark_finish(G:nx.DiGraph, n_pred_map, node):
    for succ in G.successors(node):
        n_pred_map[succ] -= 1
    G.remove_edges_from([(node, succ) for succ in G.successors(node)])
    G.remove_node(node)

curr_t = 0
while G.nodes():
    if curr_t != 0: 
        print(f"===Decision metadata at {curr_t}===")
        print(f"\tDuation:" + str(event_t[0] - curr_t))
        print(f"\tRunning_queue: {running_queue}")
        print(f"\tRes_map: {res_map}")
        print(f"\tSlack_map: {slack_map}")
    
    # status backup
    pred_rsc_map = copy.deepcopy(res_map)
    pred_t, curr_t = curr_t, event_t.pop(0)
    print(f"\nmove from {pred_t} to {curr_t}\n") 
    print(f"===At the beginning of {curr_t}===")
    
    # event checking 
    # check stimulate events
    for node in list(src_attr.keys()):
        if curr_t == src_attr[node]:
            n_pred_map.pop(node)
            mark_finish(G, n_pred_map, node)
            src_attr.pop(node)
            print(f"\tsrc {node} arrives at {curr_t}")             
                
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

    
    if sys_state != "R" and not new_ready_flag and set(res_map.keys()) == set(pre_rsc_map.keys()):
        if len(event_t) == 0 and len(n_pred_map) == 0:
            print(f"===All tasks are finished at {curr_t}===")
            break
        assert False, "a null event is detected, which should not happen"
    
    # calculate slack
    slack_map = {node:task_attr[node]['ddl'] - curr_t for node in list(running_queue.keys()) + list(ready_queue.keys())} 
    
    # calculate min_rsc requirement
    rsc_map_t = OrderedDict()
    score_dict = OrderedDict()
    constr_dict = OrderedDict()
    curr_aval_rsc = tot_core
    for node in sorted(slack_map, key=slack_map.get):
        slack = slack_map[node]
        if slack <= 0:
            req_rsc_size = curr_aval_rsc
        else:
            assert not (node in ready_queue and node in running_queue) 
            req_rsc_size = math.ceil((running_queue.get(node, 0) + ready_queue.get(node, 0))/slack)
            
        curr_aval_rsc -= req_rsc_size
        rsc_map_t[node] = req_rsc_size
        constr_dict[node] = "N/A"
        score_dict[node] = 1/slack_map[node] if slack_map[node] >0 else float("inf")
        slack_map.pop(node)
        if curr_aval_rsc <= 0:
            break
    
    # allocate resource
    if curr_aval_rsc > 0:
        # if there are still resources left, 
        # it means no late process is waiting for resources
        assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
        # also, there is no process waiting for resources in the ready queue
        assert len(slack_map) == 0
        core_distr(rsc_map_t, score_dict, curr_aval_rsc)

    # issue task: first endurance switching progress, before turely allocated resources
    if res_map != rsc_map_t:
        print(f"Reallocation is triggered at {curr_t}")
        for node, rem_t in list(running_queue.items()):
            if not node in rsc_map_t: 
                ready_queue[node] = running_queue.pop(node)
                            
        for node, rem_t in list(ready_queue.items()):
            if node in rsc_map_t: 
                running_queue[node] = ready_queue.pop(node)

        if sys_state != "R":
            print(f"\tEnter reallocation progress at {curr_t}")
            sys_state = "R"
            res_map.clear()
            res_map.update(dict.fromkeys(rsc_map_t, 0))
            duation = swt_lat
        else: 
            print(f"\texist reallocation state at {curr_t}")   
            sys_state = "S"
            res_map = rsc_map_t

            # calculate druation to next event
            # running tasks which will first finish
            duation = math.ceil(min([running_queue[pid]/res_map[pid] for pid in running_queue]))

        next_event = curr_t + duation 
        if not event_t or event_t[0] > next_event:
            event_t.insert(0, next_event)
    
        

    
    