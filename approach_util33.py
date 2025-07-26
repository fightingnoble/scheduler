import networkx as nx
from queue import Queue
from task.task_cfg import creat_logical_graph
from example.bm4 import swt_lat
from collections import defaultdict, OrderedDict
import math
from utils import core_distr

class MyGraph(nx.DiGraph):
    def __init__(self, srcs, ops, sinks, task_attr, src_attr, sink_attr=None):
        super(MyGraph, self).__init__()
        for src_n in srcs:
            self.add_node(src_n, type="src", **src_attr[src_n])
            for op_n in srcs[src_n]:
                self.add_node(op_n)
                self.add_edge(src_n, op_n, type="control")
        for op_n in ops:
            self.add_node(op_n, type="op", **task_attr[op_n])
            for sink_n in ops[op_n]:
                self.add_node(sink_n)
                self.add_edge(op_n, sink_n, type="data")
        for sink_n in sinks:
            if sink_attr is None:
                self.add_node(sink_n, type="sink")
            else:
                self.add_node(sink_n, type="sink", **sink_attr[sink_n])
            
        # self.n_pred_map trackes the non-ready, intermediate tasks in the graph
        # The value of n_pred_map is updated when tasks in graph are finished
        # A item is removed from the queue when it is moved to the ready queue
        # srcs are initially excluded from n_pred_map
        self.n_pred_map =  {node:len(list(self.predecessors(node))) for node in self.nodes() if node not in srcs}
        
        # self.srcs trackes the non-active srcs, 
        # which are removed when they are activated by external events
        self.srcs = [node for node in self.nodes() if node in srcs]

        self.sinks = [node for node in self.nodes() if node in sinks]
        self.ops = [node for node in self.nodes() if node in ops]
            

    def mark_finish(self, node):
        for succ in self.successors(node):
            self.n_pred_map[succ] -= 1
        self.remove_edges_from([(node, succ) for succ in self.successors(node)])
        self.remove_node(node)
        
    def mark_ready(self, node):
        self.n_pred_map.pop(node)


class Sen_p(object):
    """ Execution model of multi-sequential processor (MSSP), 
        where each processor only serves one task at a time,
        and scheduler will determine the which tasks are served, 
        and which processor is responsible for each task. 
    """
    def __init__(self, id, cap, base_pwr, G:MyGraph):
        self.id = id
        self.base_pwr = base_pwr
        self.G_ptr = G
        self.running = dict()
        self.ready = Queue(-1)
        self.cap = cap

    def update_run(self, pred_t, curr_t): 
        for node in list(self.running.keys()):
            rem_t  = self.running[node] - (curr_t - pred_t) * self.base_pwr
            if rem_t <= 0:
                self.G_ptr.mark_finish(node)
                self.running.pop(node)
                print(f"\tsrc {node} arrives at {curr_t}")             
            else:
                self.running[node] = rem_t

    def update_ready(self, curr_t):
        for node in list(self.G_ptr.srcs):
            if curr_t == self.G_ptr.nodes[node]["offset"]:
                load = self.G_ptr.nodes[node]["exp_comp_t"]
                self.G_ptr.srcs.remove(node)
                if load > 0:
                    self.ready.put((node, load))
                    print(f"\tsrc {node} is triggered at {curr_t}")
                else:
                    self.G_ptr.mark_finish(node)
                    print(f"\tsrc {node} arrives at {curr_t}")

    def sched(self, curr_t):
        # FCFS policy: serve a new task until last task is finished
        while not self.ready.empty() and len(self.running) < self.cap:
            node, rem_t = self.ready.get()
            self.running[node] = rem_t
            print(f"\tsen {self.id} starts task {node} at {self.G_ptr.nodes[node]['offset']}")

        # predict the next event in this queue
        duation_sen_p = float("inf")
        for node in self.running:
            duation_sen_p = min(duation_sen_p, self.running[node])
        if duation_sen_p == float("inf"):
            print(f"\tNo sensor event in future at {curr_t}")
        else:
            print(f"\tNext sensor event at {curr_t + duation_sen_p}")
        return duation_sen_p

class Acc_p(object):
    def __init__(self, id, cap, base_pwr, G:MyGraph):
        self.id = id
        self.cap = cap
        self.base_pwr = base_pwr
        self.G_ptr = G
        # running records the remaining load of task that is selected to be issued 
        # res_map records the actual resource allocation of each task
        # all keys in res_map should be in running
        self.running = dict()
        self.res_map = dict()
        self.ready = dict()
        self.sys_state = "S"
        self.alloc_map_curr = dict()
        self.sys_state_emu = ["S", "R"]
        self.slack_map = dict()

    def update_run(self, pred_t, curr_t): 
        new_complete_flag = False
        # calculate remaining workloads
        # and check finishing events
        for node in list(self.res_map.keys()):
            rem_t  = self.running[node] - (curr_t - pred_t) * self.res_map[node] * self.base_pwr
            if rem_t <= 0:
                if node != "R":
                    self.G_ptr.mark_finish(node)
                    new_complete_flag |= True
                self.running.pop(node)
                self.res_map.pop(node)
                print(f"\tTask {node} finishes at {curr_t}") 
            else:
                self.running[node] = rem_t
        return new_complete_flag

    def update_ready(self, curr_t):
        new_ready_list = []
        # put ready task to ready_queue
        for node, n_pred in list(self.G_ptr.n_pred_map.items()):
            if n_pred == 0:
                if node in self.G_ptr.sinks:
                    self.G_ptr.mark_finish(node)
                    print(f"\tsink {node} finish at {curr_t}")
                    self.G_ptr.mark_ready(node)
                elif node in self.G_ptr.ops:
                    self.ready[node] = self.G_ptr.nodes[node]["exp_comp_t"]* self.G_ptr.nodes[node]["base_size"]
                    new_ready_list.append(node)
                    print(f"\ttask {node} ready at {curr_t}")
                    self.G_ptr.mark_ready(node)
        return new_ready_list

    def sched(self, curr_t, new_comp, new_ready_list):
        """
        Allocation progress: 
            Check the event type: 
            If the event informs the completion of reallocation, 
            scheduler will directly use the cached allocation map. 
            Otherwise, scheduler will generate a new allocation map, 
            during which the allocation map in last round will be renamed as alloc_map_prev. 
            
            Switching progress: 
            Compare the alloc_map_prev with the generated alloc_map_curr, 
            if the allocation map changes, the reallocation progress will be triggered, 
            the res_map will be cleared, during which no task can be executed; 
            if the allocation map remains the same, the scheduler will continue to execute tasks. 

            case 1: complete reallocation progress, new configuration is issued, impose a finish event
            case 2: finish event happens, generate new allocation map, 
            enter reallocation progress, and impose a reallcation completion event
            case 3: finish event happens, do not change the allocation map, 
            continue to execute tasks, and impose a finish event

            Two types of tasks: 
            - "system task" that stalls the accelerator
            - "user task" that can be executed by the accelerator
        """         
        realloc = self.trigger_cond(new_comp, new_ready_list)
        self.alloc_map_curr = self.alloc_fn(curr_t, realloc)
        self.update_queue()
        self.res_map.clear()
        if realloc: 
            self.running["R"] = swt_lat
            self.res_map.update({"R": 1})
        else:
            self.res_map.update(self.alloc_map_curr) 
        # predict the next event in this queue
        duation_acc_p = math.ceil(min([self.running[pid]/self.res_map[pid] 
                                       for pid in self.res_map])) if self.res_map else float("inf")
        
        # state display 
        if self.sys_state == "R":
            print(f"\tEnter reallocation progress at {curr_t}")
            type_ = "reallocate"
        else:
            print(f"\texist reallocation state at {curr_t}")   
            type_ = "finish"        
        if duation_acc_p == float("inf"):
            print(f"\tNo accelerator event in future at {curr_t}")
        else:
            print(f"\tNext accelerator {type_} event at {curr_t + duation_acc_p}")
        return duation_acc_p, type_

    def trigger_cond(self, new_comp, new_ready_list):
        # compare the priorities of the new ready tasks with the running tasks
        # if the priority of the new ready tasks is smaller than the running tasks,
        # it means the reallcation progress is triggered. 
        # we use running as a mask to detect the reallcation progress.
        # If the minimum priority of the new ready tasks is smaller than the maximum priority of the running tasks,
        # it means the reallcation progress is triggered.

        cond1 = new_comp
        cond2 = len(self.running) == 0 and len(new_ready_list) > 0
        cond3 = len(self.running) > 0 and len(new_ready_list) > 0 \
            and min(new_ready_list, key=lambda x:self.G_ptr.nodes[x]['ddl']) < \
                max(self.running, key=lambda x:self.G_ptr.nodes[x]['ddl'])
            
        if cond1 or cond2 or cond3:
            realloc = True
            self.sys_state = "R"
        else:
            realloc = False
            self.sys_state = "S"
        return realloc

    def update_queue(self):
        # the tasks that alloted no resource is preempted, move them to ready queue
        for node, rem_t in list(self.running.items()):
            if not node in self.alloc_map_curr: 
                self.ready[node] = self.running.pop(node)
            
            # the tasks that alloted resource is executed, move them to running queue
        for node, rem_t in list(self.ready.items()):
            if node in self.alloc_map_curr: 
                self.running[node] = self.ready.pop(node)

    def alloc_fn(self, curr_t, realloc=True):
        # calculate slack 
        realloc_slack = 0 if not realloc else swt_lat
        self.slack_map = {node:self.G_ptr.nodes[node]['ddl'] - curr_t - realloc_slack
                        for node in list(self.running.keys()) + list(self.ready.keys())} 
        score = self.slack_map.copy()
        alloc_map_curr = {}
            
        # calculate min_rsc requirement
        score_dict = OrderedDict(); constr_dict = OrderedDict()
        curr_aval_rsc = self.cap
        for node in sorted(score, key=score.get):
            slack = score[node]
            if slack <= 0:
                print(f"\t{node} is timeout at {curr_t}")
                req_rsc_size = curr_aval_rsc
            else:
                assert not (node in self.ready and node in self.running) 
                req_rsc_size = math.ceil((self.running.get(node, 0) + self.ready.get(node, 0))/slack)
                if req_rsc_size > curr_aval_rsc:
                    print(f"\t{node} is hungry at {curr_t}: lack {req_rsc_size - curr_aval_rsc} tiles") 
                    req_rsc_size = curr_aval_rsc
                    
            curr_aval_rsc -= req_rsc_size
            alloc_map_curr[node] = req_rsc_size
            constr_dict[node] = "N/A"
            score_dict[node] = 1/score[node] if score[node] >0 else float("inf")
            score.pop(node)
            if curr_aval_rsc <= 0:
                break
            
            # allocate free resource
        if curr_aval_rsc > 0 and len(alloc_map_curr):
            print(f"\tMinimum resource requirement at {curr_t}: {alloc_map_curr}")
                # if there are still resources left, 
                # it means no late process is waiting for resources
            assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
                # also, there is no process waiting for resources in the ready queue
            assert len(score) == 0
            core_distr(alloc_map_curr, score_dict, curr_aval_rsc)
        return alloc_map_curr