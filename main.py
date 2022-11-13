import math
from task import Task
from collections import OrderedDict

SIMULATION_TIME = 35

# collect the task information by profiling
from task_define import *


# give a 1st round of scheduling of a initial scheduling table
# take all task out of terminated satate
# maintain a event queue
# 1. initiate the deadline and release time of each task 

# req_list = []
# deadline_list = []
# forcasted_comp_list = []
# for task in task_list:
#     req_list.append(task.release_time)
#     deadline_list.append(task.deadline)

req_dict = OrderedDict()
deadline_dict = OrderedDict()
forcasted_comp_dict = OrderedDict()
for task in task_list:
    if task.release_time in req_dict:
        req_dict[task.release_time].append(task)
    else:
        req_dict[task.release_time] = [task]
    
    if task.deadline in deadline_dict:
        deadline_dict[task.deadline].append(task)
    else:
        deadline_dict[task.deadline] = [task]

def sort_task(task_list: List[Task], key_fun: Callable[[Task],Tuple]) -> List[Task]:
    sorted_task_list = sorted(task_list, key=key_fun)
    return sorted_task_list

verbose = True
# sort the task list by deadline
schedule_queue = sort_task(task_list, lambda task: (task.deadline, task.release_time, ))
# event_list = sorted(list(set(list(req_dict.keys()) + list(deadline_dict.keys()))))
event_list = sorted(list(set().union(req_dict.keys(), deadline_dict.keys())))
running_list = []

last_process = 0
his_t = curr_t = 0
while event_list:
    curr_t, his_t = event_list.pop(0), curr_t

    print("At Time: {}, ".format(curr_t))
    if verbose:
        if curr_t in req_dict:
            print("Expected release {}".format([t.task_name+" " for t in req_dict[curr_t]]))
        if curr_t in deadline_dict:
            print("Expected deadline {}".format([t.task_name+" " for t in deadline_dict[curr_t]]))

    # scan the task list: check finish
    if running_list:
        for task_id in running_list:
            _task = schedule_queue[task_id]
            # judge if task complete: completion_count += 1, cumulative_response_time += (time + 1.0 - a_time), 
            # update arrival time, deadline, clear current execution unit
            if (_task.cumulative_executed_time == _task.exp_comp_t):
                print("\tTASK COMPLETED " + str(_task.id))
                _task.completion_count += 1
                _task.cumulative_response_time += (float(curr_t) + _task.release_time)
                _task.release_time += _task.period
                _task.deadline += _task.period
                _task.cumulative_executed_time = 0.0
                _task.set_state("suspend")
                _task.release()

    # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
    for task_i in schedule_queue:
        if(task_i.deadline < curr_t):
            print("\tTASK {:d}:{:s} MISSED DEADLINE!!".format(task_i.id, task_i.task_name));
            task_i.missed_deadline_count += 1
            task_i.release_time += task_i.period
            task_i.deadline += task_i.period
            task_i.cumulative_executed_time = 0.0
            task_i.set_state("suspend")
            task_i.release()

    # this block may be triggered by either a changes of tasks or resource
    # 1. a pre-defined task release
    # 2. a pre-defined task deadline
    # 3. a task finished
    # scan the task list: check release
    issue_list = []
    preempt_list = []

    if curr_t in req_dict:
        for task in req_dict[curr_t]:
            # activate the task
            if task_i.state == "suspend":
                task_i.set_state("ready")
        req_dict.pop(curr_t, None)

    for i, task_i in enumerate(schedule_queue):
        # task arrive at event_time
        if(task_i.release_time <= curr_t):

            # query the resource map
            # if there is enough resources
            # return the action: wait, run, or preempt
            # return the time of the action target
            action, target = task.get_available()

            if action == "run":
                issue_list.append(task_i)
            if action == "wait":
                pass
            if action == "preempt":
                # remove the target task from issue_list
                issue_list = [t for t in issue_list if t not in target]
                preempt_list = [t for t in preempt_list if t not in running_list]
                issue_list.append(task_i)
    
    if issue_list:
        # preempt the task
        for last_process in preempt_list:
            _task = schedule_queue[last_process]
            print("\tPRE-EMPTING TASK " + str(_task.id))
            _task.preemption_count += 1
            _task.set_state("ready")
            _task.allocate()

        # issue tasks
        for task_id in issue_list:
            _task = schedule_queue[task_id]
            _task.set_state("running")
            _task.start_time = curr_t
            print("\tEXECUTING TASK {:d}:{:s}, expected finish @{:f}s ".format(
                _task.id, _task.task_name, 
                _task.release_time + _task.exp_comp_t)
            )
            running_list.append(task_id)
            issue_list.remove(task_id)
            # forcast a finish event
            forcasted_comp_dict[schedule_queue[task_i].release_time + schedule_queue[task_i].exp_comp_t] = task_id

    # TODO: Check whether the expected events are met

    # update new event into the event list
    deadline_dict.pop(curr_t, None)
    forcasted_comp_dict.pop(curr_t, None)
    event_list.remove(curr_t)
    event_list = sorted(list(set().union(event_list, req_dict.keys(), deadline_dict.keys(), forcasted_comp_dict.keys())))
assert False

# check the e2e latency
e2e_latency:float = 0.09
New_e2e_latency:float = 0.12
factor = New_e2e_latency/e2e_latency

if factor > 1:
    for task in task_list:
        if task.timing_flag == "deadline":
            task.deadline = math.ceil(task.deadline * factor)
            task.ERT = math.ceil(task.ERT * factor)
            task.exp_comp_t = math.ceil(task.exp_comp_t * factor)
            allocated_cores = task.count_allocated_resource()
            if task.flops*factor < allocated_cores*FLOPS_PER_CORE:
                num_new_cores = math.ceil((task.flops*factor - allocated_cores*FLOPS_PER_CORE)/FLOPS_PER_CORE)
                task.allocate(num_new_cores)


# case 1: e2e latency spec changes, from 0.1 to x
# step 1: re-distribute the deadline of each task
# factor = e2e_latency/0.1
# if factor > 1:
# Sweep the task list, 
# first check the task that is pre-assigned cores
# if the task's e2e latency is larger than the deadline, allocate more cores in proportion
# if resource is not enough, return the latency gap 
# collect the latency gap sum 
# redistribute the latency gap to the other tasks



# case 2:
# task performance degradation is detected
# Redistribute the latency distribution on the task chain, according the the 
# if the latency of the task is larger than the deadline, allocate more cores to it

# case 3:
# the required computation power has exceeded the resource overall capacity
# sacrifice the latency or throughput of the auxiliary tasks


def __main__():
    # create a task graph
    task_graph = TaskGraph(task_list)

    # create a resource map
    resource_map = ResourceMap(rsc_map)

    # create a scheduler
    scheduler = Scheduler(task_graph, resource_map)

    # schedule the task graph
    scheduler.schedule()

    # print the schedule result
    scheduler.print_schedule()
