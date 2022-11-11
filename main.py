import math
from task import Task

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

req_dict = {}
deadline_dict = {}
forcasted_comp_dict = {}
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
event_list = sorted(list(set(list(req_dict.keys()) + list(deadline_dict.keys()))))
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
    current_process = -1


    for i, task_i in enumerate(schedule_queue):
        # task arrive at event_time
        if(task_i.release_time <= curr_t):
            current_process = i;
            # TODO: check if task has enough amount of resource to be scheduled
            task.allocate
            # TODO: check if task can be temporarily insert 
            # TODO: check if preemptable

            break;

    # judge if pre-empt
    if ((current_process != last_process) and schedule_queue[last_process].cumulative_executed_time > 0.0):
        print("\tPRE-EMPTING TASK " + str(schedule_queue[last_process].id))
        schedule_queue[last_process].preemption_n += 1
    schedule_queue[current_process].start_time = curr_t
    print("\tEXECUTING TASK {:d}:{:s}, expected finish @{:f}s ".format(
        schedule_queue[current_process].id, schedule_queue[current_process].task_name, 
        schedule_queue[current_process].release_time + schedule_queue[current_process].exp_comp_t)
    )
    # forcast a finish event
    # forcasted_comp_dict.append(schedule_queue[current_process].release_time + schedule_queue[current_process].exp_comp_t)

    if (current_process > -1):
        schedule_queue[current_process].cumulative_executed_time += curr_t - his_t
        # judge if task complete: completion_count += 1, cumulative_response_time += (time + 1.0 - a_time), 
        # update arrival time, deadline, clear current execution unit
        if (schedule_queue[current_process].cumulative_executed_time == schedule_queue[current_process].exp_comp_t):
            
            print("\tTASK COMPLETED " + str(schedule_queue[current_process].id))
            schedule_queue[current_process].completion_count += 1
            schedule_queue[current_process].cumulative_response_time += (float(curr_t) + schedule_queue[current_process].release_time)
            schedule_queue[current_process].release_time += schedule_queue[current_process].period
            schedule_queue[current_process].deadline += schedule_queue[current_process].period
            schedule_queue[current_process].cumulative_executed_time = 0.0
            
    # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
    for task_i in schedule_queue:
        if(task_i.deadline < curr_t):
            print("\tTASK {:d}:{:s} MISSED DEADLINE!!".format(task_i.id, task_i.task_name));
            task_i.missed_deadline_count = task_i.missed_deadline_count + 1
            task_i.release_time += task_i.period
            task_i.deadline += task_i.period
            task_i.cumulative_executed_time = 0.0

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
