from __future__ import annotations

from typing import Union, List, Dict, Iterator
from scheduling_table import SchedulingTableInt
from resource_agent import Resource_model_int
from task_agent import TaskInt
import numpy as np
import matplotlib.pyplot as plt
from golobal_var import *

from task_queue_agent import TaskQueue 
from task_agent import ProcessInt
import copy
from lru import LRUCache

'ID', 'Task (chain) names', 'Flops on path', 'Expected Latency (ms)', 'T release', 'Freq.', 'DDL', 'Cores/Req.', 
'Throuput factor (S)', 'Thread factor (S)', 'Min required cores', 'Timing_flag', 'Max required Cores', 'RDA./Req.', 'Resource Type', 'Pre-assigned', 'Priority'

# task_graph = {   
#     "Entry": ["Camera_pub", "LiDAR_pub"],
#     "Camera_pub": ["Pure_camera_path", "Stereo_feature_enc", "Traffic_light_detection"],
#     "LiDAR_pub": ["Lidar_based_3dDet"],
#     "Traffic_light_detection": [],
#     "ImageBB": ["MultiCameraFusion"],
#     "MultiCameraFusion": ["Pure_camera_path_head"],
#     "Pure_camera_path_head": ["Prediction"],
#     "Prediction": ["Planning"],
#     "Planning": ["Steering_speed"],
#     "Steering_speed": [],
#     "Stereo_feature_enc": ["Semantic_segm, Lane_drivable_area_det, Optical_Flow, Depth_estimation"],
#     "Semantic_segm": ["LiDAR_based_3dDet"],
#     "Lidar_based_3dDet": ["Prediction"],
#     "Lane_drivable_area_det": [],
#     "Optical_Flow": [],
#     "Depth_estimation": [],
#     : []
# }

__all__ = ['task_graph', 'affinity']


task_graph = {   
    "Traffic_light_detection": [],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["Pure_camera_path_head"],
    "Pure_camera_path_head": ["Prediction"],
    "Prediction": ["Planning"],
    "Planning": ["Steering_speed"],
    "Steering_speed": [],
    "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
    "Semantic_segm": ["LiDAR_based_3dDet"],
    "Lidar_based_3dDet": ["Prediction"],
    "Lane_drivable_area_det": [],
    "Optical_Flow": [],
    "Depth_estimation": [],
}

# affnity of a task is set to be a list that contains user-specified tasks, itself, it predecessors and its successors.
affinity = {
    "Traffic_light_detection": [],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["ImageBB", "Pure_camera_path_head"],
    "Pure_camera_path_head": ["MultiCameraFusion", "Prediction"],
    "Prediction": ["Pure_camera_path_head", "Lidar_based_3dDet", "Planning"],
    "Planning": ["Prediction", "Steering_speed"],
    "Steering_speed": ["Planning"],
    "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
    "Semantic_segm": ["Stereo_feature_enc", "LiDAR_based_3dDet"],
    "Lidar_based_3dDet": ["Semantic_segm", "Prediction"],
    "Lane_drivable_area_det": ["Optical_Flow", "Depth_estimation"],
    "Optical_Flow": ["Lane_drivable_area_det", "Depth_estimation"],
    "Depth_estimation": ["Lane_drivable_area_det", "Optical_Flow"],
}
# post-processing
# For the task that is pre-assigned with the resource, the affinity is set to be itself

def vis_task_static_timeline(task_list, show=False, save=False, save_path="task_static_timeline.pdf", **kwargs):
    sim_time = 0.2
    vertical_grid_size = 0.4
    time_grid_size = 0.004

    # build event list
    req_list = []
    ddl_list = []
    finish_list = []

    for task in task_list:
        req_list.append(task.get_release_event(sim_time))
        ddl_list.append(task.get_deadline_event(sim_time))
        finish_list.append(task.get_finish_event(sim_time))

    # print(req_list, ddl_list)


    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    # plot timeline and task name 
    # and select color for the task automatically
    horizen_grid = set()
    fig, ax = plt.subplots(figsize=(50, 10))
    vertical_offset = 0
    for i in range(len(task_list)):
        for s, e in zip(req_list[i], finish_list[i]):
            if e > sim_time:
                continue
            print("{}:{}-{}".format(task_list[i].name, s, e))
            horizen_grid.add(s)
            horizen_grid.add(e)
            ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s,
                      xmax=e, lw=2, color=mcolors.XKCD_COLORS[colors[i]]
                      )  # label=task_list[i].name)
            ax.text(# sim_time, vertical_offset*vertical_grid_size,
                    s, vertical_offset*vertical_grid_size+0.001,
                    task_list[i].name, fontsize=7)
        vertical_offset += 1
        # s = next(req_list[i])
        # e = next(ddl_list[i])
        # print("{}:{}-{}".format(task_list[i].name, s, e))
        # ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s, xmax=e, lw=2,)# label=task_list[i].name)
        # ax.text(sim_time, vertical_offset*vertical_grid_size, task_list[i].name, fontsize=7)
        # vertical_offset += 1

    # np.arange(0, sim_time+time_grid_size, time_grid_size)
    X, Y = np.meshgrid(np.array(list(horizen_grid)), np.arange(
        0, (vertical_offset+1)*vertical_grid_size, vertical_grid_size))
    ax.set(xlim=(0, 0.208), xticks=np.arange(0, 0.203, time_grid_size),)
    ax.plot(X, Y, 'k', lw=0.5, alpha=0.5)
    if show:
        plt.show()
    if "format" not in kwargs and save_path.split(".")[-1] == "pdf" and save:
        kwargs["format"] = "pdf"
    plt.savefig(save_path, **kwargs)

def load_taskint(verbose: bool = False, plot:bool = False) -> Dict[str, TaskInt]:
    import pandas as pd
    df = pd.read_csv("profiling.csv", sep=",", index_col=0) 
    if verbose:
        print(df)
    from task_agent import TaskInt
    task_dict = {}
    task_id = 0
    # print(task_attr_dict)

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    # plot timeline and task name 
    # and select color for the task automatically
    horizen_grid = set()
    fig, ax = plt.subplots(figsize=(50, 10))
    vertical_offset = 0
    sim_time = 0.2
    vertical_grid_size = 0.4
    time_grid_size = 0.004

    for task_n in df.T:
        # print(task_n)
        task_attr = df.loc[task_n].to_dict()
        task_attr["Timing_flag"] = "deadline" if task_attr["Timing_flag"]=="DDL" else "realtime"
        task_attr["Resource Type"] = "stationary" if task_attr["Resource Type"]=="S" else "moveable"
        task_attr["Pre-assigned"] = False if task_attr["Pre-assigned"]=="N" else True
        for i in range(task_attr["Throuput factor (S)"]):
            T = task_attr["Throuput factor (S)"]/task_attr["Freq."]
            phase = i/task_attr["Freq."]

            task = TaskInt(
                task_name=task_n+"_"+str(i), task_id=task_id, timing_flag=task_attr["Timing_flag"], 
                ERT=task_attr["T release"]/1000, ddl=(task_attr['DDL']-task_attr["T release"])/1000, period=T, 
                exp_comp_t=task_attr['Expected Latency (ms)']/1000, i_offset=phase, jitter_max=0,
                flops=task_attr["Flops on path"]/1e3, task_flag=task_attr["Resource Type"], 
                pre_assigned_resource_flag=task_attr["Pre-assigned"]>0, 
                RDA_size=task_attr['RDA./Req.'], main_size=task_attr['Cores/Req.'], seq_cpu_time=task_attr["Flops on path"]/1e3
            )
            task.freq = task_attr["Freq."]
            if plot:
                s = task.get_release_time()
                e = task.get_deadline_time()
                horizen_grid.add(s)
                horizen_grid.add(e)
                ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s,
                        xmax=e, lw=2, color=mcolors.XKCD_COLORS[colors[vertical_offset]]
                        )  # label=task_list[i].name)
                ax.text(s, vertical_offset*vertical_grid_size+0.001,
                        task.name, fontsize=7)
                vertical_offset+=1


            task.required_resource_size = task_attr['Cores/Req.']
            # print(str(task))
            task_id += 1
            task_dict.update({task.name: task})
        if plot:
            save_path="task_static_timeline.pdf"
            # np.arange(0, sim_time+time_grid_size, time_grid_size)
            X, Y = np.meshgrid(np.array(list(horizen_grid)), np.arange(
                0, (vertical_offset+1)*vertical_grid_size, vertical_grid_size))
            t_max = max(horizen_grid)+time_grid_size
            ax.set(xlim=(0, t_max), xticks=np.arange(0, t_max, time_grid_size),)
            ax.plot(X, Y, 'k', lw=0.5, alpha=0.5)
            plt.savefig(save_path, format="pdf")

    return task_dict

def push_task_into_scheduling_table(tasks: Union[List[TaskInt], Dict[str, TaskInt]], SchedTab: SchedulingTableInt, 
                                    quantumSize,
                                    timestep, event_range, sim_range,  verbose=False):

    # for task in tasks:
    #     for i in range(int(sim_len//task.period)):
    #         import copy
    #         task = copy.deepcopy(task)
    #         task.i_offset = i * task.period
    #         task.release_time = task.ERT + task.i_offset
    #         task.deadline = task.ddl + task.i_offset
    #         wait_list.append(task)
    # "running": 0,
    # "terminated": 3,
    # "suspend": 2,
    # "runnable": 1,
    # "throttled": 4,

    if isinstance(tasks, list):
        task_list = tasks
    elif isinstance(tasks, dict):
        task_list = list(tasks.values())

    from task_queue_agent import TaskQueue 
    import copy

    task_define_bk = copy.deepcopy(task_list) # used for detect the task changes

    # init the wait queue 
    # add1216: distinguish the task defined by user and the process in the task queue
    # generate the a serial of ideal task instances
    init_p_list = []
    pid = 0
    for task in task_list: 
        for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
            p = task.make_process(r, d, pid)
            pid += 1
            init_p_list.append(p)
    
    # monitor the wake up time: (accending)
    # activate by the new period or the arrival of the blocked io data
    # TODO: add a queue update logic
    wait_queue:TaskQueue = TaskQueue(init_p_list, sort_f=lambda x: x.release_time, decending=False)
    # monitor the deadline: (accending)
    ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, decending=False)
    # monitor the deadline for pre-emption: (decending)
    running_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline)
    
    expired_list:List[TaskInt] = []
    issue_list:List[TaskInt] = []
    completed_list:List[TaskInt] = []
    miss_list:List[TaskInt] = []
    rsc_recoder = {}

    for n_slot in range(SchedTab.scheduling_table.shape[0]):
        curr_t = n_slot * timestep

        # enqueue the process that is released in this slot
        lt = []
        for _p in wait_queue:
            if _p.release_time <= n_slot * timestep:
                lt.append(_p)
        for _p in lt:
            _p.pid = pid
            print("TASK {:d}:{:s}({:d}) RELEASEED AT {}!!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
            ready_queue.put(_p)
            wait_queue.remove(_p)
            pid += 1
        lt.clear()

        # ========================================================
        # scan the task list: check finish
        if len(running_queue):
            for _p in running_queue:
                # judge if task complete: completion_count += 1, cumulative_response_time += (time + 1.0 - a_time), 
                # update arrival time, deadline, clear current execution unit
                if (_p.cumulative_executed_time >= _p.exp_comp_t):
                    completed_list.append(_p)
                    _p.set_state("suspend")
        
        if completed_list:
            for _p in completed_list:
                # release the resource and move to the wait list
                SchedTab.release(_p, *rsc_recoder[_p.pid], verbose)
                rsc_recoder.pop(_p.pid)
                running_queue.remove(_p)
                wait_queue.put(_p)
                print("TASK {:d}:{:s}({:d}) COMPLETED!!".format(_p.task.id, _p.task.name, _p.pid))
                # update statistics
                # TODO: add lock 
                _p.task.completion_count += 1
                _p.task.cumulative_response_time += (curr_t - _p.release_time)
                # _p.task.release_time += _p.period
                # _p.task.deadline += _p.period
                _p.task.cumulative_executed_time = 0.0    

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            completed_list.clear()
        # ========================================================

        # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
        for _p in (ready_queue.queue + running_queue.queue):
            if(_p.deadline < curr_t):
                miss_list.append(_p)
                _p.set_state("suspend")
        
        if miss_list:
            for _p in miss_list:
                # release the resource and move to the wait list
                if _p in ready_queue.queue:
                    ready_queue.remove(_p)
                elif _p in running_queue.queue:
                    SchedTab.release(_p, *rsc_recoder[_p.pid], verbose)
                    rsc_recoder.pop(_p.pid)
                    running_queue.remove(_p)
                print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid));
                _p.task.missed_deadline_count += 1
                _p.task.release_time += _p.period
                _p.task.deadline += _p.period
                _p.task.cumulative_executed_time = 0.0
                wait_queue.put(_p)
            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            miss_list.clear()

        # convert the cpp to python code

        # if (CPU[i]->running != NULL && CPU[i]->running->currentburst != 0 && CPU[i]->running->currentburst % quantumSize == 0)
        # { //judge preemption on the running that have not completed but completed a Time Unit (control the pre-emption grain)
        #     if (tmp != NULL)
        #     { // next process is not NULL
        #         if ((CPU[i]->running->prio) > (tmp->prio))
        #         { // next process has lower priority
        #             CPU[i]->idle = 0;
        #             CPU[i]->running->currentburst = 0;
        #             DEBUG printf("Running: PID %d, priority %d.\n", CPU[i]->running->pid, CPU[i]->running->prio);
        #             DEBUG printf("Next   : PID %d, priority %d.\n", tmp->pid, tmp->prio);
        #             DEBUG printf("@@@----> Keep PID %d running!\n\n", CPU[i]->running->pid);
        #         }

        #         if ((CPU[i]->running->prio) <= (tmp->prio))
        #         { // running to ready, pre-empt the running and set CPU as idel and clear the currentburst
        #             Enqueue(CPU[i]->running, readyQueue);
        #             DEBUG printf("Running: PID %d, priority %d.\n", CPU[i]->running->pid, CPU[i]->running->prio);
        #             DEBUG printf("Next   : PID %d, priority %d.\n", tmp->pid, tmp->prio);
        #             DEBUG printf("###====> PID %d is going to run!\n\n", tmp->pid);
        #             CPU[i]->idle = 1;
        #             CPU[i]->running->currentburst = 0;
        #             CPU[i]->running = NULL;
        #         }
        #     }
        # }
        
        # judge if preemption: if the running task has not completed but completed a Time Unit (control the pre-emption grain) 
        # for task in running_list:
        #     if task.currentburst != 0 and task.currentburst % quantumSize == 0:
        #         pass


        # =================================================
        # push the ready task into the idle slot
        for _p in ready_queue:
            req_rsc_size = _p.required_resource_size
            # release time round up: task should not be released earlier than the release time
            time_slot_s = int((_p.release_time)//timestep+0.5)
            # deadline round down: task should not be finised later than the deadline
            time_slot_e = int((_p.deadline)//timestep) 
            # expected slot number
            expected_slot_num = int((_p.exp_comp_t)//timestep+0.5)
            # TODO: here the bug
            state, time_slot_s, alloc_size, curr_slot = SchedTab.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose) 

            if state:
                issue_list.append(_p)
                rsc_recoder[_p.pid] = [time_slot_s, alloc_size, curr_slot]

        # update the running task
        # TODO:bug here we should not use tuple to store the task
        if len(issue_list):
            for _p in issue_list:
                print("TASK {:d}:{:s}({:d}) IS ISSUED!!".format(_p.task.id, _p.task.name, _p.pid));
                running_queue.put(_p)
                ready_queue.remove(_p)
            issue_list.clear()
            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
        # =================================================
        
        # update the running task
        for _p in running_queue:
            _p.totburst += _p.required_resource_size
            _p.burst += _p.required_resource_size
            _p.currentburst += _p.required_resource_size
            _p.cumulative_executed_time += timestep

def push_task_into_scheduling_table_cyclic_preemption_disable(tasks: Union[List[TaskInt], Dict[str, TaskInt]], #SchedTab: SchedulingTableInt, 
                                    total_cores:int, quantumSize, 
                                    timestep, hyper_p, n_p=1, verbose=False, *, animation=False, warmup=False, drain=False,):
    """
    1. generate all the task instances stasticly
    2. push the ready task into the ready queue
    3. push the ready task into the idle slot in the order of the deadline
    4. update the running task
    """

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(sim_range/sim_step)
    tab_spatial_size = total_cores

    SchedTab = SchedulingTableInt(total_cores, int(sim_range/sim_step),)

    if isinstance(tasks, list):
        task_list = tasks
    elif isinstance(tasks, dict):
        task_list = list(tasks.values())

    task_define_bk = copy.deepcopy(task_list) # used for detect the task changes

    # init the wait queue 
    # add1216: distinguish the task defined by user and the process in the task queue
    # generate the a serial of ideal task instances
    init_p_list = []
    pid = 0
    for task in task_list: 
        # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
        r = task.get_release_time()
        d = task.get_deadline_time()
        p = task.make_process(r, d, pid)
        pid += 1
        init_p_list.append(p)
        if verbose:
            print("TASK {:d}:{:s}({:d}), is expected to finish {}T OPs in {:f}-{:f} !!".format(
                p.task.id, p.task.name, p.pid, p.totcpu, p.release_time, p.deadline))
    pid_max = pid
    
    # monitor the wake up time: (accending)
    # activate by the new period or the arrival of the blocked io data
    # TODO: add a queue update logic
    wait_queue:TaskQueue = TaskQueue(init_p_list, sort_f=lambda x: x.release_time, decending=False)
    # monitor the deadline: (accending)
    ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, decending=False)
    # monitor the deadline for pre-emption: (decending)
    running_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline)
    
    expired_list:List[ProcessInt] = []
    issue_list:List[ProcessInt] = []
    completed_list:List[ProcessInt] = []
    miss_list:List[ProcessInt] = []
    rsc_recoder = {}
    curr_cfg:Resource_model_int
    
    if animation:
        # for animation generation
        frame_list = []
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        plot_window = tab_spatial_size

    for n_slot in range(tab_temp_size):
        curr_t = n_slot * timestep

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")
        
        # enqueue the process that is released in this slot
        lt = []
        for _p in wait_queue:
            if _p.release_time <= n_slot * timestep and _p.release_time < event_range:
                lt.append(_p)
        for _p in lt:
            print("TASK {:d}:{:s}({:d}) RELEASEED AT {}!!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
            ready_queue.put(_p)
            wait_queue.remove(_p)
        lt.clear()

        # ========================================================
        # scan the task list: check finish
        if len(running_queue):
            for _p in running_queue:
                # judge if task complete: completion_count += 1, cumulative_response_time += (time + 1.0 - a_time), 
                # update arrival time, deadline, clear current execution unit
                if (_p.totburst >= _p.totcpu):
                    completed_list.append(_p)
                    _p.set_state("suspend")
        
        if completed_list:
            for _p in completed_list:
                # release the resource and move to the wait list
                SchedTab.release(_p, *rsc_recoder[_p.pid], verbose=False)
                print("TASK {:d}:{:s}({:d}) COMPLETED!!".format(_p.task.id, _p.task.name, _p.pid))
                # update statistics
                # TODO: add lock 
                _p.task.completion_count += 1
                _p.task.cumulative_response_time += (curr_t - _p.release_time)
                
                _p.release_time += _p.task.period
                _p.deadline += _p.task.period

                _p.end_time = curr_t * timestep
                _p.currentburst = 0
                _p.burst = 0
                _p.totburst = 0
                _p.remburst = _p.task.flops
                _p.cumulative_executed_time = 0
                _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
                rsc_recoder.pop(_p.pid)
                running_queue.remove(_p)
                wait_queue.put(_p)

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            completed_list.clear()
        # ========================================================

        # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
        for _p in (ready_queue.queue + running_queue.queue):
            if(_p.deadline < curr_t):
                miss_list.append(_p)
                _p.set_state("suspend")
        
        if miss_list:
            print("\n".join(["task {}({}) released @ {} start @ {}:\n".format(p.task.name, p.pid, p.release_time, p.start_time) +\
                   "is expected to finish {}T OPs before {} is expired, ".format(p.totcpu, p.deadline) +\
                     "but only executed {}T OPs in {}s time !!".format(p.totburst, p.cumulative_executed_time) for p in miss_list]))
            for _p in miss_list:
                # release the resource and move to the wait list
                if _p in ready_queue.queue:
                    ready_queue.remove(_p)
                elif _p in running_queue.queue:
                    SchedTab.release(_p, *rsc_recoder[_p.pid], verbose)
                    rsc_recoder.pop(_p.pid)
                    running_queue.remove(_p)
                print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid));
                _p.task.missed_deadline_count += 1
                _p.release_time += _p.task.period
                _p.deadline += _p.task.period

                _p.burst = 0
                _p.totburst = 0
                _p.remburst += _p.task.flops
                _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
                _p.cumulative_executed_time = 0
                _p.currentburst = 0

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            miss_list.clear()

        # convert the cpp to python code

        # if (CPU[i]->running != NULL && CPU[i]->running->currentburst != 0 && CPU[i]->running->currentburst % quantumSize == 0)
        # { //judge preemption on the running that have not completed but completed a Time Unit (control the pre-emption grain)
        #     if (tmp != NULL)
        #     { // next process is not NULL
        #         if ((CPU[i]->running->prio) > (tmp->prio))
        #         { // next process has lower priority
        #             CPU[i]->idle = 0;
        #             CPU[i]->running->currentburst = 0;
        #             DEBUG printf("Running: PID %d, priority %d.\n", CPU[i]->running->pid, CPU[i]->running->prio);
        #             DEBUG printf("Next   : PID %d, priority %d.\n", tmp->pid, tmp->prio);
        #             DEBUG printf("@@@----> Keep PID %d running!\n\n", CPU[i]->running->pid);
        #         }

        #         if ((CPU[i]->running->prio) <= (tmp->prio))
        #         { // running to ready, pre-empt the running and set CPU as idel and clear the currentburst
        #             Enqueue(CPU[i]->running, readyQueue);
        #             DEBUG printf("Running: PID %d, priority %d.\n", CPU[i]->running->pid, CPU[i]->running->prio);
        #             DEBUG printf("Next   : PID %d, priority %d.\n", tmp->pid, tmp->prio);
        #             DEBUG printf("###====> PID %d is going to run!\n\n", tmp->pid);
        #             CPU[i]->idle = 1;
        #             CPU[i]->running->currentburst = 0;
        #             CPU[i]->running = NULL;
        #         }
        #     }
        # }
        
        # judge if preemption: if the running task has not completed but completed a Time Unit (control the pre-emption grain) 
        # for task in running_list:
        #     if task.currentburst != 0 and task.currentburst % quantumSize == 0:
        #         pass


        # =================================================
        # push the ready task into the idle slot
        for _p in ready_queue:
            # release time round up: task should not be released earlier than the release time
            time_slot_s = int(np.ceil(_p.release_time/timestep))
            if time_slot_s < n_slot:
                time_slot_s = n_slot
            # deadline round down: task should not be finised later than the deadline
            time_slot_e = int(_p.deadline//timestep)
            req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))

            # expected slot number
            expected_slot_num = int(np.ceil(_p.exp_comp_t/timestep))
            # TODO: Add the logic to ensure the task have allocated enough resource, otherwise, 
            #         we should not issue the task or compensate the resource latter. 
            state, alloc_slot_s, alloc_size, allo_slot = SchedTab.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False) 
            total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
            total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
            if total_FLOPS_alloc < _p.remburst:
                Warning("The allocated FLOPS is not enough for the task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))

            print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
            print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
            if state:
                if not (isinstance(alloc_slot_s, list) and isinstance(alloc_size, list) and isinstance(allo_slot, list)):
                    print(f"\tgot {total_alloc_unit:d} ({alloc_size:d} cores x {allo_slot:d} slots @ {alloc_slot_s:d})\n")
                elif len(alloc_slot_s) == len(alloc_size) == len(allo_slot) == 1:
                    print(f"\tgot {total_alloc_unit:d} ({alloc_size[0]:d} cores x {allo_slot[0]:d} slots @ {alloc_slot_s[0]:d})\n")
                else:
                    alloc_slot_s_str = (r"{},"*len(alloc_slot_s)).format(*alloc_slot_s)
                    alloc_size_str = (r"{},"*len(alloc_size)).format(*alloc_size)
                    allo_slot_str = (r"{},"*len(allo_slot)).format(*allo_slot)
                    print(f"\tgot {total_alloc_unit:d} ({alloc_size_str:s} cores x {allo_slot_str:s} slots @ {alloc_slot_s_str:s})\n")
            else:
                print("\t[{:s}]\n".format("FAILED" if not state else "SUCCESS"))

            if state:
                issue_list.append(_p)
                rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot]
            else:
                Warning("TASK {:d}:{:s}({:d}) IS DELAY ISSUED!!".format(_p.task.id, _p.task.name, _p.pid))

        # update the running task
        # TODO:bug here we should not use tuple to store the task
        if len(issue_list):
            for _p in issue_list:
                print("TASK {:d}:{:s}({:d}) IS ISSUED @ {}!!".format(_p.task.id, _p.task.name, _p.pid, n_slot*timestep))
                running_queue.put(_p)
                ready_queue.remove(_p)
                if _p.totburst == 0:
                    _p.start_time = n_slot*timestep
                _p.waitTime = 0
            issue_list.clear()
            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
        # =================================================
                
        curr_cfg = SchedTab.scheduling_table[n_slot]
        if curr_cfg.rsc_map:
            # update the running task
            _p_dict = {p.pid:p for p in running_queue}
            # _p_dict_n = {p.task.name:p for p in running_queue}
            # if "MultiCameraFusion_0" in _p_dict_n.keys():
            #     if not _p_dict_n["MultiCameraFusion_0"].pid in curr_cfg.rsc_map.keys():
            #         print("MultiCameraFusion_0 is not in the current configuration")
            for pid in curr_cfg.rsc_map.keys():
                _p = _p_dict[pid]
                _p.currentburst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                _p.burst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                _p.totburst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                _p.remburst -= curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                _p.cumulative_executed_time += timestep

        # SchedTab.step()
        if animation: 
            if n_slot % 8 == 0 and n_slot+plot_window < tab_temp_size:
                frm = ax.imshow(SchedTab.get_plot_frame(n_slot, n_slot+plot_window), animated=True, vmax=pid_max)
                frame_list.append([frm])
    if animation:
        ani = animation.ArtistAnimation(fig, frame_list, interval=20, blit=False,
                                    repeat_delay=1000)
        writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("movie.mp4", writer=writer)

    SchedTab.print_scheduling_table()


def new_bin(spatial_size:int, temporal_size:int, id:int = 0, name:str = "bin"):
    SchedTab = SchedulingTableInt(spatial_size, temporal_size, id=id, name=name)
    return SchedTab

def traverse_task_graph(task_list):
    pass

def push_task_into_bins(tasks: Union[List[TaskInt], Dict[str, TaskInt]], #SchedTab: SchedulingTableInt, 
                                    total_cores:int, quantumSize, 
                                    timestep, hyper_p, n_p=1, verbose=False, *, animation=False, warmup=False, drain=False,):

    """
    implement a naive 2d bin packing algorithm
    input: task_list, that is already arranged in the topological order
    output: a list of bins, each bin is a list of tasks
    """
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(sim_range/sim_step)
    tab_spatial_size = total_cores

    if isinstance(tasks, list):
        task_list = tasks
    elif isinstance(tasks, dict):
        task_list = list(tasks.values())

    task_define_bk = copy.deepcopy(task_list) # used for detect the task changes

    # init the wait queue 
    # add1216: distinguish the task defined by user and the process in the task queue
    # generate the a serial of ideal task instances
    init_p_list = []
    pid = 0
    for task in task_list: 
        # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
        r = task.get_release_time()
        d = task.get_deadline_time()
        p = task.make_process(r, d, pid)
        pid += 1
        init_p_list.append(p)
        if verbose:
            print("TASK {:d}:{:s}({:d}), is expected to finish {}T OPs in {:f}-{:f} !!".format(
                p.task.id, p.task.name, p.pid, p.totcpu, p.release_time, p.deadline))
    pid_idx = {_p.task.name:_p.pid for _p in init_p_list}
    pid_max = pid
    
    # monitor the wake up time: (accending)
    # activate by the new period or the arrival of the blocked io data
    # TODO: add a queue update logic
    wait_queue:TaskQueue = TaskQueue(init_p_list, sort_f=lambda x: x.release_time, decending=False)
    # monitor the deadline: (accending)
    ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, decending=False)
    # monitor the deadline for pre-emption: (decending)
    running_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline)
    
    expired_list:List[ProcessInt] = []
    issue_list:List[ProcessInt] = []
    completed_list:List[ProcessInt] = []
    miss_list:List[ProcessInt] = []
    preempt_list:List[ProcessInt] = []
    rsc_recoder = {}
    rsc_recoder_his = {}
    curr_cfg:Resource_model_int
 
    # try to push the task into the bins in the bin_list
    # if the task cannot be pushed into any bin, create a new bin
    # _new_bins = lambda id: new_bins(total_cores, int(sim_range/sim_step), id=id, name="bin"+str(id))
    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, tab_temp_size, id=id, name=name)

    def _next_bin_obj():
        # release time round up: task should not be released earlier than the release time
        # deadline round down: task should not be finised later than the deadline
    # aa = lambda _p: int(np.ceil(_p.task.flops/(int(np.ceil(_p.release_time/timestep))-int(_p.deadline//timestep))/timestep/FLOPS_PER_CORE))
    def get_core_size(_p):
        # release time round up: task should not be released earlier than the release time
        time_slot_s = int(np.ceil(_p.release_time/timestep))
        # deadline round down: task should not be finised later than the deadline
        time_slot_e = int(_p.deadline//timestep)
        req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
        return req_rsc_size

        p_list = [(get_core_size(_p), _p) for _p in init_p_list]
        print("&*(&*(^(*^(*&^*(^*(&")
        p_list.sort(key=lambda x: x[0], reverse=True)
        # TODO: bug here, name confilit
        yield from (_new_bin(bin_id,x[0],x[1].task.name) for bin_id, x in enumerate(p_list))

    init_bin_id = 0
    bin_id = init_bin_id
    iter_next_bin_obj = _next_bin_obj()
    bin_list:List[SchedulingTableInt] = [next(iter_next_bin_obj)]
    bin_name_list = [bin_list[0].name]
   
    if animation:
        # for animation generation
        frame_list = []
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        plot_window = tab_spatial_size

    for n_slot in range(tab_temp_size):
        curr_t = n_slot * timestep

        if (n_slot - 1) * timestep < event_range and n_slot * timestep >= event_range: 
            print("="*20, "DRAIN", "="*20, "\n")
        elif n_slot == 0 and warmup:
            print("="*20, "WARMUP", "="*20, "\n")
        elif (n_slot * timestep)//hyper_p > (n_slot-1)*timestep//hyper_p:
            print("="*20, "PERIOD {:d}".format(int((n_slot * timestep)//hyper_p)), "="*20, "\n")
        
        # enqueue the process that is released in this slot
        lt = []
        for _p in wait_queue:
            if _p.release_time <= n_slot * timestep and _p.release_time < event_range:
                lt.append(_p)
        for _p in lt:
            print("TASK {:d}:{:s}({:d}) RELEASEED AT {}!!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
            ready_queue.put(_p)
            wait_queue.remove(_p)
        lt.clear()

        # ========================================================
        # scan the task list: check finish
        if len(running_queue):
            for _p in running_queue:
                # judge if task complete: completion_count += 1, cumulative_response_time += (time + 1.0 - a_time), 
                # update arrival time, deadline, clear current execution unit
                if (_p.totburst >= _p.totcpu):
                    completed_list.append(_p)
                    _p.set_state("suspend")
        
        if completed_list:
            for _p in completed_list:
                # release the resource and move to the wait list
                alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[_p.pid]
                if isinstance(alloc_slot_s_t, list):
                    alloc_slot_s, alloc_size, allo_slot = [], [], []
                    for i in range(len(alloc_slot_s_t)):
                        if alloc_slot_s_t[i]+allo_slot_t[i] > n_slot: 
                            alloc_slot_s.append(alloc_slot_s_t[i] if alloc_slot_s_t[i] > n_slot else n_slot )
                            alloc_size.append(alloc_size_t[i] )
                            allo_slot.append(allo_slot_t[i] if alloc_slot_s_t[i] > n_slot else allo_slot_t[i]-(n_slot - alloc_slot_s_t[i]) )
                else:
                    alloc_slot_s = alloc_slot_s_t if alloc_slot_s_t > n_slot else n_slot
                    alloc_size = alloc_size_t
                    allo_slot = allo_slot_t if alloc_slot_s_t > n_slot else allo_slot_t-(n_slot - alloc_slot_s_t)
                
                _SchedTab = bin_list[bin_id_t]
                _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                print("TASK {:d}:{:s}({:d}) COMPLETED!!".format(_p.task.id, _p.task.name, _p.pid))
                # update statistics
                # TODO: add lock 
                _p.task.completion_count += 1
                _p.task.cumulative_response_time += (curr_t - _p.release_time)
                
                _p.release_time += _p.task.period
                _p.deadline += _p.task.period

                _p.end_time = curr_t * timestep
                _p.currentburst = 0
                _p.burst = 0
                _p.totburst = 0
                _p.remburst = _p.task.flops
                _p.cumulative_executed_time = 0
                _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
                rsc_recoder.pop(_p.pid)
                running_queue.remove(_p)
                wait_queue.put(_p)

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            completed_list.clear()
        # ========================================================

        # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
        for _p in (ready_queue.queue + running_queue.queue):
            if(_p.deadline < curr_t):
                miss_list.append(_p)
                _p.set_state("suspend")
        
        if miss_list:
            print("\n".join(["task {}({}) released @ {} start @ {}:\n".format(p.task.name, p.pid, p.release_time, p.start_time) +\
                   "is expected to finish {}T OPs before {} is expired, ".format(p.totcpu, p.deadline) +\
                     "but only executed {}T OPs in {}s time !!".format(p.totburst, p.cumulative_executed_time) for p in miss_list]))
            for _p in miss_list:
                # release the resource and move to the wait list
                if _p in ready_queue.queue:
                    ready_queue.remove(_p)
                elif _p in running_queue.queue:

                    alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[_p.pid]
                    if isinstance(alloc_slot_s_t, list):
                        alloc_slot_s, alloc_size, allo_slot = [], [], []
                        for i in range(len(alloc_slot_s_t)):
                            if alloc_slot_s_t[i]+allo_slot_t[i] > n_slot: 
                                alloc_slot_s.append(alloc_slot_s_t[i] if alloc_slot_s_t[i] > n_slot else n_slot )
                                alloc_size.append(alloc_size_t[i] )
                                allo_slot.append(allo_slot_t[i] if alloc_slot_s_t[i] > n_slot else allo_slot_t[i]-(n_slot - alloc_slot_s_t[i]) )
                    else:
                        alloc_slot_s = alloc_slot_s_t if alloc_slot_s_t > n_slot else n_slot
                        alloc_size = alloc_size_t
                        allo_slot = allo_slot_t if alloc_slot_s_t > n_slot else allo_slot_t-(n_slot - alloc_slot_s_t)
                    
                    _SchedTab = bin_list[bin_id_t]
                    _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                    rsc_recoder.pop(_p.pid)
                    running_queue.remove(_p)
                print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid));
                _p.task.missed_deadline_count += 1
                _p.release_time += _p.task.period
                _p.deadline += _p.task.period

                _p.burst = 0
                _p.totburst = 0
                _p.remburst += _p.task.flops
                _p.required_resource_size = np.ceil(_p.remburst/_p.exp_comp_t/FLOPS_PER_CORE)
                _p.cumulative_executed_time = 0
                _p.currentburst = 0

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            miss_list.clear()

        # convert the cpp to python code

        # if (CPU[i]->running != NULL && CPU[i]->running->currentburst != 0 && CPU[i]->running->currentburst % quantumSize == 0)
        # { //judge preemption on the running that have not completed but completed a Time Unit (control the pre-emption grain)
        #     if (tmp != NULL)
        #     { // next process is not NULL
        #         if ((CPU[i]->running->prio) > (tmp->prio))
        #         { // next process has lower priority
        #             CPU[i]->idle = 0;
        #             CPU[i]->running->currentburst = 0;
        #             DEBUG printf("Running: PID %d, priority %d.\n", CPU[i]->running->pid, CPU[i]->running->prio);
        #             DEBUG printf("Next   : PID %d, priority %d.\n", tmp->pid, tmp->prio);
        #             DEBUG printf("@@@----> Keep PID %d running!\n\n", CPU[i]->running->pid);
        #         }

        #         if ((CPU[i]->running->prio) <= (tmp->prio))
        #         { // running to ready, pre-empt the running and set CPU as idel and clear the currentburst
        #             Enqueue(CPU[i]->running, readyQueue);
        #             DEBUG printf("Running: PID %d, priority %d.\n", CPU[i]->running->pid, CPU[i]->running->prio);
        #             DEBUG printf("Next   : PID %d, priority %d.\n", tmp->pid, tmp->prio);
        #             DEBUG printf("###====> PID %d is going to run!\n\n", tmp->pid);
        #             CPU[i]->idle = 1;
        #             CPU[i]->running->currentburst = 0;
        #             CPU[i]->running = NULL;
        #         }
        #     }
        # }
        
        # judge if preemption: if the running task has not completed but completed a Time Unit (control the pre-emption grain) 
        # for task in running_list:
        #     if task.currentburst != 0 and task.currentburst % quantumSize == 0:
        #         pass


        # =================================================
        # push the ready task into the idle slot
        # input: 
        #   deadline (hard or expected), release time(real or expected)
        # output: 
        #   resource size, start time, end time
        # action: 
        #   try to allocate the enough resources for the task to finish before the deadline
        # strategy:
        #   timing constraint first: use just enough resources to finish the task before the deadline
        #   estimate the resource size according to the remaining operation and the relative deadline
        #   allow the task to execute even though the resource is not enough
        #   allow the lateness

        for _p in ready_queue:
            issue_process(_p, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)

        # update the running task
        if len(preempt_list):
            for _p in preempt_list:
                if _p in running_queue.queue:
                    running_queue.remove(_p)
                    ready_queue.put(_p)
            preempt_list.clear()
        # TODO:bug here we should not use tuple to store the task
        if len(issue_list):
            for _p in issue_list:
                print("TASK {:d}:{:s}({:d}) IS ISSUED @ {}!!".format(_p.task.id, _p.task.name, _p.pid, n_slot*timestep))
                running_queue.put(_p)
                ready_queue.remove(_p)
                if _p.totburst == 0:
                    _p.start_time = n_slot*timestep
                _p.waitTime = 0
            issue_list.clear()
            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            # if animation:
            #     frm_arr = []
            #     for _SchedTab in bin_list:
            #         frm_arr.append(_SchedTab.get_plot_frame(n_slot, n_slot+plot_window))
            #     frm_arr = np.concatenate(frm_arr, axis=0)
            #     frm = ax.imshow(frm_arr, animated=True, vmax=pid_max)
            #     frame_list.append([frm])

        # =================================================
        for _SchedTab in bin_list:
            curr_cfg = _SchedTab.scheduling_table[n_slot]
            if curr_cfg.rsc_map:
                # update the running task
                _p_dict = {p.pid:p for p in running_queue}
                # _p_dict_n = {p.task.name:p for p in running_queue}
                # if "MultiCameraFusion_0" in _p_dict_n.keys():
                #     if not _p_dict_n["MultiCameraFusion_0"].pid in curr_cfg.rsc_map.keys():
                #         print("MultiCameraFusion_0 is not in the current configuration")
                for pid in curr_cfg.rsc_map.keys():
                    _p = _p_dict[pid]
                    _p.currentburst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.burst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.totburst += curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.remburst -= curr_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
                    _p.cumulative_executed_time += timestep

        # SchedTab.step()
        if animation: 
            if n_slot % 8 == 0 and n_slot+plot_window < tab_temp_size:
                frm_arr = []
                for _SchedTab in bin_list:
                    frm_arr.append(_SchedTab.get_plot_frame(n_slot, n_slot+plot_window))
                frm_arr = np.concatenate(frm_arr, axis=0)
                frm = ax.imshow(frm_arr, animated=True, vmax=pid_max)
                frame_list.append([frm])
    if animation:
        ani = animation.ArtistAnimation(fig, frame_list, interval=20, blit=False,
                                    repeat_delay=1000)
        writer = animation.FFMpegWriter(fps=50, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("movie.mp4", writer=writer)

    print("=====================================\n")
    print("bin_pack_result:")
    print("=====================================\n")
    for _SchedTab in bin_list:
        bin_pack_result = _SchedTab.index_occupy_by_id()

        # sort the result by the start time
        # item[1] is alloc_slot_s_t, alloc_size_t, allo_slot_t
        # item[1][0] is alloc_slot_s_t
        sorted_task_pid = [k for k, v in sorted(bin_pack_result.items(), key=lambda item: item[1][0])]

        # replace the pid with the task name
        # and print the result
        print(f"bin: {_SchedTab.name}({_SchedTab.id})")
        for pid in list(sorted_task_pid):
            _result = bin_pack_result.pop(pid)
            bin_pack_result[init_p_list[pid].task.name] = _result
            print("task: {:s}({:d})".format(init_p_list[pid].task.name, pid))
            print("\tstart time: {:s}".format(", ".join([f"{x*timestep:.6f}" for x in _result[0]])))
            print("\talloc cores: {:s}".format(", ".join([f"{x:d}" for x in _result[1]])))
            print("\tused time: {:s}".format(", ".join([f"{x*timestep:.6f}" for x in _result[2]])))
        print("=====================================\n")
    
    from scheduling_table import get_task_layout_compact
    get_task_layout_compact(bin_list, init_p_list) 
    

def issue_process(_p:ProcessInt, n_slot:int, 
                affinity:Dict[str, List[str]], pid_idx:dict, init_p_list:List[ProcessInt], 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
                ready_queue:TaskQueue, running_queue:TaskQueue, 
                issue_list:List[ProcessInt], preempt_list:List[ProcessInt],
                iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], ):
    # release time round up: task should not be released earlier than the release time
    time_slot_s = int(np.ceil(_p.release_time/timestep))
    if time_slot_s < n_slot:
        time_slot_s = n_slot
    # deadline round down: task should not be finised later than the deadline
    time_slot_e = int(_p.deadline//timestep)
    req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))

    # expected slot number
    expected_slot_num = time_slot_e-time_slot_s # int(np.ceil(_p.exp_comp_t/timestep))
    # TODO: Add the logic to ensure the task have allocated enough resource, otherwise, 
    #         we should not issue the task or compensate the resource latter. 

    # try to push the task into the bins in the bin_list
    state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
    # TODO: arange the bin_list according to the affinity of the class
    # name parse
    thread_n = _p.task.name.split('_')[-1]
    p_name = _p.task.name

    # For the task that is pre-assigned with the resource, the affinity is set to be itself
    if p_name in bin_name_list:
        _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
        bin_id = bin_name_list.index(p_name)
        bin = bin_list[bin_id]
        state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
        total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
        total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
        # check if the task is allocated successfully
        # if fail collect the tasks in the target bin and preempt the tasks
        if state and (total_FLOPS_alloc >= _p.remburst):
            pass
        else: 
            # release the resource
            bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
            # reset the state
            state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
            occupation_candi_dict:Dict[int, List[int]] = bin.index_occupy_by_id(time_slot_s, time_slot_e)
            occupation_candi = list(occupation_candi_dict.keys())
            # decide which task to preempt
            # select the task with the latest deadline
            # TODO: evaluate more strategies
            occupation_candi.sort(key=lambda x: _p_index_by_pid[x].deadline, reverse=True)
            # note that here is an assumption that current bin is designed for the task, all the candidate tasks are preempted
            aval_flops = sum(bin.idx_free_by_slot(time_slot_s, time_slot_e, _p.pid)) * timestep * FLOPS_PER_CORE
            flops_2b_preempt = (time_slot_e - time_slot_s) * bin.scheduling_table[0].size * timestep * FLOPS_PER_CORE - aval_flops
            p_2b_preempt = []
            p_2b_realloc = []
            for pid in occupation_candi:
                # load allocation history
                _p_2b_preempt = _p_index_by_pid[pid]
                alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[pid]
                assert bin_id == bin_id_t
                
                # task is in the ready queue and issue list
                # judge if preemption: 
                #   task is runnning but has executed for an integer multiples of the quantum size (control the pre-emption grain)
                cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
                reach_preempt_grain = np.allclose(cum_exec_quantum, round(cum_exec_quantum), atol=1e-2)
                if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                    continue
                
                # calculate the interval intersection of the two tasks
                # [time_slot_s, time_slot_e]
                # [alloc_slot_s_t, alloc_slot_s_t + len(alloc_size_t)]
                # total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
                alloc_slot_s_t, alloc_size_t, allo_slot_t = occupation_candi_dict[pid]
                total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
                total_FLOPS_alloc_t = total_alloc_unit_t * timestep * FLOPS_PER_CORE
                flops_2b_preempt -= total_FLOPS_alloc_t

                # pop the task from the bin
                print(f"pop the task {_p_2b_preempt.task.id}:{_p_2b_preempt.task.name}({pid})from the bin {bin_id}")
                # release all the tasks in the p_2b_preempt
                bin.release(_p_2b_preempt, alloc_slot_s_t, alloc_size_t, allo_slot_t, verbose=False)
                # update the rsc_recoder
                rsc_recoder.pop(_p_2b_preempt.pid)
                
                if _p_2b_preempt.currentburst == 0: 
                    # task is in the ready queue and issue list
                    issue_list.remove(_p_2b_preempt)
                    # update the rsc_recoder_his
                    rsc_recoder_his[pid].withdraw()
                    # update the running_queue
                    if _p_2b_preempt in running_queue:
                        raise ValueError("A unexpected situation happens, task is not executed but in the running queue")
                        # maybe blocked by the I/O
                        # TODO: add the logic to handle the blocked task
                        # put into waiting list


                elif _p_2b_preempt.currentburst != 0 and reach_preempt_grain:
                    # task is in the running queue
                    # update the task status
                    _p_2b_preempt.task.preemption_count += 1
                    _p_2b_preempt.currentburst = 0 

                    # update the running_queue
                    preempt_list.append(_p_2b_preempt)
                    print(f"task {_p_2b_preempt.task.id}:{_p_2b_preempt.task.name}({pid}) is preempted and put into the ready queue")
                
                # mark for reallocation in current time slot
                p_2b_realloc.append(_p_2b_preempt)
                p_2b_preempt.append(pid)
                if flops_2b_preempt <= 1e-2*timestep*FLOPS_PER_CORE:
                    break

            # reallocate the current task
            state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
            total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
            total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
            # check if the task is allocated successfully
            # if fail collect the tasks in the target bin and preempt the tasks
            if state and (total_FLOPS_alloc >= _p.remburst):
                if bin.locker == _p.pid:
                    bin.release_lock(_p, time_slot_s, time_slot_e)
            else: 
                # In this condition, some tasks are not preempted successfully, and the current task is blocked by these tasks, 
                # the current task will be delayed to issue
                # release the resource
                bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
                # reset the state
                state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
                bin.add_lock(_p, time_slot_s, time_slot_e)
                Warning(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.task.pid}) is not allocated successfully after preemption in its own bin")
                # raise ValueError(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.task.pid}) is not allocated successfully after preemption in its own bin")
            
            # reallocate the tasks in the P_2b_realloc
            for _p_2b_realloc in p_2b_realloc:
                issue_process(_p_2b_realloc, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                    rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)

    else:
        # remove the thread number at the end of the name
        task_base_name = p_name.replace("_"+thread_n, "")
        thread_n = int(thread_n)
        affinity_tgt_n_list = affinity[task_base_name]        
        affinity_tgt_n_list = [n+'_'+str(thread_n) for n in affinity_tgt_n_list]
        # get affinity target id
        # TODO: bug here， key error when the affinity target is not in the pid_idx
        affinity_tgt_id_list = [pid_idx[n] for n in affinity_tgt_n_list if n in pid_idx]
        # find the target bin of the affinity target
        affinity_tgt_bin_id_list = [rsc_recoder_his[n].get_mru() for n in affinity_tgt_id_list if n in rsc_recoder_his] 
        affinity_search_bin_id_list = [n for n in range(len(bin_list)) if n not in affinity_tgt_bin_id_list] 
        # arrange the search targets in the order of the fitness of the size
        affinity_search_bin_id_list.sort(key=lambda x: abs(bin_list[x].scheduling_table[0].size - req_rsc_size))

        # search the bin in the affinity target bin list
        for bin_id in affinity_tgt_bin_id_list: 
            bin = bin_list[bin_id]
            state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
            total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
            total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
            if state and (total_FLOPS_alloc >= _p.remburst):
                break
            else:
                # release the resources if the task cannot be pushed into the bin
                # TODO: merge this function into the scheduling table class
                # TODO: distinguish the state of risking the lack of resources and the state of the task cannot be pushed into the bin
                bin.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                state = False
        
        if not state:
            # search the bin in the affinity search bin list
            for bin_id in affinity_search_bin_id_list: 
                bin = bin_list[bin_id]
                state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
                total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
                total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
                if state and (total_FLOPS_alloc >= _p.remburst):
                    break
                else:
                    # release the resources if the task cannot be pushed into the bin
                    # TODO: merge this function into the scheduling table class
                    # TODO: distinguish the state of risking the lack of resources and the state of the task cannot be pushed into the bin
                    bin.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                    state = False
                    
        # check the state; if the task cannot be pushed into any bin, create a new bin
        if not state:
            bin = next(iter_next_bin_obj)
            bin_id = bin.id
            state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False) 
            bin_list.append(bin)
            bin_name_list.append(bin.name)
            total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
            total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE

    # if the task is allowed to execute under insufficient resources
    if state and total_FLOPS_alloc < _p.remburst:
        Warning("The allocated FLOPS is not enough for the task {:d}:{:s}({:d})".format(_p.task.id, _p.task.name, _p.pid))
        
    print(f"TASK {_p.task.id:d}:{_p.task.name:s}({_p.pid:d}) tries to allocate\n")
    print(f"\t{req_rsc_size * expected_slot_num:d} ({req_rsc_size:d} cores x {expected_slot_num:d} slots) from {time_slot_s:d} to {time_slot_e:d}")
    if state:
        if not (isinstance(alloc_slot_s, list) and isinstance(alloc_size, list) and isinstance(allo_slot, list)):
            print(f"\tgot {total_alloc_unit:d} ({alloc_size:d} cores x {allo_slot:d} slots @ {alloc_slot_s:d}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
        elif len(alloc_slot_s) == len(alloc_size) == len(allo_slot) == 1:
            print(f"\tgot {total_alloc_unit:d} ({alloc_size[0]:d} cores x {allo_slot[0]:d} slots @ {alloc_slot_s[0]:d}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
        else:
            alloc_slot_s_str = (r"{},"*len(alloc_slot_s)).format(*alloc_slot_s)
            alloc_size_str = (r"{},"*len(alloc_size)).format(*alloc_size)
            allo_slot_str = (r"{},"*len(allo_slot)).format(*allo_slot)
            print(f"\tgot {total_alloc_unit:d} ({alloc_size_str:s} cores x {allo_slot_str:s} slots @ {alloc_slot_s_str:s}, Bin({bin_id}):{bin_name_list[bin_id]})\n")
    else:
        print("\t[{:s}]\n".format("FAILED" if not state else "SUCCESS"))

    if state:
        issue_list.append(_p)
        rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot, bin_id]
        if _p.pid in rsc_recoder_his:
            rsc_recoder_his[_p.pid].put(bin_id)
        else:
            rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)
    else:
        Warning("TASK {:d}:{:s}({:d}) IS DELAY ISSUED!!".format(_p.task.id, _p.task.name, _p.pid))
        
def build_task_graph_and_packing(verbose: bool = False, plot:bool = False) -> Dict[str, TaskInt]:
    pass


if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="task1", help="task name")
    parser.add_argument("--plot", action="store_true", help="plot the task timeline")
    args = parser.parse_args() 
    task_dict = load_taskint(args.verbose)
    if args.plot:
        vis_task_static_timeline(list(task_dict.values()), save=True, save_path="task_static_timeline_cyclic.pdf")
    import numpy as np 
    f_gcd = np.gcd.reduce([task_dict[task].freq for task in task_dict])
    f_max = max([task_dict[task].freq for task in task_dict])
    hyper_p = 1/f_gcd
    sim_step = min([task_dict[task].exp_comp_t for task in task_dict])/32
    # push_task_into_scheduling_table_cyclic_preemption_disable(task_dict, 256, sim_step*1, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)
    push_task_into_bins(task_dict, 256, sim_step*1, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)