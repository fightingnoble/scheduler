from __future__ import annotations

from typing import Union, List, Dict, Tuple, Any, Optional
from scheduling_table import SchedulingTableInt
from task_agent import TaskInt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

task_graph = {   
    "Traffic_light_detection": [],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["Pure_camera_path_head"],
    "Pure_camera_path_head": ["Prediction"],
    "Prediction": ["Planning"],
    "Planning": ["Steering_speed"],
    "Steering_speed": [],
    "Stereo_feature_enc": ["Semantic_segm, Lane_drivable_area_det, Optical_Flow, Depth_estimation"],
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
    "Stereo_feature_enc": ["Semantic_segm, Lane_drivable_area_det, Optical_Flow, Depth_estimation"],
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
                RDA_size=task_attr['RDA./Req.'], main_size=task_attr['Cores/Req.']
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

    sim_len = SchedTab.scheduling_table.shape[0] * timestep
    from task_queue_agent import TaskQueue
    
    import copy
    task_define_bk = copy.deepcopy(task_list)
    # monitor the wake up time: (accending)
    # activate by the new period or the arrival of the blocked io data
    # TODO: distinguish the task defined by user and the process in the task queue
    # TODO: add a queue update logic
    wait_queue:TaskQueue = TaskQueue(copy.deepcopy(task_list), sort_f=lambda x: x.release_time, decending=False)
    # monitor the deadline: (accending)
    ready_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline, decending=False)
    # monitor the deadline for pre-emption: (decending)
    running_queue:TaskQueue = TaskQueue(sort_f=lambda x: x.deadline)
    
    expired_list:List[TaskInt] = []
    issue_list:List[TaskInt] = []
    completed_list:List[TaskInt] = []
    miss_list:List[TaskInt] = []
    rsc_recoder = {}
    pid = 0

    for n_slot in range(SchedTab.scheduling_table.shape[0]):
        curr_t = n_slot * timestep

        # enqueue the task that is released in this slot
        lt = []
        for task in wait_queue:
            if task.release_time <= n_slot * timestep and task.release_time <= event_range+timestep:
                lt.append(task)
        for task in lt:
            task.pid = pid
            print("TASK {:d}:{:s}({:d}) RELEASEED AT {}!!".format(task.id, task.name, task.pid, curr_t))
            ready_queue.put(task)
            wait_queue.remove(task)
            pid += 1
        lt.clear()

        # ========================================================
        # scan the task list: check finish
        if len(running_queue):
            for _task in running_queue:
                # judge if task complete: completion_count += 1, cumulative_response_time += (time + 1.0 - a_time), 
                # update arrival time, deadline, clear current execution unit
                if (_task.cumulative_executed_time >= _task.exp_comp_t):
                    completed_list.append(_task)
        
        if completed_list:
            for _task in completed_list:
                # release the resource and move to the wait list
                sche_tab.release(_task, *rsc_recoder[_task.pid], verbose)
                rsc_recoder.pop(_task.pid)
                running_queue.get(_task)
                wait_queue.put(_task)
                print("TASK {:d}:{:s}({:d}) COMPLETED!!".format(_task.id, _task.name, _task.pid))
                # update statistics
                _task.completion_count += 1
                _task.cumulative_response_time += (curr_t - _task.release_time)
                _task.release_time += _task.period
                _task.deadline += _task.period
                _task.cumulative_executed_time = 0.0
                _task.set_state("suspend")

            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
            completed_list.clear()
        # ========================================================

        # judge if deadline miss: deadline_misses += 1, update arrival time, deadline, clear current execution unit
        for task_i in (ready_queue.queue + running_queue.queue):
            if(task_i.deadline < curr_t):
                miss_list.append(task_i)
        
        if miss_list:
            for _task in miss_list:
                # release the resource and move to the wait list
                if _task in ready_queue.queue:
                    ready_queue.get(_task)
                elif _task in running_queue.queue:
                    sche_tab.release(_task, *rsc_recoder[_task.pid], verbose)
                    rsc_recoder.pop(_task.pid)
                    running_queue.get(_task)
                print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(task_i.id, task_i.name, task_i.pid));
                _task.missed_deadline_count += 1
                _task.release_time += task_i.period
                _task.deadline += task_i.period
                _task.cumulative_executed_time = 0.0
                _task.set_state("suspend")
                wait_queue.put(task_i)
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
        for task_i in ready_queue:
            req_rsc_size = task_i.required_resource_size
            # release time round up: task should not be released earlier than the release time
            time_slot_s = int((task_i.release_time)//timestep+0.5)
            # deadline round down: task should not be finised later than the deadline
            time_slot_e = int((task_i.deadline)//timestep) 
            # expected slot number
            expected_slot_num = int((task_i.exp_comp_t)//timestep+0.5)
            # TODO: here the bug
            state, time_slot_s, alloc_size, curr_slot = SchedTab.insert_task(task_i, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose) 
            if state:
                issue_list.append(task_i)
                rsc_recoder[task_i.pid] = [time_slot_s, alloc_size, curr_slot]

        # update the running task
        # TODO:bug here we should not use tuple to store the task
        if len(issue_list):
            for task_i in issue_list:
                print("TASK {:d}:{:s}({:d}) IS ISSUED!!".format(task_i.id, task_i.name, task_i.pid));
                running_queue.put(task_i)
                ready_queue.remove(task_i)
            issue_list.clear()
            # print("Scheduling Table:")
            # print(SchedTab.print_scheduling_table())
        # =================================================
        
        # update the running task
        for task in running_queue:
            task.totburst += task.required_resource_size
            task.burst += task.required_resource_size
            task.currentburst += task.required_resource_size
            task.cumulative_executed_time += timestep


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
    event_range = 1/f_gcd
    sim_range = 1/f_gcd * 2
    sim_step = min([task_dict[task].exp_comp_t for task in task_dict])/32
    sche_tab = SchedulingTableInt(256, int(sim_range/sim_step),)
    push_task_into_scheduling_table(task_dict, sche_tab, sim_step*10, sim_step, event_range, sim_range, args.verbose)
    