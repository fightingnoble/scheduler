from __future__ import annotations

from typing import Union, List, Dict, Iterator, Callable, Union
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from global_var import *
from lru import LRUCache
from scheduling_table import SchedulingTableInt
from resource_agent import Resource_model_int
from task_agent import TaskInt
from task_queue_agent import TaskQueue 
from task_agent import ProcessInt

# 'ID', 'Task (chain) names', 'Flops on path', 'Expected Latency (ms)', 'T release', 'Freq.', 'DDL', 'Cores/Req.', 
# 'Throuput factor (S)', 'Thread factor (S)', 'Min required cores', 'Timing_flag', 'Max required Cores', 'RDA./Req.', 'Resource Type', 'Pre-assigned', 'Priority'

# task_graph = {   
#     "Entry": ["surr_view_camera_pub", "streo_camera_pub", "LiDAR_pub"],
#     "surr_view_camera_pub": ["ImageBB"],
#     "streo_camera_pub": ["Stereo_feature_enc", "Traffic_light_detection"],
#     "LiDAR_pub": ["Lidar_based_3dDet"],
#     "Traffic_light_detection": ["Exit"],
#     "ImageBB": ["MultiCameraFusion"],
#     "MultiCameraFusion": ["Pure_camera_path_head"],
#     "Pure_camera_path_head": ["Prediction"],
#     "Prediction": ["Planning"],
#     "Planning": ["Steering_speed"],
#     "Steering_speed": ["Exit"],
#     "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
#     "Semantic_segm": ["Lidar_based_3dDet"],
#     "Lidar_based_3dDet": ["Prediction"],
#     "Lane_drivable_area_det": ["Exit"],
#     "Optical_Flow": ["Exit"],
#     "Depth_estimation": ["Exit"],
#     "Exit": []
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
    "Semantic_segm": ["Lidar_based_3dDet"],
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

pre_assign_priority = {
    "ImageBB", 
    "Semantic_segm",
    "Traffic_light_detection",
    "Lane_drivable_area_det",
}


# def vis_task_static_timeline(task_list, show=False, save=False, save_path="task_static_timeline.pdf", **kwargs):
#     sim_time = 0.2
#     vertical_grid_size = 0.4
#     time_grid_size = 0.004

#     # build event list
#     req_list = []
#     ddl_list = []
#     finish_list = []

#     for task in task_list:
#         req_list.append(task.get_release_event(sim_time))
#         ddl_list.append(task.get_deadline_event(sim_time))
#         finish_list.append(task.get_finish_event(sim_time))

#     # print(req_list, ddl_list)


#     import matplotlib.colors as mcolors
#     import matplotlib as mpl
#     cmap = mpl.colormaps['viridis']
#     colors=list(mcolors.XKCD_COLORS.keys())
    
#     # plot timeline and task name 
#     # and select color for the task automatically
#     horizen_grid = set()
#     fig, ax = plt.subplots(figsize=(50, 10))
#     vertical_offset = 0
#     for i in range(len(task_list)):
#         for s, e in zip(req_list[i], finish_list[i]):
#             if e > sim_time:
#                 continue
#             print("{}:{}-{}".format(task_list[i].name, s, e))
#             horizen_grid.add(s)
#             horizen_grid.add(e)
#             ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s,
#                       xmax=e, lw=2, color=mcolors.XKCD_COLORS[colors[i]]
#                       )  # label=task_list[i].name)
#             ax.text(# sim_time, vertical_offset*vertical_grid_size,
#                     s, vertical_offset*vertical_grid_size+0.001,
#                     task_list[i].name, fontsize=7)
#         vertical_offset += 1
#         # s = next(req_list[i])
#         # e = next(ddl_list[i])
#         # print("{}:{}-{}".format(task_list[i].name, s, e))
#         # ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s, xmax=e, lw=2,)# label=task_list[i].name)
#         # ax.text(sim_time, vertical_offset*vertical_grid_size, task_list[i].name, fontsize=7)
#         # vertical_offset += 1

#     # np.arange(0, sim_time+time_grid_size, time_grid_size)
#     X, Y = np.meshgrid(np.array(list(horizen_grid)), np.arange(
#         0, (vertical_offset+1)*vertical_grid_size, vertical_grid_size))
#     ax.set(xlim=(0, 0.208), xticks=np.arange(0, 0.203, time_grid_size),)
#     ax.plot(X, Y, 'k', lw=0.5, alpha=0.5)
#     if show:
#         plt.show()
#     if "format" not in kwargs and save_path.split(".")[-1] == "pdf" and save:
#         kwargs["format"] = "pdf"
#     plt.savefig(save_path, **kwargs)

def vis_task_static_timeline(task_list, show=False, save=False, save_path="task_static_timeline_cyclic.pdf", 
                            hyper_p=0.1, n_p=1, warmup=False, drain=False, plot_legend=False,
                            plot_start=None, plot_end=None, 
                            tick_dens = 1, txt_size = 30,
                            **kwargs):
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = 0
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

    vertical_grid_size = 0.4
    time_grid_size = 0.004

    # build event list
    req_list = []
    ddl_list = []
    finish_list = []

    for task in task_list:
        req_list.append(task.get_release_event(event_range))
        ddl_list.append(task.get_deadline_event(event_range))
        finish_list.append(task.get_finish_event(event_range))

    # print(req_list, ddl_list)

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    # plot timeline and task name 
    # and select color for the task automatically
    horizen_grid = set()
    fig, ax = plt.subplots(figsize=(50, 15))
    vertical_offset = 0
    for i in range(len(task_list)):
        for s, e in zip(req_list[i], finish_list[i]):
            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            if s < plot_start and e > plot_start:
                s = plot_start
            if e > plot_end and s < plot_end:
                e = plot_end
            if s > plot_end or e < plot_start: 
                continue
            print("{}:{}-{}".format(task_list[i].name, s, e))
            horizen_grid.add(s)
            horizen_grid.add(e)
            # plot task
            ax.broken_barh([(s, e-s)], (vertical_offset*vertical_grid_size, vertical_grid_size), facecolors=mcolors.XKCD_COLORS[colors[i]])
            # add task name
            if not plot_legend:
                ax.text(s, vertical_offset*vertical_grid_size+0.001, task_list[i].name, ha='center', va='center', fontsize=7)
        vertical_offset += 1

    # np.arange(0, sim_time+time_grid_size, time_grid_size)
    X, Y = np.meshgrid(np.array(list(horizen_grid)), np.arange(
        0, (vertical_offset+1)*vertical_grid_size, vertical_grid_size))
    # set x range
    ax.set(xlim=(plot_start, plot_end), xticks=np.arange(plot_start, plot_end+time_grid_size, time_grid_size*tick_dens),)
    ax.plot(X, Y, 'k', lw=0.5, alpha=0.5)
    # add legend at the top as wide as the plot, text size 30
    if plot_legend:
        from matplotlib.lines import Line2D
        legend_elements = []
        for i in range(len(task_list)): 
            legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=task_list[i].name))
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
        # remove y axis
        ax.get_yaxis().set_visible(False)
        # reset x ticks: text size 30, rotation 45, distance time_grid_size * 2
        # ticks format: .3f
        ax.set_xticks(np.arange(plot_start, plot_end+time_grid_size, time_grid_size*tick_dens))
        ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end+time_grid_size, time_grid_size*tick_dens)]
        ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
        ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
        # remove frame
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # set x axis label as Time (s), text size 30
        ax.set_xlabel("Time (s)", fontsize=txt_size) 


    if show:
        plt.show()
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if "format" in kwargs and isinstance(kwargs["format"], list):
            fmt_list = kwargs.pop("format")
            if path_parse[-1] not in fmt_list:
                kwargs["format"].append(path_parse[-1])
            for f in fmt_list:
                save_path = ".".join(path_parse[:-1]) + "." + f
                plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
        elif "format" not in kwargs and len(path_parse) > 1: 
            kwargs["format"] = path_parse[-1]
        else:
            kwargs["format"] = "pdf"
            save_path = save_path + ".pdf"        
            plt.savefig(save_path, bbox_inches='tight', **kwargs)

def creat_jobTask_graph(task_graph:Dict[str, List[str]], plot:bool=False):
    # create task graph from task_graph
    task_graph_nx = nx.DiGraph(task_graph)

    job_graph_nx = nx.DiGraph()
    df = pd.read_csv("profiling.csv", sep=",", index_col=0) 

    for task_n in task_graph: 
        if task_n in df.index:
            task_attr = df.loc[task_n].to_dict()
            factor = task_attr["Throuput factor (S)"]
            for i in range(factor):
                task_name = task_n+"_"+str(i)
                job_graph_nx.add_node(task_name)
                # add control dependency
                if i < factor-1:
                    # job_graph_nx.add_edge(task_name, task_n+"_"+str(i+1), type="control")
                    pass

                # add dependency
                for succ_n in task_graph[task_n]:
                    # select instance of succ 
                    # 1. in the same sub-period; 
                    # 2. or the nearest period before the current time. 
                    # get the factor of succ
                    if succ_n == "Exit":
                        job_graph_nx.add_edge(task_name, "Exit", type="control")
                        continue
                    succ_factor = df.loc[succ_n]["Throuput factor (S)"]
                    if succ_factor == factor:
                        succ_name = succ_n+"_"+str(i)
                    else:
                        # just like quantization
                        curr_t = i/factor
                        succ_t = int(curr_t*succ_factor)
                        succ_name = succ_n+"_"+str(succ_t)
                    job_graph_nx.add_edge(task_name, succ_name, type="data")
        else:
            job_graph_nx.add_node(task_n)
            edge_type = "control" if task_n == "Entry" else "data"
            for succ_n in task_graph[task_n]:
                if succ_n in df.index:
                    task_attr = df.loc[succ_n].to_dict()
                    factor = task_attr["Throuput factor (S)"]
                    for i in range(factor):
                        succ_name = succ_n+"_"+str(i)
                        job_graph_nx.add_edge(task_n, succ_name, type=edge_type)
                else:
                    job_graph_nx.add_edge(task_n, succ_n, type=edge_type)

    # plot the job graph and the task graph, then save to pdf
    if plot: 
        color_map = {"control": "r", "data": "b"}
        edge_colors = [color_map[d] for u,v,d in job_graph_nx.edges(data="type")]
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for layer, nodes in enumerate(nx.topological_generations(job_graph_nx)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                job_graph_nx.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(job_graph_nx, subset_key="layer") 
        nx.draw(job_graph_nx, pos, with_labels=True, node_size=100, node_color='r', edge_color=edge_colors, font_size=10, ax=ax1)
        fig.tight_layout()
        # plt.savefig("job_graph.pdf", format="pdf")
        
        for layer, nodes in enumerate(nx.topological_generations(task_graph_nx)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                task_graph_nx.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(task_graph_nx, subset_key="layer")
        nx.draw(task_graph_nx, pos, with_labels=True, node_size=100, node_color='r', font_size=10, ax=ax2)
        fig.tight_layout()
        plt.savefig("jobTask_graph.pdf", format="pdf")

    return task_graph_nx, job_graph_nx

def load_taskint(verbose: bool = False, plot:bool = False) -> Dict[str, TaskInt]:

    df = pd.read_csv("profiling.csv", sep=",", index_col=0) 
    if verbose:
        print(df)
    task_dict = {}
    task_id = 0
    # print(task_attr_dict)

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    if plot:
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
                RDA_size=task_attr['RDA./Req.'], main_size=task_attr['Cores/Req.'], seq_cpu_time=task_attr["Flops on path"]/1e3,
                op_cpu_time=task_attr["Flops on path"]/1e3, op_io_time=1e-6,
            )
            task.freq = task_attr["Freq."]
            # initialize task affinity list
            thread_n = int(i)
            affinity_tgt_n_list = affinity[task_n]        
            affinity_tgt_n_list = [n+'_'+str(thread_n) for n in affinity_tgt_n_list]
            task.affinity_n = affinity_tgt_n_list
            # initialize dependency list


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

    for task_n, task in task_dict.items(): 
        # get affinity target id
        # TODO: bug here， key error when the affinity target is not in the pid_idx
        affinity_tgt_id_list = [task_dict[n].id for n in task.affinity_n if n in task_dict]
        task.affinity = affinity_tgt_id_list

    return task_dict

def init_depen(taskJobs:Union[Dict[str, Union[TaskInt,ProcessInt]], List[Union[TaskInt,ProcessInt]]], job_graph_nx:nx.DiGraph, verbose=False):
    # if taskJobs is a list, convert it to a dict
    if isinstance(taskJobs, list):
        if taskJobs[0].__class__.__name__ == "ProcessInt":
            taskJobs = {i.task.name:i for i in taskJobs}
        elif taskJobs[0].__class__.__name__ == "TaskInt":
            taskJobs = {i.name:i for i in taskJobs}
    
    for task_n, task in taskJobs.items():
        for pre_n, datadict in job_graph_nx.pred[task_n].items():
            dep_t = datadict["type"]
            if dep_t == "data":
                task.pred_data.update({pre_n:False})
            elif dep_t == "control":
                task.pred_ctrl.update({pre_n:False})
            else:
                raise Exception("Unknown dependency type")
        for succ_n, datadict in job_graph_nx.succ[task_n].items():
            dep_t = datadict["type"]
            if dep_t == "data":
                task.succ_data.update({succ_n:False})
            elif dep_t == "control":
                task.succ_ctrl.update({succ_n:False})
            else:
                raise Exception("Unknown dependency type")
        
        if verbose:
            print(task_n, task.pred_data, task.pred_ctrl, task.succ_data, task.succ_ctrl)


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
                # judge if task complete: completion_count += 1, cum_trunAroundTime += (time + 1.0 - a_time), 
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
                _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
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
    tab_temp_size = int(sim_range/timestep)
    tab_spatial_size = total_cores

    SchedTab = SchedulingTableInt(total_cores, int(sim_range/timestep),)

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
                # judge if task complete: completion_count += 1, cum_trunAroundTime += (time + 1.0 - a_time), 
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
                _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
                
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
                print("TASK {:d}:{:s}({:d}) IS ISSUED @ {}!!".format(_p.task.id, _p.task.name, _p.pid, curr_t))
                running_queue.put(_p)
                ready_queue.remove(_p)
                if _p.totburst == 0:
                    _p.start_time = curr_t
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

def push_task_into_bins(tasks: Union[List[TaskInt], Dict[str, TaskInt]], affinity, #SchedTab: SchedulingTableInt, 
                                    total_cores:int, quantumSize, 
                                    timestep, hyper_p, n_p=1, verbose=False, *, animation=False, warmup=False, drain=False,):

    """
    implement a naive 2d bin packing algorithm
    input: task_list, that is already arranged in the topological order
    output: a list of bins, each bin is a list of tasks
    """
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(sim_range/timestep)
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
    
    rsc_recoder = {}
    rsc_recoder_his = {}

    expired_list:List[ProcessInt] = []
    def issue_sort_fn(x:ProcessInt):
        alloc_slot_s, alloc_size, allo_slot, bin_id = rsc_recoder[x.pid]
        if isinstance(alloc_slot_s, int):
            return alloc_slot_s
        else:
            return alloc_slot_s[0]
    # monitor the issue time: (accending)
    issue_list:TaskQueue = TaskQueue(sort_f=issue_sort_fn, decending=False)
    completed_list:List[ProcessInt] = []
    miss_list:List[ProcessInt] = []
    preempt_list:List[ProcessInt] = []
    curr_cfg:Resource_model_int
 
    # try to push the task into the bins in the bin_list
    # if the task cannot be pushed into any bin, create a new bin
    # _new_bins = lambda id: new_bins(total_cores, int(sim_range/timestep), id=id, name="bin"+str(id))
    def _new_bin(id, size=tab_spatial_size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, tab_temp_size, id=id, name=name)

    # aa = lambda _p: int(np.ceil(_p.task.flops/(int(np.ceil(_p.release_time/timestep))-int(_p.deadline//timestep))/timestep/FLOPS_PER_CORE))
    def get_core_size(_p):
        # release time round up: task should not be released earlier than the release time
        time_slot_s = int(np.ceil(_p.release_time/timestep))
        # deadline round down: task should not be finised later than the deadline
        time_slot_e = int(_p.deadline//timestep)
        req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
        return req_rsc_size

    def _next_bin_obj_0():
        """
        generate the bin list in the descending order of the core size
        """        
        p_list = [(get_core_size(_p), _p) for _p in init_p_list]
        print("&*(&*(^(*^(*&^*(^*(&")
        p_list.sort(key=lambda x: x[0], reverse=True)
        # TODO: bug here, name confilit
        yield from (_new_bin(bin_id,x[0],x[1].task.name) for bin_id, x in enumerate(p_list))

    # iter_next_bin_obj = _next_bin_obj_0()
    # bin_list:List[SchedulingTableInt] = [next(iter_next_bin_obj)]
    # bin_name_list = [bin_list[0].name]

    def _next_bin_obj_1(max_core_size:int, size_list:List[int], name_list:List[str]): 
        """
        genrate the bin list according to the size_list until the max_core_size is reached
        """ 
        cum_size = 0
        bin_id = 0
        for size, name in zip(size_list, name_list): 
            # if cum_size + size > max_core_size:
            #     yield _new_bin(bin_id, max_core_size - cum_size, name)
            #     break
            yield _new_bin(bin_id, size, name)
            bin_id += 1
            cum_size += size

    def _next_bin_obj_2(max_core_size:int, p_list:List[ProcessInt], RDA_ratio:float=1.2): 
        """
        genrate the bin list according to the size_list until the max_core_size is reached
        """ 
        cum_size = 0
        bin_list = []
        for _p in p_list:
            size_main = get_core_size(_p)
            task_type = _p.task.timing_flag
            # get RDA size
            if task_type == "deadline": 
                size_RDA = int(np.ceil(size_main * RDA_ratio))
            else:
                size_RDA = 0
            size = size_main + size_RDA
            # get bin
            if cum_size + size > max_core_size: 
                yield _new_bin(bin_id, max_core_size - cum_size, _p.task.name)
                break
            yield _new_bin(bin_id, size, _p.task.name)
            bin_id += 1
            cum_size += size

    # iter_next_bin_obj = _next_bin_obj_0()
    # bin_list:List[SchedulingTableInt] = [next(iter_next_bin_obj)]
    # bin_name_list = [bin_list[0].name]

    size_l = []
    name_l = []
    for _p in init_p_list:
        if _p.task.pre_assigned_resource_flag:
            size_l.append(_p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size)
            name_l.append(_p.task.name)
    iter_next_bin_obj = _next_bin_obj_1(max_core_size=256, size_list=size_l, name_list=name_l)
    bin_list:List[SchedulingTableInt] = list(iter_next_bin_obj)
    bin_name_list = [bin.name for bin in bin_list]
    # bin_list:List[SchedulingTableInt] = [next(iter_next_bin_obj)]
    # bin_name_list = [bin_list[0].name]

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
                # judge if task complete: completion_count += 1, cum_trunAroundTime += (time + 1.0 - a_time), 
                # update arrival time, deadline, clear current execution unit
                if (_p.totburst >= _p.totcpu):
                    completed_list.append(_p)
                    _p.set_state("suspend")
        
        if completed_list:
            for _p in completed_list:
                # release the resource and move to the wait list
                bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                
                _SchedTab = bin_list[bin_id_t]
                _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                print("TASK {:d}:{:s}({:d}) COMPLETED!!".format(_p.task.id, _p.task.name, _p.pid))
                # update statistics
                # TODO: add lock 
                _p.task.completion_count += 1
                _p.task.cum_trunAroundTime += (curr_t - _p.release_time)
                
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
            for _p in miss_list:
                # release the resource and move to the wait list
                if _p in ready_queue.queue:
                    ready_queue.remove(_p)
                elif _p in running_queue.queue:
                    bin_id_t, alloc_slot_s, alloc_size, allo_slot = get_rsc_2b_released(rsc_recoder, n_slot, _p)
                    
                    _SchedTab = bin_list[bin_id_t]
                    _SchedTab.release(_p, alloc_slot_s, alloc_size, allo_slot, verbose=False)
                    rsc_recoder.pop(_p.pid)
                    running_queue.remove(_p)
                # print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid));
                print("TASK {:d}:{:s}({:d}) MISSED DEADLINE!!".format(_p.task.id, _p.task.name, _p.pid)+\
                        "released @ {} start @ {}:\n".format(_p.release_time, _p.start_time) +\
                        "is expected to finish {}T OPs before {} is expired, ".format(_p.totcpu, _p.deadline) +\
                        "but only executed {}T OPs in {}s time !!".format(_p.totburst, _p.cumulative_executed_time))
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


        # A. rearange the task in the ready queue
        #   1. sort the task according to the deadline
        #   2. sort the task according to the affinity to the existing bins                        
        # B. calculate the affinity preference core
        # C. priorize the task that has firm affinity with the existing bins

        # rearange the task in the ready queue
        cond_fn1 = lambda x: (x.task.deadline - curr_t)/x.exp_comp_t
        cond_fn2 = lambda x: get_target_bin_score(x, bin_name_list=bin_name_list, rsc_recoder_his=rsc_recoder_his, reverse=True)
        sorted_ready_queue = sorted(ready_queue.queue, key=lambda x: (cond_fn1(x), cond_fn2(x)))
        
        for _p in sorted_ready_queue: 
            # issue the task
            allocate_rsc_4_process(_p, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)

        # update the running task
        if len(preempt_list):
            for _p in preempt_list:
                if _p in running_queue.queue:
                    running_queue.remove(_p)
                    ready_queue.put(_p)
            preempt_list.clear()
        
        # issue the task
        # if the task of the queue equals to the current slot, then issue the task
        if len(issue_list):
            for _p in issue_list:
                if _p in ready_queue:
                    ready_queue.remove(_p)
            while len(issue_list):
                _p = issue_list[0]
                if issue_sort_fn(_p) == n_slot: 
                    running_queue.put(_p)
                    if _p.totburst == 0:
                        _p.start_time = curr_t
                    _p.waitTime = 0
                    issue_list.get()
                else:
                    break
            # issue_list.clear()
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
        _p_dict = {p.pid:p for p in running_queue}
        for _SchedTab in bin_list:
            curr_cfg = _SchedTab.scheduling_table[n_slot]
            if curr_cfg.rsc_map:
                # update the running task
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

    for _SchedTab in bin_list:
        print("=====================================\n")
        print(f"Scheduling Table of {_SchedTab.name}({_SchedTab.id}):")
        _SchedTab.print_scheduling_table()
        print("=====================================\n")
    
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
    
    return bin_list, init_p_list

def get_rsc_2b_released(rsc_recoder, n_slot, _p):
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
    return bin_id_t,alloc_slot_s,alloc_size,allo_slot


def allocate_rsc_4_process(_p:ProcessInt, n_slot:int, 
                affinity:Dict[str, List[str]], pid_idx:dict, init_p_list:List[ProcessInt], 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
                ready_queue:TaskQueue, running_queue:TaskQueue, 
                issue_list:TaskQueue, preempt_list:List[ProcessInt], 
                iter_next_bin_obj:Iterator, bin_list:List[SchedulingTableInt], bin_name_list:List[str], ):
    # initialize the resource request parameters
    p_name = _p.task.name
    time_slot_s, time_slot_e, req_rsc_size = rsc_req_estm(_p, n_slot, timestep, FLOPS_PER_CORE)

    # expected slot number
    expected_slot_num = time_slot_e-time_slot_s # int(np.ceil(_p.exp_comp_t/timestep))
    # TODO: Add the logic to ensure the task have allocated enough resource, otherwise, 
    #         we should not issue the task or compensate the resource latter. 

    # try to push the task into the bins in the bin_list
    state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
    # pre-allocation strategy: 
    # 1. the resource constraint should be respected
    # 2. the pre-defined resource preservation should be respected 
    # 3. the affinity settings of all the tasks should be respected 
    # 4. all tasks should be allocated with the resource
    # 5. tasks is epected to migrate as less as possible
    # 6. the resource should be allocated as compact as possible
    # 7. the resource should be allocated as balanced as possible

    _p_index_by_pid = {_p.pid: _p for _p in init_p_list}
    # TODO: arange the bin_list according to the affinity of the class

    # 2. the pre-defined resource preservation should be respected 
    #   For the task that is pre-assigned with the resource, the affinity is set to be itself
    if p_name in bin_name_list:
        bin_id = bin_name_list.index(p_name)
        bin = bin_list[bin_id]
        state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
        total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
        total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
        # check if the task is allocated successfully
        # if fail collect the tasks in the target bin and preempt the tasks
        if state and (total_FLOPS_alloc >= _p.remburst):
            pass
            print("Task {} is allocated successfully".format(_p.pid))
        else: 
            # release the resource
            bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
            # reset the state
            state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
            p_2b_realloc = []
            state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc = preempt_the_conflicts(_p, bin, 
                req_rsc_size, expected_slot_num, 
                time_slot_s, time_slot_e, n_slot, 
                timestep, FLOPS_PER_CORE, quantumSize, 
                rsc_recoder, rsc_recoder_his, 
                running_queue, 
                issue_list, preempt_list, p_2b_realloc, 
                _p_index_by_pid, quantum_check_en=False) 

            # reallocate the tasks in the P_2b_realloc
            for _p_2b_realloc in p_2b_realloc:
                allocate_rsc_4_process(_p_2b_realloc, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                    rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)
    else: 
        # find the target bin of the affinity target
        def get_target_bin_id(_p=_p, bin_name_list=bin_name_list, rsc_recoder_his=rsc_recoder_his):
            affinity_tgt_bin_id_list = []
            for task_n in _p.task.affinity_n:
                # suppose the target is pre-assigned with the resource but is not allocated
                if task_n in bin_name_list:
                    affinity_tgt_bin_id_list.append(bin_name_list.index(task_n))
            # suppose the target was allocated with the resource
            for _pid in _p.task.affinity:
                if _pid in rsc_recoder_his:
                    affinity_tgt_bin_id_list.append(rsc_recoder_his[_pid].get_mru())
            return affinity_tgt_bin_id_list

        affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)

        # # remove the thread number at the end of the name
        # # name parse
        # thread_n = _p.task.name.split('_')[-1]
        # p_name = _p.task.name
        # task_base_name = p_name.replace("_"+thread_n, "")
        # thread_n = int(thread_n)
        # affinity_tgt_n_list = affinity[task_base_name]        
        # affinity_tgt_n_list = [n+'_'+str(thread_n) for n in affinity_tgt_n_list]
        # # get affinity target id
        # # TODO: bug here， key error when the affinity target is not in the pid_idx
        # affinity_tgt_id_list = [pid_idx[n] for n in affinity_tgt_n_list if n in pid_idx]
        # # find the target bin of the affinity target
        # affinity_tgt_bin_id_list = [rsc_recoder_his[n].get_mru() for n in affinity_tgt_id_list if n in rsc_recoder_his] 

        # mark other bins as the targets of the search
        affinity_search_bin_id_list = [n for n in range(len(bin_list)) if n not in affinity_tgt_bin_id_list] 
        # arrange the targets of the search in the order of the fitness of the size
        affinity_search_bin_id_list.sort(key=lambda x: abs(bin_list[x].scheduling_table[0].size - req_rsc_size))

        # 3. the affinity settings of all the tasks should be respected 
        # search the bin in the affinity target bin list to find bin with best affinity
        for bin_id in affinity_tgt_bin_id_list: 
            bin = bin_list[bin_id]
            state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
            total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
            total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
            # check if the task is allocated successfully
            # if fail collect the tasks in the target bin and preempt the tasks
            if state and (total_FLOPS_alloc >= _p.remburst):
                print("Task {} is allocated successfully".format(_p.pid))
                break
            else:
                # release the resource
                # TODO: merge this function into the scheduling table class
                # TODO: distinguish the state of risking the lack of resources and the state of the task cannot be pushed into the bin
                bin.release(_p, alloc_slot_s, alloc_size, allo_slot)
                # reset the state
                state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
                # p_2b_realloc = []
                # state, alloc_slot_s, alloc_size, allo_slot, \
                #     total_alloc_unit, total_FLOPS_alloc = preempt_the_conflicts(_p, bin, 
                #     req_rsc_size, expected_slot_num, 
                #     time_slot_s, time_slot_e, n_slot, 
                #     timestep, FLOPS_PER_CORE, quantumSize, 
                #     rsc_recoder, rsc_recoder_his, 
                #     running_queue, 
                #     issue_list, preempt_list, p_2b_realloc, 
                #     _p_index_by_pid, key=lambda x: affinity_fn(x, bin, rsc_recoder_his)) 
                # # reallocate the tasks in the P_2b_realloc
                # for _p_2b_realloc in p_2b_realloc:
                #     allocate_rsc_4_process(_p_2b_realloc, n_slot, affinity, pid_idx, init_p_list, timestep, FLOPS_PER_CORE, quantumSize,  
                #         rsc_recoder, rsc_recoder_his, ready_queue, running_queue, issue_list, preempt_list, iter_next_bin_obj, bin_list, bin_name_list)


        if not state:
            # search the bin in the affinity search bin list
            for bin_id in affinity_search_bin_id_list: 
                bin = bin_list[bin_id]
                state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
                total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
                total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
                if state and (total_FLOPS_alloc >= _p.remburst):
                    print("Task {} is allocated successfully".format(_p.pid))
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
        
    # print the allocation result
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

    # record the allocation result and prepare the issue list
    if state:
        rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot, bin_id]
        issue_list.put(_p)
        if _p.pid in rsc_recoder_his:
            rsc_recoder_his[_p.pid].put(bin_id)
        else:
            rsc_recoder_his[_p.pid] = LRUCache(3)
            rsc_recoder_his[_p.pid].put(bin_id)
    else:
        Warning("TASK {:d}:{:s}({:d}) IS DELAY ISSUED!!".format(_p.task.id, _p.task.name, _p.pid))

def rsc_req_estm(_p, n_slot, timestep, FLOPS_PER_CORE):
    # release time round up: task should not be released earlier than the release time
    time_slot_s = int(np.ceil(_p.release_time/timestep))
    if time_slot_s < n_slot:
        time_slot_s = n_slot
    # deadline round down: task should not be finised later than the deadline
    time_slot_e = int(_p.deadline//timestep)
    req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
    return time_slot_s,time_slot_e,req_rsc_size
        
def build_task_graph_and_packing(verbose: bool = False, plot:bool = False) -> Dict[str, TaskInt]:
    pass

def get_process_affinity_idx(_p:ProcessInt, affinity:Dict[str, List[str]], pid_idx:Dict[str, int]) -> List[int]:
    # name parse
    p_name = _p.task.name
    # remove the thread number at the end of the name
    thread_n = _p.task.name.split('_')[-1]
    task_base_name = p_name.replace("_"+thread_n, "")
    thread_n = int(thread_n)
    affinity_tgt_n_list = affinity[task_base_name]        
    affinity_tgt_n_list = [n+'_'+str(thread_n) for n in affinity_tgt_n_list]
    # get affinity target id
    # TODO: bug here， key error when the affinity target is not in the pid_idx
    affinity_tgt_id_list = [pid_idx[n] for n in affinity_tgt_n_list if n in pid_idx]
    return affinity_tgt_id_list

def preempt_the_conflicts(_p:ProcessInt, bin:SchedulingTableInt, 
        req_rsc_size:int, expected_slot_num:int, 
        time_slot_s:int, time_slot_e:int, n_slot:int,
        timestep, FLOPS_PER_CORE, quantumSize, 
        rsc_recoder:dict, rsc_recoder_his:Dict[int, LRUCache], 
        running_queue:TaskQueue, 
        issue_list:TaskQueue, preempt_list:List[ProcessInt], p_2b_realloc:List[ProcessInt],
        _p_index_by_pid:Dict[int, ProcessInt], key:Callable[[ProcessInt], int]=None, quantum_check_en:bool = True):
    """
    Preempt the conflict tasks and reallocate the resources to the current task
    quantum_check_en:
        disable the quantum check in the stage of preallocation
    """
    
    bin_id = bin.id
    # find out the conflict tasks
    occupation_candi_dict:Dict[int, List[int]] = bin.index_occupy_by_id(time_slot_s, time_slot_e)
    # filter the confict tasks that have lower affinity with the current bin compared with the current task
    if key is not None:
        occupation_candi_dict = {k:v for k,v in occupation_candi_dict.items() if key(_p) > key(_p_index_by_pid[k])}
    occupation_candi = list(occupation_candi_dict.keys())

    aval_flops = sum(bin.idx_free_by_slot(time_slot_s, time_slot_e, _p.pid)) * timestep * FLOPS_PER_CORE
    # note that here is an assumption that current bin is designed for the task, all the candidate tasks are preempted
    # flops_2b_preempt = (time_slot_e - time_slot_s) * bin.scheduling_table[0].size * timestep * FLOPS_PER_CORE - aval_flops
    flops_2b_preempt = _p.remburst - aval_flops
    
    # evaluate the preemption possibility
    # if the preemptable conflict task can provide enough FLOPS to the current task
    total_FLOPS_occupied = 0
    for pid in occupation_candi:
        alloc_slot_s_t, alloc_size_t, allo_slot_t = occupation_candi_dict[pid]
        total_alloc_unit_t = np.sum(np.array(alloc_size_t) * np.array(allo_slot_t))
        total_FLOPS_alloc_t = total_alloc_unit_t * timestep * FLOPS_PER_CORE
        total_FLOPS_occupied += total_FLOPS_alloc_t
    if flops_2b_preempt-total_FLOPS_occupied > 1e-2*timestep*FLOPS_PER_CORE: 
        state, alloc_slot_s, alloc_size, allo_slot = False, None, None, None
        total_alloc_unit, total_FLOPS_alloc = 0, 0
        return state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc

    # decide which task to preempt
    # select the task with the latest deadline
    # TODO: evaluate more strategies
    occupation_candi.sort(key=lambda x: _p_index_by_pid[x].deadline, reverse=True)
    
    for pid in occupation_candi:
        # load allocation history
        _p_2b_preempt = _p_index_by_pid[pid]
        alloc_slot_s_t, alloc_size_t, allo_slot_t, bin_id_t = rsc_recoder[pid]
        assert bin_id == bin_id_t
        
        # task is in the ready queue and issue list
        # judge if preemptable: 

        # task is runnning but has executed for an integer multiples of the quantum size (control the pre-emption grain)
        if quantum_check_en:
            cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
            reach_preempt_grain = np.allclose(cum_exec_quantum, round(cum_exec_quantum), atol=1e-2)
            if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain: 
                continue
        else:
            reach_preempt_grain = True
        
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
        bin_id_t, alloc_slot_s_t, alloc_size_t, allo_slot_t = get_rsc_2b_released(rsc_recoder, n_slot, _p_2b_preempt)
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
        if flops_2b_preempt <= 1e-2*timestep*FLOPS_PER_CORE:
            break

    # reallocate the current task
    state, alloc_slot_s, alloc_size, allo_slot = bin.insert_task(_p, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, verbose=False)
    total_alloc_unit = np.sum(np.array(alloc_size) * np.array(allo_slot))
    total_FLOPS_alloc = total_alloc_unit * timestep * FLOPS_PER_CORE
    # check if the task is allocated successfully
    # if fail collect the tasks in the target bin and preempt the tasks
    if state and (total_FLOPS_alloc >= _p.remburst): 
        print(f"task {_p.task.id}:{_p.task.name}({_p.pid}) is allocated successfully in the bin {bin_id}")
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
        Warning(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.pid}) is not allocated successfully after preemption in its expected bin")
        # raise ValueError(f"A unexpected situation happens, task {_p.task.id}:{_p.task.name}({_p.pid}) is not allocated successfully after preemption in its own bin")
    return state, alloc_slot_s, alloc_size, allo_slot, total_alloc_unit, total_FLOPS_alloc

# how does task affinity match with the existing bins
def get_target_bin_score(_p:ProcessInt, bin_name_list:List[str], rsc_recoder_his:Dict[int, LRUCache], reverse=True): 
    """
    match the affinity targets list with the existing bins list
    """
    affinity_tgt_bin_id_list = []
    # case 1: task is pre-assigned with the resource
    p_name = _p.task.name
    if p_name in bin_name_list: 
        affinity_tgt_bin_id_list.append(bin_name_list.index(p_name))
    else:
        affinity_tgt_bin_id_list.append(-1)
    
    for task_n, task_id in zip(_p.task.affinity_n, _p.task.affinity): 
        # case 2: suppose the target is pre-assigned with the resource but is not allocated
        if task_n in bin_name_list:
            affinity_tgt_bin_id_list.append(bin_name_list.index(task_n))
        # case 3: suppose the target was allocated with the resource
        elif task_id in rsc_recoder_his:
            affinity_tgt_bin_id_list.append(rsc_recoder_his[task_id].get_mru())
        else:
            affinity_tgt_bin_id_list.append(-1)

    # score function
    # if p_name in bin_name_list, score = 1
    # else set weight of each affinity target as the reciprocal of the 2^i, i is the index of the affinity target
    if affinity_tgt_bin_id_list[0] != -1:
        score = 1.0
    else:
        weight = 1/2**np.arange(len(_p.task.affinity))
        score = np.sum(weight*(np.array(affinity_tgt_bin_id_list)!=-1)[1:][::-1])
    if reverse:
        return 1 - score
    return score


def affinity_fn(_p, bin, rsc_recoder_his): 
    affinity_tgt_id_list = _p.task.affinity
    if len(affinity_tgt_id_list) == 0:
        return 0
    # find the target bin of the affinity target
    affinity_tgt_bin_id_list = [rsc_recoder_his[n].get_mru() for n in affinity_tgt_id_list if n in rsc_recoder_his] 
    if bin.id in affinity_tgt_bin_id_list:
        return 1 - affinity_tgt_bin_id_list.index(bin.id)/len(affinity_tgt_bin_id_list)
    else:
        return 0


if __name__ == "__main__": 
    import argparse
    import numpy as np 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="all", help="task name")
    parser.add_argument("--plot", action="store_true", help="plot the task timeline")
    parser.add_argument("--bin_pack", action="store_true", help="plot the task timeline")
    parser.add_argument("--test_all", default=False, help="test all the task")
    args = parser.parse_args() 
    task_dict = load_taskint(args.verbose)

    if args.test_case == "all":
        args.test_all = True
    f_gcd = np.gcd.reduce([task_dict[task].freq for task in task_dict])
    f_max = max([task_dict[task].freq for task in task_dict])
    hyper_p = 1/f_gcd
    sim_step = min([task_dict[task].exp_comp_t for task in task_dict])/32

    if args.test_case == "timeline" or args.test_all:
        vis_task_static_timeline(list(task_dict.values()), save=True, save_path="task_static_timeline_cyclic.pdf", hyper_p=hyper_p, n_p=1, warmup=False, drain=True, )
    elif args.test_case == "liveness" or args.test_all:
        vis_task_static_timeline(list(task_dict.values()), save=True, save_path="task_liveness_timeline_cyclic.svg", 
        hyper_p=hyper_p, n_p=1, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
        txt_size=40, tick_dens=4)
    elif args.test_case == "bin_pack" or args.test_all:
        # push_task_into_scheduling_table_cyclic_preemption_disable(task_dict, 256, sim_step*1, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)
        bin_list, init_p_list = push_task_into_bins(task_dict, affinity, 256, sim_step*2, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)
        from scheduling_table import get_task_layout_compact
        get_task_layout_compact(bin_list, init_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=1, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
        txt_size=40, tick_dens=2, save_path="task_bin_pack_cyclic.pdf") 

        get_task_layout_compact(bin_list, init_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=1, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path="task_bin_pack_full.pdf")

        # save the bin_list and the init_p_list
        with open("bin_list.pkl", "wb") as f:
            pickle.dump(bin_list, f)
        with open("init_p_list.pkl", "wb") as f:
            pickle.dump(init_p_list, f)
        try:
            # load the bin_list and the init_p_list
            with open("bin_list.pkl", "rb") as f:
                bin_list = pickle.load(f)
            with open("init_p_list.pkl", "rb") as f:
                init_p_list = pickle.load(f)
            print("bin_list.pkl and init_p_list.pkl saved and loaded successfully")
        except:
            print("bin_list.pkl or init_p_list.pkl not found")
            exit()

    elif args.test_case == "graph" or args.test_all:
        task_graph_nx, job_graph_nx = creat_jobTask_graph(task_graph, plot=True)
        init_depen(task_dict, job_graph_nx, verbose=args.verbose)
    elif args.test_case == "dynamic" or args.test_all:
        try:
            # load the bin_list and the init_p_list
            with open("bin_list.pkl", "rb") as f:
                bin_list = pickle.load(f)
            with open("init_p_list.pkl", "rb") as f:
                init_p_list = pickle.load(f)
        except:
            print("bin_list.pkl or init_p_list.pkl not found")
            bin_list, init_p_list = push_task_into_bins(task_dict, affinity, 256, sim_step*2, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)

        task_graph_nx, job_graph_nx = creat_jobTask_graph(task_graph, plot=False)
        init_depen(init_p_list, job_graph_nx)
        from allocator_agent import cyclic_sched
        from scheduler_agent import Scheduler 
        from monitor_agent import Monitor
        from spec import Spec
        rsc_recoder = {}
        rsc_recoder_his = {}
        scheduler_list = [Scheduler() for _ in range(len(bin_list))]
        monitor_list = [Monitor() for _ in range(len(bin_list))]
        task_spec = Spec(0.1, [1 for _ in init_p_list]) 
        rsc_list = [Resource_model_int(size=sched_tab.scheduling_table[0].size) for sched_tab in bin_list]

        print("sim_step: ", sim_step)
        cyclic_sched(task_spec, affinity, 
                bin_list, scheduler_list, monitor_list,
                rsc_list,
                rsc_recoder, rsc_recoder_his,
                256, sim_step*2, init_p_list,
                sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)
        
