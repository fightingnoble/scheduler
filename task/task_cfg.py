from __future__ import annotations

from typing import Union, List, Dict, Iterator, Callable, Tuple, Optional
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from global_var import *
from task.task_agent import TaskInt, TaskIntAttr
from model.task_queue_agent import TaskQueue 
from task.task_agent import ProcessInt
from task.graph_scaling import build_node_relationship
from sched.slack_estim import estim_release_dll_time, duduce_cfg

# 'ID', 'Task (chain) names', 'Flops on path (G)', 'Expected Latency (ms)', 'T release (ms)', 'Freq.', 'DDL (ms)', 'Cores/Req.', 
# 'Throuput factor (Spat.)', 'Thread factor (S)', 'Min required cores', 'Timing_flag', 'Max required Cores', 'RDA./Req.', 'Resource Type', 'Pre-assigned', 'Priority'

task_graph_srcs = {
    # "Entry": ["surr_view_camera_pub", "streo_camera_pub", "LiDAR_pub"],
    "LiDAR_pub": ["Lidar_based_3dDet"],
    "surr_view_camera_pub": ["Traffic_light_detection", "ImageBB", ],
    "IMU_pub": ["Steering_speed"],
    "streo_camera_pub": ["Stereo_feature_enc"],
}
task_graph_sinks = {
    "Sink_control": [],
    "Sink_screen": [],
}
task_graph_ops = {   
    "Traffic_light_detection": ["Sink_control"],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["Pure_camera_path_head"],
    "Pure_camera_path_head": ["Prediction"],
    "Prediction": ["Planning"],
    "Planning": ["Steering_speed"],
    "Steering_speed": ["Sink_control"],
    "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
    "Semantic_segm": ["Lidar_based_3dDet","Sink_screen"],
    "Lidar_based_3dDet": ["Prediction"],
    "Lane_drivable_area_det": ["Sink_screen"],
    "Optical_Flow": ["Sink_screen"],
    "Depth_estimation": ["Sink_screen"],
}

__all__ = ['task_graph', 'affinity_cfg']


# task_graph = {   
#     "Traffic_light_detection": [],
#     "ImageBB": ["MultiCameraFusion"],
#     "MultiCameraFusion": ["Pure_camera_path_head"],
#     "Pure_camera_path_head": ["Prediction"],
#     "Prediction": ["Planning"],
#     "Planning": ["Steering_speed"],
#     "Steering_speed": [],
#     "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
#     "Semantic_segm": ["Lidar_based_3dDet"],
#     "Lidar_based_3dDet": ["Prediction"],
#     "Lane_drivable_area_det": [],
#     "Optical_Flow": [],
#     "Depth_estimation": [],
# }

# affnity of a task is set to be a list that contains user-specified tasks, itself, it predecessors and its successors.
affinity_cfg = {
    "Traffic_light_detection": [],
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["ImageBB", "Pure_camera_path_head"],
    "Pure_camera_path_head": ["MultiCameraFusion", "Prediction"],
    "Prediction": ["Pure_camera_path_head", "Lidar_based_3dDet", "Planning"],
    "Planning": ["Prediction", "Steering_speed"],
    "Steering_speed": ["Planning"],
    "Stereo_feature_enc": ["Semantic_segm", "Lane_drivable_area_det", "Optical_Flow", "Depth_estimation"],
    "Semantic_segm": ["Stereo_feature_enc", "LiDAR_based_3dDet"],
    "Lidar_based_3dDet": ["Stereo_feature_enc", "Semantic_segm", "Prediction"],
    "Lane_drivable_area_det": ["Stereo_feature_enc"],
    "Optical_Flow": ["Stereo_feature_enc", "Lane_drivable_area_det",],
    "Depth_estimation": ["Stereo_feature_enc", "Lane_drivable_area_det",],
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

def vis_task_static_timeline(task_list:List[TaskInt], show=False, save=False, save_path="task_static_timeline_cyclic.pdf", 
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
    stimu_list = []
    req_list = []
    ddl_list = []
    finish_list = []
    
    for task in task_list:
        stimu_tab = task.extract_sensor_event(event_range)
        start_tab = [t+task.ERT for t in stimu_tab]
        ddl_tab = [t+task.ERT+task.ddl for t in stimu_tab]
        finish_tab = [t+task.ERT+task.exp_comp_t for t in stimu_tab]
        stimu_list.append(stimu_tab)
        req_list.append(start_tab)
        ddl_list.append(ddl_tab)
        finish_list.append(finish_tab)

    # print(req_list, ddl_list)

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    # plot timeline and task name 
    # and select color for the task automatically
    horizen_grid = set()
    vertical_grid = set()
    fig, ax = plt.subplots(figsize=(50, 15))
    vertical_offset = 0
    for i in range(len(task_list)):
        for stimu, s, f, ddl in zip(stimu_list[i], req_list[i], finish_list[i], ddl_list[i]):
            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            if s < plot_start and f > plot_start:
                s = plot_start
            if f > plot_end and s < plot_end:
                f = plot_end
            if s > plot_end or f < plot_start: 
                continue
            print("{}:{}-{}".format(task_list[i].name, s, f))
            horizen_grid.add(s)
            horizen_grid.add(f)
            vertical_grid.add(vertical_offset*vertical_grid_size)
            # plot task
            ax.broken_barh([(s, f-s)], (vertical_offset*vertical_grid_size, vertical_grid_size), facecolors=mcolors.XKCD_COLORS[colors[i]])
            # add up arrow and down arrow for stimu and ddl event respectively, with same color as the task
            # arrow length is 1.4 times of the vertical grid size,
            arrowprops=dict(arrowstyle="->")
            top_ = (vertical_offset-0.2)*vertical_grid_size
            bottom_ = (vertical_offset+1.2)*vertical_grid_size
            ax.annotate("", xy=(stimu, top_), xytext=(stimu, bottom_), arrowprops=arrowprops)
            ax.annotate("", xy=(ddl, bottom_), xytext=(ddl, top_), arrowprops=arrowprops)
            # add task name
            if not plot_legend:
                ax.text(s, vertical_offset*vertical_grid_size+0.001, task_list[i].name, ha='center', va='center', fontsize=7)
        vertical_offset += 1

    # np.arange(0, sim_time+time_grid_size, time_grid_size)
    # draw horizontal grid
    X, Y = np.meshgrid(np.array(list(horizen_grid)), np.arange(
        0, (vertical_offset+1)*vertical_grid_size, vertical_grid_size))
    # set x range
    ax.set(xlim=(plot_start, plot_end), xticks=np.arange(plot_start, plot_end+time_grid_size, time_grid_size*tick_dens),)
    ax.plot(X, Y, 'k', lw=0.5, alpha=0.5)
    # draw vertical grid
    X, Y = np.meshgrid(np.arange(plot_start, plot_end+time_grid_size, time_grid_size), np.array(list(vertical_grid)))
    ax.plot(X.T, Y.T, 'k', lw=0.5, alpha=0.5)
    
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
            save_path = path_parse[0] + ".pdf"        
            plt.savefig(save_path, bbox_inches='tight', **kwargs)
        else:
            kwargs["format"] = "pdf"
            save_path = save_path + ".pdf"        
            plt.savefig(save_path, bbox_inches='tight', **kwargs)

def creat_logical_graph(srcs:Dict[str, List[str]], ops:Dict[str, List[str]], sinks:Dict[str, List[str]]): 
    """
    Logical Graph:
        A logical graph is a directed graph where the nodes are Operators and the edges define 
        input/output-relationships of the operators and correspond to data streams or data sets. 
    Function: 
        create logical graph from srcs, ops, sinks
    """
    logical_graph_nx = nx.DiGraph()
    for src_n in srcs:
        logical_graph_nx.add_node(src_n, type="src")
        for op_n in srcs[src_n]:
            logical_graph_nx.add_node(op_n)
            logical_graph_nx.add_edge(src_n, op_n, type="control")
    for op_n in ops:
        logical_graph_nx.add_node(op_n, type="op")
        for sink_n in ops[op_n]:
            logical_graph_nx.add_node(sink_n)
            logical_graph_nx.add_edge(op_n, sink_n, type="data")
    for sink_n in sinks:
        logical_graph_nx.add_node(sink_n, type="sink")
        for op_n in sinks[sink_n]:
            logical_graph_nx.add_node(op_n)
            logical_graph_nx.add_edge(op_n, sink_n, type="data")
    return logical_graph_nx

def creat_physical_graph(logical_graph_nx:nx.DiGraph, f_gcd:int, profiling_filename:str="profiling.csv", 
                         taskattr_dict:Dict[str, TaskIntAttr]=None):
    """
    Physical Graph:
        A physical graph is the result of translating a Logical Graph for execution in a distributed runtime. 
        The nodes are Tasks and the edges indicate input/output-relationships or partitions of data streams or data sets.
    """
    physical_graph_nx = nx.DiGraph()

    node_parall_dict = {}
    if taskattr_dict is not None:
        for node_n, node_attr in taskattr_dict.items():
            node_attr:TaskIntAttr
            factor = node_attr.freq_division_factor
            copy_n = node_attr.thread_scaling_factor
            freq = int(node_attr.freq/f_gcd)
            node_parall_dict[node_n] = [copy_n, factor, freq]
    else:
        # extract the parallelism of each node
        df = pd.read_csv(profiling_filename, sep=",", index_col=0)
        for node_n in df.index:
            node_attr = df.loc[node_n].to_dict()
            factor = node_attr["Throuput factor (Spat.)"]
            copy_n = node_attr['Thread factor (Spat.)']
            freq = int(node_attr["Freq."]/f_gcd)
            node_parall_dict[node_n] = [copy_n, factor, freq]


    # add nodes
    for node_n, t in logical_graph_nx.nodes(data="type"):
        if node_n in node_parall_dict:
            copy_n, factor, freq = node_parall_dict[node_n]
            for copy_j in range(copy_n):
                for exe_k in range(factor):
                    node_name = node_n+"_"+str(copy_j)+"_"+str(exe_k)
                    physical_graph_nx.add_node(node_name, type=t)
                    # add control dependency
                    if exe_k < factor-1:
                        # physical_graph_nx.add_edge(node_name, node_n+"_"+str(copy_j)+"_"+str(exe_k+1))
                        pass
        else:
            physical_graph_nx.add_node(node_n, type=t)

    # add data dependency, rescale the parallelism
    for pred_n, succ_n, edge_attr in logical_graph_nx.edges(data=True):
        if pred_n in node_parall_dict and succ_n in node_parall_dict:
            pred_copy_n, pred_factor, pred_freq = node_parall_dict[pred_n]
            succ_copy_n, succ_factor, succ_freq = node_parall_dict[succ_n]
            for pred_copy_j in range(pred_copy_n):
                for succ_copy_j in range(succ_copy_n):   
                    build_node_relationship(physical_graph_nx, 
                                            pred_freq, succ_freq, 
                                            pred_factor, succ_factor,
                                            # pred_n+"_"+pred_copy_j, 
                                            f"{pred_n}_{pred_copy_j}",
                                            f"{succ_n}_{succ_copy_j}", 'repeat')
            # count the number of edges from pred to each succ
            for succ_exe_k in range(succ_factor):
                count = 0
                succ_node_name = succ_n+"_"+str(0)+"_"+str(succ_exe_k)
                for pred_exe_k in range(pred_factor):
                    pred_node_name = pred_n+"_"+str(0)+"_"+str(pred_exe_k)
                    if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                        count += 1
                # add attribute to the edge, type: data, factor: count
                # reDistPattn: one2one (count==1), 
                # reDistPattn: downscaling (count>1)
                if count == 1:
                    reDistPattn = "one2one"
                else:
                    reDistPattn = "downscaling"
                for pred_copy_j in range(pred_copy_n):
                    for succ_copy_j in range(succ_copy_n): 
                        for pred_exe_k in range(pred_factor):
                            succ_node_name = succ_n+"_"+str(succ_copy_j)+"_"+str(succ_exe_k)
                            pred_node_name = pred_n+"_"+str(pred_copy_j)+"_"+str(pred_exe_k)
                            if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                                physical_graph_nx.edges[pred_node_name, succ_node_name]["reDistPattn"] = reDistPattn
                                physical_graph_nx.edges[pred_node_name, succ_node_name]["type"] = "data"
                                physical_graph_nx.edges[pred_node_name, succ_node_name]["factor"] = count
            
            for pred_exe_k in range(pred_factor):
                count = 0
                pred_node_name = pred_n+"_"+str(0)+"_"+str(pred_exe_k)
                for succ_exe_k in range(succ_factor):
                    succ_node_name = succ_n+"_"+str(0)+"_"+str(succ_exe_k)
                    if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                        count += 1
                        if count > 1:
                            break
                # add attribute to the edge, type: data, factor: count
                # reDistPattn: upscaling (count>1)
                if count > 1:
                    reDistPattn = "upscaling"
                    for pred_copy_j in range(pred_copy_n):
                        for succ_copy_j in range(succ_copy_n): 
                            for succ_exe_k in range(succ_factor):
                                succ_node_name = succ_n+"_"+str(succ_copy_j)+"_"+str(succ_exe_k)
                                pred_node_name = pred_n+"_"+str(pred_copy_j)+"_"+str(pred_exe_k)
                                if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                                    physical_graph_nx.edges[pred_node_name, succ_node_name]["reDistPattn"] = reDistPattn
                                    physical_graph_nx.edges[pred_node_name, succ_node_name]["type"] = "data"
                                    physical_graph_nx.edges[pred_node_name, succ_node_name]["factor"] = count

        elif pred_n in node_parall_dict and succ_n not in node_parall_dict:
            pred_copy_n, pred_factor, pred_freq = node_parall_dict[pred_n]
            for pred_copy_j in range(pred_copy_n):
                for exe_k in range(pred_factor):
                    pred_node_name = pred_n+"_"+str(pred_copy_j)+"_"+str(exe_k)
                    physical_graph_nx.add_edge(pred_node_name, succ_n, type="control", reDistPattn="none")
        elif pred_n not in node_parall_dict and succ_n in node_parall_dict:
            succ_copy_n, succ_factor, succ_freq = node_parall_dict[succ_n]
            for succ_copy_j in range(succ_copy_n): 
                for exe_k in range(succ_factor):
                    succ_node_name = succ_n+"_"+str(succ_copy_j)+"_"+str(exe_k)
                    physical_graph_nx.add_edge(pred_n, succ_node_name, type="control", reDistPattn="none")
        else:
            physical_graph_nx.add_edge(pred_n, succ_n, type="control", reDistPattn="none")
    
    return physical_graph_nx

def creat_jobTask_graph(task_graph:Dict[str, List[str]], f_gcd, plot:bool=False, profiling_filename:str="profiling.csv"):
    # create task graph from task_graph
    task_graph_nx = nx.DiGraph(task_graph)

    job_graph_nx = nx.DiGraph()
    df = pd.read_csv(profiling_filename, sep=",", index_col=0) 

    for task_n in task_graph: 
        if task_n in df.index:
            task_attr = df.loc[task_n].to_dict()
            factor = task_attr["Throuput factor (Spat.)"]
            copy_n = task_attr['Thread factor (Spat.)']
            freq  = task_attr["Freq."]/f_gcd
            num_per_group = int(np.ceil(freq / factor))
            for copy_j in range(copy_n):
                for exe_k in range(factor):
                    task_name = task_n+"_"+str(copy_j)+"_"+str(exe_k)
                    job_graph_nx.add_node(task_name)
                    # add control dependency
                    if exe_k < factor-1:
                        # job_graph_nx.add_edge(task_name, task_n+"_"+str(copy_j)+"_"+str(exe_k+1), type="control")
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
                succ_factor = df.loc[succ_n]["Throuput factor (Spat.)"]
                succ_copy_n = df.loc[succ_n]['Thread factor (Spat.)']
                succ_freq  = int(df.loc[succ_n]["Freq."]/f_gcd)
                succ_num_per_group = int(np.ceil(succ_freq/succ_factor))
                for copy_j in range(succ_copy_n):
                    for exe_k in range(succ_freq):
                        no_succ = int(exe_k // succ_num_per_group)
                        succ_job_name = succ_n+"_"+str(copy_j)+"_"+str(no_succ)
                        
                        # just like quantization
                        succ_t = exe_k/succ_freq
                        pred_t = int(succ_t*freq)

                        no_ = int(pred_t // num_per_group)
                        for copy_i in range(copy_n):
                            pre_job_name = task_n+"_"+str(copy_i)+"_"+str(no_)
                            job_graph_nx.add_edge(pre_job_name, succ_job_name, type="data")
                            # add attribute "reDistPattn", to classify the redistributing pattern
                            # downstream <- upstream
                            if succ_factor < factor:
                                # set reDistPattn as "downscaling"
                                job_graph_nx[pre_job_name][succ_job_name]["reDistPattn"] = "downscaling"
                            elif succ_factor > factor:
                                # set reDistPattn as "upscalling"
                                job_graph_nx[pre_job_name][succ_job_name]["reDistPattn"] = "upscaling"
                            else:
                                # set reDistPattn as "one2one" 
                                job_graph_nx[pre_job_name][succ_job_name]["reDistPattn"] = "one2one"
        else:
            job_graph_nx.add_node(task_n)
            edge_type = "control" if task_n == "Entry" else "data"
            for succ_n in task_graph[task_n]:
                if succ_n in df.index:
                    task_attr = df.loc[succ_n].to_dict()
                    factor = task_attr["Throuput factor (Spat.)"]
                    copy_n = task_attr['Thread factor (Spat.)']
                    for copy_j in range(copy_n):
                        for exe_k in range(factor):
                            succ_job_name = succ_n+"_"+str(copy_j)+"_"+str(exe_k)
                            job_graph_nx.add_edge(task_n, succ_job_name, type=edge_type)
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

    """
    Basic properties of each tasks:
        1. id
        2. name
        3. flops_atomic
        4. freq
        5. thread
        6. parallel_type
        7. parallel_range
        8. timing_flag
        9. paramter_size
        8. forward_size
        9. input_size
        10. criticality

        ID,Task (chain) names,Flops (G),Freq.,Thread,Parallel type,Parallel range,Timing_flag,
        Pre-assigned,Priority,Criti_flag,Cbs_en,Trigger_mode,
        Parallel_type,Parallel_range,Parallel_range_prealloc
    Hand-crafted properties of each tasks:
        1. freq_division_factor 
        2. thread_scaling_factor 
        3. var_factor
        
        Thread factor (Tmp.),Throuput factor (Tmp.),Throuput factor (Spat.),Thread factor (Spat.),Var. factor (Tmp.),

    Deduced properties (compute with complex formula): 
        1. N_exec
        2. flops
        3. required_resource_size
        4. pre_assigned_resource.main_size 
        5. pre_assigned_resource.RDA_size 

        Number of execution,Flops on path (G),Total Ops(typical)(T),Total Ops(Max)(T),Cores/Req.,Max required Cores,RDA./Req.,
        Dynamic loading overhead,Forward Data Size

    Trival properties:
        1. ERT
        2. ddl
        3. period

        T release (ms),DDL (ms),Cores/Req.,No-stall latency (ms),Util.,Min required cores,Equavalent used cores,No-stall Bandwidth

    """
def load_taskint(profiling_filename:str="profiling.csv", 
                 freq_div_en:bool=True, thread_scaling_en:bool=True, 
                 plot:bool = False, verbose: bool = False) -> Dict[str, TaskInt]:

    df = pd.read_csv(profiling_filename, sep=",", index_col=0) 
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

    # calculate the gcd of all the task's frequency
    f_gcd = np.gcd.reduce(df["Freq."].to_list())
    hyper_p = 1/f_gcd

    for task_n in df.T:
        # print(task_n)
        task_attr = df.loc[task_n].to_dict()
        task_attr["Timing_flag"] = "deadline" if task_attr["Timing_flag"]=="DDL" else "realtime"
        task_attr["Resource Type"] = "stationary" if task_attr["Resource Type"]=="S" else "moveable"
        task_attr["Pre-assigned"] = False if task_attr["Pre-assigned"]=="N" else True
        if thread_scaling_en:
            thread_scaling_factor = task_attr["Thread factor (Spat.)"]
        else:
            thread_scaling_factor = 1
        parallel_cfg = extract_parallel_cfg(task_attr, "runtime")
        parallel_cfg_compile = extract_parallel_cfg(task_attr, "compile")
        for thread_j in range(thread_scaling_factor):
            # for exe_k in range(task_attr["Throuput factor (Spat.)"]):
                # T = task_attr["Throuput factor (Spat.)"]/task_attr["Freq."]
                # phase = exe_k/task_attr["Freq."]
                T = 1/task_attr["Freq."]
                phase = 0
                flops_on_path = task_attr["Flops on path (G)"]/1e3
                # flops_on_path = task_attr["Flops (G)"]*task_attr["Thread factor (Tmp.)"]/1e3
                task = TaskInt(
                    task_name=task_n + (f"_{thread_j}" if thread_scaling_en else ''),
                    task_id=task_id, timing_flag=task_attr["Timing_flag"], 
                    ERT=task_attr["T release (ms)"]/1000, ddl=(task_attr['DDL (ms)']-task_attr["T release (ms)"])/1000, 
                    period=T, 
                    exp_comp_t=task_attr['Expected Latency (ms)']/1000, i_offset=phase, jitter_max=0,
                    flops=flops_on_path, task_flag=task_attr["Resource Type"], 
                    pre_assigned_resource_flag=task_attr["Pre-assigned"]>0, 
                    RDA_size=task_attr['RDA./Req.'], main_size=task_attr['Cores/Req.'], seq_cpu_time=flops_on_path,
                    op_cpu_time=flops_on_path, op_io_time=1e-6,
                    criti_flag="soft" if task_attr["Criti_flag"]=='S' else "hard", 
                    cbs_en=True, # if task_attr["Cbs_en"]=='Y' else False, 
                    trigger_mode=task_attr["Trigger_mode"], 
                    parallel_cfg=parallel_cfg,
                    parallel_cfg_compile=parallel_cfg_compile,

                )
                task.freq = task_attr["Freq."]
                task.thread_scaling_factor = task_attr["Thread factor (Spat.)"]
                task.freq_division_factor = task_attr["Throuput factor (Spat.)"] 
                task.var_factor = task_attr["Var. factor (Tmp.)"] 
                task.required_resource_size = task_attr['Cores/Req.']
                if freq_div_en:
                    division_factor = task_attr["Throuput factor (Spat.)"]
                    freq_div_mode = 'interleave' if task_attr["Freq."]/f_gcd <= task_attr["Throuput factor (Spat.)"] else 'repeat'
                    task_list = task.freq_division(division_factor, hyper_p, mode=freq_div_mode)
                else:
                    division_factor = 1
                    task_list = [task]
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


                # print(str(task))
                # task_id += 1
                task_id += division_factor
                # task_dict.update({task.name: task})
                task_dict.update({task.name:task for task in task_list})
        if plot:
            save_path="task_static_timeline.pdf"
            # np.arange(0, sim_time+time_grid_size, time_grid_size)
            X, Y = np.meshgrid(np.array(list(horizen_grid)), np.arange(
                0, (vertical_offset+1)*vertical_grid_size, vertical_grid_size))
            t_max = max(horizen_grid)+time_grid_size
            ax.set(xlim=(0, t_max), xticks=np.arange(0, t_max, time_grid_size),)
            ax.plot(X, Y, 'k', lw=0.5, alpha=0.5)
            plt.savefig(save_path, format="pdf")

    return task_dict, f_gcd


def load_taskattrib(profiling_filename:str="profiling.csv", verbose: bool = False) -> Dict[str, TaskIntAttr]:

    df = pd.read_csv(profiling_filename, sep=",", index_col=0) 
    if verbose:
        print(df)
    task_dict = {}
    task_id = 0
    # print(task_attr_dict)

    # calculate the gcd of all the task's frequency
    f_gcd = np.gcd.reduce(df["Freq."].to_list())

    for task_n in df.T:
        # print(task_n)
        task_attr = df.loc[task_n].to_dict()
        task_attr["Timing_flag"] = "deadline" if task_attr["Timing_flag"]=="DDL" else "realtime"
        task_attr["Resource Type"] = "stationary" if task_attr["Resource Type"]=="S" else "moveable"
        task_attr["Pre-assigned"] = False if task_attr["Pre-assigned"]=="N" else True
        thread_scaling_factor = task_attr["Thread factor (Spat.)"]
        parallel_cfg = extract_parallel_cfg(task_attr, "runtime")
        parallel_cfg_compile = extract_parallel_cfg(task_attr, "compile")
        period = 1/task_attr["Freq."]
        flops_on_path = task_attr["Flops on path (G)"]/1e3
        # flops_on_path = task_attr["Flops (G)"]*task_attr["Thread factor (Tmp.)"]/1e3
        # T = task_attr["Throuput factor (Spat.)"]/task_attr["Freq."]
        # phase = exe_k/task_attr["Freq."]

        task_name=task_n

        timing_flag=task_attr["Timing_flag"]        
        ERT=task_attr["T release (ms)"]/1000
        ddl=(task_attr['DDL (ms)']-task_attr["T release (ms)"])/1000
        RDA_size=task_attr['RDA./Req.']
        main_size=task_attr['Cores/Req.']
        exp_comp_t=task_attr['Expected Latency (ms)']/1000
        required_resource_size = task_attr['Cores/Req.']
        i_offset=0
        task_flag=task_attr["Resource Type"]
        pre_assigned_resource_flag=task_attr["Pre-assigned"]>0
                
        # flops=flops_on_path        
        # seq_cpu_time=flops_on_path
        # op_cpu_time=flops_on_path
        op_io_time=1e-6
        jitter_max=0
        criti_flag="soft" if task_attr["Criti_flag"]=='S' else "hard"
        
        cbs_en=True
        # if task_attr["Cbs_en"]=='Y' else False
        
        trigger_mode=task_attr["Trigger_mode"] # event-triggered or periodic
        freq = task_attr["Freq."]
        thread_scaling_factor = task_attr["Thread factor (Spat.)"]
        freq_division_factor = task_attr["Throuput factor (Spat.)"] 
        var_factor = task_attr["Var. factor (Tmp.)"] 
        
        task = TaskIntAttr(name=task_name, 
                        freq=freq, 
                        timing_flag=timing_flag, 
                        criticality = criti_flag, # unused
                        trigger_mode=trigger_mode, # unused
                        
                        core_max = parallel_cfg["max"] if "max" in parallel_cfg else 1e3,
                        core_min = parallel_cfg["min"] if "min" in parallel_cfg else 0,
                        core_list = parallel_cfg["list"] if "list" in parallel_cfg else None,
                        parallel_mode = parallel_cfg["mode"] if "mode" in parallel_cfg else None,

                        core_max_compile = parallel_cfg_compile["max"] if "max" in parallel_cfg else 1e3,
                        core_min_compile = parallel_cfg_compile["min"] if "min" in parallel_cfg else 0,
                        core_list_compile = parallel_cfg_compile["list"] if "list" in parallel_cfg else None,

                        thread_scaling_factor=thread_scaling_factor, # unused
                        freq_division_factor=freq_division_factor, 
                        var_factor=var_factor, # unused

                        jitter_max=jitter_max, # unused
                        flops=flops_on_path, 
                        task_flag=task_flag, # unused
                        pre_assigned_resource_flag=pre_assigned_resource_flag, # unused
                        )
        # print(str(task))
        task_id += 1
        task_dict.update({task.name: task})

    return task_dict, f_gcd

def gen_taskint_from_cfg(taskattr_dict:Dict[str, TaskIntAttr], f_gcd: int,
                 plot:bool = False, verbose: bool = False) -> Dict[str, TaskInt]:

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

    hyper_p = 1/f_gcd
    for task_n,task_attr in taskattr_dict.items():
        # print(task_n)
        task_attr:TaskIntAttr
        thread_scaling_factor = task_attr.thread_scaling_factor
        for thread_j in range(thread_scaling_factor):
            # for exe_k in range(task_attr["Throuput factor (Spat.)"]):
                # T = task_attr["Throuput factor (Spat.)"]/task_attr["Freq."]
                # phase = exe_k/task_attr["Freq."]
                T = 1/task_attr.freq
                phase = 0
                flops_on_path = task_attr.flops
                task = TaskInt(
                    task_name=task_attr.name + f"_{thread_j}", 
                    task_id=task_id, timing_flag=task_attr.timing_flag,
                    ERT=task_attr.ERT, 
                    ddl=task_attr.ddl, 
                    period=T, 
                    exp_comp_t=task_attr.exp_comp_t, 
                    i_offset=phase, jitter_max=0,
                    flops=task_attr.flops, 
                    task_flag=task_attr.task_flag, 
                    pre_assigned_resource_flag=task_attr.pre_assigned_resource_flag, 
                    RDA_size=task_attr.rda_size, 
                    main_size=task_attr.main_size, 
                    seq_cpu_time=flops_on_path,
                    op_cpu_time=task_attr.flops, op_io_time=1e-6,
                    criti_flag=task_attr.criticality, 
                    cbs_en=True, # if task_attr["Cbs_en"]=='Y' else False, 
                    trigger_mode=task_attr.trigger_mode, 
                    parallel_cfg={"max":task_attr.core_max, "min":task_attr.core_min, "list":task_attr.core_list, "mode":task_attr.parallel_mode},
                    parallel_cfg_compile={"max":task_attr.core_max_compile, "min":task_attr.core_min_compile, "list":task_attr.core_list_compile},
                )
                task.freq = task_attr.freq
                task.thread_scaling_factor = task_attr.thread_scaling_factor
                task.required_resource_size = task_attr.main_size

                task.freq_division_factor = task_attr.freq_division_factor
                task.var_factor = task_attr.var_factor
                task.required_resource_size = task_attr.main_size

                division_factor = task_attr.freq_division_factor
                freq_div_mode = 'interleave' if task_attr.freq/f_gcd <= task_attr.freq_division_factor else 'repeat'
                task_list = task.freq_division(division_factor, hyper_p, mode=freq_div_mode)

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

                # print(str(task))
                # task_id += 1
                task_id += division_factor
                # task_dict.update({task.name: task})
                task_dict.update({task.name:task for task in task_list})
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

def gen_workloads(args, slack_threshold):
    taskattr_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=args.verbose) 
    hyper_p = 1/f_gcd
    if args.aux_scale_factor > 1:
        for node, taskattr in taskattr_dict.items():
            # scale up the thread scaling factor
            if taskattr.timing_flag == "realtime":
                taskattr.thread_scaling_factor *= args.aux_scale_factor

    logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
    duduce_cfg(taskattr_dict, f_gcd, hyper_p, logical_graph_nx, task_graph_srcs, 
                         task_graph_sinks, slack_threshold, args.e2e_latency, args.temporal_rda_ratio, args.wsc_slack_ratio)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)
    physical_graph_nx = creat_physical_graph(logical_graph_nx, int(f_gcd), taskattr_dict=taskattr_dict)
    init_depen(glb_n_task_dict, physical_graph_nx, verbose=args.verbose)
    return hyper_p,glb_n_task_dict,physical_graph_nx


def extract_parallel_cfg(task_attr, mode="runtime"):
    parallel_cfg = {}
    if mode == "runtime":
        tgt_title = "Parallel_range"
    elif mode == "compile":
        tgt_title = "Parallel_range_prealloc"
    if task_attr["Parallel_type"] == "Upb":
        parallel_cfg["mode"] = "upb"
        parallel_cfg["max"] = int(task_attr[tgt_title])
    elif task_attr["Parallel_type"] == "Lwb":
        parallel_cfg["mode"] = "lwb"
        parallel_cfg["min"] = int(task_attr[tgt_title])
    elif task_attr["Parallel_type"] == "Range":
        parallel_cfg["mode"] = "range"
                    # split the range into two parts
        parallel_cfg["min"], parallel_cfg["max"] = map(int, task_attr[tgt_title].split(","))
    elif task_attr["Parallel_type"] == "list":
        parallel_cfg["mode"] = "list"
        parallel_cfg["list"] = map(int, task_attr[tgt_title].split(","))
    return parallel_cfg

# initialize dependency list
def init_depen(taskJobs:Union[Dict[str, Union[TaskInt,ProcessInt]], List[Union[TaskInt,ProcessInt]]], job_graph_nx:nx.DiGraph, verbose=False):
    # if taskJobs is a list, convert it to a dict
    if isinstance(taskJobs, list):
        if taskJobs[0].__class__.__name__ == "ProcessInt":
            taskJobs = {i.task.name:i for i in taskJobs}
        elif taskJobs[0].__class__.__name__ == "TaskInt":
            taskJobs = {i.name:i for i in taskJobs}
    
    for job_n, job in taskJobs.items():

        # group the edge with attribution "reDistPattn" eq "downscaling"
        for pre_n, datadict in job_graph_nx.pred[job_n].items():
            dep_t = datadict["type"]
            attr = copy.deepcopy(datadict)
            attr.update({"valid":False})
            if dep_t == "data":
                attr.update({"event_queue":TaskQueue(sort_f=lambda x: x.ctx.get_timestamp(), descending=False)})
                job.pred_data.update({pre_n:attr})
            elif dep_t == "control":
                attr.update({"event_queue":TaskQueue(sort_f=lambda x: x.get_timestamp(), descending=False)})
                job.pred_ctrl.update({pre_n:attr})
            else:
                raise Exception("Unknown dependency type")
        for succ_n, datadict in job_graph_nx.succ[job_n].items():
            dep_t = datadict["type"]
            attr = copy.deepcopy(datadict)
            attr.update({"valid":False})
            if dep_t == "data":
                job.succ_data.update({succ_n:attr})
            elif dep_t == "control":
                job.succ_ctrl.update({succ_n:attr})
            else:
                raise Exception("Unknown dependency type")
        
        if verbose:
            print(job_n, job.pred_data, job.pred_ctrl, job.succ_data, job.succ_ctrl)

def init_affinity(taskJobs:Union[Dict[str, Union[TaskInt,ProcessInt]], List[Union[TaskInt,ProcessInt]]]=None, 
                  mode="task",
                  job_graph_nx:nx.DiGraph=None, 
                  task_graph_nx:nx.DiGraph=None,
                  task_custom_affinity_cfg:Dict[str, List[str]]=None, 
                  job_custom_affinity_cfg:Dict[str, List[str]]=None,
                  verbose=False):
    # affinity task/job(s) selection machanism:
    #   Basic principle: 
    #       Besides the custom affinity configuration, prioritize the tasks/jobs with the highest probability 
    #       of running simultaneously.
    #   Positive affinity:
    #       1. the predecesor of the task/job
    #       2. the successor of the task/job
    #   negative affinity:
    #       1. the slibling of the task/job
    pos_affinity_cfg = {}
    neg_affinity_cfg = {}
    if mode == "task":
        assert task_graph_nx is not None
        for node_n in task_graph_nx.nodes():
            # get predecesor, successor and slibling
            pred_n_list = [pred for pred in task_graph_nx.pred[node_n].keys() if pred not in task_graph_srcs.keys()]
            succ_n_list = [succ for succ in task_graph_nx.succ[node_n].keys() if succ not in task_graph_sinks.keys()]
            slib_n_list = [sibling for pred_n in pred_n_list for sibling in task_graph_nx.succ[pred_n].keys() if sibling != node_n and sibling not in task_graph_sinks.keys()]
            
            # 0. custom affinity configuration
            pos_affinity_cfg.update({node_n:task_custom_affinity_cfg[node_n]})
            # 1. the predecesor of the task/job
            pos_affinity_cfg.update({node_n:pred_n_list})
            # 2. the successor of the task/job
            pos_affinity_cfg[node_n].extend(succ_n_list)
            # 3. the slibling of the task/job
            neg_affinity_cfg.update({node_n:slib_n_list})
    elif mode == "job":
        assert taskJobs is not None
        if isinstance(taskJobs, list):
            if taskJobs[0].__class__.__name__ == "ProcessInt":
                taskJobs = {i.task.name:i for i in taskJobs}
            elif taskJobs[0].__class__.__name__ == "TaskInt":
                taskJobs = {i.name:i for i in taskJobs}
        assert job_graph_nx is not None
        for job_n, job in taskJobs.items():
            # get predecesor, successor and slibling
            pred_n_list = [pred_n for pred_n in job_graph_nx.pred[job_n].keys() if pred_n not in task_graph_srcs.keys()]
            succ_n_list = [succ_n for succ_n in job_graph_nx.succ[job_n].keys() if succ_n not in task_graph_sinks.keys()]
            slib_n_list = [sibling for pred_n in pred_n_list for sibling in job_graph_nx.succ[pred_n].keys() if sibling != job_n and sibling not in task_graph_sinks.keys()]
            
            # 0. custom affinity configuration
            if job_custom_affinity_cfg is not None:
                pos_affinity_cfg.update({job_n:job_custom_affinity_cfg[job_n]})
            # 1. the predecesor of the task/job
            pos_affinity_cfg.update({job_n:pred_n_list})
            # 2. the successor of the task/job
            pos_affinity_cfg[job_n].extend(succ_n_list)
            # 3. the slibling of the task/job
            neg_affinity_cfg.update({job_n:slib_n_list})

            # initialize task affinity list
            job.task.affinity_n = pos_affinity_cfg[job_n]

            # get affinity target id
            # TODO: bug here， key error when the affinity target is not in the pid_idx
            job.task.affinity = [taskJobs[n].task.id for n in pos_affinity_cfg[job_n] if n in taskJobs]

    else:
        raise Exception("Unknown mode")
    return pos_affinity_cfg, neg_affinity_cfg

def redist_ert_dll(taskJobs:Union[Dict[str, Union[TaskInt,ProcessInt]], List[Union[TaskInt,ProcessInt]]],
        logical_graph_nx:nx.DiGraph=None, temporal_rda_ratio=0, sched_step_comp=0, 
        comm_compen_en=False, profiling_filename:str="profiling.csv", verbose=False):

    df:pd.DataFrame = pd.read_csv(profiling_filename, sep=",", index_col=0) 
    comp_time: Dict[str, float] = {task_n:df.loc[task_n, "Expected Latency (ms)"]/1000 for task_n in df.T}
    io_time: Dict[str, float] = {task_n:1e-6 for task_n in df.T}
    task_type: Dict[str, str] = {task_n:df.loc[task_n, "Timing_flag"] for task_n in df.T}

    ert, ddl = estim_release_dll_time(logical_graph_nx, comp_time, io_time, task_type,
                                      temporal_rda_ratio, sched_step_comp, comm_compen_en, profiling_filename, verbose)
    # if taskJobs is a list, convert it to a dict
    if isinstance(taskJobs, list):
        if taskJobs[0].__class__.__name__ == "ProcessInt":
            taskJobs = {i.task.name:i for i in taskJobs}
        elif taskJobs[0].__class__.__name__ == "TaskInt":
            taskJobs = {i.name:i for i in taskJobs}
    
    for job_n, job in taskJobs.items():
        # parse the sub task number from the job name
        thread_n = job_n.split('_')[-2]
        troughput_n = job_n.split('_')[-1]
        task_n = job_n.replace("_"+thread_n, "").replace("_"+troughput_n, "")
        job.ERT = ert[task_n]
        job.ddl = ddl[task_n] - ert[task_n]

def create_init_p_list(tasks: Union[List[TaskInt], Dict[str, TaskInt]], verbose:bool):
    if isinstance(tasks, list):
        task_list = tasks
    elif isinstance(tasks, dict):
        task_list = list(tasks.values())

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
        # init the fork_pid_candi with the random value
        p.fork_pid_candi = p.pid * fork_pid_base + np.random.randint(0, max_fork_pid, size=max_fork_candi_num)
        p.fork_pid_candi = np.unique(p.fork_pid_candi).tolist()
    return init_p_list


if __name__ == "__main__": 
    import argparse
    import numpy as np 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--test_case", type=str, default="all", help="task name")
    parser.add_argument("--plot", action="store_true", help="plot the task timeline")
    parser.add_argument("--bin_pack", action="store_true", help="plot the task timeline")
    parser.add_argument("--test_all", default=False, help="test all the task")
    parser.add_argument("--profiling_filename", type=str, default="profiling.csv", help="profiling filename")
    args = parser.parse_args() 

    if args.profiling_filename == "profiling.csv":
        cfg_n = "heavy"
    else:
        cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]

    glb_n_task_dict, f_gcd = load_taskint(args.profiling_filename, verbose=args.verbose) 
    hyper_p = 1/f_gcd

    if args.test_case == "all":
        args.test_all = True
    sim_step = min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32

    if args.test_case == "ert_ddl" or args.test_all:
        logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
        ert, ddl = estim_release_dll_time(logical_graph_nx, temporal_rda_ratio=0.05, sched_step_comp=sim_step, 
                                          profiling_filename=args.profiling_filename, verbose=args.verbose)
        df = pd.read_csv(args.profiling_filename, sep=",", index_col=0) 
        for task_n in ert: 
            print(f"W/ T_comm: {task_n}: {ert[task_n]:.8f} - {ddl[task_n]:.8f}({ddl[task_n]-ert[task_n]:.8f})")
            if task_n in df.T:
                print(f"w/o T_comm: {task_n}: {df.loc[task_n, 'T release (ms)']/1000:.8f} - {df.loc[task_n, 'DDL (ms)']/1000:.8f}({df.loc[task_n, 'DDL (ms)']/1000-df.loc[task_n, 'T release (ms)']/1000:.8f})") 
            print("------------------")
    elif args.test_case == "timeline" or args.test_all:
        vis_task_static_timeline(list(glb_n_task_dict.values()), save=True, save_path="plot/{cfg_n}/task_static_timeline_cyclic.pdf", hyper_p=hyper_p, n_p=2, warmup=False, drain=True, )
    elif args.test_case == "liveness" or args.test_all:
        vis_task_static_timeline(list(glb_n_task_dict.values()), save=True, save_path="plot/{cfg_n}/task_liveness_timeline_cyclic.svg", 
        hyper_p=hyper_p, n_p=1, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
        txt_size=40, tick_dens=4)
    elif args.test_case == "graph" or args.test_all:
        # task_graph_nx, job_graph_nx = creat_jobTask_graph(task_graph, int(f_gcd), plot=True)
        # init_depen(task_dict, job_graph_nx, verbose=args.verbose)
        logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
        physical_graph_nx = creat_physical_graph(logical_graph_nx, int(f_gcd), args.profiling_filename)
        init_depen(glb_n_task_dict, physical_graph_nx, verbose=args.verbose)

        node_color_map = {"op": "red", "sink": "blue", "src": "green"}
        edge_color_map = {"data": "red", "control": "blue"}
        node_colors = [node_color_map[d] for n, d in logical_graph_nx.nodes(data="type")]
        edge_colors = [edge_color_map[d] for u,v,d in logical_graph_nx.edges(data="type")]
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        for layer, nodes in enumerate(nx.topological_generations(logical_graph_nx)):
            for node in nodes:
                logical_graph_nx.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(logical_graph_nx, subset_key="layer")
        # text with 45 degree rotation
        nx.draw(logical_graph_nx, pos, with_labels=False, node_size=100, node_color=node_colors, edge_color=edge_colors, font_size=10, ax=ax1)
        text = nx.draw_networkx_labels(logical_graph_nx, pos, font_size=10, ax=ax1)
        for _, t in text.items():
            t.set_rotation(60)
        fig.tight_layout()


        node_colors = [node_color_map[d] for n, d in physical_graph_nx.nodes(data="type")]
        edge_colors = [edge_color_map[d] for u,v,d in physical_graph_nx.edges(data="type")]
        for layer, nodes in enumerate(nx.topological_generations(physical_graph_nx)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                physical_graph_nx.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(physical_graph_nx, subset_key="layer")
        nx.draw(physical_graph_nx, pos, with_labels=False, node_size=100, node_color=node_colors, edge_color=edge_colors, font_size=10, ax=ax2)
        text = nx.draw_networkx_labels(physical_graph_nx, pos, font_size=10, ax=ax2)
        for _, t in text.items():
            t.set_rotation(60)
        fig.tight_layout()
        plt.savefig("plot/{cfg_n}/jobTask_graph.pdf", format="pdf")
