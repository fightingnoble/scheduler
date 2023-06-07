from __future__ import annotations

from typing import Union, List, Dict, Iterator, Callable, Union
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from global_var import *
from model.lru import LRUCache
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from task.task_agent import TaskInt
from model.task_queue_agent import TaskQueue 
from task.task_agent import ProcessInt
from task.graph_scaling import build_node_relationship

# 'ID', 'Task (chain) names', 'Flops on path', 'Expected Latency (ms)', 'T release', 'Freq.', 'DDL', 'Cores/Req.', 
# 'Throuput factor (S)', 'Thread factor (S)', 'Min required cores', 'Timing_flag', 'Max required Cores', 'RDA./Req.', 'Resource Type', 'Pre-assigned', 'Priority'

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

def creat_physical_graph(logical_graph_nx:nx.DiGraph, f_gcd:int):
    """
    Physical Graph:
        A physical graph is the result of translating a Logical Graph for execution in a distributed runtime. 
        The nodes are Tasks and the edges indicate input/output-relationships or partitions of data streams or data sets.
    """
    physical_graph_nx = nx.DiGraph()

    # extract the parallelism of each node
    df = pd.read_csv("profiling.csv", sep=",", index_col=0)
    node_parall_dict = {}
    for node_n in df.index:
        node_attr = df.loc[node_n].to_dict()
        factor = node_attr["Throuput factor (S)"]
        freq = int(node_attr["Freq."]/f_gcd)
        node_parall_dict[node_n] = [factor, freq]


    for node_n, t in logical_graph_nx.nodes(data="type"):
        if node_n in df.index:
            node_attr = df.loc[node_n].to_dict()
            factor = node_attr["Throuput factor (S)"]
            for i in range(factor):
                node_name = node_n+"_"+str(i)
                physical_graph_nx.add_node(node_name, type=t)
                # add control dependency
                if i < factor-1:
                    # physical_graph_nx.add_edge(node_name, node_n+"_"+str(i+1))
                    pass
        else:
            physical_graph_nx.add_node(node_n, type=t)

    # add data dependency, rescale the parallelism
    for pred_n, succ_n, edge_attr in logical_graph_nx.edges(data=True):
        if pred_n in df.index and succ_n in df.index:
            pred_factor, pred_freq = node_parall_dict[pred_n]
            succ_factor, succ_freq = node_parall_dict[succ_n]
            build_node_relationship(physical_graph_nx, 
                                    pred_freq, succ_freq, 
                                    pred_factor, succ_factor,
                                    pred_n, succ_n,)
            # count the number of edges from pred to each succ
            for i in range(succ_factor):
                count = 0
                succ_node_name = succ_n+"_"+str(i)
                for j in range(pred_factor):
                    pred_node_name = pred_n+"_"+str(j)
                    if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                        count += 1
                # add attribute to the edge, type: data, factor: count
                # reDistPattn: one2one (count==1), 
                # reDistPattn: downscaling (count>1)
                if count == 1:
                    reDistPattn = "one2one"
                else:
                    reDistPattn = "downscaling"
                for j in range(pred_factor):
                    pred_node_name = pred_n+"_"+str(j)
                    if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                        physical_graph_nx.edges[pred_node_name, succ_node_name]["reDistPattn"] = reDistPattn
                        physical_graph_nx.edges[pred_node_name, succ_node_name]["type"] = "data"
                        physical_graph_nx.edges[pred_node_name, succ_node_name]["factor"] = count
            
            for i in range(pred_factor):
                count = 0
                pred_node_name = pred_n+"_"+str(i)
                for j in range(succ_factor):
                    succ_node_name = succ_n+"_"+str(j)
                    if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                        count += 1
                        if count > 1:
                            break
                # add attribute to the edge, type: data, factor: count
                # reDistPattn: upscaling (count>1)
                if count > 1:
                    reDistPattn = "upscaling"
                    for j in range(succ_factor):
                        succ_node_name = succ_n+"_"+str(j)
                        if physical_graph_nx.has_edge(pred_node_name, succ_node_name):
                            physical_graph_nx.edges[pred_node_name, succ_node_name]["reDistPattn"] = reDistPattn
                            physical_graph_nx.edges[pred_node_name, succ_node_name]["type"] = "data"
                            physical_graph_nx.edges[pred_node_name, succ_node_name]["factor"] = count

        elif pred_n in df.index and succ_n not in df.index:
            pred_factor, pred_freq = node_parall_dict[pred_n]
            for i in range(pred_factor):
                pred_node_name = pred_n+"_"+str(i)
                physical_graph_nx.add_edge(pred_node_name, succ_n, type="control", reDistPattn="none")
        elif pred_n not in df.index and succ_n in df.index:
            succ_factor, succ_freq = node_parall_dict[succ_n]
            for i in range(succ_factor):
                succ_node_name = succ_n+"_"+str(i)
                physical_graph_nx.add_edge(pred_n, succ_node_name, type="control", reDistPattn="none")
        else:
            physical_graph_nx.add_edge(pred_n, succ_n, type="control", reDistPattn="none")
    
    return physical_graph_nx

def creat_jobTask_graph(task_graph:Dict[str, List[str]], f_gcd, plot:bool=False):
    # create task graph from task_graph
    task_graph_nx = nx.DiGraph(task_graph)

    job_graph_nx = nx.DiGraph()
    df = pd.read_csv("profiling.csv", sep=",", index_col=0) 

    for task_n in task_graph: 
        if task_n in df.index:
            task_attr = df.loc[task_n].to_dict()
            factor = task_attr["Throuput factor (S)"]
            freq  = task_attr["Freq."]/f_gcd
            num_per_group = int(np.ceil(freq / factor))
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
                succ_freq  = int(df.loc[succ_n]["Freq."]/f_gcd)
                succ_num_per_group = int(np.ceil(succ_freq/succ_factor))
                for i_s in range(succ_freq):
                    no_succ = int(i_s // succ_num_per_group)
                    succ_job_name = succ_n+"_"+str(no_succ)
                    
                    # just like quantization
                    succ_t = i_s/succ_freq
                    pred_t = int(succ_t*freq)

                    no_ = int(pred_t // num_per_group)
                    pre_job_name = task_n+"_"+str(no_)
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
                    factor = task_attr["Throuput factor (S)"]
                    for i in range(factor):
                        succ_job_name = succ_n+"_"+str(i)
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
            parallel_cfg = {}
            if task_attr["Parallel_type"] == "Upb":
                parallel_cfg["mode"] = "upb"
                parallel_cfg["max"] = int(task_attr["Parallel_range"])
            elif task_attr["Parallel_type"] == "Lwb":
                parallel_cfg["mode"] = "lwb"
                parallel_cfg["min"] = int(task_attr["Parallel_range"])
            elif task_attr["Parallel_type"] == "Range":
                parallel_cfg["mode"] = "range"
                # split the range into two parts
                parallel_cfg["min"], parallel_cfg["max"] = map(int, task_attr["Parallel_range"].split(","))
            elif task_attr["Parallel_type"] == "list":
                parallel_cfg["mode"] = "list"
                parallel_cfg["list"] = map(int, task_attr["Parallel_range"].split(","))
            task = TaskInt(
                task_name=task_n+"_"+str(i), task_id=task_id, timing_flag=task_attr["Timing_flag"], 
                ERT=task_attr["T release"]/1000, ddl=(task_attr['DDL']-task_attr["T release"])/1000, period=T, 
                exp_comp_t=task_attr['Expected Latency (ms)']/1000, i_offset=phase, jitter_max=0,
                flops=task_attr["Flops on path"]/1e3, task_flag=task_attr["Resource Type"], 
                pre_assigned_resource_flag=task_attr["Pre-assigned"]>0, 
                RDA_size=task_attr['RDA./Req.'], main_size=task_attr['Cores/Req.'], seq_cpu_time=task_attr["Flops on path"]/1e3,
                op_cpu_time=task_attr["Flops on path"]/1e3, op_io_time=1e-6,
                criti_flag="soft" if task_attr["Criti_flag"]=='S' else "hard", 
                cbs_en=True, # if task_attr["Cbs_en"]=='Y' else False, 
                trigger_mode=task_attr["Trigger_mode"], 
                parallel_cfg=parallel_cfg,

            )
            task.freq = task_attr["Freq."]
            # initialize task affinity list
            thread_n = int(i)
            affinity_tgt_n_list = affinity_cfg[task_n]        
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
    
    for job_n, job in taskJobs.items():

        # group the edge with attribution "reDistPattn" eq "downscaling"
        for pre_n, datadict in job_graph_nx.pred[job_n].items():
            dep_t = datadict["type"]
            attr = copy.deepcopy(datadict)
            attr.update({"valid":False})
            attr.update({"event_queue":TaskQueue(sort_f=lambda x: x.ctx.get_timestamp(), descending=False)})
            if dep_t == "data":
                job.pred_data.update({pre_n:attr})
            elif dep_t == "control":
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
    args = parser.parse_args() 
    glb_n_task_dict = load_taskint(args.verbose)

    if args.test_case == "all":
        args.test_all = True
    f_gcd = np.gcd.reduce([glb_n_task_dict[task].freq for task in glb_n_task_dict])
    f_max = max([glb_n_task_dict[task].freq for task in glb_n_task_dict])
    hyper_p = 1/f_gcd
    sim_step = min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32

    if args.test_case == "timeline" or args.test_all:
        vis_task_static_timeline(list(glb_n_task_dict.values()), save=True, save_path="plot/task_static_timeline_cyclic.pdf", hyper_p=hyper_p, n_p=1, warmup=False, drain=True, )
    elif args.test_case == "liveness" or args.test_all:
        vis_task_static_timeline(list(glb_n_task_dict.values()), save=True, save_path="plot/task_liveness_timeline_cyclic.svg", 
        hyper_p=hyper_p, n_p=1, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
        txt_size=40, tick_dens=4)
    elif args.test_case == "graph" or args.test_all:
        # task_graph_nx, job_graph_nx = creat_jobTask_graph(task_graph, int(f_gcd), plot=True)
        # init_depen(task_dict, job_graph_nx, verbose=args.verbose)
        logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
        physical_graph_nx = creat_physical_graph(logical_graph_nx, int(f_gcd))
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
            t.set_rotation(30)
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
            t.set_rotation(30)
        fig.tight_layout()
        plt.savefig("plot/jobTask_graph.pdf", format="pdf")
