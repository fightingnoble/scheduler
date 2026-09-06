"""Historical graph construction and superseded timeline reference."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


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


def creat_jobTask_graph(task_graph:Dict[str, List[str]], f_gcd, plot:bool=False, profiling_filename:str="profiling/profiling.csv"):
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
