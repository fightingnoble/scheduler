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
