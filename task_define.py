import networkx as nx
from task import Task
from typing import Dict, List, Tuple, Union, Callable
import numpy as np
import matplotlib.pyplot as plt
from hw_rsc import FLOPS_PER_CORE

def init_rectangle_shaped_cores(
    task_index_table_by_name:Dict[str, Tuple[int, Task]], 
    x_range: Union[Tuple[int, int], List[Tuple[int, int]]],
    y_range: Union[Tuple[int, int], List[Tuple[int, int]]],
    task_name: str, attr_name: str, key: str=""
):
    x_range = x_range if isinstance(x_range, list) else [x_range]
    y_range = y_range if isinstance(y_range, list) else [y_range]
    assert len(x_range) == len(y_range)
    rsc_list = []
    for i in range(len(x_range)):
        x_range_t = x_range[i]
        y_range_t = y_range[i]
        x, y = np.meshgrid(np.arange(*x_range_t), np.arange(*y_range_t))
        rsc_list += [(i, j) for i, j in zip(x.reshape(-1), y.reshape(-1))]
    target_len = len(task_index_table_by_name[task_name][1].__dict__[attr_name])
    if target_len:
        assert len(rsc_list) == target_len
    # task_index_table_by_name[task_name][1].__dict__[attr_name] = rsc_list
    if not key:
        task_index_table_by_name[task_name][1].add_resource(attr_name, rsc_list)
    else:
        task_index_table_by_name[task_name][1].add_resource(attr_name, [key], [rsc_list])


def get_affinity(
    task_list: List[Task], similarity_func: Callable,
    sim_time: float = 1.0, sim_step: float = 1e-6, **kwargs
):
    # build affinity graph
    task_seq_id_list_n = []
    task_seq_id_list_p = []
    for i, task in enumerate(task_list):
        if task.task_flag == "fixed":
            continue
        task_seq_rls_gen = task.get_release_event(sim_time)
        task_seq_ddl_gen = task.get_deadline_event(sim_time)
        # fill the interval between rls and ddl with 1 and interval between ddl and rls with 0
        task_seq_id = []
        # from start to 1st event
        # record history time and current time
        sim_time = round(sim_time/sim_step)
        t_his = t_curr = 0
        rls = next(task_seq_rls_gen)
        t_curr, t_his = round(rls/sim_step), t_curr
        task_seq_id.extend([0]*round((rls-0)/sim_step))
        while True:
            try:
                # add event interval
                ddl = next(task_seq_ddl_gen)
                t_curr, t_his = round(ddl/sim_step), t_curr

                if t_curr > sim_time:
                    task_seq_id.extend([1]*(sim_time-t_his))
                    break
                else:
                    task_seq_id.extend([1]*(t_curr-t_his))

                # add free interval
                rls = next(task_seq_rls_gen)
                t_curr, t_his = round(rls/sim_step), t_curr
                if rls > sim_time:
                    task_seq_id.extend([0]*(sim_time-t_his))
                    break
                else:
                    task_seq_id.extend([0]*(t_curr-t_his))
            except StopIteration:
                t_his = t_curr
                task_seq_id.extend([0]*(sim_time-t_his))
                break
        print("a step!!!")
        task_seq_id_list_n.append(np.array(task_seq_id))
        task_seq_id_list_p.append(1-np.array(task_seq_id))
    lack, surplus = similarity_func(
        task_seq_id_list_p, task_seq_id_list_n, **kwargs)
    return lack, surplus


def similarity_func(task_seq_id_list_p, task_seq_id_list_n, **kwargs):
    """
    similarity function for task sequence
    """
    # M_A = np.zeros((len(task_seq_id_list_p), len(task_seq_id_list_n)))
    # M_B = np.zeros((len(task_seq_id_list_p), len(task_seq_id_list_n)))
    array_p = np.array(task_seq_id_list_p).reshape(
        len(task_seq_id_list_p), 1, -1)
    array_n = np.array(task_seq_id_list_n).reshape(
        1, len(task_seq_id_list_n), -1)

    M_A = np.sum(array_p == array_n, axis=-1)
    M_B = np.sum(array_p != array_n, axis=-1)
    # for i, seq_n in enumerate(task_seq_id_list_n):
    #     for j, seq_p in enumerate(task_seq_id_list_p):
    #         if np.array_equal(seq_n, seq_p):
    #             return 1
    #         M_A[i, j] = ((seq_n - seq_p)==-1).sum()/seq_p.sum()
    #         M_B[i, j] = ((seq_n - seq_p)==1).sum()/seq_p.sum()
    return M_A, M_B


# given the new e2e constraint, redistribute the ddl to each task
def redistribute_ddl():
    pass


# Section 1: Task properties
sim_time = 0.2
vertical_grid_size = 0.4
time_grid_size = 0.004

task_n = [
    "Pure_camera_path",
    "Lidar_camera_path",
    "Traffic_light_detection",
    "Lane_drivable_area",
    "Depth_estimation",
    "Pure_camera_path_head",
    "Prediction",
    "Optical_Flow",
    "Planning",
]
task_attr = {
    k: {} for k in task_n
}


timing_k = ["period", "flops", "release_t", "ddl", "exe_t"]
timing_v = [[1/30, 2721.5, 0, 90.8041529, 90.7166666666667, ],
            [1/10, 264.793, 0, 98.05223011, 88.2643333333333, ],
            [1/30, 233.24, 0, 100, 93.296, ],
            [1/20, 64.2818, 61.2717904750299, 100, 32.1409, ],
            [1/20, 64.2818, 61.2717904750299, 100, 32.1409, ],
            [1/30, 3.504, 90.8041529, 98.05223011, 7.008, ],
            [1/30, 1.70769268, 97.8121529, 99.51984558, 1.70769268, ],
            [1/20, 79.514, 61.2717904750299, 100, 31.8056, ],
            [1/240, 0.24007721, 99.51984558, 100, 0.48015442, ], ]

req_attr_k = ["Thread_factor", "Cores_req", "Util", "Min_required_cores",
              "Equavalent_used_cores", "Maximum", "Equavalent_used_cores", "Redundent", "Redundent_req"]
req_attr_v = [
    [3, 60, 91, 180, 163.45, 216, 195.95, 36, 12],
    [1, 6, 88, 6, 5.45, 8, 7.06, 2, 2.],
    [3, 5, 93, 15, 15.00, 18, 16.79, 3, 1],
    [2, 8, 32, 16, 6.20, 20, 6.43, 4, 2],
    [2, 4, 32, 8, 3.10, 10, 3.21, 2, 1],
    [3, 6, 7, 18, 1.26, 24, 1.68, 6, 2],
    [3, 2, 2, 6, 0.03, 24, 0.41, 18, 6],
    [2, 5, 32, 10, 3.87, 12, 3.82, 2, 1],
    [1, 1, 12, 1, 0.04, 1, 0.12, 0, 0],
]


pre_assign_v = [
    ["deadline", 216, "stationary"],
    ["deadline", 8, "stationary"],
    ["deadline", 18, "fixed"],
    ["realtime", 20, "stationary"],
    ["realtime", 0, "moveable"],
    ["deadline", 0, "stationary"],
    ["deadline", 0, "moveable"],
    ["realtime", 0, "moveable"],
    ["deadline", 0, "hard"],
]
pre_assign_k = ["timing_type", "pre_signed", "task_flag"]


# intergrate all the task properties
for i in range(len(req_attr_v)):
    task_attr[task_n[i]].update(dict(zip(timing_k, timing_v[i])))
    task_attr[task_n[i]].update(dict(zip(req_attr_k, req_attr_v[i])))
    task_attr[task_n[i]].update(dict(zip(pre_assign_k, pre_assign_v[i])))


# get task list
task_list: List[Task] = []
task_id = 0
for task in task_attr.keys():
    for i in range(task_attr[task]["Thread_factor"]):
        T = task_attr[task]["period"]*task_attr[task]["Thread_factor"]
        phase = task_attr[task]["period"]*i
        task_t = Task(
            flops=task_attr[task]["flops"]/1e3,
            ERT=task_attr[task]["release_t"]/1000,
            i_offset=phase,
            exp_comp_t=task_attr[task]["exe_t"]/1000,
            period=T,
            ddl=task_attr[task]["ddl"]/1000,
            jitter_max=0,
            task_flag=task_attr[task]['task_flag'],
            timing_flag=task_attr[task]['timing_type'],)
        task_t.set_task_name(task+"_"+str(i))
        # task_t.set_task_name(task+"_"+str(0))
        if task_attr[task]['pre_signed'] > 0:
            task_t.pre_assigned_resource_flag = True
            task_t.expected_redundancy_size = task_attr[task]['Redundent_req']
            task_t.main_resource_size = task_attr[task]['Cores_req']
        task_id += 1
        task_t.set_task_id(task_id)
        task_list.append(task_t)


# hash table for task id and task object
task_index_table_by_name: Dict[str, Tuple[int, Task]] = {
    task.task_name: (task.id, task) for i, task in enumerate(task_list)}

task_index_table_by_id: Dict[int, Tuple[str, Task, Node]] = {
    task.id: (task.task_name, task) for i, task in enumerate(task_list)}

task_chains: List = [

]

# Section 2: define resource pre-assignment

# Section 2-1: mapping task that is pre-assigned cores
# TODO: assign the resources with a function rather than by hand

rsc_map = np.full((16, 16), -1, dtype=int)
# row: 0-12, col: 0-6, assign to task "Pure_camera_path_0"
# row: 0-12, col: 6-12, assign to task "Pure_camera_path_1"
# row: 0-12, col: 12-16, assign to task "Pure_camera_path_2"
# row: 12-16, col: 10-16, assign to task "Pure_camera_path_2"
# row: 11-16, col: 0-2, assign to task "Lane_drivable_area_0"
# row: 11-16, col: 2-4, assign to task "Lane_drivable_area_1"
# row: 13-14, col: 4-10, assign to task "Traffic_light_detection_0"
# row: 14-15, col: 4-10, assign to task "Traffic_light_detection_1"
# row: 15-16, col: 4-10, assign to task "Traffic_light_detection_2"
# row: 12-13, col: 4-12, assign to task "Lidar_camera_path_0"

rsc_map[0:12, 0:6] = task_index_table_by_name["Pure_camera_path_0"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (0, 12), (0, 6), "Pure_camera_path_0", "main_resource")

rsc_map[0:12, 6:12] = task_index_table_by_name["Pure_camera_path_1"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (0, 12), (6, 12), "Pure_camera_path_1", "main_resource")

rsc_map[0:12, 12:16] = task_index_table_by_name["Pure_camera_path_2"][0]
rsc_map[12:16, 10:16] = task_index_table_by_name["Pure_camera_path_2"][0]
init_rectangle_shaped_cores(task_index_table_by_name, [(0, 12), (12, 16)], [(12, 16), (10, 16)], "Pure_camera_path_2", "main_resource")

rsc_map[11:16, 0:2] = task_index_table_by_name["Lane_drivable_area_0"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (11, 16), (0, 2), "Lane_drivable_area_0", "main_resource")

rsc_map[11:16, 2:4] = task_index_table_by_name["Lane_drivable_area_1"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (11, 16), (2, 4), "Lane_drivable_area_1", "main_resource")

rsc_map[13:14, 4:10] = task_index_table_by_name["Traffic_light_detection_0"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (13, 14), (4, 10), "Traffic_light_detection_0", "main_resource")

rsc_map[14:15, 4:10] = task_index_table_by_name["Traffic_light_detection_1"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (14, 15), (4, 10), "Traffic_light_detection_1", "main_resource")

rsc_map[15:16, 4:10] = task_index_table_by_name["Traffic_light_detection_2"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (15, 16), (4, 10), "Traffic_light_detection_2", "main_resource")

rsc_map[12:13, 4:12] = task_index_table_by_name["Lidar_camera_path_0"][0]
init_rectangle_shaped_cores(task_index_table_by_name, (12, 13), (4, 12), "Lidar_camera_path_0", "main_resource")


# mark redundant cores
# row: 10-12, col: 0-6, assign to task "Pure_camera_path_0"
# row: 10-12, col: 6-12, assign to task "Pure_camera_path_1"
# row: 12-16, col: 10-14, assign to task "Pure_camera_path_2"
# row: 11-12, col: 0-2, assign to task "Lane_drivable_area_0"
# row: 11-12, col: 2-4, assign to task "Lane_drivable_area_1"
# row: 13-14, col: 9-10, assign to task "Traffic_light_detection_0"
# row: 14-15, col: 9-10, assign to task "Traffic_light_detection_1"
# row: 15-16, col: 9-10, assign to task "Traffic_light_detection_2"
# row: 12-13, col: 10-12, assign to task "Lidar_camera_path_0"
init_rectangle_shaped_cores(task_index_table_by_name, (10, 12), (0, 6), "Pure_camera_path_0", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (10, 12), (6, 12), "Pure_camera_path_1", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (12, 16), (10, 14), "Pure_camera_path_2", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (11, 12), (0, 2), "Lane_drivable_area_0", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (11, 12), (2, 4), "Lane_drivable_area_1", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (13, 14),(9, 10), "Traffic_light_detection_0", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (14, 15),(9, 10), "Traffic_light_detection_1", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (15, 16),(9, 10), "Traffic_light_detection_2", "redundant_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (12, 13), (10, 12), "Lidar_camera_path_0", "redundant_resource")


# mark overlap cores
# row: 12-13, col: 10-12, assign to task "Lidar_camera_path_0"
# row: 11-12, col: 0-4, assign to task "Pure_camera_path_0"
overlap_cores = np.full((16, 16), -1, dtype=int)
overlap_cores[12:13, 10:12] = task_index_table_by_name["Lidar_camera_path_0"][0]
overlap_cores[11:12, 0:4] = task_index_table_by_name["Pure_camera_path_0"][0]
rsc_map[12:13, 10:12] = task_index_table_by_name["Lidar_camera_path_0"][0]
rsc_map[11:12, 0:4] = task_index_table_by_name["Pure_camera_path_0"][0]

# TODO: mark the overlap cores in the task rsc map, and add the allocation logic
init_rectangle_shaped_cores(task_index_table_by_name, (12, 13), (10, 12), "Lidar_camera_path_0", "overlap_resource")
init_rectangle_shaped_cores(task_index_table_by_name, (11, 12), (0, 4), "Pure_camera_path_0", "overlap_resource")

# Section 2-2: set task groups that is not pre-assigned cores

# define color map
color_map = {
    "fixed": "red",
    "hard": "blue",
    "stationary": "green",
    "moveable": "yellow",
}


# generate a fully connected resource sharing graph
# get the id of the task that is pre-assigned cores
pre_assigned_task_id: List[Tuple[int, str]] = []
non_pre_assigned_task_id: List[Tuple[int, str]] = []
for task_name in task_index_table_by_name.keys():
    task = task_index_table_by_name[task_name][1]
    if task.pre_assigned_resource_flag:
        pre_assigned_task_id.append((task_index_table_by_name[task_name][0], {
                                    "name": task_name, "color": color_map[task_index_table_by_name[task_name][1].task_flag]}))
    else:
        non_pre_assigned_task_id.append((task_index_table_by_name[task_name][0], {
                                        "name": task_name, "color": color_map[task_index_table_by_name[task_name][1].task_flag]}))

# only one resource group when the resource sharing graph is fully connected
# TODO: add affinity analysis to generate a more accurate resource sharing graph: such as Cosine Similarity, etc.

# add all nodes
graph = nx.DiGraph()
graph.add_nodes_from(pre_assigned_task_id, **{"pre_assigned": True, })
graph.add_nodes_from(non_pre_assigned_task_id, **{"pre_assigned": False, })


# assign resources for stationary tasks that is not pre-assigned cores
# TODO: use a more general function
unsigned_stationanry_task_name = [
    "Pure_camera_path_head_0", "Pure_camera_path_head_1", "Pure_camera_path_head_2"]
for task_name in unsigned_stationanry_task_name:
    ua_task = task_index_table_by_name[task_name][1]
    for id_as in pre_assigned_task_id:
        # j[1]: task attr, j[1]["name"]: task name
        # task_index_table_by_name[j[1]["name"]][1]: task object
        S_task = task_index_table_by_name[id_as[1]['name']][1]

        # for hard and movable, to find S_tasks with lable [stationary, moveable]
        if S_task.task_flag in ["stationary", "moveable"]:
            graph.add_edge(
                ua_task.id, id_as[0], **{"type": ua_task.task_flag, "color": color_map[ua_task.task_flag]})
            if ua_task.task_flag != "hard":
                graph.add_edge(
                    id_as[0], ua_task.id, **{"type": S_task.task_flag, "color": color_map[S_task.task_flag]})


# add edges
# build a fully connected graph between [hard, moveable] and [stationary, moveable]
for id_us in non_pre_assigned_task_id:
    ua_task = task_index_table_by_name[id_us[1]['name']][1]
    for id_as in pre_assigned_task_id:
        # j[1]: task attr, j[1]["name"]: task name
        # task_index_table_by_name[j[1]["name"]][1]: task object
        S_task = task_index_table_by_name[id_as[1]['name']][1]
        # print(ua_task, S_task)
        # check legalty: task that is not pre-assigned should not be fixed and stationary
        if ua_task.task_flag not in ["fixed", "stationary"]:
            # for hard and movable, to find S_tasks with lable [stationary, moveable]
            if S_task.task_flag in ["stationary", "moveable"]:
                graph.add_edge(
                    id_us[0], id_as[0], **{"type": ua_task.task_flag, "color": color_map[ua_task.task_flag]})
                if ua_task.task_flag != "hard":
                    graph.add_edge(
                        id_as[0], id_us[0], **{"type": S_task.task_flag, "color": color_map[S_task.task_flag]})
            else:
                pass
        else:
            assert "Error: task {} of type {} is not pre-assigned resources".format(
                id_us[1]['name'], ua_task.task_flag)


# set the edge weight with reource affinity
# TODO: replace the vector compare with interval algorithm: vector genneration is too slow!!!
# m1, m2 = get_affinity(task_list, similarity_func=similarity_func, sim_time=0.2, sim_step=1e-5)
# print(m1)
# print(m2)


# Function for visulization
def rsc_affinity_graph_viz(graph: nx.DiGraph, show=False, save=False, save_path="rsc_affinity_graph.pdf", **kwargs):

    # print(graph.nodes(data=True))
    # print(graph.edges())

    # pos = nx.nx_agraph.graphviz_layout(graph, prog="twopi", args="")
    pos = nx.bipartite_layout(graph, [i[0] for i in pre_assigned_task_id])
    plt.figure(figsize=(8, 8))
    node_color = [v[-1] for v in graph.nodes(data="color")]
    edge_color = [e[-1] for e in graph.edges(data="color")]
    # print(node_color, edge_color)
    nx.draw(graph, pos, node_size=20, alpha=0.5, with_labels=False,
            node_color=node_color, edge_color=edge_color)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_color="black", labels={
                            v: data["name"] for v, data in graph.nodes(data=True)})
    # nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u,v):ddict["type"] for u, v, ddict in graph.edges(data=True)}, font_size=8, font_color="black")
    plt.axis("equal")
    if "format" not in kwargs and save_path.split(".")[-1] == "pdf" and save:
        kwargs["format"] = "pdf"
    if show:
        plt.show()
    plt.savefig(save_path, **kwargs)


def vis_task_static_timeline(task_list, show=False, save=False, save_path="task_static_timeline.pdf", **kwargs):
    # build event list
    req_list = []
    ddl_list = []
    finish_list = []

    for task in task_list:
        req_list.append(task.get_release_event(sim_time))
        ddl_list.append(task.get_deadline_event(sim_time))
        finish_list.append(task.get_finish_event(sim_time))

    # print(req_list, ddl_list)

    # plot
    horizen_grid = set()
    fig, ax = plt.subplots(figsize=(50, 10))
    vertical_offset = 0
    for i in range(len(task_list)):
        for s, e in zip(req_list[i], finish_list[i]):
            if e > sim_time:
                continue
            print("{}:{}-{}".format(task_list[i].task_name, s, e))
            horizen_grid.add(s)
            horizen_grid.add(e)
            ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s,
                      xmax=e, lw=2,)  # label=task_list[i].task_name)
            ax.text(sim_time, vertical_offset*vertical_grid_size,
                    task_list[i].task_name, fontsize=7)
        vertical_offset += 1
        # s = next(req_list[i])
        # e = next(ddl_list[i])
        # print("{}:{}-{}".format(task_list[i].task_name, s, e))
        # ax.hlines(y=vertical_offset*vertical_grid_size, xmin=s, xmax=e, lw=2,)# label=task_list[i].task_name)
        # ax.text(sim_time, vertical_offset*vertical_grid_size, task_list[i].task_name, fontsize=7)
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


def __main__():
    # vis_task_static_timeline()
    rsc_affinity_graph_viz(graph, save=True)


if __name__ == "__main__":
    __main__()
