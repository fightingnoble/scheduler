import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Iterable, Iterator
from collections import OrderedDict
from resource_agent import Resource_model_int
from functools import reduce
from task_agent import ProcessInt, TaskInt
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, Range1d, LabelSet, Label, Legend
from bokeh.layouts import row, column, gridplot
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class SchedulingTableInt(object): 
    """
    =============== 1. scheduling table ===============
    scheduling table is a 2D array, each row is a resource, each column is a time slot
    the value of each cell is the task id that occupies the resource in the time slot
    the size of the scheduling table is determined by the number of resources and the number of time slots
    the number of time slots is determined by the hyper-period of the tasks
    the number of resources is determined by the number of resources that are available
    """
    def __init__(self, num_resources: int, num_time_slots:int, id: int = None, name: str = None, hp:int=None):
        # self.scheduling_table = np.zeros((num_resources, num_time_slots), dtype=int)
        # self.scheduling_table = np.full((num_time_slots), Resource_model_int(num_resources, id, name), dtype=Resource_model_int)
        self.scheduling_table = np.array([Resource_model_int(num_resources, ) for _ in range(num_time_slots)], dtype=Resource_model_int)
        self.id = id
        self.name = name
        self.num_resources = num_resources
        self.temp_size = num_time_slots
        self.locker = None
        self.lock_mask = np.ones((num_time_slots), dtype=bool)
        self.sparse_list = OrderedDict()
        # current index of the sparse list
        self.sparse_idx = 0
        # previous index of the sparse list
        self.sparse_idx_prev = 0
        # next index of the sparse list
        self.sparse_idx_next = 0
        self.sparse_mode = False

        # available when used as a recorder
        self.wr_pointer = 0
        self.hypper_period = hp

    def append(self, rsc_agent: Resource_model_int):
        # Too slow
        # self.scheduling_table = np.append(self.scheduling_table, rsc_agent)
        
        if self.wr_pointer >= self.temp_size:
            assert self.hypper_period is not None
            self.temp_size += self.hypper_period
            self.scheduling_table = np.append(self.scheduling_table, np.array([Resource_model_int(self.num_resources, ) for _ in range(self.hypper_period)], dtype=Resource_model_int))
        self.scheduling_table[self.wr_pointer].update(rsc_agent.rsc_map)
        self.wr_pointer += 1

    def index_occupy_by_id(self, time_slot_s:int=None, time_slot_e:int=None) -> Dict[int, List[int]]:
        """
        get the task id that occupies the resource
        """
        # rsc_array = self.scheduling_table[:, time_slot_s:time_slot_e]
        # rsc_occp = []
        # for id in np.unique(rsc_array[rsc_array != 0]):
        #     rsc_occp.append((id, np.where(rsc_array == id)[0]))
        # return np.where(rsc_array == 0)[0], rsc_occp
        
        # get task_id set
        task_id_set = set()
        assert not (time_slot_s is None and time_slot_e is not None)
        if time_slot_s is None:
            time_slot_s = 0

        if time_slot_e is None:
            rsc_agents_arr = self.scheduling_table[time_slot_s:]
        else:
            rsc_agents_arr = self.scheduling_table[time_slot_s:time_slot_e]
        
        # get task_id set
        for rsc_agent in rsc_agents_arr:
            # print(rsc_agent.rsc_map.keys())
            task_id_set.update(rsc_agent.rsc_map.keys())
        # return list(task_id_set)
        
        Scheduling_table_index_by_task_id = OrderedDict()
        for task_id in task_id_set: 
            Scheduling_table_index_by_task_id[task_id] = []
        
        for rsc_agent in rsc_agents_arr:
            for task_id in task_id_set:
                Scheduling_table_index_by_task_id[task_id].append(rsc_agent.rsc_map.get(task_id, 0))

        # divide the time slot into intervals
        # get the size and the start and length of each interval
        for task_id, rsc_occ in Scheduling_table_index_by_task_id.items():
            rsc_occ = np.array(rsc_occ)
            boader = (rsc_occ[0:-1] != rsc_occ[1:]).nonzero()[0] + 1
            s = [0] + boader.tolist() 
            e = boader.tolist() + [len(rsc_occ)] 
            l = [e[i] - s[i] for i in range(len(s))]
            size = [rsc_occ[s[i]] for i in range(len(s))]
            # get non zero intervals
            s = [s[i]+time_slot_s for i in range(len(s)) if size[i] != 0]
            l = [l[i] for i in range(len(l)) if size[i] != 0]
            size = [size[i] for i in range(len(size)) if size[i] != 0]            
            Scheduling_table_index_by_task_id[task_id] = (s, size, l)


        return Scheduling_table_index_by_task_id

    def idx_free_by_slot(self, time_slot_s, time_slot_e, key=None):
        """
        get the available resources in the time slot
        """
        # culculate available resources at each time slot
        rsc_maps_arr = self.scheduling_table[time_slot_s:time_slot_e]
        rsc_avl = []
        # check lock: 
        #   if the lock is free in the target interval or the query is from the locker, return all available resources 
        if np.all(self.lock_mask[time_slot_s:time_slot_e]) or key == self.locker:
            for idx, rsc_map in enumerate(rsc_maps_arr):
                rsc_avl.append(rsc_map.get_available_rsc())
        else: 
            for rsc_map, lock in zip(rsc_maps_arr, self.lock_mask[time_slot_s:time_slot_e]):
                if lock:
                    rsc_avl.append(rsc_map.get_available_rsc())
                else:
                    rsc_avl.append(0)
        return rsc_avl
    
    def insert_task(self, task:ProcessInt, req_rsc_size:int, time_slot_s:int, time_slot_e:int, expected_slot_num:int, 
                    verbose=False, DEBUG=False)->Tuple[bool, Union[int,List[int]], Union[int,List[int]], Union[int,List[int]]]: 
        """
        play a insert-based scheduling: 
        1. search available tensor cores at each slot
        2. insert the task into the scheduling table at a proper interval (here we adapt First-Fit)
            return success or not, the start time slot, the allocated resources, the allocated time slots
        3. release the resources given the list of allocated resources and the list of allocated time slots
        """ 
        rsc_avl = self.idx_free_by_slot(time_slot_s, time_slot_e, key=task.pid)
        rsc_avl = np.array(rsc_avl)
        # check if the task can be scheduled with expected resources
        # the task can be scheduled at any time slot
        # bug
        if np.all(rsc_avl >= req_rsc_size) and (time_slot_e - time_slot_s) >= expected_slot_num:
            # allocate resources 
            for rsc_map in self.scheduling_table[time_slot_s:time_slot_s+expected_slot_num]:
                rsc_map.allocate(task.pid, req_rsc_size, verbose)
            return True, time_slot_s, req_rsc_size, expected_slot_num
        else: 
            # divide the rsc_avl into intervals
            boader = (rsc_avl[0:-1] != rsc_avl[1:]).nonzero()[0] + 1
            s = [0] + boader.tolist() 
            e = boader.tolist() + [len(rsc_avl)] 
            # check if the task can be executed on the current interval
            for i in range(len(s)):
                if (e[i] - s[i]) >= expected_slot_num and np.all(rsc_avl[s[i]:e[i]] > req_rsc_size):
                    # allocate resources 
                    alloc_slot_s = time_slot_s+s[i]
                    for rsc_map in self.scheduling_table[alloc_slot_s:alloc_slot_s+expected_slot_num]:
                        rsc_map.allocate(task.pid, req_rsc_size, verbose)
                    return True, alloc_slot_s, req_rsc_size, expected_slot_num

            # redistribute the resources to the intervals
            # based on the priciple of as soon as possible
            
            # current allocation (C)
            curr_alloc = np.zeros(len(s), dtype=int)
            curr_slot = np.zeros(len(s), dtype=int)

            # avalable (A)
            rsc_avl_tmp = np.zeros(len(s), dtype=int) 

            # required (R)
            expected_req_rsc_size = req_rsc_size * expected_slot_num
            cum_rsc_alloc = 0
            cum_slot_length = 0

            # warning: expected_slot_num may be larger than the available slots
            if rsc_avl[:expected_slot_num].sum() < expected_req_rsc_size: 
                # as soon as possible
                if DEBUG:
                    print("Not enough resources: as soon as possible")
                for i in range(len(s)): 
                    if rsc_avl[s[i]] > 0:
                        cond2 = rsc_avl[:e[i]].sum() >= expected_req_rsc_size
                        # rsc size 
                        curr_alloc[i] = rsc_avl[s[i]]
                        # slot length
                        if cond2:
                            curr_slot[i] = np.ceil((expected_req_rsc_size - rsc_avl[:e[i-1]].sum())/rsc_avl[s[i]]).astype(int)
                            break
                        else:
                            curr_slot[i] = int(e[i] - s[i])
            else: 
                # there is enough rsc in 
                # [0, expected_slot_num] + time_slot_s
                # try to distribute the rsc_lack to the intervals as evenly as possible
                if DEBUG:
                    print("Enough resources: as evenly as possible")
                # calculate the lacked resources in the interval: 
                for i in range(len(s)): 
                    if rsc_avl[s[i]] > 0:
                        # rsc size 
                        if rsc_avl[s[i]] < req_rsc_size:
                            curr_alloc[i] = rsc_avl[s[i]]
                            rsc_avl_tmp[i] = 0
                        else:
                            curr_alloc[i] = req_rsc_size
                            rsc_avl_tmp[i] = rsc_avl[s[i]] - req_rsc_size
                        # slot length
                        
                        cond1 = (e[i] - s[i]) >= (expected_slot_num - cum_slot_length)
                        if cond1:
                            curr_slot[i] = int(expected_slot_num - cum_slot_length)
                        else: 
                            curr_slot[i] = int(e[i] - s[i])

                        cum_rsc_alloc += curr_alloc[i] * curr_slot[i]
                        cum_slot_length += curr_slot[i]
                        # stop if the available resources are enough 
                        # and the slot length is enough
                        if cond1: 
                            break
                
                rsc_lack = expected_req_rsc_size - cum_rsc_alloc

                # try to distribute the rsc_lack to the intervals as evenly as possible
                while cum_rsc_alloc < expected_req_rsc_size: 
                    size_t = rsc_avl_tmp[rsc_avl_tmp>0].min()
                    avl_slot_idx = np.where(rsc_avl_tmp >0)[0]
                    slot_size_sum = np.sum(curr_slot[avl_slot_idx])
                    if slot_size_sum * size_t >= rsc_lack:
                        size_t = np.ceil(rsc_lack / slot_size_sum).astype(int)
                        rsc_lack = 0
                    else: 
                        rsc_lack -= slot_size_sum * size_t
                    curr_alloc[avl_slot_idx] += size_t
                    rsc_avl_tmp[avl_slot_idx] -= size_t
                    cum_rsc_alloc += size_t * slot_size_sum

            # allocate resources
            idx = curr_alloc.nonzero()[0]
            for i in range(len(idx)):
                alloc_slot_s = time_slot_s+s[idx[i]]
                for rsc_map in self.scheduling_table[alloc_slot_s:alloc_slot_s+int(curr_slot[idx[i]])]:
                    rsc_map.allocate(task.pid, curr_alloc[idx[i]], verbose)
            
        self.sparse_mode = False
        return True, (time_slot_s+np.array(s)[idx]).tolist(), curr_alloc[idx].tolist(), curr_slot[idx].tolist()

    def release(self, task: ProcessInt, time_slot_s:Union[int,List[int]], curr_alloc:Union[int,List[int]], curr_slot:Union[int,List[int]], verbose: bool = False):
        if isinstance(curr_alloc, int) and isinstance(curr_slot, int) and isinstance(time_slot_s, int):
            for rsc_map in self.scheduling_table[time_slot_s:time_slot_s+curr_slot]:
                rsc_map.release(task.pid, curr_alloc, verbose)
        else:
            assert len(curr_alloc) == len(curr_slot) == len(time_slot_s)
            for i in range(len(curr_alloc)):
                for rsc_map in self.scheduling_table[time_slot_s[i]:time_slot_s[i]+curr_slot[i]]:
                    rsc_map.release(task.pid, curr_alloc[i], verbose)
        self.sparse_mode = False
    
    def step(self, mode:str="cyclic"): 
        assert mode in ["cyclic", "dynamic"]
        running = self.scheduling_table[0]
        self.scheduling_table = np.roll(self.scheduling_table, -1, axis=0)
        if mode == "dynamic": 
            self.scheduling_table[-1].clear()
        return running

    def print_scheduling_table(self, pid2name:Dict[int,str]=None, timestep=None):
        empty_boader_s = []
        empty_boader_e = []
        title_line = False

        pre_rsc = self.scheduling_table[0].rsc_map
        pre_idx = 0
        empty_flag = len(pre_rsc) == 0
        if empty_flag:
            empty_boader_s.append(0)
        
        for rsc_map_idx in range(len(self.scheduling_table)):
            rsc_map = self.scheduling_table[rsc_map_idx].rsc_map

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    if timestep is not None:
                        _str = f"time:[{pre_idx*timestep:.6f}-{rsc_map_idx*timestep:.6f}), slot:[{pre_idx}-{rsc_map_idx})\n"
                    else:
                        _str = f"slot:[{pre_idx}-{rsc_map_idx})\n"
                    if title_line: 
                        _str += pre_rsc.title_line
                        title_line = True
                    _str += f"{pre_rsc.print_simple(pid2name)}"

                    print(_str)
                pre_idx = rsc_map_idx
                pre_rsc = rsc_map
                empty_flag = len(pre_rsc) == 0
                if empty_flag:
                    empty_boader_s.append(rsc_map_idx)
        if empty_flag:
            empty_boader_e.append(len(self.scheduling_table))
        else:
            print(f"slot:[{pre_idx}-{len(self.scheduling_table)})\n{str(pre_rsc)}")

        _str = [f"[{empty_boader_s[i]}-{empty_boader_e[i]})" for i in range(len(empty_boader_s))]
        print("slot:{} Empty".format(",".join(_str,)))

    def print_alloc_detail(self, pid2name:Dict[int,str], timestep):
        bin_pack_result = self.index_occupy_by_id()

        # sort the result by the start time
        # item[1] is alloc_slot_s_t, alloc_size_t, allo_slot_t
        # item[1][0] is alloc_slot_s_t
        sorted_task_pid = [k for k, v in sorted(bin_pack_result.items(), key=lambda item: item[1][0])]

        # replace the pid with the task name
        # and print the result
        print(f"bin: {self.name}({self.id})")
        for pid in list(sorted_task_pid):
            _result = bin_pack_result.pop(pid)
            bin_pack_result[pid2name[pid]] = _result
            print("task: {:s}({:d})".format(pid2name[pid], pid))
            print("\tstart time: {:s}".format(", ".join([f"{x*timestep:.6f}" for x in _result[0]])))
            print("\talloc cores: {:s}".format(", ".join([f"{x:d}" for x in _result[1]])))
            print("\tused time: {:s}".format(", ".join([f"{x*timestep:.6f}" for x in _result[2]])))
        print("=====================================\n")

    def to_sparse_dict(self, init_pos: int = 0, verbose: bool = False):
        # sparst_dict = {}
        sparse_list = []
        empty_boader_s = []
        empty_boader_e = []
        title_line = False

        pre_rsc = self.scheduling_table[0].rsc_map
        pre_idx = 0
        empty_flag = len(pre_rsc) == 0
        if empty_flag:
            empty_boader_s.append(0)
        
        for rsc_map_idx in range(len(self.scheduling_table)):
            rsc_map = self.scheduling_table[rsc_map_idx].rsc_map

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    # if title_line: 
                    #     _str = f"slot:[{pre_idx}-{rsc_map_idx})\n{pre_rsc.print_simple()}"
                    # else:
                    #     _str = f"slot:[{pre_idx}-{rsc_map_idx})\n{str(pre_rsc)}"
                    #     title_line = True
                    # print(_str)
                    # sparse_dict[pre_idx] = [pre_idx, pre_rsc, rsc_map_idx-pre_idx]
                    sparse_list.append([pre_idx, pre_rsc, rsc_map_idx-pre_idx])
                pre_idx = rsc_map_idx
                pre_rsc = rsc_map
                empty_flag = len(pre_rsc) == 0
                if empty_flag:
                    empty_boader_s.append(rsc_map_idx)
        if empty_flag:
            empty_boader_e.append(len(self.scheduling_table))
        else:
            print(f"slot:[{pre_idx}-{len(self.scheduling_table)})\n{str(pre_rsc)}")
            # sparse_dict[pre_idx] = [pre_rsc, len(self.scheduling_table)-pre_idx]
            sparse_list.append([pre_idx, pre_rsc, len(self.scheduling_table)-pre_idx])

        # _str = [f"[{empty_boader_s[i]}-{empty_boader_e[i]})" for i in range(len(empty_boader_s))]
        # print("slot:{} Empty".format(",".join(_str,)))
        self.sparse_list = sparse_list
        self.sparse_idx = init_pos
        self.sparse_idx_next = init_pos + 1
        self.sparse_mode = True

    # add spase inded by 1
    def idx_plus_1(self,):
        assert self.sparse_mode
        self.sparse_idx_prev = self.sparse_idx
        self.sparse_idx = self.sparse_idx_next
        self.sparse_idx_next += 1
        if self.sparse_idx_next == len(self.sparse_list):
            self.sparse_idx_next = 0

    def idx_minus_1(self,):
        assert self.sparse_mode
        self.sparse_idx_next = self.sparse_idx
        self.sparse_idx = self.sparse_idx_prev
        self.sparse_idx_prev -= 1
        if self.sparse_idx_prev == -1:
            self.sparse_idx_prev = len(self.sparse_list) - 1
    
    def next_item(self,):
        self.idx_plus_1()
        # return self.sparse_list[list(self.sparse_dict.keys())[self.sparse_dict_idx]]
        return self.sparse_list[self.sparse_idx]


    def get_plot_frame(self, start:int=0, end:int=-1):
        frame = []
        for rsc_map in self.scheduling_table[start:end]:
            frame.append(rsc_map.get_plot_col())
        frame = np.array(frame).T
        return frame

    def __eq__(self, __o: object) -> bool:
        for i in range(len(self.scheduling_table)):
            if self.scheduling_table[i] != __o.scheduling_table[i]:
                return False
        return True
    
    def add_lock(self, task:ProcessInt, time_slot_s, time_slot_e):
        # check if the lock is valid
        if self.locker is not None:
            return False
        # set lock mask
        self.lock_mask[time_slot_s:time_slot_e] = False
        self.locker = task.pid
        return True

    def release_lock(self, task:ProcessInt, time_slot_s, time_slot_e):
        if self.locker == task.pid:
            self.lock_mask[time_slot_s:time_slot_e] = True
            self.locker = None
            return True
        else:
            return False


    @staticmethod
    def get_core_size(_p, timestep, FLOPS_PER_CORE):
        # release time round up: task should not be released earlier than the release time
        time_slot_s = int(np.ceil(_p.release_time/timestep))
        # deadline round down: task should not be finised later than the deadline
        time_slot_e = int(_p.deadline//timestep)
        req_rsc_size = int(np.ceil(_p.remburst/(time_slot_e-time_slot_s)/timestep/FLOPS_PER_CORE))
        return req_rsc_size



def get_task_layout_compact(bin_list:List[SchedulingTableInt], init_p_list:List[ProcessInt], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = hyper_p * (warmup)
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    colors=list(mcolors.XKCD_COLORS.keys())
    
    base_vertical_offset = 0
    y_margin = 0.5 
    time_grid_size = 0.004
    x_margin = time_grid_size

    # plot timeline and task name bin by bin
    # and select color for the task automatically
    if plot_legend:
        fig_size = (40, 50)
    else:
        fig_size = (60, 30)
    if tool == "matplotlib":
        fig = plt.figure(figsize=fig_size)
    elif tool == "bokeh":
        fig = []
    elif tool == "plotly":
        fig = make_subplots(rows=len(bin_list), cols=1, shared_xaxes=True, vertical_spacing=0.01)
    

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    for bin_idx, _SchedTab in enumerate(bin_list): 
        bin_temp_size = len(_SchedTab.scheduling_table)
        bin_spatial_size = _SchedTab.scheduling_table[0].size

        if tool == "matplotlib":
            ax = fig.add_subplot(len(bin_list), 1, len(bin_list)-_SchedTab.id)
        elif tool == "bokeh":
            ax = figure()
            fig.append(ax)
        elif tool == "plotly":
            pass
                        

        empty_boader_s = []
        empty_boader_e = []
        title_line = False

        pre_rsc = _SchedTab.scheduling_table[0].rsc_map
        # build a position dict
        position_dict = {}
        cum_pos = 0
        for k,v in pre_rsc.items():
            position_dict[k] = [[cum_pos], [v], True] 
            cum_pos += v

        pre_idx = 0
        empty_flag = len(pre_rsc) == 0

        if empty_flag:
            empty_boader_s.append(0)
        
        for rsc_map_idx in range(bin_temp_size):
            rsc_map = _SchedTab.scheduling_table[rsc_map_idx].rsc_map

            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            s, e = pre_idx*time_step, rsc_map_idx*time_step
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end
            if s >= plot_end or e <= plot_start: 
                continue

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    # plot the task layout
                    for pid, size in pre_rsc.items():
                        
                        _p = init_p_list[pid]
                        _p_name = _p.task.name
                        _p_color = mcolors.XKCD_COLORS[colors[pid]]
                        is_new = position_dict[pid][-1]

                        for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                            # culculate the position of the bar
                            bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                            text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                            # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                            if tool == "matplotlib":
                                ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)
                            elif tool == "bokeh":
                                ax.rect(x=s, y=bar_vertical_offset, width=e-s, height=vertical_size*vertical_grid_size, color=_p_color)
                            elif tool == "plotly":
                                fig.add_trace(go.Bar(x=[e-s], y=[bar_vertical_offset], marker_color=_p_color, orientation='h',
                                                     width=vertical_size*vertical_grid_size, base=s), row=len(bin_list)-_SchedTab.id, col=1)
                            # plot the text
                            if not plot_legend:
                                if is_new:
                                    position_dict[pid][-1] = False
                                    if tool == "matplotlib":
                                        ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)
                                    elif tool == "bokeh":
                                        ax.text(x=(s+e)/2, y=text_vertical_offset, text=_p_name, text_color='black', text_font_size='10pt', text_align='center')
                                    elif tool == "plotly":
                                        fig.add_annotation(x=(s+e)/2, y=text_vertical_offset, text=_p_name, showarrow=False, font=dict(color='black', size=txt_size), 
                                                                                            xref='x', yref='y', row=len(bin_list)-_SchedTab.id, col=1)

                # the vertical grid at the end of the bar
                if tool == "matplotlib":
                    ax.axvline(e, color='black', linestyle='-', linewidth=0.5)
                elif tool == "bokeh": 
                    ax.line(x=[e, e], y=[bin_vertical_offset, bin_vertical_offset+bin_spatial_size*vertical_grid_size], color='black', line_width=0.5)
                elif tool == "plotly":
                    fig.add_trace(go.Scatter(x=[e, e], y=[bin_vertical_offset, bin_vertical_offset+bin_spatial_size*vertical_grid_size], mode='lines', 
                                             line=dict(color='black', width=0.5)), row=len(bin_list)-_SchedTab.id, col=1)

                # update the position dict
                new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
                expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
                old_pid = set(pre_rsc.keys()) - expired_pid

                used_position = []
                for pid in old_pid:
                    p_size = rsc_map[pid]
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        used_position += [i for i in range(s, e)]

                # remove the expired task from the position dict
                for pid in expired_pid:
                    position_dict.pop(pid)
                
                aval_pos = [i for i in range(bin_spatial_size) if i not in used_position]
                # check if the old task's allocation is changed

                size_plus = []
                size_minus = []
                for pid in old_pid:
                    old_size = pre_rsc[pid]
                    new_size = rsc_map[pid]
                    if new_size > old_size:
                        size_plus.append(pid)
                    elif new_size < old_size:
                        size_minus.append(pid)

                for group in [size_minus, size_plus]:
                    for pid in group: 
                        old_size = pre_rsc[pid]
                        new_size = rsc_map[pid]

                        # release the old position
                        for s, size in zip(*position_dict[pid][:-1]):
                            e = s + size
                            aval_pos += [i for i in range(s, e)]
                        aval_pos.sort()
                        # get the start position of the old task
                        cum_pos = position_dict[pid][0][0]
                        # divide the avaliable position into two parts
                        left_pos = aval_pos[:aval_pos.index(cum_pos)]
                        right_pos = aval_pos[aval_pos.index(cum_pos):]
                        # select the leftmost position from cum_pos
                        interval_picked = aval_pos[aval_pos.index(cum_pos):aval_pos.index(cum_pos)+new_size]
                        if len(interval_picked) < new_size:
                            # select the leftmost position from left_pos
                            interval_picked = left_pos[-(new_size-len(interval_picked)):] + interval_picked
                        # check if the position is continuous
                        interval_picked.sort()
                        # remove selected position from aval_pos
                        aval_pos = [i for i in aval_pos if i not in interval_picked]
                        start = [interval_picked[0]]
                        size = []
                        for i in range(new_size-1):
                            if interval_picked[i] != interval_picked[i+1]-1:
                                size.append(interval_picked[i]-start[-1]+1)
                                start.append(interval_picked[i+1])
                        size.append(interval_picked[-1]-start[-1]+1)
                        position_dict[pid] = [start, size, position_dict[pid][-1]]

                # pick a proper position for the new task in the available position
                for pid in new_pid:
                    p_size = rsc_map[pid]
                    # select the leftmost position
                    interval_picked = aval_pos[:p_size]
                    # check if the position is continuous
                    interval_picked.sort()
                    # remove selected position from aval_pos
                    aval_pos = [i for i in aval_pos if i not in interval_picked]
                    start = [interval_picked[0]]
                    size = []
                    for i in range(p_size-1):
                        if interval_picked[i] != interval_picked[i+1]-1:
                            size.append(interval_picked[i]-start[-1]+1)
                            start.append(interval_picked[i+1])
                    size.append(interval_picked[-1]-start[-1]+1)
                    position_dict[pid] = [start, size, True]
                
                # update the pre_rsc                                                
                pre_idx = rsc_map_idx
                pre_rsc = rsc_map
                empty_flag = len(pre_rsc) == 0
                if empty_flag:
                    empty_boader_s.append(rsc_map_idx)


        # set start and end time for each task: 
        #   if part of the task is in the warmup cycle or drain cycle, 
            # set the start and end time to the start and end time of the plot
        s, e = pre_idx*time_step, bin_temp_size*time_step
        if s < plot_start:
            s = plot_start
        if e > plot_end:
            e = plot_end
        if s >= plot_end or e <= plot_start: 
            continue

        if empty_flag:
            empty_boader_e.append(bin_temp_size)
        else:
            # plot the task layout
            for pid, size in pre_rsc.items():
                
                _p = init_p_list[pid]
                _p_name = _p.task.name
                _p_color = mcolors.XKCD_COLORS[colors[pid]]
                is_new = position_dict[pid][-1]

                for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                    # culculate the position of the bar
                    bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                    text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                    # plot the bar
                    if tool == "matplotlib":
                        ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)
                    elif tool == "bokeh":
                        ax.rect(x=s, y=bar_vertical_offset, width=e-s, height=vertical_size*vertical_grid_size, color=_p_color)
                    elif tool == "plotly":
                        fig.add_trace(go.Bar(x=[e-s], y=[bar_vertical_offset], marker_color=_p_color, orientation='h',
                                                width=vertical_size*vertical_grid_size, base=s), row=len(bin_list)-_SchedTab.id, col=1, show_trace_name=False)

                    # plot the text
                    if not plot_legend:
                        if is_new:
                            position_dict[pid][-1] = False
                            if tool == "matplotlib":
                                ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)
                            elif tool == "bokeh": 
                                ax.text(x=(s+e)/2, y=text_vertical_offset, text=_p_name, text_color='black', text_font_size='10pt', text_align='center')
                            elif tool == "plotly":
                                fig.add_annotation(x=(s+e)/2, y=text_vertical_offset, text=_p_name, showarrow=False, font=dict(color='black', size=txt_size), 
                                                    xref='x', yref='y', row=len(bin_list)-_SchedTab.id, col=1)

        # the vertical grid at the end of the bar
        if tool == "matplotlib":
            ax.axvline(e, color='black', linestyle='-', linewidth=0.5)
        elif tool == "bokeh":
            ax.line(x=[e, e], y=[bin_vertical_offset, bin_vertical_offset+bin_spatial_size*vertical_grid_size], color='black', line_width=0.5)
        elif tool == "plotly":
            fig.add_trace(go.Scatter(x=[e, e], y=[bin_vertical_offset, bin_vertical_offset+bin_spatial_size*vertical_grid_size], mode='lines',
                                        line=dict(color='black', width=0.5)), row=len(bin_list)-_SchedTab.id, col=1)

        # set the axis and title
        vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
        ticks = np.linspace(vs, vs+bin_spatial_size-1, 4, dtype=int)
        if tool == "matplotlib":
            ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
            ax.set_ylim(bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks, fontsize=txt_size)
        elif tool == "bokeh":
            ax.x_range = Range1d(plot_start-x_margin, plot_end+x_margin)
            ax.y_range = Range1d(bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin)
            ax.yaxis.ticker = ticks
            ax.yaxis.major_label_overrides = {i: str(i) for i in ticks}
            ax.yaxis.major_label_text_font_size = f"{txt_size}pt"
        elif tool == "plotly":            
            fig.update_xaxes(range=[plot_start-x_margin, plot_end+x_margin], row=len(bin_list)-_SchedTab.id, col=1)
            fig.update_yaxes(range=[bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin],
                                tickvals=ticks, ticktext=ticks, row=len(bin_list)-_SchedTab.id, col=1)
            
        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        if tool == "matplotlib":
            ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)
        elif tool == "bokeh":
            ax.text(x=plot_start, y=bin_vertical_offset, text=f"bin: {_SchedTab.name}({_SchedTab.id})", text_font_size=f"{txt_size}pt", text_align='left')
        elif tool == "plotly":
            fig.add_annotation(x=plot_start, y=bin_vertical_offset, text=f"bin: {_SchedTab.name}({_SchedTab.id})", showarrow=False, 
                                font=dict(color='black',), xref='x', yref='y', row=len(bin_list)-_SchedTab.id, col=1)
            
    # sub-figures shares the same x-axis
    # fig.subplots_adjust(hspace=0)
    if tool == "matplotlib":
        plt.setp([a.get_xticklabels() for a in fig.axes[1:]], visible=False)
    elif tool == "bokeh":
        pass

    # remove y axis
    # ax.get_yaxis().set_visible(False)

    if plot_legend:
        if tool == "matplotlib":
            # add legend
            ax = fig.axes[-1]
            from matplotlib.lines import Line2D
            legend_elements = []
            for i in range(len(init_p_list)): 
                legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=init_p_list[i].task.name))
            ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.2),
            ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
            # reset x ticks: text size 30, rotation 45, distance time_grid_size * 2
            # ticks format: .3f
            ax = fig.axes[0]
            ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
            ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
            ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
            ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
            # remove frame
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # set x axis label as Time (s), text size 30
            ax.set_xlabel("Time (s)", fontsize=txt_size) 
        elif tool == "bokeh":
            # add legend
            legend_it = []
            for i in range(len(init_p_list)): 
                legend_it.append((init_p_list[i].task.name, [fig.circle(x=[], y=[], color=colors[i], size=10)]))
            legend = Legend(items=legend_it, location=(0, 0))
            fig.add_layout(legend, 'right')
            # reset x ticks: text size 30, rotation 45, distance time_grid_size * 2
            # ticks format: .3f
            ax = fig.axes[0]
            ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
            ax.xaxis.ticker = np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end]
            ax.xaxis.major_label_overrides = {t: str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end]}
            ax.xaxis.major_label_text_font_size = f"{txt_size}pt"
            ax.xaxis.major_label_orientation = np.pi/4
            ax.xaxis.major_label_standoff = time_grid_size * tick_dens
            # remove frame
            # ax.outline_line_color = None
            # set x axis label as Time (s), text size 30
            ax.xaxis.axis_label = "Time (s)"
            ax.xaxis.axis_label_text_font_size = f"{txt_size}pt"
            

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
        elif tool == "bokeh":
            show(column(fig))
    # save the figure
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if tool == "matplotlib" or tool == "plotly":
            if "format" in kwargs and isinstance(kwargs["format"], list):
                fmt_list = kwargs.pop("format")
                if path_parse[-1] not in fmt_list:
                    kwargs["format"].append(path_parse[-1])
                for f in fmt_list:
                    save_path = ".".join(path_parse[:-1]) + "." + f
                    if tool == "matplotlib":
                        plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
                    elif tool == "plotly":
                        fig.write_image(save_path)
            elif "format" not in kwargs and len(path_parse) > 1: 
                kwargs["format"] = path_parse[-1]
            else:
                kwargs["format"] = "pdf"
                save_path = save_path + ".pdf"        
                if tool == "matplotlib":
                    plt.savefig(save_path, bbox_inches='tight', **kwargs)
                elif tool == "plotly":
                    fig.write_image(save_path)

        elif tool == "bokeh":
            # export svg
            # activate the SVG backend
            from bokeh.io import export_svgs, export_png
            export_svgs(column(fig), filename=save_path+".svg")
            export_png(column(fig), filename=save_path+".png")



def get_task_layout(bin_list:List[SchedulingTableInt], init_p_list:List[ProcessInt]
                        , show:bool=False, save:bool=False, save_path:str="task_layout.pdf", **kwargs):

    import matplotlib.colors as mcolors
    import matplotlib as mpl
    cmap = mpl.colormaps['viridis']
    colors=list(mcolors.XKCD_COLORS.keys())
    
    # plot timeline and task name bin by bin
    # and select color for the task automatically
    fig = plt.figure(figsize=(50, 20))
    vertical_grid_size = 1
    
    for _SchedTab in bin_list:
        tab_temp_size = len(_SchedTab.scheduling_table)
        ax = fig.add_subplot(len(bin_list), 1, len(bin_list)-_SchedTab.id)
        vertical_offset = 0
        horizen_grid = set()
        bin_pack_result = _SchedTab.index_occupy_by_id()
        ordered_k = [k for k, v in sorted(bin_pack_result.items(), key=lambda item: item[1][1])]
        for pid in ordered_k:
            alloc_info = bin_pack_result[pid]
            _p = init_p_list[pid]
            _p_name = _p.task.name
            _p_color = mcolors.XKCD_COLORS[colors[pid]]
            for s, size, l in zip(*alloc_info):
                ax.broken_barh([(s, l)], (vertical_offset*vertical_grid_size, size*vertical_grid_size), facecolors=_p_color)
                ax.text(s+l//2, (vertical_offset+size//2)*vertical_grid_size, _p_name, ha='center', va='center', color='black', fontsize=10)
                horizen_grid.add(s)
                horizen_grid.add(s+l)
            vertical_offset += size
        # ax.set_ylim(0, vertical_offset*vertical_grid_size)
        ax.set_xlim(0, tab_temp_size)
        ax.set_xlabel('Time')
        # add bin name
        ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})")
        # plot the grid
        for x in horizen_grid:
            ax.axvline(x, color='black', linestyle='-', linewidth=0.5)
    # sub-figures shares the same x-axis
    # fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    # plot the result
    if show:
        plt.show()
    # save the figure
    if "format" not in kwargs and save_path.split(".")[-1] == "pdf" and save:
        kwargs["format"] = "pdf"
    plt.savefig(save_path, **kwargs)


class BinGenSelInt(object):
    def __init__(self, tab_temp_size:int):
        self.tab_temp_size = tab_temp_size
        self.init_bin_list()
    
    def _new_bin(self, id, size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, self.tab_temp_size, id=id, name=name)
    
    def init_bin_list(self, bin_list=[], bin_name_list=[]):
        self.bin_list = bin_list
        self.bin_name_list = bin_name_list

    def init_gen(self, gen:Iterator):
        self.iter_next_bin_obj = gen

class BinSelInt(BinGenSelInt): 
    """
    Given a set of bins, select a bin to allocate a task
    """
    pass
    
class BinGenInt(BinGenSelInt):
    """
    Given a generator, find a proper bins to allocate a task, if not found, create a new bin
    """
    def pick(self,):
        bin = next(self.iter_next_bin_obj)
        self.bin_list.append(bin)
        self.bin_name_list.append(bin.name)
        return bin

def new_bin(spatial_size:int, temporal_size:int, id:int = 0, name:str = "bin"):
    SchedTab = SchedulingTableInt(spatial_size, temporal_size, id=id, name=name)
    return SchedTab


if __name__ == "__main__": 
    # create a task set
    # first branch: have free cores and free slots at beginning
    # t1 [15, 19] 25 rsc and 2 slot
    # t2 [3, 12] 24 rsc and 7 slot
    # second branch: select a interval with enough free cores and free slots
    # t3 [0, 20] 10 rsc and 4 slot
    # thrird branch: evently distribute the resources in the expected interval
    # t4 [0, 7] 10 rsc and 4 slot
    # last branch: As soon as possible
    # t5 [0, 16] 10 rsc and 8 slot


    t1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                ERT=15, ddl=19, period=30, exp_comp_t=2, 
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t2 = TaskInt(task_name="task2", task_id=2, task_flag="moveable", timing_flag="deadline",
                ERT=3, ddl=12, period=30, exp_comp_t=7,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t3 = TaskInt(task_name="task3", task_id=3, task_flag="moveable", timing_flag="deadline", 
                ERT=0, ddl=20, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t4 = TaskInt(task_name="task4", task_id=4, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=7, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t5 = TaskInt(task_name="task5", task_id=5, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=16, period=30, exp_comp_t=8,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)

    task_list:List[TaskInt] = [None for i in range(10)]
    alloc_info = [None for i in range(10)]
    require_rsc_size = [0 for i in range(10)]
    task_list[0:5] = [t1, t2, t3, t4, t5]
    require_rsc_size[0:5] = [25, 24, 10, 10, 10]

    # create a scheduling table
    scheduling_table = SchedulingTableInt(30, 20)
    # allocate resources

    # alloc_info[0] = scheduling_table.insert_task(t1, 25, t1.ERT, t1.ddl, t1.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[1] = scheduling_table.insert_task(t2, 24, t2.ERT, t2.ddl, t2.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[2] = scheduling_table.insert_task(t3, 10, t3.ERT, t3.ddl, t3.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[3] = scheduling_table.insert_task(t4, 10, t4.ERT, t4.ddl, t4.exp_comp_t, verbose=True)
    # alloc_info[4] = scheduling_table.print_scheduling_table()
    # alloc_info[5] = scheduling_table.insert_task(t5, 10, t5.ERT, t5.ddl, t5.exp_comp_t, verbose=True)
    # # print the scheduling table
    # scheduling_table.print_scheduling_table()

    pid2name = []
    pid = 0
    for task in task_list[0:5]: 
        # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
        r = task.get_release_time()
        d = task.get_deadline_time()
        p = task.make_process(r, d, pid)
        pid += 1
        pid2name.append(p)

    for i in range(5):
        print(f"task {i} allocation\n")
        alloc_info[i] = scheduling_table.insert_task(pid2name[i], require_rsc_size[i], 
                                                     pid2name[i].release_time, pid2name[i].deadline, 
                                                     pid2name[i].exp_comp_t, verbose=True)
        print("="*20)
        scheduling_table.print_scheduling_table()
    
    print("occupy by id:", scheduling_table.index_occupy_by_id())
    
    # release resources
    for i in range(5):
        print("before release")
        scheduling_table.print_scheduling_table()
        if alloc_info[i] is not None:
            print(f"task {pid2name[i].pid}({i}) release:")
            print([*alloc_info[i]])
            scheduling_table.release(pid2name[i], *alloc_info[i][1:], verbose=True)
        print("after release")
        scheduling_table.print_scheduling_table()