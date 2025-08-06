from typing import TYPE_CHECKING
from sched.scheduling_table import SchedulingTableInt

from typing import List, Dict, Tuple, Union, Optional, Iterable, Iterator, Collection
from collections import OrderedDict

from functools import reduce
import math
import os
import numpy as np
from model.resource_agent import Resource_model_int
from task.task_agent import ProcessInt, TaskInt
from global_var import fork_pid_base
from global_var import *
from utils import load_pickle

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
# import networkx as nx
# import matplotlib as mpl
# from bokeh.plotting import figure, show
# from bokeh.models import ColumnDataSource, HoverTool, Range1d, LabelSet, Label, Legend
# from bokeh.layouts import row, column, gridplot
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
from model.lru import LRUCache

def get_task_layout_compact(bin_list:List[SchedulingTableInt], pid2name:Dict[int, str], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = hyper_p * (warmup)
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

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
    fig, axes = plt.subplots(nrows=len(bin_list),ncols=1,sharex=True,figsize=fig_size) 

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    for bin_idx, _SchedTab in enumerate(bin_list): 
        # bin_temp_size = len(_SchedTab.scheduling_table)
        # bin_spatial_size = _SchedTab.scheduling_table[0].size
        bin_spatial_size = _SchedTab.num_resources
        bin_temp_size = _SchedTab.temp_size

        ax = axes[len(bin_list)-bin_idx-1] if len(bin_list) > 1 else axes                     

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

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    # plot the task layout
                    for pid, size in pre_rsc.items():
                        _p_name = pid2name[pid]
                        _p_color = mcolors.XKCD_COLORS[colors[pid%len(colors)]]
                        is_new = position_dict[pid][-1]

                        for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                            # culculate the position of the bar
                            bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                            text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                            # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                            ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)
                            # plot the text
                            if not plot_legend:
                                if is_new:
                                    position_dict[pid][-1] = False
                                    ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

                # the vertical grid at the end of the bar
                ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

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
                for pid in sorted(old_pid):
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
                        # divide the available position into two parts
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
                    # search for a gap in the available positions that can accommodate the task's new size
                    gap_start = None
                    gap_size = 0
                    for pos in aval_pos:
                        if gap_start is None:
                            gap_start = pos
                        gap_size += 1
                        if gap_size == p_size:
                            break
                        if pos + 1 not in aval_pos:
                            gap_start = None
                            gap_size = 0

                    if gap_start is not None and gap_size == p_size:
                        # allocate the task to the found gap
                        start = [gap_start]
                        size = [gap_size]
                        position_dict[pid] = [start, size, True]
                        # remove the selected positions from aval_pos
                        aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                    else:
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
        if s < plot_end or e > plot_start: 
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end

            if empty_flag:
                empty_boader_e.append(bin_temp_size)
            else:
                # plot the task layout
                for pid, size in pre_rsc.items():
                    _p_name = pid2name[pid]
                    _p_color = mcolors.XKCD_COLORS[[colors[pid%len(colors)]]]
                    is_new = position_dict[pid][-1]

                    for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                        # culculate the position of the bar
                        bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                        text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                        # plot the bar
                        ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)

                        # plot the text
                        if not plot_legend:
                            if is_new:
                                position_dict[pid][-1] = False
                                ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

            # the vertical grid at the end of the bar
            ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

        # set the axis and title
        vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
        ticks = np.linspace(vs, vs+bin_spatial_size-1, 4, dtype=int)
        ax.set_ylim(bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=txt_size, rotation=45)

        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)

    # only set x axis for the bottom plot
    ax = axes[len(bin_list)-1] if len(bin_list) > 1 else axes
    ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
    ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
    ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
    ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
    ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
    ax.set_xlabel("Time (s)", fontsize=txt_size) 

    if plot_legend:
        # add legend to the top plot
        ax = axes[0] if len(bin_list) > 1 else axes
        from matplotlib.lines import Line2D
        legend_elements = []
        # for i in range(len(init_p_list)): 
        for i, pid in enumerate(pid2name.keys()):
            legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=pid2name[pid]))
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.2),
        ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
            

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
    # save the figure
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if tool == "matplotlib":
            if "format" in kwargs and isinstance(kwargs["format"], list):
                fmt_list = kwargs.pop("format")
                if path_parse[-1] not in fmt_list:
                    kwargs["format"].append(path_parse[-1])
                for f in fmt_list:
                    save_path = ".".join(path_parse[:-1]) + "." + f
                    if tool == "matplotlib":
                        plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
            elif "format" not in kwargs and len(path_parse) > 1: 
                kwargs["format"] = path_parse[-1]
            else:
                kwargs["format"] = "pdf"
                save_path = save_path + ".pdf"        
                if tool == "matplotlib":
                    plt.savefig(save_path, bbox_inches='tight', **kwargs)

def get_task_layout_compact1bin(bin_list:List[SchedulingTableInt], pid2name:Dict[int, str], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    if plot_start is None:
        plot_start = hyper_p * (warmup)
    if plot_end is None:
        plot_end = hyper_p * (n_p+warmup+drain)

    colors=list(mcolors.XKCD_COLORS.keys())
    
    base_vertical_offset = 0
    y_margin = 0.5 
    time_grid_size = 0.004
    x_margin = time_grid_size

    # plot timeline and task name bin by bin
    # and select color for the task automatically
    tot_cores = sum([_SchedTab.num_resources for _SchedTab in bin_list])
    height = math.ceil(tot_cores//256)
    if plot_legend:
        fig_size = (40, 20+30*height)
    else:
        fig_size = (60, 30*height)
    fig, axes = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=fig_size) 

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    ax = axes           
    for bin_idx, _SchedTab in enumerate(bin_list): 
        # bin_temp_size = len(_SchedTab.scheduling_table)
        # bin_spatial_size = _SchedTab.scheduling_table[0].size
        bin_spatial_size = _SchedTab.num_resources
        bin_temp_size = _SchedTab.temp_size


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

            if rsc_map == pre_rsc:
                continue
            else:
                if empty_flag:
                    empty_boader_e.append(rsc_map_idx)
                else:
                    # plot the task layout
                    for pid, size in pre_rsc.items():
                        _p_name = pid2name[pid]
                        _p_color = mcolors.XKCD_COLORS[colors[pid%len(colors)]]
                        is_new = position_dict[pid][-1]

                        for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                            # culculate the position of the bar
                            bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                            text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                            # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                            ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)
                            # plot the text
                            if not plot_legend:
                                if is_new:
                                    position_dict[pid][-1] = False
                                    ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

                # the vertical grid at the end of the bar
                ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

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
                for pid in sorted(old_pid):
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
                        # divide the available position into two parts
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
                    # search for a gap in the available positions that can accommodate the task's new size
                    gap_start = None
                    gap_size = 0
                    for pos in aval_pos:
                        if gap_start is None:
                            gap_start = pos
                        gap_size += 1
                        if gap_size == p_size:
                            break
                        if pos + 1 not in aval_pos:
                            gap_start = None
                            gap_size = 0

                    if gap_start is not None and gap_size == p_size:
                        # allocate the task to the found gap
                        start = [gap_start]
                        size = [gap_size]
                        position_dict[pid] = [start, size, True]
                        # remove the selected positions from aval_pos
                        aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                    else:
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
        if s < plot_end or e > plot_start: 
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end

            if empty_flag:
                empty_boader_e.append(bin_temp_size)
            else:
                # plot the task layout
                for pid, size in pre_rsc.items():
                    _p_name = pid2name[pid]
                    _p_color = mcolors.XKCD_COLORS[[colors[pid%len(colors)]]]
                    is_new = position_dict[pid][-1]

                    for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                        # culculate the position of the bar
                        bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                        text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                        # plot the bar
                        ax.broken_barh([(s, e-s)], (bar_vertical_offset, vertical_size*vertical_grid_size), facecolors=_p_color)

                        # plot the text
                        if not plot_legend:
                            if is_new:
                                position_dict[pid][-1] = False
                                ax.text((s+e)/2, text_vertical_offset, _p_name, ha='center', va='center', color='black', fontsize=10)

            # the vertical grid at the end of the bar
            ax.axvline(e, color='black', linestyle='-', linewidth=0.5)

        # add a horizontal line to seperate bins
        ax.axhline(bin_vertical_offset, color='black', linestyle='-', linewidth=0.5)
        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)

    # only set x axis for the bottom plot
    ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
    ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
    ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
    ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
    ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
    ax.set_xlabel("Time (s)", fontsize=txt_size) 

    # set the axis and title
    vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
    ticks = np.linspace(0, vs, 4, dtype=int)
    ax.set_ylim(-y_margin, bin_vertical_offset+y_margin)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=txt_size, rotation=45)

    if plot_legend:
        # add legend to the top plot
        from matplotlib.lines import Line2D
        legend_elements = []
        # for i in range(len(init_p_list)): 
        for i, pid in enumerate(pid2name.keys()):
            legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=pid2name[pid]))
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.2),
        ncol=4, fancybox=True, shadow=True, fontsize=txt_size)
            

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
    # save the figure
    if save: 
        # if format is given in file name, use it
        # by default, use pdf
        path_parse = save_path.split(".")
        if tool == "matplotlib":
            if "format" in kwargs and isinstance(kwargs["format"], list):
                fmt_list = kwargs.pop("format")
                if path_parse[-1] not in fmt_list:
                    kwargs["format"].append(path_parse[-1])
                for f in fmt_list:
                    save_path = ".".join(path_parse[:-1]) + "." + f
                    if tool == "matplotlib":
                        plt.savefig(save_path, bbox_inches='tight', format=f,**kwargs)
            elif "format" not in kwargs and len(path_parse) > 1: 
                kwargs["format"] = path_parse[-1]
            else:
                kwargs["format"] = "pdf"
                save_path = save_path + ".pdf"        
                if tool == "matplotlib":
                    plt.savefig(save_path, bbox_inches='tight', **kwargs)

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
            _p_color = mcolors.XKCD_COLORS[[colors[pid%len(colors)]]]
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
    print("save figure to", save_path)

def get_task_layout_sparse(bin_list:List[SchedulingTableInt], pid2name:Dict[int, str], time_step:float = 1e-6,
                    show=False, save=False, save_path="task_layout_compact.pdf", 
                    hyper_p=0.1, n_p=1, warmup=False, drain=False,
                    plot_legend=False,
                    plot_start=None, plot_end=None, 
                    tick_dens = 1, txt_size = 30, *, tool="matplotlib",
                    **kwargs):

    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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
    

    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    for bin_idx, _SchedTab in enumerate(bin_list): 
        _SchedTab: SchedulingTableInt
        # bin_temp_size = len(_SchedTab.scheduling_table)
        # bin_spatial_size = _SchedTab.scheduling_table[0].size
        bin_spatial_size = _SchedTab.num_resources

        if tool == "matplotlib":
            ax = fig.add_subplot(len(bin_list), 1, len(bin_list)-_SchedTab.id)
        
        _SchedTab.to_sparse_dict(0)
        # build a position dict
        position_dict = {}
        aval_pos = [i for i in range(bin_spatial_size)]

        while _SchedTab.sparse_idx_next:
            pre_idx, curr_cfg, slot_num = _SchedTab.sparse_list[_SchedTab.sparse_idx]
            if _SchedTab.sparse_idx == 0:
                rsc_map_idx = 0
                cum_pos = 0
                for k,v in curr_cfg.items():
                    position_dict[k] = [[cum_pos], [v], True] 
                    aval_pos = [i for i in aval_pos if i not in range(cum_pos, cum_pos+v)]
                    cum_pos += v
            else:
                rsc_map_idx = _SchedTab.sparse_list[_SchedTab.sparse_idx_prev][0] + _SchedTab.sparse_list[_SchedTab.sparse_idx_prev][2] 
                rsc_map = curr_cfg
                pre_rsc = _SchedTab.sparse_list[_SchedTab.sparse_idx_prev][1]

                # update the position dict
                new_pid = set(rsc_map.keys()) - set(pre_rsc.keys())
                expired_pid = set(pre_rsc.keys()) - set(rsc_map.keys())
                old_pid = set(pre_rsc.keys()) - expired_pid

                # remove the expired task from the position dict
                for pid in expired_pid:
                    # release the old position
                    for s, size in zip(*position_dict[pid][:-1]):
                        e = s + size
                        aval_pos += [i for i in range(s, e)]
                    position_dict.pop(pid)
                
                # check if the old task's allocation is changed
                size_plus = []
                size_minus = []
                for pid in sorted(old_pid):
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
                        # divide the available position into two parts
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
                    # search for a gap in the available positions that can accommodate the task's new size
                    gap_start = None
                    gap_size = 0
                    for pos in aval_pos:
                        if gap_start is None:
                            gap_start = pos
                        gap_size += 1
                        if gap_size == p_size:
                            break
                        if pos + 1 not in aval_pos:
                            gap_start = None
                            gap_size = 0

                    if gap_start is not None and gap_size == p_size:
                        # allocate the task to the found gap
                        start = [gap_start]
                        size = [gap_size]
                        position_dict[pid] = [start, size, True]
                        # remove the selected positions from aval_pos
                        aval_pos = [i for i in aval_pos if not gap_start <= i < gap_start + gap_size]
                    else:
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
                
            _SchedTab.idx_plus_1()

            s = pre_idx*time_step
            if pre_idx != rsc_map_idx:
                # add the vertical grid at the beginning of the bar
                if s >= plot_start and s <= plot_end:
                    add_v_grid(s, ax)

            rsc_map_idx = pre_idx + slot_num
            e = rsc_map_idx*time_step
            # sparse list contains no empty cfg
            assert len(curr_cfg) > 0 

            # set start and end time for each task: 
            #   if part of the task is in the warmup cycle or drain cycle, 
                # set the start and end time to the start and end time of the plot
            if s < plot_start:
                s = plot_start
            if e > plot_end:
                e = plot_end
            if s >= plot_end or e <= plot_start: 
                continue

            # plot the task layout
            for pid, size in curr_cfg.items():
                _p_name = pid2name[pid]
                _p_color = mcolors.XKCD_COLORS[colors[pid%len(colors)]]
                is_new = position_dict[pid][-1]

                for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
                    # culculate the position of the bar
                    bar_vertical_offset = bin_vertical_offset + vertical_s*vertical_grid_size
                    text_vertical_offset = bar_vertical_offset + 0.5*vertical_size*vertical_grid_size
                    # plot the horizontal bar: from s to e, with height vertical_size*vertical_grid_size, and vertical offset bar_vertical_offset
                    h_pos, h_size, v_pos, v_size = s, e-s, bar_vertical_offset, vertical_size*vertical_grid_size
                    add_bar(h_pos, h_size, v_pos, v_size, _p_color, ax)
                    # plot the text
                    if not plot_legend:
                        if is_new:
                            position_dict[pid][-1] = False
                            add_text((s+e)/2, text_vertical_offset, _p_name, ax)
            # the vertical grid at the end of the bar
            add_v_grid(e, ax)

            
        # set the axis and title
        vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
        ticks = np.linspace(vs, vs+bin_spatial_size-1, 4, dtype=int)
        if tool == "matplotlib":
            ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
            ax.set_ylim(bin_vertical_offset-y_margin, bin_vertical_offset+bin_spatial_size*vertical_grid_size+y_margin)
            ax.set_yticks(ticks)
            ax.set_yticklabels(ticks, fontsize=txt_size)

        # set yticks
        # add bin name
        # ax.set_title(f"bin: {_SchedTab.name}({_SchedTab.id})", fontsize=txt_size)
        bin_vertical_offset += bin_spatial_size* vertical_grid_size 
        if tool == "matplotlib":
            ax.text(plot_start, bin_vertical_offset, f"bin: {_SchedTab.name}({_SchedTab.id})", ha='left', va='top', fontsize=txt_size)

    # sub-figures shares the same x-axis
    # fig.subplots_adjust(hspace=0)
    if tool == "matplotlib":
        plt.setp([a.get_xticklabels() for a in fig.axes[1:]], visible=False)

    if plot_legend:
        if tool == "matplotlib":
            # add legend
            ax = fig.axes[-1]
            from matplotlib.lines import Line2D
            legend_elements = []
            # for i in range(len(init_p_list)): 
            for i, pid in enumerate(pid2name.keys()):
                legend_elements.append(Line2D([0], [0], color=mcolors.XKCD_COLORS[colors[i]], lw=4, label=pid2name[pid]))
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

    # plot the result
    if show:
        if tool == "matplotlib":
            plt.show()
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


def add_bar(h_pos, h_size, v_pos, v_size, _p_color, ax, backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        ax.broken_barh([(h_pos, h_size)], (v_pos, v_size), facecolors=_p_color)
    else:
        raise NotImplementedError

def add_text(h_pos, v_pos, _p_name, ax, backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        ax.text(h_pos, v_pos, _p_name, ha='center', va='center', color='black', fontsize=10)

def add_v_grid(v_pos, ax, backend="matplotlib", **kwargs):
    if backend == "matplotlib":
        ax.axvline(v_pos, color='black', linestyle='-', linewidth=0.5)

def Bin_list_print(bin_list, glb_p_list, timestep):
    pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
    for _SchedTab in bin_list:
        print("=====================================\n")
        print(f"Scheduling Table of {_SchedTab.name}({_SchedTab.id}):")
        _SchedTab.print_scheduling_table(pid2name, timestep)
        print("=====================================\n")
    
    print("=====================================\n")
    print("bin_pack_result:")
    print("=====================================\n")
    for _SchedTab in bin_list:
        _SchedTab.print_alloc_detail(pid2name, timestep)
    layout = {_bin.name:_bin.num_resources for _bin in bin_list}
    print("max_core_num:", sum(layout.values()))
    print(f"max_core_layout: {layout}")

# def generate_dependency_table(bin_list, job_graph_nx:nx.DiGraph, pname2pid, pid2pname): 
#     num_processors = len(bin_list)
#     num_timesteps = len(bin_list[0])
#     dependency_table = np.empty_like(bin_list, dtype=object)
#     dependency_table.fill([])  # Initialize all items with an empty list
    
#     num_processors = bin_list.shape[1]

#     for _Sched_tab in bin_list:
#         for cfg_slot_s, next_cfg, cfg_slot_num in _Sched_tab.sparse_list: 
#             next_cfg:RscMapInt
#             for pid in next_cfg.keys():
#                 job_n = pid2pname[pid]
#                 for succ_n, datadict in job_graph_nx.succ[job_n].items(): 
#                     succ_pid = pname2pid[succ_n]
    
#     # Iterate over each timestep and processor in the scheduling table
#     for timestep, processor in np.ndindex(bin_list.shape):
#         jobs_executed = bin_list[timestep, processor]
        
#         # Iterate over each job executed by the current processor
#         for job_id in jobs_executed:
#             dependencies = job_graph.get_dependencies(job_id)  # Get the dependencies of the current job
            
#             # Iterate over each dependency of the current job
#             for dependency in dependencies:
#                 # Find the processors that executed the predecessor jobs
#                 predecessor_processors = np.where(bin_list[:, :] == dependency)[1]
                
#                 # Append the predecessor processors to the dependency table
#                 dependency_table[timestep, processor].extend(predecessor_processors)
    
#     return dependency_table

def get_sparse_flops(bin_list:List[SchedulingTableInt], glb_p_list: List[ProcessInt], timestep, event_range):

    process_dict = {_p.pid:_p for _p in glb_p_list}
    # build event list
    process_stim_dict = OrderedDict()
    for pid, _p in process_dict.items():
        _p:ProcessInt
        stimu_tab = _p.task.extract_sensor_event(event_range, verbose=False)
        l = []
        for stimu_t in stimu_tab:
            start_t, ddl_t = elim_nume_error(stimu_t+_p.task.ERT), elim_nume_error(stimu_t+_p.task.ERT+_p.task.ddl)
            # quantize the start time and ddl time
            slot_s = int(math.ceil(start_t/timestep)) 
            slot_e = int(math.floor(ddl_t/timestep))
            l.append([_p.pid, slot_s, slot_e, _p.task.name, stimu_t]) 
        process_stim_dict[pid] = l

    rsc_recoder = {}
    for _bin in bin_list:
        _bin:SchedulingTableInt
        # # [task_id] = (s, size, l)
        # bin_pack_result:Dict[int, Tuple[List[int], List[int], List[int]]] = _bin.index_occupy_by_id()
        # # merge the result to the recoder, also add a extra item of bin_id 
        # for pid, item in bin_pack_result.items():
        #     bin_id = [_bin.id] * len(item[0])
        #     if pid in rsc_recoder:
        #         s, size, l, b_id = rsc_recoder[pid]
        #         rsc_recoder[pid] = [s+item[0], size+item[1], l+item[2], b_id+bin_id]
        #     else:
        #         rsc_recoder[pid] = [item[0], item[1], item[2], bin_id]

        # update the rsc_recoder by the sparse list
        sparse_list = _bin.sparse_list
        for pre_idx, pre_rsc, slot_num in sparse_list:
            for pid, size in pre_rsc.items():
                if pid in rsc_recoder:
                    s, size_l, l, b_id = rsc_recoder[pid]
                    rsc_recoder[pid] = [s+[pre_idx], size_l+[size], l+[slot_num], b_id+[_bin.id]]
                else:
                    rsc_recoder[pid] = [[pre_idx], [size], [slot_num], [_bin.id]]


    for pid, item in rsc_recoder.items():
        s, size, l, b_id = item
        # sort the result by the start time
        # sort s and update the size and l
        s, size, l, b_id = zip(*sorted(zip(s, size, l, b_id), key=lambda x: x[0]))
        rsc_recoder[pid] = [s, size, l, b_id]
    # sort the record by pid
    rsc_recoder = OrderedDict(sorted(rsc_recoder.items(), key=lambda item: item[0]))
    
    # inner dict is indexed by slot index
    # outter dict is indexed by bin index
    sparse_flops_dict = OrderedDict()
    sparse_cores_dict = OrderedDict()
    sparse_event_dict = OrderedDict()
    # init the above dict
    for bin_idx in range(len(bin_list)):
        sparse_flops_dict[bin_idx] = {}
        sparse_cores_dict[bin_idx] = {}
        sparse_event_dict[bin_idx] = {}
    # event pattern:
    # f"start-{pid:d}_{j:d}", f"complete-{pid:d}_{j:d}", f"migrate-{pid:d}_from_{pre_bin_idx}", f"migrate-{pid:d}_to_{next_bin_idx}"
    #  start
    #  ts
    #  complete
    #  migrate
    
    # check legelty
    assert set(rsc_recoder.keys()) == set(process_stim_dict.keys())
    for pid in rsc_recoder.keys():
        alloc_item = rsc_recoder[pid]
        stim_item = process_stim_dict[pid]
        i=0
        cum_ops = 0
        ref_flops = process_dict[pid].totcpu
        s_j_istart = 0 # find the start chunk idx in the alloc_item for jth stimulus
        
        for j in range(len(stim_item)):
            # start the jth stimulus
            
            # get stimu info
            s_j = stim_item[j][1]
            e_j = stim_item[j][2]
            pid = stim_item[j][0]
            s_j_istart = i
            
            # set the start event
            if i < len(alloc_item[0]):
                s_i = alloc_item[0][i]
                bin_idx = alloc_item[3][i]
                update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, item={f"start-{pid:d}_{j:d}"}, item_type=set)
            
            while i < len(alloc_item[0]):
                # get the i-th slot info, from the s_j_start
                s_i = alloc_item[0][i]
                size_i = alloc_item[1][i]
                l_i = alloc_item[2][i]
                e_i = alloc_item[0][i] + alloc_item[2][i]
                pre_bin_idx = alloc_item[3][i-1 if i>0 else 0]
                bin_idx = alloc_item[3][i]
                next_bin_idx = alloc_item[3][i+1 if i<len(alloc_item[0])-1 else len(alloc_item[0])-1]
                
                if pre_bin_idx != bin_idx and i > s_j_istart:
                    update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, 
                                       item={f"migrate-{pid:d}_from_{pre_bin_idx}"}, item_type=set)
                if cum_ops >= ref_flops:
                    cum_ops = 0
                    break
                if e_i <= e_j:
                    ops = size_i * l_i * timestep * FLOPS_PER_CORE
                    ops_tbd = min(ops, ref_flops-cum_ops)
                    cum_ops += ops
                    update_sparse_dict(bin_idx, pid, s_i, sparse_flops_dict, item=ops_tbd, item_type=float)
                    update_sparse_dict(bin_idx, pid, s_i, sparse_cores_dict, item=size_i, item_type=int)
                    if ops>ops_tbd:
                        update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, item={f"complete-{pid:d}_{j:d}"}, item_type=set)
                    elif next_bin_idx != bin_idx:
                        update_sparse_dict(bin_idx, pid, s_i, sparse_event_dict, 
                                        item={f"migrate-{pid:d}_to_{next_bin_idx:d}"}, item_type=set)

                    i += 1
                else:
                    assert cum_ops >= ref_flops
                    cum_ops = 0
                    break
        # NOTE: check if all the allocation are found the corresponding stimulus
        assert i == len(alloc_item[0])
    # NOTE:check whether the sparse_cores and the sparse_flops have the same length with the sparse_list
    for bin_idx in range(len(bin_list)):
        # verify the sparse index matching
        assert set(sparse_flops_dict[bin_idx].keys()) == set(sparse_cores_dict[bin_idx].keys()) 
        # event keys is a subset of slot keys
        assert set(sparse_event_dict[bin_idx].keys()) <= set(sparse_flops_dict[bin_idx].keys())

        # sort the dict by key
        sorted_slot_n = sorted(sparse_flops_dict[bin_idx].keys())
        sparse_cores = [sparse_cores_dict[bin_idx][k] for k in sorted_slot_n]
        sparse_flops = [sparse_flops_dict[bin_idx][k] for k in sorted_slot_n]
        sparse_event = [sparse_event_dict[bin_idx][k] if k in sparse_event_dict[bin_idx] else {} for k in sorted_slot_n ]

        sparse_list = bin_list[bin_idx].sparse_list
        assert len(sparse_flops) == len(sparse_list)
        assert len(sparse_cores) == len(sparse_list)
        # check have same keys
        for i in range(len(sparse_list)): 
            # has the same slot index
            assert sparse_list[i][0] == sorted_slot_n[i]
            # has the same task id
            assert set(sparse_list[i][1].keys()) == set(sparse_flops[i].keys())
            slot_n = sparse_list[i][0]
            assert set(sparse_list[i][1].keys()) >= set(sparse_event[i].keys())
        # sort the dict by key
        sparse_flops_dict[bin_idx] = OrderedDict(zip(sorted_slot_n, sparse_flops))
        sparse_cores_dict[bin_idx] = OrderedDict(zip(sorted_slot_n, sparse_cores))
        sparse_event_dict[bin_idx] = OrderedDict(zip(sorted_slot_n, sparse_event))
        bin_list[bin_idx].sparse_flops = list(zip(sorted_slot_n, sparse_flops))
        bin_list[bin_idx].sparse_cores = list(zip(sorted_slot_n, sparse_cores))
        bin_list[bin_idx].sparse_event = list(zip(sorted_slot_n, sparse_event))
        bin_list[bin_idx].alloc_mod = "proper"
    return sparse_flops_dict, sparse_cores_dict, sparse_event_dict

def update_sparse_dict(bin_idx, pid, s_i, sparse_dict, item, item_type):
    if s_i not in sparse_dict[bin_idx]:
        sparse_dict[bin_idx][s_i] = {}
    if hasattr(item_type, "update"):
        old = sparse_dict[bin_idx][s_i].get(pid, item_type())
        old.update(item)
        sparse_dict[bin_idx][s_i].update({pid:old})
    else:
        sparse_dict[bin_idx][s_i].update({pid:item})

