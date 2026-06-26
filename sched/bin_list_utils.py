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



