"""bin_list_utils_unused.py — moved dead code from bin_list_utils.py (B5-MOVE-002, unfinished standalone).
Contents: get_task_layout_sparse + add_bar/add_text/add_v_grid (multi-backend viz attempt, never landed; backend param only has matplotlib branch).

Moved, not rewritten. Archive only.
Recover: git checkout archive/test_pipeline-20260612 -- sched/bin_list_utils.py
"""

"""bin_list_utils_old.py — moved dead code from bin_list_utils.py (B5-CLEANUP).
Contents (historical versions, superseded by compact variants / dead sim-chain):
  get_task_layout (pre-compact), get_task_layout_sparse + add_bar/add_text/add_v_grid (old viz),
  get_sparse_flops + update_sparse_dict (old sim-chain sparse flops).
Moved, not rewritten. Archive only. Recover: git checkout archive/test_pipeline-20260612 -- sched/bin_list_utils.py
"""

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

