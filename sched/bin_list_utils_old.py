"""bin_list_utils_old.py — moved dead code from bin_list_utils.py (B5-MOVE-001, historical versions).
Contents: get_task_layout (pre-compact), get_sparse_flops + update_sparse_dict (old sim-chain sparse flops).

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

