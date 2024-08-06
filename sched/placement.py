from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from sched.scheduler_agent import Scheduler

def core_mapping_1d(core_num_list:List[int]):
    """
    map the cores to the partitions
    """
    core_mapping = {}
    start = 0
    for i in range(len(core_num_list)):
        end = start + core_num_list[i]
        core_mapping[i] = list(range(start, end))
    return core_mapping


def update_phy_posi(sched:Scheduler, position_dict, pre_rsc, rsc_map, expired_pid, old_pid, new_pid):
    # update the position dict
    used_position = []
    # for pid in old_pid:
    #     p_size = rsc_map[pid]
    #     # set the is_new flag to False
    #     position_dict[pid][-1] = False
    #     for s, size in zip(*position_dict[pid][:-1]):
    #         e = s + size
    #         used_position += [i for i in range(s, e)]

    # remove the expired task from the position dict
    for pid in expired_pid:
        position_dict.pop(pid)

    for pid in position_dict.keys():
        for s, size in zip(*position_dict[pid][:-1]):
            e = s + size
            used_position += [i for i in range(s, e)]               

    # aval_pos = [i for i in range(bin_spatial_size) if i not in used_position]
    aval_pos = [i for i in sched.core_map if i not in used_position]
    
    # check if the old task's allocation is changed
    # if so, release the old position and allocate the new one
    # to release the data transfering overhead, we try to allocate the new position as close as possible to the old one
    # TODO: consider the data transfering overhead
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
            # position_dict[pid] = [start, size, True]
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

