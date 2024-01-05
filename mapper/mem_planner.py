
"""
Input: block list I: [(s1, r1, c1), (s2, r2, c2), ...]
    Each triple (si,ri,ci) is an axis-parallel rectangle: 
        (ri,ci) is the interval on the x-axis
        si is the length on the y-axis
        the rectangles along the y-axis while stay fixed on x-axis
Output: a projection alpha of I onto the y-axis, which organized as 
    1. a a list of y position
    2. or a list of intervals consisting of interv_s, interv_e, and a dict of 
        block positions (y_pos) indexed by block_idx
    selected by the parameter "return_type"
Constraint:
    Ensure the rectangle containment: 
    no two items in the same bin overlap
    if (ri; ci) overlap with (rj, cj),
    then either i=j or 
    alpha((si,ri, ci))+si <= alpha((sj,rj,cj)) or 
    alpha((sj,rj,cj))+sj <= alpha((si,ri, ci))
Objective: 
    minimize the highest point of the packing
"""

from typing import List, Dict, Set
from dataclasses import dataclass, field
from collections import OrderedDict
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@dataclass
class Block(object):
    s:float # size
    r:float # release time
    c:float # deadline
    idx:int # index
    n_conflict:int = 0 # number of conflicts
    lifetime:int = field(init=False) 

    def __post_init__(self):
        self.lifetime = self.c - self.r
        
@dataclass
class ContentionGroup(object):
    start:float
    duatation:float
    tot_size:int
    block_list:List[Block]

def stat_overlapping(
    block_list:List[Block], 
    timestep:float, 
    block_contention:Dict[int, int],
    conflict_graph:Dict[int, Set[int]] = None,
    event_list:List[float] = None
    ):
    """
    Input: block list I: [(s1, r1, c1), (s2, r2, c2), ...]
        Each triple (si,ri,ci) is an axis-parallel rectangle: 
            (ri,ci) is the interval on the x-axis
            si is the length on the y-axis
            the rectangles along the y-axis while stay fixed on x-axis
    Output: the blocks in each interval, the contention of each task
    """
    # collect the start and end of the intervals
    block_list.sort(key=lambda x: x.r)
    event_list = []
    for item in block_list:
        s, r, c, idx = item.s, item.r, item.c, item.idx
        event_list.extend([r,c])
    event_list = list(set(event_list))
    event_list.sort()
    
    # some cache
    curr_items = [] # cache the current items
    pending_items = [] # cache the pending items
    max_core_num = 0 # cache the max core number
    
    prev_t:float = 0
    curr_t:float = 0
    prev_slot_n:int = 0
    curr_slot_n:int = 0
    sparse_list:List[ContentionGroup] = []
    # scan the event list, place, and pop the task into the bin, 
    # expanding the bin size if necessary
    for curr_t in list(event_list):
        curr_slot_n = int(curr_t/timestep)

        if prev_t < curr_t and curr_items:
            sparse_list.append(ContentionGroup(prev_t, curr_t - prev_t, max_core_num, curr_items.copy()))
            # set block contention
            for item in curr_items:
                s, r, c, idx = item.s, item.r, item.c, item.idx
                # record the max degree of contention of each block and set of conflicted blocks
                block_contention[idx] = max(block_contention.get(idx, 0), len(curr_items))
                # get the union of the conflicted blocks, and remove the current block
                conflict_graph[idx] = conflict_graph.get(idx, set()).union(set([item.idx for item in curr_items]))#.remove(idx)
                conflict_graph[idx].remove(idx)
        
        # get the pending items
        while len(block_list):
            s, r, c, idx = block_list[0].s, block_list[0].r, block_list[0].c, block_list[0].idx
            if r <= curr_t:
                pending_items.append(block_list.pop(0))
                print("s:", s, "r:", r, "c:", c, "idx:", idx)
            else:
                break

        # pop the items that are finished regarding the ddl
        while len(curr_items):
            s, r, c, idx = curr_items[0].s, curr_items[0].r, curr_items[0].c, curr_items[0].idx
            if c <= curr_t:
                curr_items.pop(0)
                print("pop item: s:", s, "r:", r, "c:", c, "idx:", idx)
            else:
                break
        
        all_items = curr_items + pending_items
        max_core_num_tmp = sum([item.s for item in all_items])

        # update the bin size
        if max_core_num_tmp > max_core_num:
            print(f"curr_t:{curr_t}, max_core_num: {max_core_num} -> {max_core_num_tmp}")
            max_core_num = max_core_num_tmp

        # place the pending items into the bin
        while len(pending_items):
            item = pending_items.pop(0)
            curr_items.append(item)
            print("place item: s:", item.s, "r:", item.r, "c:", item.c, "idx:", item.idx)
            
        # sort the current items and pending items by the ddl
        curr_items.sort(key=lambda x: x.c)
        prev_t = curr_t
        prev_slot_n = curr_slot_n
    
    return max_core_num, sparse_list

@dataclass
class Memalloc:
    offset: int
    size: int
    nextoffset: int = field(init=False)

    def __post_init__(self):
        self.nextoffset = self.offset + self.size

@dataclass
class MemRegion:
    offset: int
    nextoffset: int
    size: int = field(init=False)

    def __post_init__(self):
        self.size = self.nextoffset - self.offset
    
    def update_size(self):
        self.size = self.nextoffset - self.offset

class MemMap(list): 
    def __init__(self, max_size=0):
        self.position_recoder = OrderedDict()
        self.free_interv = [] if not max_size else [MemRegion(0, max_size)]
        self.max_offset = max_size
    
    def insert(self, idx, offset, size, free_interv_idx):
        new_alloc = Memalloc(offset, size) 
        self.position_recoder[idx] = new_alloc
        # update the free interval
        region = self.free_interv[free_interv_idx]
        # if size is smaller than the free interval
        if size < region.size:
            if offset > region.offset:
                region.nextoffset = offset
                region.update_size()
            if offset + size < region.nextoffset:
                self.free_interv.insert(free_interv_idx+1, MemRegion(offset+size, region.nextoffset))
                
    def remove(self, idx):
        region = self.position_recoder.pop(idx)
        # update the free interval
        free_interv_idx = -1
        if len(self.free_interv) == 0:
            self.free_interv.append(region)
            return
        else:
            # check the index the region should be inserted
            for i in range(len(self.free_interv)):
                if region.offset < self.free_interv[i].offset:
                    free_interv_idx = i
                    break
            if free_interv_idx == -1:
                self.free_interv.append(region)
                return
            # interval merge: region, free_interv[free_interv_idx-1] and free_interv[free_interv_idx]
            if free_interv_idx > 0:
                region_left = region.offset == self.free_interv[free_interv_idx-1].nextoffset
            else:
                region_left = False
            if free_interv_idx < len(self.free_interv):
                region_right = region.nextoffset == self.free_interv[free_interv_idx].offset
            else:
                region_right = False
            if region_left and region_right:
                # merge the three intervals
                self.free_interv[free_interv_idx-1].nextoffset = self.free_interv[free_interv_idx].nextoffset
                self.free_interv[free_interv_idx-1].update_size()
                self.free_interv.pop(free_interv_idx)
            elif region_left:
                # merge the region with the left interval
                self.free_interv[free_interv_idx-1].nextoffset = region.nextoffset
                self.free_interv[free_interv_idx-1].update_size()
            elif region_right:
                # merge the region with the right interval
                self.free_interv[free_interv_idx].offset = region.offset
                self.free_interv[free_interv_idx].update_size()
            else:
                # insert the region into the free interval
                self.free_interv.insert(free_interv_idx, region)
    
    def incr_max_offset(self, amount, force_incr=False):
        """
        Function:
            increase the max offset of the bin
            check if the last free interval can be merged
            if force_incr is False, and the last free interval is ended with the max offset 
        """

        if force_incr or len(self.free_interv) == 0 or self.free_interv[-1].nextoffset != self.max_offset:
            max_offset_new = self.max_offset + amount
            self.free_interv.append(MemRegion(self.max_offset, max_offset_new))
            self.max_offset = max_offset_new
        else:
            # extend the last free interval to the size of amount
            # the highest free offset that not larger than the max offset
            self.max_offset = self.free_interv[-1].nextoffset = self.free_interv[-1].offset + amount
            self.free_interv[-1].update_size()
                            
    def first_fit_placement(self,
        block: Block): 
        """
        Input: 
            position_dict: a dict recorder, each item is 
                keyed by idx of the block, 
                values is (position and size) of the block
            block: a block to be placed
        Output:
            searched best position for the block
        """

        # get the avaliable interval and check if the block is legal
        for interv_idx, interv in enumerate(self.free_interv):
            # the free interval can contain the block
            if interv.size >= block.s:
                # the block is legal
                return interv.offset, interv_idx
        raise ValueError("No legal interval for the block")
        
    def worst_fit_placement(self,
        block: Block): 
        """
        Input: 
            position_dict: a dict recorder, each item is 
                keyed by idx of the block, 
                values is (position and size) of the block
            block: a block to be placed
        Output:
            searched best position for the block
        """

        size_cache = np.array([interv.size for interv in self.free_interv if interv.size >= block.s])
        # get the legal interval with minimal size
        max_size = np.max(size_cache)
        max_size_idx = np.where(size_cache == max_size)[0][0]
        return self.free_interv[max_size_idx].offset, max_size_idx
    
    def best_fit_placement(self,
        block: Block): 
        """
        Input: 
            position_dict: a dict recorder, each item is 
                keyed by idx of the block, 
                values is (position and size) of the block
            block: a block to be placed
        Output:
            searched best position for the block
        """

        size_cache = np.array([interv.size for interv in self.free_interv if interv.size >= block.s])
        # get the legal interval with minimal size
        min_size = np.min(size_cache)
        min_size_idx = np.where(size_cache == min_size)[0][0]
        return self.free_interv[min_size_idx].offset, min_size_idx

    def try_to_place(self, 
                     block: Block, strategy: str = 'first_fit'): 
        """
        Function:
            try to place the block into the current bin, with current max_offset;
            if the block can be placed, 
            else, stack the block on the top of the bin, and increase the bin size. 
        Return: 
            the offset and the interval index
        """
        # get the best position for the block
        try:
            if strategy == 'best_fit':
                offset, interv_idx = self.best_fit_placement(block)
            elif strategy == 'worst_fit':
                offset, interv_idx = self.worst_fit_placement(block)
            else:
                offset, interv_idx = self.first_fit_placement(block)
        except ValueError: 
            self.incr_max_offset(block.s) 
            offset = self.free_interv[-1].offset
            interv_idx = len(self.free_interv) - 1
        return offset, interv_idx
        
    # lambda x: (x.n_conflict, -x.r, -x.s)
    # lambda x: x.s
    def seq_mapper(self, block_list, time_step, sort_fn:callable=lambda x: x.s):
        # get block confliction
        block_contention = {}
        # perform a shallow copy
        block_list_copy = block_list.copy()
        contention_graph = {}
        event_list = []
        max_core_num, sparse_list = stat_overlapping(block_list_copy, time_step, block_contention, contention_graph, event_list)
        # update the block_contention to the blocks
        for item in block_list:
            # update the block contention
            item.n_conflict = block_contention.get(item.idx, 0)
            
        for contention_group in sparse_list:
            block_group = contention_group.block_list
            block_group.sort(key=sort_fn, reverse=True)
            # compare previous block with the current block
            # pop the old block
            # for idx in self.position_recoder:
            #     if block_list[idx] not in block_group:
            #         self.remove(idx)
            for idx in self.position_recoder:
                
            # insert the new block
            for block in block_group:
                if block.idx not in self.position_recoder:
                    # get the best position for the block
                    offset, interv_idx = self.try_to_place(block, str)
                    # insert the block into the position
                    self.insert(block.idx, offset, block.s, interv_idx)
                            
        # sort by xx
        block_list.sort(key=sort_fn, reverse=True)
        for item in block_list:
            # get the best position for the block
            offset, interv_idx = self.try_to_place(item)
            # insert the block into the position
            self.insert(item.idx, offset, item.s, interv_idx)
        return self.position_recoder

    def prority_mapper(self, block_list, time_step, sort_fn:callable=lambda x: x.s):
        # get block confliction
        block_contention = {}
        # perform a shallow copy
        block_list_copy = block_list.copy()
        conflict_graph = {}
        event_list = []
        max_core_num, sparse_list = stat_overlapping(block_list_copy, time_step, block_contention, conflict_graph, event_list)
        # update the block_contention to the blocks
        for item in block_list:
            # update the block contention
            item.n_conflict = block_contention.get(item.idx, 0)
        # sort by xx
        block_list.sort(key=sort_fn, reverse=True)
        i=0
        for item in block_list:
            # get the best position for the block
            offset = search_interval(item, block_list, conflict_graph, self.position_recoder)
            
            # insert the block into the position
            self.position_recoder[item.idx] = Memalloc(offset, item.s)
            i+=1
        return self.position_recoder, conflict_graph

    # def check_conflict(self, confict_graph:Dict[int, Set[int]]):
    #     """
    #     check if the placement is legal, 
    #     """
    #     tmp_graph = confict_graph.copy()
    #     while len(tmp_graph):
    #         idx, conflict_idx_list = tmp_graph.popitem()  
    #         Block_A = self.position_recoder[idx]
    #         for idx_B in conflict_idx_list:
    #             Block_B = self.position_recoder[idx_B]
    #             # also remove the reverse edge
    #             tmp_graph[idx_B].remove(idx)
    #             if Block_A.offset < Block_B.nextoffset and Block_A.nextoffset > Block_B.offset:
    #                 return False
    
    # def check_legalty(self):
    #     """
    #     check if the placement is legal: 
    #         sort the allocated Memalloc by the offset
    #         check the adjacent Memallocs if they overlap
    #     """
    #     sorted_memalloc = list(sorted(self.position_recoder.values(), key=lambda x: x.offset))
    #     for i in range(len(sorted_memalloc)-1):
    #         if sorted_memalloc[i].nextoffset > sorted_memalloc[i+1].offset:
    #             return False
    
        

def search_interval(block, block_list:List[Block], confict_graph:Dict[int, Set[int]], 
                        position_dict:Dict[int, Memalloc], strategy:str='first_fit'):
    """
    get the free interval among the blocks
    """
    # get the confict blocks
    confict_idx_list:List[int] = list(confict_graph[block.idx])
    overlapping_blocks = []
    for confict_idx in confict_idx_list:
        if confict_idx in position_dict:
            block = block_list[confict_idx]
            overlapping_blocks.append(position_dict[confict_idx])
    overlapping_blocks.sort(key=lambda x: x.offset)
    if len(overlapping_blocks) == 0:
        return 0
    
    # get the free interval
    free_interval = []
    prev_offset = 0
    # CAUTION: the overlapping_blocks may not overlap with each other, i.e., 
    # their allocated memory region may overlap with each other
    best_offset = -1
    for item in overlapping_blocks:
        if item.offset > prev_offset:
            if block.s <= item.offset - prev_offset:
                if strategy == 'first_fit':
                    best_offset = prev_offset
                    return best_offset     
                else:
                    free_interval.append(MemRegion(prev_offset, item.offset))
        prev_offset = max(prev_offset, item.nextoffset)
    
    if (strategy == 'first_fit' and best_offset == -1) or (strategy != 'first_fit' and len(free_interval) == 0):
        # Case: get no candidate
        best_offset = prev_offset
    elif strategy == 'best_fit':
        free_interval.sort(key=lambda x: x.size)
        best_offset = free_interval[0].offset
    elif strategy == 'worst_fit':
        free_interval.sort(key=lambda x: x.size, reverse=True)
        best_offset = free_interval[0].offset
    assert best_offset != -1
    return best_offset
        
def scan_overlap_2d(position_recoder:Dict[int, Memalloc], block_list:List[Block]):
    """
    check if the placement is legal: 
    check the any two Memallocs if they overlap with each other in both x and y axis
    """
    idx_list = list(position_recoder.keys())
    for i in range(len(idx_list)-1):
        idx_a = idx_list[i]
        a_bot, a_top = position_recoder[idx_a].offset, position_recoder[idx_a].nextoffset
        a_ed, a_st = block_list[idx_a].c, block_list[idx_a].r
        for j in range(i+1, len(idx_list)):
            idx_b = idx_list[j]
            b_bot, b_top = position_recoder[idx_b].offset, position_recoder[idx_b].nextoffset
            x_overlap = a_bot < b_top and b_bot < a_top
            b_ed, b_st = block_list[idx_b].c, block_list[idx_b].r
            y_overlap = a_ed < b_st and b_ed < a_st
            if x_overlap and y_overlap:
                return False
    return True

def scan_overlap_1d(position_recoder):
    """
    check if the placement is legal: 
        sort the allocated Memalloc by the offset
        check the adjacent Memallocs if they overlap
    """
    sorted_memalloc = list(sorted(position_recoder.values(), key=lambda x: x.offset))
    for i in range(len(sorted_memalloc)-1):
        if sorted_memalloc[i].nextoffset > sorted_memalloc[i+1].offset:
            return False

def layout_plot(position_recoder:Dict[int, Memalloc], block_list:List[Block], 
                tick_dens = 1, txt_size = 30,
                show=False, save=False, save_path="task_layout_compact.pdf", 
                ):
    """
    plot the block in the block list as a horizontal bar
        the x-axis of the bar is the start (r) and end (c) time of the block
        the y-axis of bottom line is the offset of its Memalloc
        the y-axis of top line is the nextoffset of its Memalloc
    """
    
    base_vertical_offset = 0
    y_margin = 0.5 
    time_grid_size = 0.004
    x_margin = time_grid_size
    vertical_grid_size = 1
    bin_vertical_offset = base_vertical_offset

    # one bin version
    get_vertical_offset = lambda x: x*vertical_grid_size + bin_vertical_offset
    get_vertical_size = lambda x: x*vertical_grid_size

    colors=list(mcolors.XKCD_COLORS.keys())
    fig, ax = plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(20, 10)) 
    # plot the task layout
    for idx, memalloc in position_recoder.items():
        color = mcolors.XKCD_COLORS[colors[idx%len(colors)]]
        w = block_list[idx].lifetime
        l, r = block_list[idx].r, block_list[idx].c
        h = memalloc.size
        b, t = memalloc.offset, memalloc.nextoffset
        # convert the l, r, b, t to the plot coordinate
        ax.broken_barh([(l, w)], (get_vertical_offset(b), get_vertical_size(h)), facecolors=color)
        ax.text((l+r)/2, get_vertical_offset(b)+get_vertical_size(h)/2, str(idx), ha='center', va='center', color='black', fontsize=10)
        
    # get the plot start and the plot end 
    plot_start = min([block.r for block in block_list])
    plot_end = max([block.c for block in block_list])
    # only set x axis for the bottom plot
    ax.set_xlim(plot_start-x_margin, plot_end+x_margin)
    ticks = [str(round(t, 3)) for t in np.arange(plot_start, plot_end, time_grid_size*tick_dens)] + [str(round(plot_end, 3))]
    ax.set_xticks(np.arange(plot_start, plot_end, time_grid_size*tick_dens).tolist()+[plot_end])
    ax.set_xticklabels(ticks, fontsize=txt_size, rotation=45)
    ax.tick_params(axis='x', which='major', pad=time_grid_size * tick_dens)
    ax.set_xlabel("Time (s)", fontsize=txt_size) 

    # set the axis and title
    bin_vertical_offset += max([memalloc.nextoffset for memalloc in position_recoder.values()])
    vs = int((bin_vertical_offset-base_vertical_offset)//vertical_grid_size)
    ticks = np.linspace(0, vs, 4, dtype=int)
    ax.set_ylim(-y_margin, bin_vertical_offset+y_margin)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=txt_size, rotation=45)

    # plot the result
    if show:
        plt.show()
    # save the figure
    if save: 
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fmt = save_path.split(".")[-1]
        plt.savefig(save_path, bbox_inches='tight', format=fmt)
