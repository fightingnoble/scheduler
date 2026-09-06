"""Historical event-sweep helper moved without logic changes (B24)."""
from __future__ import annotations
from typing import List, Dict, Set
from mapper.mem_planner import Block, ContentionGroup, MemMap


def stat_overlapping_old(
    block_list:List[Block], 
    timestep:float, 
    block_contention:Dict[int, int] = dict(),
    conflict_graph:Dict[int, Set[Block]] = dict(),
    sparse_list:List[ContentionGroup] = [],
    mapper:MemMap = None, sort_fn:callable=lambda x: x.s, strategy:str='first_fit'
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
    # scan the event list, place, and pop the task into the bin, 
    # expanding the bin size if necessary
    for curr_t in list(event_list):
        curr_slot_n = int(curr_t/timestep)

        if prev_t < curr_t and curr_items:
            if sparse_list is not None:
                sparse_list.append(ContentionGroup(prev_t, curr_t - prev_t, max_core_num, curr_items.copy()))

            if block_contention is not None or conflict_graph is not None:
                # set block contention
                for item in curr_items:
                    s, r, c, idx = item.s, item.r, item.c, item.idx
                    if block_contention is not None:
                        # record the max degree of contention of each block and set of conflicted blocks
                        block_contention[idx] = max(block_contention.get(idx, 0), len(curr_items))
                    if conflict_graph is not None:
                        # get the union of the conflicted blocks, and remove the current block
                        conflict_graph[idx] = conflict_graph.get(idx, set()).union(set([item for item in curr_items])).difference(set([item]))
        
        # get the pending items
        while len(block_list):
            s, r, c, idx = block_list[0].s, block_list[0].r, block_list[0].c, block_list[0].idx
            if r <= curr_t:
                pending_items.append(block_list.pop(0))
                # print("s:", s, "r:", r, "c:", c, "idx:", idx)
            else:
                break

        # pop the items that are finished regarding the ddl
        while len(curr_items):
            s, r, c, idx = curr_items[0].s, curr_items[0].r, curr_items[0].c, curr_items[0].idx
            if c <= curr_t:
                if mapper is not None:
                    # pop the old block
                    mapper.release(idx)
                curr_items.pop(0)
                # print("pop item: s:", s, "r:", r, "c:", c, "idx:", idx)
            else:
                break
            1+1
        
        all_items = curr_items + pending_items
        max_core_num_tmp = sum([item.s for item in all_items])

        # update the bin size
        if max_core_num_tmp > max_core_num:
            print(f"curr_t:{curr_t}, max_core_num: {max_core_num} -> {max_core_num_tmp}")
            max_core_num = max_core_num_tmp

        # place the pending items into the bin
        if mapper is not None:
            pending_items.sort(key=sort_fn, reverse=True)
            
        
        while len(pending_items):
            item = pending_items.pop(0)
            curr_items.append(item)
            if mapper is not None:
                # get the best position for the block
                offset, interv_idx = mapper.try_to_place(item, strategy)
                # insert the block into the position
                mapper.alloc(item.idx, offset, item.s, interv_idx)
                1+1
            
            
            # print("place item: s:", item.s, "r:", item.r, "c:", item.c, "idx:", item.idx)
            
        # sort the current items and pending items by the ddl
        curr_items.sort(key=lambda x: x.c)
        prev_t = curr_t
        prev_slot_n = curr_slot_n
    
    return max_core_num
