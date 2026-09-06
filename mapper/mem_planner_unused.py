"""Independent unused mapper helpers preserved without logic changes (B24)."""
from __future__ import annotations
from mapper.mem_planner import Block, CyclicBlock


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


def sort_fn_conflict_s_r(x:Block):
    return (x.n_conflict, x.s, -x.r, -x.idx if not isinstance(x, CyclicBlock) else -x.pid)
