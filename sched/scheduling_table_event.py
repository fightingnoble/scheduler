"""scheduling_table_event.py — scheduling-table event handling (self-contained group).

Moved from scheduling_table.py (B6-SPLIT-002, 2026-06-29). Byte-identical relocation.
Functions: init_event, parse_event_msg, filter_msg, process_tab_event,
event_handle_migrate_from, event_handle_migrate_to, check_legality.
Recovery: git checkout archive/test_pipeline-20260612 -- sched/scheduling_table.py
"""
from typing import List, Dict, Tuple, Union, Optional, Iterable, Iterator, Collection, Set
from collections import OrderedDict, defaultdict

from functools import reduce
import math
import bisect
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

def init_event(_p, curr_t:float, type:str, src:int=-1, tgt:int=-1) -> str:
    if type == "migrate_from":
        assert src >= 0
        return f"{_p.task.name:s}({_p.pid:d}) migrate from {src:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    elif type == "migrate_to":
        assert tgt >= 0
        return f"{_p.task.name:s}({_p.pid:d}) migrate to {tgt:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    elif type == "start":
        assert src >= 0
        return f"{_p.task.name:s}({_p.pid:d}) start on {src:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    elif type == "complete":
        assert src >= 0
        return f"{_p.task.name:s}({_p.pid:d}) complete on {src:d} @ {curr_t:.6f}/{_p.get_timestamp():.6f}!!"
    else:
        raise ValueError(f"Unknown event type {type}")

def parse_event_msg(msg:str):
    """
    use re to match event pattern, and extract 
     start, ts, complete, migrate
    f"start-{pid:d}_{j:d}", 
    f"complete-{pid:d}_{j:d}", 
    f"migrate-{pid:d}_from_{pre_bin_idx}", 
    f"migrate-{pid:d}_to_{next_bin_idx}"
    """ 
    import re
    # pattern = re.compile(r"^(?P<event_type>\w+)-(?P<pid>\d+)(_(?P<ts>\d+))?(_from_(?P<from>\d+))?(_to_(?P<to>\d+))?$")
    pattern = re.compile(tab_event_re)
    match = pattern.match(msg)
    # {k: t(v) for k,v,t in zip(pattern_keys, match.groups(), pattern_type) if v is not None}
    try:
        result = {k: t(v) for k,v,t in zip(tab_event_pattern_keys, match.groups(), tab_event_pattern_type)}
    except TypeError:
        assert False, f"TypeError: {msg}"
    # replace "migrate from" with "migrate_from", "migrate to" with "migrate_to"
    result['event_type'] = result['event_type'].replace("on", "")
    result['event_type'] = result['event_type'].replace(" ", "_")
    return result

def filter_msg(tab_event_msg:str, msg_filter:Optional[Union[None, Dict[str, str]]]): 
    """
    Parse the event message and filter the message by the filter dict
    Return True if the message is filtered, False otherwise
    """
    results = parse_event_msg(tab_event_msg)
    for k,v in msg_filter.items(): 
        if k not in results or (results[k] != '?' and results[k]!= v):
            return False, {}
    return True, results

def process_tab_event(sched, curr_t, ready_queue, running_queue, throttle_list, event_list, _SchedTab, process_dict, bin_id, 
                      msg_filter:Optional[Union[None, Dict[str, str]]]):
    """
    process old events -> clear -> process new events
    NOTE: should not directly rob the budget from all the other partitions
    CASE: suppose a task A executes on 1 -> 2 -> 3
    and the task is late and misses the budeget on (1), then at arrival of A, 
    it should be executed on 2, with the budget of sum of 1 and 2, without 3
    CASE: 2 -> 1
    scheduler scan (1), take from bk but got nothing
    then scheduler scan (2), put the budget to (2) ruther than bk
    Event format: 
        {
            "name": str,
            "pid": int,
            "event_type": str,
            "bin_id": int,
            "curr_t": float,
            "ts": float
        }
    """
    if _SchedTab.alloc_mod == 'N/A':
        return 

    matched_events = []
    for msg in event_list:
        match, results = filter_msg(msg, msg_filter) 
        if not match:
            continue
        matched_events.append(msg)
        # "migrate_from", "start", "complete", "migrate_to"
        if "migrate" in results['event_type']:
            if 'to' in results['event_type']:
                print(msg)
                event_handle_migrate_to(sched, curr_t, ready_queue, running_queue, throttle_list, process_dict, bin_id, results)
            elif 'from' in results['event_type']:
                print(msg)
                event_handle_migrate_from(process_dict, bin_id, results)
        elif results['event_type'] == "start":
            pass
        elif results['event_type'] == "complete":
            pass
        else:
            raise ValueError(f"Unknown event type {results['event_type']}")
    # clear the matched events
    for idx in reversed(matched_events):
        event_list.remove(idx)

def event_handle_migrate_from(process_dict, bin_id, results):
    _p = check_legality(process_dict, results)
    # restore the _p.rem_flop_budget from BK
    _p.rem_flop_budget[bin_id] += _p.rem_flop_budget.pop('bk', 0.)

def check_legality(process_dict, results):
    _p = process_dict[results['pid']]
    # rem_flop_budget = {k:v for k,v in _p.rem_flop_budget.items() if v > numerical_error_tol_abs}
    # try:
    #     assert len(rem_flop_budget) <= 2
    # except AssertionError:
    #     print(f"20231126: CodingError, try to gurrante the budget only on one partition at a time")
    return _p

def event_handle_migrate_to(sched, curr_t, ready_queue, running_queue, throttle_list, process_dict, bin_id, results):
    _p:ProcessInt = check_legality(process_dict, results)
    # throttle the task to be migrated
    if _p in (running_queue.queue+ready_queue.queue):
        _p.throttle_util(throttle_list, curr_t)
        _p.task.migration_count += 1
        if _p in running_queue.queue:
            sched.res_release(_p.pid)
            # budget_recoder.pop(_p.pid)
            running_queue.remove(_p)
        elif _p in ready_queue.queue:
            ready_queue.remove(_p)
        else:
            raise ValueError("Task is not in the running queue or ready queue")
    # backup the _p.rem_flop_budget
    rem_flop_budget = _p.rem_flop_budget[bin_id]
    if results['bin_id'] in _p.rem_flop_budget:
        _p.rem_flop_budget[results['bin_id']] += rem_flop_budget
    else:
        _p.rem_flop_budget['bk'] = rem_flop_budget
    _p.rem_flop_budget[bin_id] = 0
