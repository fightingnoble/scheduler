"""global_sched_alloc.py — Step 2 预分配 (Task-to-Bin Clustering).

Moved from global_sched.py (B6-SPLIT-001, 2026-06-29). Byte-identical relocation.
Functions: coleasing_alloc_cluster, coleasing_alloc_1bin, gurobi_split_solver,
update_bp_result2_schedtab, rename_bins_and_relable_assignments, build_greedy_obj, build_search_obj.
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from typing import List, Dict
    from sched.scheduling_table import SchedulingTableInt

import math
from collections import OrderedDict, defaultdict
import numpy as np
from global_var import *
from model.resource_agent import Resource_model_int
from model.task_queue_agent import TaskQueue 
from model.buffer import Buffer
from model.message.msg_dispatcher import MsgDispatcher
from model.message.message_handler import message_trigger_event_new
from model.streaming_processing.wartermark_strategy import WatermarkStrategy
from model.message.data_pipe import DataPipe
from model.position_table import PosTableInt
from model.performance import slack_comp

from sched.monitor_agent import Monitor
from sched.scheduler_agent import Scheduler, check_miss, check_complete
from sched.scheduler_agent import data_pipe_read, pendingToReady
from sched.pre_alloc_new import glb_alloc_new2
from sched.bin_ops import new_bin, get_initlist_and_biniter, static_1_bin
from sched.sort_function import get_process_sort
from sched.packing_solver.gurobi_MP_semi2DClst import ClusterGurobiSolverSemi2D
from sched.monitor_agent import get_rsc_2b_released, get_target_bin_id
from sched.bin_ops import bin_iter_list
from networkx import DiGraph
from functools import reduce
from sched.slack_estim import get_chains
# (init_event import was dead — discarded on split)
from sched.binpack_config import BinPackConfig

# 默认配置实例（函数签名 fallback；实际运行时由 input_parser 从 JSON 加载）
from sched.binpack_config import BinPackConfig
default_binpack_cfg = BinPackConfig()

def coleasing_alloc_1bin(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, old_flops_relase_rate,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
        ):
    # --- 入口参数检查与读取 ---
    reservation_policy = binpack_cfg.get("reservation_policy", "manual")
    # -----------------------
    event_range = hyper_p * (n_p+warmup)
    sim_range = hyper_p * (n_p+warmup+drain)
    tab_temp_size = int(hyper_p//timestep)
    # assert math.isclose(hyper_p, tab_temp_size*timestep, abs_tol=numerical_error_tol_abs), \
    #         "hyper_p should be the multiple of timestep"
    sim_slot_num = int(sim_range/timestep)
    # glb_name_p_dict = {p.task.name:p for p in glb_p_list}

    def _new_bin(id, size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(
        bin_list, glb_p_list, 0, 
        _new_bin, reservation_policy)
    _bin = bin_list[0]

    # build event list
    process_block_sortby_start_slot = []    
    for _p in glb_p_list:
        _p:ProcessInt
        stimu_tab = _p.task.extract_sensor_event(event_range)
        # (task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t)
        for stimu_t in stimu_tab:
            item = (_p.task.name, _p.pid, 
              _p.task.pre_assigned_resource.main_size + _p.task.pre_assigned_resource.RDA_size, 
              elim_nume_error(stimu_t), elim_nume_error(stimu_t+_p.task.ERT), elim_nume_error(stimu_t+_p.task.ERT+_p.task.ddl), _p.task.exp_comp_t)
            start_t, ddl_t = item[4], item[5]
            # quantize the start time and ddl time
            slot_s = int(math.ceil(start_t/timestep)) * timestep
            slot_e = int(math.floor(ddl_t/timestep)) * timestep
            process_block_sortby_start_slot.append((item[0], item[1], item[2], item[3], slot_s, slot_e, item[6]))

    # sort the event list by the start time, ddl, exp_comp_t
    process_block_sortby_start_slot.sort(key=lambda x: (x[4], x[5], x[6]))
    event_set = OrderedDict()
    for item in process_block_sortby_start_slot:
        event_set.update({item[4]:None})
        event_set.update({item[5]:None})
    event_set = sorted(event_set.keys())
    
    # convergence flag
    bin_size_ok = False
    slot_ok = np.full(len(event_set), False)
    # some cache
    curr_items = [] # cache the current items
    pending_items = [] # cache the pending items
    max_core_num = _bin.num_resources # cache the max core number
    max_core_layout = {} # start, core_size, slot_num
    position_recoder = PosTableInt(len(event_set)) # cache the current usage of the cores
    prev_t = 0
    cum_flops = {}
    process_dict = {p.pid:p for p in glb_p_list}

    # scan the event list, place, and pop the task into the bin, 
    # expanding the bin size if necessary
    for curr_t in list(event_set):
        # curr_t = n_slot * timestep
        n_slot = int(curr_t/timestep)
        tab_pointer = n_slot % tab_temp_size
        hyper_p_n = int(n_slot/tab_temp_size)

        curr_n_period = int(curr_t/hyper_p)
        prev_n_period = int(prev_t/hyper_p)
        if warmup and prev_t == 0 and curr_n_period == 0:
            print("="*20, "WARMUP", "="*20, "\n")
        elif curr_n_period > prev_n_period:
            if curr_t > event_range:
                print("="*20, "DRAIN", "="*20, "\n")
            else:
                print("="*20, "PERIOD {:d}".format(curr_n_period), "="*20, "\n")

        if prev_t < curr_t and curr_items:
            flops_per_core = (curr_t-prev_t) * FLOPS_PER_CORE 
            flops_dict = {}
            cores_dict = {}
            for item in curr_items:
                # task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = item
                req_size = item[2]
                pid = item[1]
                start_t, ddl_t = item[4], item[5]
                # size_del_rda = process_dict[pid].task.flops/FLOPS_PER_CORE/(ddl_t-start_t)
                # cores_dict[pid] = int(math.ceil(size_del_rda/(1-exec_t_comp_ratioB)))
                # ================= Calculate the fueling rate =================
                # ratioB is used for adjusting the rate for fueling the budget, 
                # rather than the bw of the core
                # such rate should be less than bw * flops_per_core, but > truely allocated number of ops
                slack = slack_comp((ddl_t-start_t), 0, old_flops_relase_rate)
                size_del_rda = process_dict[pid].task.flops/FLOPS_PER_CORE/slack
                # size_del_rda = item[2] 
                cores_dict[pid] = int(math.ceil(item[2]))
                if cum_flops[pid] > 0:
                    flops = flops_per_core*math.ceil(size_del_rda)
                    flops = min(flops, cum_flops[pid])
                    flops_dict.update({pid:flops})
                    cum_flops[pid] -= flops
                else:
                    flops_dict.update({pid:0})
            _bin.sparse_flops.append([round(prev_t/timestep), flops_dict])
            _bin.sparse_cores.append([round(prev_t/timestep), cores_dict])

        # get the pending items
        while len(process_block_sortby_start_slot):
            task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = process_block_sortby_start_slot[0]
            if start_t <= curr_t:
                pending_items.append(process_block_sortby_start_slot.pop(0))
                print("task_name:", task_name, "stimu_t:", stimu_t, "start_t:", start_t, "ddl_t:", ddl_t, "exp_comp_t:", exp_comp_t)
            else:
                break
        
        # pop the items that are finished regarding the ddl
        while len(curr_items):
            if curr_items[0][5] <= curr_t:
                pid = curr_items[0][1]
                assert cum_flops[pid] == 0
                curr_items.pop(0)
                cum_flops.pop(pid)
            else:
                break

        # get the current items
        # use cache curr_items
        # ordered_occupant_dict = _bin.index_occupy_by_id(n_slot)

        # get the history placement of the current items
        if bin_size_ok and slot_ok.all():
            position_dict = position_recoder.get_record(n_slot)
            

        # vefify the legality of the placement of the history items

        # determine the placement of the pending items, as well as the bin size

        all_items = curr_items + pending_items
        # max_core_num
            # without laxity, a tasks is placed iff it is released at the current slot
            # no neet to scan the bin in the future monents
        max_core_num_tmp = sum([item[2] for item in all_items])

        # update the bin size
        if max_core_num_tmp > max_core_num:
            print(f"max_core_num: {max_core_num} -> {max_core_num_tmp}")
            _bin = bin_list[0]
            _bin.add_rsc_num(max_core_num_tmp - max_core_num)
            max_core_num = max_core_num_tmp
            max_core_layout = [n_slot, {
                item[1]:item[2] for item in all_items
            }, min({int(math.floor(item[5]/timestep)) for item in all_items}) - n_slot]

        # update the position table
        if bin_size_ok and slot_ok.all():
            position_recoder.set_record(tab_pointer, position_dict)

        # place the pending items into the bin

        for item in pending_items:
            task_name, pid, req_size, stimu_t, start_t, ddl_t, exp_comp_t = item
            slot_s = round(start_t/timestep)
            slot_e = round(ddl_t/timestep)
            slot_n = slot_e - slot_s
            _bin.allocate(pid, [slot_s,], [req_size,], [slot_n,], DEBUG_FG)
            curr_items.append(item)
            cum_flops.update({pid:process_dict[pid].task.flops})
        pending_items.clear()

        # sort the current items and pending items by the ddl
        curr_items.sort(key=lambda x: x[5])
        prev_t = curr_t
    
    # sparsify the Scheduling table
    # initialize the interval info: slot_s, slot_e, core_size, flops
    _bin.alloc_mod = "compress"
    
    assert max_core_num == sum(max_core_layout[1].values()), "max_core_num should be equal to the sum of the core size of the current items"
    _bin.to_sparse_dict()

    bin_size_list = {_bin.id: _bin.num_resources}
    pid2_bin_id = {p.pid: _bin.id for p in glb_p_list}
    print("max_core_num:", max_core_num)
    print(f"max_core_layout: {bin_size_list}: \n {max_core_layout}")
    print(f"pid2_bin_id: all in one bin")
    return max_core_num, pid2_bin_id, bin_size_list

def coleasing_alloc_cluster(
        bin_list: List[SchedulingTableInt], 
        glb_p_list: List[ProcessInt], affinity, event_iter_dict:Dict,
        total_cores:int, quantum_check_en, quantumSize, 
        timestep, hyper_p, quantile,

        scheduler_list: List[Scheduler], monitor_list:List[Monitor],
        msg_dispatcher:MsgDispatcher=None, # msg_pipe:Message=Message(),
        a_data_pipe:DataPipe=None,
        w_data_pipe:DataPipe=None, 

        n_p=1, binpack_cfg:Dict=default_binpack_cfg,
        job_graph:DiGraph=None, n_partition:int=9999,
        show_warnings=True, 
        verbose=False, DEBUG_FG=False, *, 
        warmup=False, drain=False,                     
):
    # --- 入口参数检查与读取 ---
    # 目前此函数主要作为包装器，参数通过 quantile 和 n_partition 显式传递
    # -----------------------

    max_core_num, pid2_bin_id, bin_size_list = coleasing_alloc_1bin(
        bin_list,
        glb_p_list, affinity, event_iter_dict,
        total_cores, quantum_check_en, quantumSize, 
        timestep, hyper_p, quantile,

        scheduler_list, monitor_list,
        msg_dispatcher,
        a_data_pipe, w_data_pipe,

        n_p, binpack_cfg, 
        show_warnings, 
        verbose, DEBUG_FG,
        warmup=warmup, drain=drain
        )

    # bypass clustering when target partition number is 1 or less
    if n_partition == 1:
        return max_core_num, pid2_bin_id, bin_size_list

    # split the bin
    sim_range = hyper_p * (n_p+warmup+drain)
    sim_slot_num = int(sim_range/timestep)
    assert len(bin_list) == 1
    _bin_tb_split = bin_list[0]

    def _new_bin(id, size, name=None): 
        if name is None:
            name = "bin"+str(id)
        print("Create a new bin: ", id, "name:", name, "size:", size)
        return new_bin(size, sim_slot_num, id=id, name=name)

    # use a gurobi solver to determine the placement of the tasks
    src_nodes = [n for n, x in job_graph.in_degree() if x == 0]
    end_nodes = [n for n, x in job_graph.out_degree() if x == 0]
    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))
    task_dict = {_p.task.name: _p.task for _p in process_dict.values()}
    node_var_dists = [job_graph.nodes[p.task.name]['var_dist'] for p in glb_p_list]

    sorted_chains = get_chains(job_graph, src_nodes, end_nodes, task_dict, 
                            quantile=quantile,
                            node_var_dists=node_var_dists,
                            remove_src_sink=True)

    placed_p, bin_name_list, sol, bin_size = gurobi_split_solver(
        glb_p_list, n_partition, _bin_tb_split, job_graph, sorted_chains
    )

    update_bp_result2_schedtab(bin_list, _bin_tb_split, _new_bin, bin_name_list, sol, bin_size)

    # layout format: start slot, allocation map, duation slot
    bin_size_list = {_bin.id:_bin.num_resources for _bin in bin_list}
    pid2_bin_id = sol
    max_core_num = sum(bin_size_list.values())
    print("max_core_num:", max_core_num)
    print(f"max_core_layout: {bin_size_list}")
    print(f"pid2_bin_id: {pid2_bin_id}")
    return max_core_num, pid2_bin_id, bin_size_list

def update_bp_result2_schedtab(bin_list, _bin_tb_split, _new_bin, bin_name_list, sel, bin_size):

    # create the bins from the bin size and the bin name    
    iter_next_bin_obj =bin_iter_list(_new_bin, bin_size, bin_name_list)
    planed_bin_list = list(iter_next_bin_obj)

    # rebuild the scheduling table list
    for cfg_slot_s, cached_cfg, cfg_slot_num in _bin_tb_split.sparse_list:
        for pid, size in cached_cfg.items():
            _bin:SchedulingTableInt = planed_bin_list[sel[pid]]
            _bin.allocate(pid, [cfg_slot_s,], [size,], [cfg_slot_num,], False)
    # for _bin in planed_bin_list:
    #     _bin: SchedulingTableInt
    
    # rebuild the sparse_cores and sparse_flops
    for cores, flops, cfgs in zip(_bin_tb_split.sparse_cores, _bin_tb_split.sparse_flops, _bin_tb_split.sparse_list):
        cfg_slot_s, cached_cfg, cfg_slot_num = cfgs
        cores_slot_s, cores_dict = cores
        flops_slot_s, flops_dict = flops
        assert cores_slot_s == cfg_slot_s == flops_slot_s
        bins_core_recoder = defaultdict(dict)
        bins_flops_recoder = defaultdict(dict)
        for pid, size in cached_cfg.items():
            bin:SchedulingTableInt = planed_bin_list[sel[pid]]
            bin.allocate(pid, [cfg_slot_s,], [size,], [cfg_slot_num,], False)
            bins_core_recoder[bin.id].update({pid:cores_dict[pid]})
            bins_flops_recoder[bin.id].update({pid:flops_dict[pid]})
        for bin_id in bins_core_recoder:
            bin:SchedulingTableInt = planed_bin_list[bin_id]
            bin.sparse_cores.append([cores_slot_s, bins_core_recoder[bin_id]])
            bin.sparse_flops.append([flops_slot_s, bins_flops_recoder[bin_id]])
            
    for bin in planed_bin_list:
        bin: SchedulingTableInt
        bin.to_sparse_dict()
        # sort 
        bin.sparse_cores.sort(key=lambda x: x[0])
        bin.sparse_flops.sort(key=lambda x: x[0])
        # merge the sparse_cores and sparse_flops refer to the new spase_list
        core_idx = 0
        flops_idx = 0
        assert len(bin.sparse_cores) == len(bin.sparse_flops)
        oringinal_sparse_size = len(bin.sparse_cores)
        for cfg_slot_s, next_cfg, cfg_slot_num in bin.sparse_list:
            new_sparse_cores = []
            new_sparse_flops = []
            # for cores_slot_s, cores_dict in bin.sparse_cores:
            #     if cores_slot_s >= cfg_slot_s and cores_slot_s < cfg_slot_s + cfg_slot_num:
            #         new_sparse_cores.append([cores_slot_s, cores_dict])
            # for flops_slot_s, flops_dict in bin.sparse_flops:
            #     if flops_slot_s >= cfg_slot_s and flops_slot_s < cfg_slot_s + cfg_slot_num:
            #         new_sparse_flops.append([flops_slot_s, flops_dict])
            while core_idx < oringinal_sparse_size: 
                cores_slot_s, cores_dict = bin.sparse_cores[0]
                if cores_slot_s >= cfg_slot_s and cores_slot_s < cfg_slot_s + cfg_slot_num:
                    bin.sparse_cores.pop(0)
                    new_sparse_cores.append([cores_slot_s, cores_dict])
                    core_idx += 1
                else:
                    break
                
            while flops_idx < oringinal_sparse_size: 
                flops_slot_s, flops_dict = bin.sparse_flops[0]
                if flops_slot_s >= cfg_slot_s and flops_slot_s < cfg_slot_s + cfg_slot_num:
                    bin.sparse_flops.pop(0)
                    new_sparse_flops.append([flops_slot_s, flops_dict])
                    flops_idx += 1
                else:
                    break
            assert flops_idx == core_idx
            
            # Merge dictionaries of the same slot
            assert len(new_sparse_cores) >= 1
            assert len(new_sparse_flops) == len(new_sparse_cores)
            assert new_sparse_cores[0][0] == cfg_slot_s
            bin.sparse_cores.append(new_sparse_cores[0])
            
            merged_flops_dict = {}
            for _, flops_dict in new_sparse_flops:
                for key, value in flops_dict.items():
                    if key in merged_flops_dict:
                        merged_flops_dict[key] += value
                    else:
                        merged_flops_dict[key] = value
            bin.sparse_flops.append([cfg_slot_s, merged_flops_dict])
        assert len(bin.sparse_cores) == len(bin.sparse_flops)
        assert core_idx == flops_idx
        # set {} for sparse_event
        for slot_s, cfg, slot_num in bin.sparse_list:
            bin.sparse_event.append([slot_s, {}])
        # set the alloc_mod
        # set the alloc_mod
        bin.alloc_mod = "compress"
    
    bin_list.clear()
    bin_list.extend(planed_bin_list)

def gurobi_split_solver(glb_p_list, n_partition, _bin_tb_split, job_graph, sorted_chains):

    probs = [list(cfg.keys()) for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    J = len(probs)
    duation = [slot_num for slot_s, cfg, slot_num in _bin_tb_split.sparse_list]
    affinity_mode = "search" 
    assert affinity_mode in ["manual", "search", "greedy"]
    # collect used items from problems
    col_pid = set(reduce(lambda x,y: x+y, probs))
    
    if n_partition == partition_max:
        bin_name_list,affinity_dict1, affinity_dict2, placed_p, tbd_p = build_search_obj(glb_p_list, sorted_chains, job_graph, col_pid)
        
        gurobi_obj = "mux_min_nbin"
    else: 
        bin_name_list,affinity_dict1, affinity_dict2, placed_p, tbd_p = build_greedy_obj(n_partition, glb_p_list, sorted_chains, col_pid)
        gurobi_obj = "colocate_fix_nbin_min_size"

    M = len(bin_name_list)
    N = len(tbd_p)
    K = len(placed_p)

    solver = ClusterGurobiSolverSemi2D(
        M, N, K, J, tbd_p, placed_p, probs, duation, 1000, affinity_dict1, affinity_dict2, mode=gurobi_obj
    )
    solver.create_variables()
    solver.create_constraints()
    sel, bin_size = solver.solve()    # filter the empty bins
    if gurobi_obj == "mux_min_nbin":
        # classify the items into the bins
        sel, bin_size, bin_name_list = rename_bins_and_relable_assignments(glb_p_list, placed_p, tbd_p, M, sel, bin_size)
    else: 
        # update the placed items to the sol
        for pid, (size, bin_id) in placed_p.items():
            assert pid not in sel
            sel.update({pid:bin_id})
    # print(sol)
    return placed_p,bin_name_list,sel,bin_size

def rename_bins_and_relable_assignments(glb_p_list, placed_p, tbd_p, M, sel, bin_size):
    binid2pidgroups = {}
    for pid, bin_id in sel.items():
        if bin_id not in binid2pidgroups:
            binid2pidgroups[bin_id] = []
        binid2pidgroups[bin_id].append(pid)
    # remane
    # for each bin, find the max item, and set the bin name as the name of the max item
    name2pid = {p.task.name:p.pid for p in glb_p_list}
    pid2_name = {v:k for k,v in name2pid.items()}
    bin_dict = {}
    pid2_size = {**placed_p, **tbd_p}
    for bin_id, pidgroup in binid2pidgroups.items():
        if pidgroup:
            # get the max item
            max_pid = max(pidgroup, key=lambda x: pid2_size[x])
            # get the name of the max item
            bin_dict[bin_id] = (pid2_name[max_pid], bin_size[bin_id])

    # reindex the bins, by the corresponding pid 
    pid2_bin_name = {} 
    for pid, bin_id in sel.items():
        assert bin_id in range(M)
        pid2_bin_name[pid] = bin_dict[bin_id][0]
        # sort the name,size tuple by the pid
        # zip(sorted(bin_dict.values(), key=lambda x: name2pid[x[0]]))
    bin_name_list, bin_size = zip(*sorted(bin_dict.values(), key=lambda x: name2pid[x[0]]))
    sel = {pid:bin_name_list.index(pid2_bin_name[pid]) for pid in sel}
    return sel,bin_size,bin_name_list

def build_greedy_obj(
        n_partition, glb_p_list, sorted_chains, 
        col_pid):
    placed_p, tbd_p = {}, {}
    affinity_dict1, affinity_dict2 = {}, {}
    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))
    name2pid = {p.task.name:p.pid for p in glb_p_list}
    # generate the bin name list, 
    # for the chain that contain the node allowed to start a new bin, 
    # occupy the a quota to use a bin. Continue this process until the quota of bin creation is exhausted.
    # If a node located in a chain that has already been placed in a bin, mark it as placed.
    # otherwise, mark it as tbd.
        
    # create bins
    # get allowed Bin names and sizes
    allowed_bin_name = []
    for _p in glb_p_list:
        if _p.task.pre_assigned_resource_flag:
            allowed_bin_name.append(_p.task.name)
            
    bin_name_list = []
    for info in sorted_chains:
        # check if the chain is allowed to start a new bin, and occupy the quota
        chain = info["chain_nodes"]
        tgt_bin_id = None
        for node in chain:
            pid = name2pid[node]
            if pid in placed_p: # the node has been placed in another chain, skip, also skip the partition
                continue
            if node in allowed_bin_name:
                tgt_bin_id = len(bin_name_list)
                bin_name_list.append(node)
                break
            # set the placement
        if tgt_bin_id is not None:
                # the placed items
            for node in chain:
                pid = name2pid[node]
                if pid not in placed_p:
                    _p:ProcessInt = process_dict[pid]
                    size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
                    placed_p.update({pid:[size, tgt_bin_id]})
                    print(f"{node} -> {bin_name_list[tgt_bin_id]}")
            print(f"Bin {tgt_bin_id}:{bin_name_list[tgt_bin_id]} is used for the chain {chain}")
            n_partition -= 1
        if n_partition == 0:
            break
    # mark others as tbd
    for pid, _p in process_dict.items():
        if pid not in col_pid:
            continue
        _p:ProcessInt
        if pid not in placed_p:
            size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
            tbd_p.update({pid:size})

    return bin_name_list,affinity_dict1, affinity_dict2, placed_p, tbd_p

def build_search_obj(
    glb_p_list, sorted_chains, 
        job_graph, col_pid): 

    placed_p, tbd_p = {}, {}
    affinity_dict1, affinity_dict2 = {}, {}
    process_dict = OrderedDict(sorted([(p.pid, p) for p in glb_p_list]))
    name2pid = {p.task.name:p.pid for p in glb_p_list}
    # create bins
    # get allowed Bin names and sizes
    # find the largest item on each chain
    used = []
    bin_name_list = []
    end_nodes = [n for n, x in job_graph.out_degree() if x == 0]
    src_nodes = [n for n, x in job_graph.in_degree() if x == 0]
    size_dict = {x: process_dict[name2pid[x]].task.pre_assigned_resource.main_size\
             +process_dict[name2pid[x]].task.pre_assigned_resource.RDA_size 
                         for x in job_graph.nodes if x not in end_nodes and x not in src_nodes}
    for info in sorted_chains:
        chain = info["chain_nodes"]
        free_nodes = [x for x in chain if x not in used]
        if free_nodes:
            bgest = max(free_nodes, key=lambda x: size_dict[x])
            used.append(bgest)
            # affinity_dict1.update({name2pid[bgest]:len(bin_name_list)})
            bin_name_list.append(bgest)
            
        
        # set the affinity2 by the connectivities of the tasks in the job graph
        # using bfs 
    from collections import deque
    q = deque([node for node in src_nodes])
    visited = []
    while q:
        node = q.popleft()
        if node in visited:
            continue
        visited.append(node)
        for successor in job_graph.successors(node):
            if node in name2pid and successor in name2pid:
                pid = name2pid[node]
                tgt_pid = name2pid[successor]
                if node not in bin_name_list and successor not in bin_name_list:
                    affinity_dict2[(pid, tgt_pid)] = 1
                # elif node in bin_name_list and successor not in bin_name_list:
                #     bin_id = bin_name_list.index(node)
                #     affinity_dict1[(tgt_pid, bin_id)] = 1 
                # elif node not in bin_name_list and successor in bin_name_list:
                #     bin_id = bin_name_list.index(successor)
                #     affinity_dict1[(pid, bin_id)] = 1 
            if successor not in visited:
                q.append(successor)
    # mark others as tbd
    for pid, _p in process_dict.items():
        if pid not in col_pid:
            continue
        _p:ProcessInt
        if pid not in placed_p:
            size = _p.task.pre_assigned_resource.main_size+_p.task.pre_assigned_resource.RDA_size
            tbd_p.update({pid:size})
    return bin_name_list,affinity_dict1, affinity_dict2, placed_p, tbd_p
