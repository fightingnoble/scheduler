from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from approach_util import BaseProcessor
from typing import List
from approach_util import Acc_p, Sen_p, MyGraph, PartitionConfig, acc_p_factory, GlobalEvent_t, old_timestep
from global_var import elim_nume_error, BW_DRAM, GLB_BUFFER_SIZE_PER_CORE
from task_estimation import trasfer_realloc_as_task

def print_progress(curr_t, pred_t, processors):
    print(f"===Decision metadata at {curr_t}===")
    if curr_t > 0:
        print(f"\tDuration: {curr_t - pred_t}")
    else:
        print(f"\tInitial decision")
    
    # 只打印有实际动作的处理器
    active_processors = []
    for i, proc in enumerate(processors):
        # 检查处理器是否有实际动作
        has_action = False
        
        # 检查是否有运行中的任务
        if hasattr(proc, 'running') and proc.running:
            has_action = True
        
        # 检查是否有就绪任务
        if hasattr(proc, 'ready'):
            if isinstance(proc.ready, dict) and proc.ready:
                has_action = True
            elif hasattr(proc.ready, 'qsize') and proc.ready.qsize() > 0:
                has_action = True
        
        # 检查是否有资源分配
        if hasattr(proc, 'res_map') and proc.res_map:
            has_action = True
        
        # 检查是否有状态变化
        if hasattr(proc, 'sys_state') and proc.sys_state == "R":
            has_action = True
        
        if has_action:
            active_processors.append((i, proc))
    
    # 只打印活跃的处理器
    for i, proc in active_processors:
        print(f"\n\tProcessor {i}: {proc}")
        # Print sys_state if it exists
        if hasattr(proc, 'sys_state'):
            print(f"\t\tstate: {proc.sys_state}")
        # Print running if it exists
        if hasattr(proc, 'running'):
            print(f"\t\tRunning_queue: {proc.running}")
        # Print res_map if it exists
        if hasattr(proc, 'res_map'):
            print(f"\t\tRes_map: {proc.res_map}")
        # Print slack_map if it exists
        if hasattr(proc, 'slack_map'):
            print(f"\t\tSlack_map: {proc.slack_map}")
    
    # 如果没有活跃处理器，打印一个简短的提示
    if not active_processors:
        print(f"\tNo active processors")


def instantiate_processors(G, partition_cfg, event_t, policy="glb"):
    # 创建传感器处理器，使用动态映射模式
    src_nodes = [n for n, x in G.logical_graph.in_degree() if x == 0]
    sen_p0 = Sen_p("sen_p", 3, 1, G, set(src_nodes))
    acc_p_list = acc_p_factory(policy, partition_cfg)
    processors = [sen_p0] + acc_p_list
    for i, proc in enumerate(processors):
        print(f"Processor {i}: {proc}")
    event_t = GlobalEvent_t(event_t)
    return processors, event_t

def run_simulation(processors:List[BaseProcessor], event_t:GlobalEvent_t, G:MyGraph, num_hp=1, T_hp=float('inf')):
    pred_t = 0
    curr_t = 0
    curr_hp = 0
    # TODO: check the condition (num_hp + 1) * T_hp
    sim_hp = num_hp + 1
    while (G.nodes() and curr_hp < sim_hp) or curr_t == 0:
        if curr_t != 0:
            print(f"\nmove from {pred_t} to {curr_t}\n") 

        if (curr_t >= (curr_hp) * T_hp and pred_t < (curr_hp) * T_hp) and curr_hp < num_hp or curr_t == 0:
            # update the graph
            G.duplicate_for_hyperperiod(curr_hp, 0, T_hp)
            # 更新处理器映射
            for processor in processors:
                processor.update_mapped_nodes_for_hyperperiod(curr_hp)
            # 更新事件队列
            event_t.add_events_for_hyperperiod(curr_hp, T_hp)
            curr_hp += 1


        print(f"===At the beginning of {curr_t}===")
        
        # Two types of divice is considered: sen_p0 and acc_p0
        # sen_p0: sensor processor
        # acc_p0: accelerator     
        # Each processor folows update -> schedule -> execute -> update cycle. 
        
        # During the update phase, the runtime will 
        # 1. calculate remaining workloads and check finishing events. 
        # process the event (1. external, 2. finish and reallocate)
        
        # The most important difference of the accelerator processor, compared to the sensor processor, is that 
        # it has to go through a extra step of reallocation, before launching the different allocation map. 
        # The reallocation is a special case of execution, which can be seen as a "system task" that reallocate the resources, 
        # and will preempt other computational tasks until it finishes. 
        # However, I don't know how to unify two types of tasks. 
        
        # A minor difference is the scheduling policy: Sensor processor follows the first-come-first-served policy (FCFS), 
        # while accelerator follows a priority-based policy with parameterized priority defination. 

        # Task dispatching: 
        # As for our ADS scenario, we mark the periodic timer event as "external", 
        # which informs the sensor processor to access and pre-process sensor data. 
        # While the accelerator processor is responsible for the remaining DNN tasks, 
        # which are triggered to schedule incomming tasks, when "finish" events occur. 
        # Specially, "reallocate" means the completion of system reallocation, 
        # which is also need to be unified with the the concept of "finish" event. 
        
        # update the running queue, calculate remaining workloads and check finishing events
        # sen_p0.update_run(pred_t, curr_t)
        # new_comp = acc_p0.update_run(pred_t, curr_t)
        new_comp_dict = {}
        for proc in processors:
            new_comp = proc.update_run(pred_t, curr_t)
            new_comp_dict[proc] = new_comp

        # process the event, update the ready queue
        # sen_p0.update_ready(curr_t)
        # new_ready_list = acc_p0.update_ready(curr_t)
        new_ready_dict = {}
        for proc in processors:
            new_ready = proc.update_ready(curr_t)
            new_ready_dict[proc] = new_ready

        # TODO: collective update ready flag accross all processors

        # allocation progress
        # duation_sen_p = sen_p0.sched(curr_t) 
        # duation_acc_p= acc_p0.sched(curr_t, new_comp, new_ready_list)

        # 统一 sched
        duation_dict = {}
        for proc in processors:
            # 兼容不同类型的参数
            if isinstance(proc, Acc_p):
                proc: Acc_p
                duation = proc.sched(curr_t, new_comp_dict[proc], new_ready_dict[proc])
            else:
                proc: Sen_p
                duation = proc.sched(curr_t)
            duation_dict[proc] = duation

        # calculate the next 
        # duation = min(duation_acc_p, duation_sen_p)
        # 统一事件推进
        duation = min(duation_dict.values())

        # 统一事件队列（可扩展为每个处理器独立event_t）
        next_timer_event = event_t.get_next_event_time(curr_t)

        print_progress(curr_t, pred_t, processors)
        
        # status backup
        curr_t, pred_t = min(curr_t + duation, next_timer_event[0], sim_hp * T_hp), curr_t
        assert curr_t != pred_t
        event_type = next_timer_event[1] if curr_t == next_timer_event[0] else "finish"
        # check the event type: finish, external
        assert event_type in ["finish", "external"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--case", type=str, default="case1", help="Case name")
    args = parser.parse_args()

    from example.bm4 import *
    if args.case == "case1":
        G = MyGraph(
            task_graph_srcs, task_graph_ops, task_graph_sinks, 
            task_attr, src_attr, sink_attr
        )

        # # Suppose there is only one partition
        sink_and_op_nodes = [n for n, x in G.logical_graph.in_degree() if x > 0]
        partition_cfg = PartitionConfig(
            num_partitions=1,
            cap_list=[tot_core],
            base_pwr_list=[1],
            mapped_node_list=[sink_and_op_nodes],
            TSmap_list=[None],  # 如果用cyclic策略可传入具体map
            swt_lat_list=[5],  # case1使用固定值5
            G=G, # nodes other than sources
        )
        event_t = [(0, "external")]

        processors, event_t = instantiate_processors(
            G, partition_cfg, list(event_t), policy="pglb"
            )
        run_simulation(processors, event_t, G, num_hp=1, T_hp=100)

    elif args.case == "case2":
        print("case2: single partition glb")
        # similar to case1, but with a actual load graph 
        from approach_util import instantiate_mygraph_from_json, get_partition_info, set_time_unit
        from utils import load_pickle
        from global_var import FLOPS_PER_CORE
        
        time_unit, time_norm_factor = set_time_unit(1e-6, False)

        G, pid2name = instantiate_mygraph_from_json(
            'cache/graph_w_ert_ddl.json',
            time_norm_factor=time_norm_factor
            )

        # # Suppose there is only one partition
        swt_lat = trasfer_realloc_as_task(BW_DRAM, 500, GLB_BUFFER_SIZE_PER_CORE, time_norm_factor)
        partition_cfg = PartitionConfig(
            num_partitions=1,
            cap_list=[500],
            base_pwr_list=[FLOPS_PER_CORE],
            mapped_node_list=[G.sinks+G.ops],
            TSmap_list=[None],  # 如果用cyclic策略可传入具体map
            swt_lat_list=[swt_lat],  # case2使用计算值
            G=G, # nodes other than sources
        )

        # collect the event_t from:
        event_t = set()
        # 1. bin_list (from the sparse_list of each bin)
        for bin in bin_list:
            for cfg_slot_s, next_cfg, cfg_slot_num in bin.sparse_list:
                event_t.add((elim_nume_error(cfg_slot_s*old_timestep), "external"))
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in G.srcs:
            t = elim_nume_error(G.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        processors, event_t = instantiate_processors(
            G, partition_cfg, list(event_t), policy="pglb"
            )

    elif args.case == "case3":
        print("case3: multi partition pglb")
        from approach_util import instantiate_mygraph_from_json, get_partition_info, set_time_unit
        from utils import load_pickle
        
        time_unit, time_norm_factor = set_time_unit(1e-6, False)
        G, pid2name = instantiate_mygraph_from_json(
            'cache/graph_w_ert_ddl.json',
            time_norm_factor=time_norm_factor
            )

        bin_list = load_pickle('./cache/coalescing_scan/n_bins_max/x1_0.1s_rda-20.00%(J)_100.00%(T)_30.00%(S)_ignore/bin_list_477.pkl')
        num_partitions, partition_size, flops_per_core, partition_task_map, TSMap_list = get_partition_info(bin_list, G, pid2name)

        # 为每个分区计算swt_lat
        swt_lat_list = []
        for i in range(num_partitions):
            swt_lat = trasfer_realloc_as_task(BW_DRAM, partition_size[i], GLB_BUFFER_SIZE_PER_CORE, time_norm_factor)
            swt_lat_list.append(swt_lat)

        partition_cfg = PartitionConfig(
            num_partitions=num_partitions,
            cap_list=partition_size,
            base_pwr_list=flops_per_core,
            mapped_node_list=partition_task_map,
            TSmap_list=TSMap_list,
            swt_lat_list=swt_lat_list,  # case3使用计算值
            G=G,
        )
        # collect the event_t from:
        event_t = set()
        # 1. bin_list (from the sparse_list of each bin)
        for bin in bin_list:
            for cfg_slot_s, next_cfg, cfg_slot_num in bin.sparse_list:
                event_t.add((elim_nume_error(cfg_slot_s*old_timestep), "external"))
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in G.srcs:
            t = elim_nume_error(G.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        
        processors, event_t = instantiate_processors(
            G, partition_cfg, list(event_t), policy="pglb"
            )
    elif args.case in ["case4", "case5"]:
        print(f"case{args.case}: {'single' if args.case == 'case4' else 'multi'}-partition cyclic")
        from approach_util import instantiate_mygraph_from_json, get_partition_info, set_time_unit
        from utils import load_pickle
        
        time_unit, time_norm_factor = set_time_unit(1e-6, False)
        G, pid2name = instantiate_mygraph_from_json(
            'cache/graph_w_ert_ddl.json',
            time_norm_factor=time_norm_factor
            )
        if args.case == "case5":
            bin_list = load_pickle('./cache/coalescing_scan/n_bins_max/x1_0.1s_rda-20.00%(J)_100.00%(T)_30.00%(S)_ignore/bin_list_477.pkl')
        else:
            bin_list = load_pickle('./cache/coalescing_scan/n_bins_1/x1_0.1s_rda-20.00%(J)_100.00%(T)_30.00%(S)_ignore/bin_list_371.pkl')
        
        num_partitions, partition_size, flops_per_core, partition_task_map, TSMap_list = get_partition_info(bin_list, G, pid2name)

        # 为每个分区计算swt_lat
        swt_lat_list = []
        for i in range(num_partitions):
            swt_lat = trasfer_realloc_as_task(BW_DRAM, partition_size[i], GLB_BUFFER_SIZE_PER_CORE, time_norm_factor)
            swt_lat_list.append(swt_lat)
        
        partition_cfg = PartitionConfig(
            num_partitions=num_partitions,
            cap_list=partition_size,
            base_pwr_list=flops_per_core,
            mapped_node_list=partition_task_map,
            TSmap_list=TSMap_list,
            swt_lat_list=swt_lat_list,  # case3使用计算值
            G=G,
        )
        # collect the event_t from:
        event_t = set()
        # 1. bin_list (from the sparse_list of each bin)
        for bin in bin_list:
            for cfg_slot_s, next_cfg, cfg_slot_num in bin.sparse_list:
                event_t.add((elim_nume_error(cfg_slot_s*old_timestep), "external"))
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in G.srcs:
            t = elim_nume_error(G.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        processors, event_t = instantiate_processors(
            G, partition_cfg, list(event_t), policy="cyc"
            )
    elif args.case in ["case6", "case7"]:
        print(f"case{args.case}: {'single' if args.case == 'case6' else 'multi'}-partition reservation")
        from approach_util import instantiate_mygraph_from_json, get_partition_info, set_time_unit
        from utils import load_pickle
        
        time_unit, time_norm_factor = set_time_unit(1e-6, False)
        G, pid2name = instantiate_mygraph_from_json(
            'cache/graph_w_ert_ddl.json',
            time_norm_factor=time_norm_factor
            )
        if args.case == "case7":
            bin_list = load_pickle('./cache/coalescing_scan/n_bins_8/x1_0.1s_rda-20.00%(J)_100.00%(T)_30.00%(S)_ignore/bin_list_371.pkl')
        else:
            bin_list = load_pickle('./cache/coalescing_scan/n_bins_1/x1_0.1s_rda-20.00%(J)_100.00%(T)_30.00%(S)_ignore/bin_list_371.pkl')
        
        num_partitions, partition_size, flops_per_core, partition_task_map, TSMap_list = get_partition_info(bin_list, G, pid2name)

        # 为每个分区计算swt_lat
        swt_lat_list = []
        for i in range(num_partitions):
            swt_lat = trasfer_realloc_as_task(BW_DRAM, partition_size[i], GLB_BUFFER_SIZE_PER_CORE, time_norm_factor)
            swt_lat_list.append(swt_lat)
        
        partition_cfg = PartitionConfig(
            num_partitions=num_partitions,
            cap_list=partition_size,
            base_pwr_list=flops_per_core,
            mapped_node_list=partition_task_map,
            TSmap_list=TSMap_list,
            swt_lat_list=swt_lat_list,  # case3使用计算值
            G=G,
        )
        # collect the event_t from:
        event_t = set()
        # 1. bin_list (from the sparse_list of each bin)
        for bin in bin_list:
            for cfg_slot_s, next_cfg, cfg_slot_num in bin.sparse_list:
                event_t.add((elim_nume_error(cfg_slot_s*old_timestep), "external"))
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in G.srcs:
            t = elim_nume_error(G.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        processors, event_t = instantiate_processors(
            G, partition_cfg, list(event_t), policy="reserv"
            )

    else:
        raise ValueError(f"Invalid case: {args.case}")

