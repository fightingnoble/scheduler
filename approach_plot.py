from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from approach_util import BaseProcessor
    from approach_collector import StatisticsCollector
from typing import List
from approach_util import Acc_p, Sen_p, MyGraph, PartitionConfig, GlobalEvent_t
from global_var import elim_nume_error, BW_DRAM, GLB_BUFFER_SIZE_PER_CORE
from task_estimation import trasfer_realloc_as_task, time_eq, time_gt, time_gtq, time_lt, time_ltq, time_add, time_sub
from approach_initiator import instantiate_processors, instantiate_mygraph_from_json, get_partition_info
from task_estimation import set_time_unit
from utils import load_pickle
from global_var import FLOPS_PER_CORE


def print_progress(curr_t, pred_t, processors):
    print(f"===Decision metadata at {curr_t}===")
    if time_gt(curr_t, 0):
        duration = time_sub(curr_t, pred_t)
        print(f"\tDuration: {duration}")
    else:
        print(f"\tInitial decision")
    
    # 只打印有实际动作的处理器
    active_processors = []
    for i, proc in enumerate(processors):
        # 检查处理器是否有实际动作
        if proc.has_action():
            active_processors.append((i, proc))
    
    # 只打印活跃的处理器
    for i, proc in active_processors:
        proc.repr_info() # 调用新函数

    # 如果没有活跃处理器，打印一个简短的提示
    if not active_processors:
        print(f"\tNo active processors")



def run_simulation(processors:List[BaseProcessor], event_t:GlobalEvent_t, G:MyGraph, num_hp=1, T_hp=float('inf'), verbose=False):
    pred_t = -float('inf')
    curr_t = -T_hp
    curr_hp = -2
    sim_hp = num_hp + 1
    stats_collector: StatisticsCollector = processors[0].stats_collector  # 假设共享

    while (G.nodes() or curr_hp < num_hp) or time_eq(curr_t, 0):
        # if curr_t != 0:
        #     print(f"\nmove from {pred_t} to {curr_t}\n") 

        next_hp_boundary = elim_nume_error((curr_hp+1) * T_hp)
        # next_hp_boundary: -T_hp, 0, T_hp, 2T_hp, 3T_hp, 
        # curr_t: -T_hp, 0, T_hp, 2T_hp, 3T_hp, 
        if (time_gtq(curr_t, next_hp_boundary) and time_lt(pred_t, next_hp_boundary)):
            if curr_hp >= 0:
                stats_collector.forward_hyperperiod()
            
            # -2, -1, 0, 1, 2 -> -1, 0, 1, 2, 3
            curr_hp += 1
            # 0, 1, 2, 3, 4,
            tgt_hp = curr_hp + 1
            # the final hp is used for draining the unscheduled tasks
            if tgt_hp < num_hp:
                # a) duplicate the graph
                G.duplicate_for_hyperperiod(tgt_hp, 0, T_hp)
                # b) update the mapping
                for processor in processors:
                    processor.update_mapped_nodes_for_hyperperiod(tgt_hp)
                # c) add the external events
                event_t.add_events_for_hyperperiod(tgt_hp, T_hp, curr_t)
            elif tgt_hp < sim_hp:
                # curr_hp >= num_hp: only add the table events
                event_t.add_events_for_hyperperiod(tgt_hp, T_hp, curr_t, type_list=["table"])

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

        if verbose:
            print_progress(curr_t, pred_t, processors)
        
        # status backup
        next_curr_t = time_add(curr_t, duation)
        curr_t, pred_t = min(next_curr_t, next_timer_event, elim_nume_error(sim_hp * T_hp)), curr_t
        assert curr_t != pred_t
        if time_eq(curr_t, sim_hp * T_hp):
            break
        if curr_t == next_timer_event:
            _, event_type = event_t.confirm_next_event(pred_t)
        else:
            event_type = "finish"
        # check the event type: finish, external
        assert event_type in ["finish", "external", "table"]

    # 仿真结束后输出统计摘要
    if stats_collector:
        stats_collector.export_summary()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--case", type=str, default="case1", help="Case name")
    parser.add_argument("--var_en", type=bool, default=False, help="Enable variable execution time")
    parser.add_argument("--num_hp", type=int, default=1, help="Number of hyperperiods")
    args = parser.parse_args()
    num_hp = args.num_hp
    T_hp = 0.1 if args.case != "case1" else 100

    from example.bm4 import *
    if args.case == "case1":
        policy = "pglb"
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
            T_hp=T_hp,
        )
        event_t = [(0, "external")]

    elif args.case == "case2":
        print("case2: single partition glb")
        # similar to case1, but with a actual load graph 
        policy = "pglb"
        
        time_unit, time_norm_factor = set_time_unit(1e-6, False)

        G, pid2name = instantiate_mygraph_from_json(
            'cache/graph_w_ert_ddl.json',
            time_norm_factor=time_norm_factor
            )

        # # Suppose there is only one partition
        swt_lat = trasfer_realloc_as_task(BW_DRAM, 500, GLB_BUFFER_SIZE_PER_CORE, time_norm_factor)
        sink_and_op_nodes = [n for n, x in G.logical_graph.in_degree() if x > 0]
        partition_cfg = PartitionConfig(
            num_partitions=1,
            cap_list=[190],
            base_pwr_list=[FLOPS_PER_CORE],
            mapped_node_list=[sink_and_op_nodes],
            TSmap_list=[None],  # 如果用cyclic策略可传入具体map
            swt_lat_list=[swt_lat],  # case2使用计算值
            G=G, # nodes other than sources
            T_hp=T_hp,
        )

        # collect the event_t from:
        event_t = set()
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "src"]:
            t = elim_nume_error(G.logical_graph.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        
    elif args.case == "case3":
        print("case3: multi partition pglb")
        policy = "pglb"

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
            T_hp=T_hp,
        )
        # collect the event_t from:
        event_t = set()
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "src"]:
            t = elim_nume_error(G.logical_graph.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        
    elif args.case in ["case4", "case5"]:
        print(f"case{args.case}: {'single' if args.case == 'case4' else 'multi'}-partition cyclic")
        policy = "cyc"
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
            T_hp=T_hp,
        )
        # collect the event_t from:
        event_t = set()
        # 1. bin_list (from the sparse_list of each bin)
        for TSmap in TSMap_list:
            for t, cfg in TSmap:
                event_t.add((t, "table"))
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "src"]:
            t = elim_nume_error(G.logical_graph.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s 0th event at {t}")
        
        
    elif args.case in ["case6", "case7"]:
        print(f"case{args.case}: {'single' if args.case == 'case6' else 'multi'}-partition reservation")
        policy = "reserv"
        
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
            T_hp=T_hp,
        )
        # collect the event_t from:
        event_t = set()
        # 2. the offset of the sensor and the length to be simulated 
        # 只初始化第一个超周期的事件，后续事件将动态添加
        for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "src"]:
            t = elim_nume_error(G.logical_graph.nodes[node]['offset'])
            event_t.add((t, "external"))
            print(f"{node}'s trigger event at {t}")

        for node in [n_ for n_ in G.logical_graph.nodes if G.logical_graph.nodes[n_]['type'] == "op"]:
            t = elim_nume_error(G.logical_graph.nodes[node]['ert'])
            event_t.add((t, "external"))
            print(f"{node}'s ert event at {t}")
        
    else:
        raise ValueError(f"Invalid case: {args.case}")

    processors, event_t, stats_collector = instantiate_processors(
        G, partition_cfg, list(event_t), policy=policy
        )
    run_simulation(processors, event_t, G, num_hp=num_hp, T_hp=T_hp)
    # 仿真结束后输出统计摘要
    if stats_collector:
        stats_collector.export_summary()
