import argparse
import os
import pickle
from approach_def import set_verbose_output, set_realloc_disabled
from approach_sched import PartitionConfig
from global_var import BW_DRAM, GLB_BUFFER_SIZE_PER_CORE, case_name_bp
from approach_Eq import trasfer_realloc_as_task, set_time_unit
from approach_initiator import instantiate_processors, get_partition_info, initialize_events
from approach_sim import run_simulation
from approach_setup import setup_benchmark
from utils import check_parents_path
from utils import input_parser

def main():
    # This is the main entry point for the refactored, policy-driven simulation flow.
    # It uses the original argument parser from sim_main.py for full compatibility.
    args = input_parser()
    
    set_verbose_output(args.verbose)
    
    # 设置切换开销控制（用于Case 3对照实验）
    realloc_disabled = getattr(args, 'barrier_dis', False)
    set_realloc_disabled(realloc_disabled)

    # Set time unit and normalization factor first, as it's needed for graph loading
    time_unit, time_norm_factor = set_time_unit(1e-6, False) 

    # 2. Setup benchmark: generate/load scheduling solution (bin_list)
    args.test_case = case_name_bp
    n_p_bk, args.n_p = args.n_p, 3
    G, pid2name, bin_list, policy, path_ctx, T_hp = setup_benchmark(args, time_norm_factor)    
    args.n_p = n_p_bk
    
    # 3. Create PartitionConfig from the results of the setup
    num_partitions, partition_size, flops_per_core, partition_task_map, TSMap_list = get_partition_info(bin_list, G, pid2name)

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
        swt_lat_list=swt_lat_list,
        G=G,
        T_hp=T_hp,
    )

    # 3. Initialize simulation events based on the policy
    int_event_t = initialize_events(policy, G, TSMap_list)

    # 4. Instantiate processors and run simulation
    processors, event_t_rt, stats_collector = instantiate_processors(
        partition_cfg, list(int_event_t), policy=policy, stat_param=args.stat_param
    )
    # 使用n_p参数而不是硬编码的100
    num_hp = args.n_p
    run_simulation(processors, event_t_rt, G, num_hp=num_hp, T_hp=T_hp, verbose=args.verbose, var_en=True)     

    # # 仿真结束后保存collector对象
    # if not getattr(args, 'dry_run', False):
    #     collector_path = path_ctx.get_collector_path()
    #     check_parents_path(collector_path)
    #     print(f"INFO: Saving collector object to {collector_path}")
    #     with open(collector_path, 'wb') as f:
    #         pickle.dump(stats_collector, f)

    return stats_collector

if __name__ == "__main__":
    main()