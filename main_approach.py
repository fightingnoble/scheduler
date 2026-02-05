import argparse
import os
from approach_def import set_verbose_output
from approach_sched import PartitionConfig
from global_var import BW_DRAM, GLB_BUFFER_SIZE_PER_CORE
from approach_Eq import trasfer_realloc_as_task, set_time_unit
from approach_initiator import instantiate_processors, get_partition_info, initialize_events
from approach_sim import run_simulation
from approach_setup import setup_benchmark
from utils import check_parents_path


if __name__ == "__main__":
    # This is the main entry point for the refactored, policy-driven simulation flow.
    # It uses the original argument parser from sim_main.py for full compatibility.
    from utils import input_parser
    target_sim_n_p = 100
    args = input_parser()
    
    set_verbose_output(args.verbose)

    # Set time unit and normalization factor first, as it's needed for graph loading
    time_unit, time_norm_factor = set_time_unit(1e-6, False) 

    # 2. Setup benchmark: generate/load scheduling solution (bin_list)
    G, pid2name, bin_list, policy, path_ctx, T_hp = setup_benchmark(args, time_norm_factor)    
    
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
        G, partition_cfg, list(int_event_t), policy=policy
    )
    stat_log_path = path_ctx.get_stat_log_path()
    motiv3_plot_path = path_ctx.get_plot_path("motiv3")
    check_parents_path(stat_log_path)
    check_parents_path(motiv3_plot_path)
    stats_collector.set_path("stat", stat_log_path)
    stats_collector.set_path("motiv3", motiv3_plot_path)
    run_simulation(processors, event_t_rt, G, num_hp=100, T_hp=T_hp, verbose=args.verbose, var_en=args.exec_var_en)
