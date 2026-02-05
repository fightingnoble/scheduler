
from task.task_cfg import export_json_graph_utils
from sim_main import (build_paths_and_ctx, build_workload_and_criticality, 
                     build_simulation_env, perform_bin_packing, preprocess_args,
                     determine_resource_config, create_scheduler_elements_with_config)
from global_var import *
from approach_initiator import instantiate_mygraph_from_json

def setup_benchmark(args, time_norm_factor):
    """
    Main function to setup the benchmark. It generates workloads, creates or loads
    the scheduling information (bin_list), and prepares the logical graph.

    Returns:
        tuple: (G, pid2name, bin_list, policy, log_path_root, hyper_p)
    """
    print_title("Benchmark Setup Started")
    
    # ======================== 使用 sim_main API ========================
    # call api from sim_main.py, 
    # build the workload graph, setup the path, 
    # generate the bin_list
    preprocess_args(args)

    # 1. 构建路径和上下文
    path_params, path_ctx = build_paths_and_ctx(args)
    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root, case_pth = path_params

    # redirect standard output
    from contextlib import redirect_stdout
    with open(path_ctx.get_log_path(), 'w') as f:
        with redirect_stdout(f):
            # 2. 生成 workload 并设置 criticality
            workload = build_workload_and_criticality(args)
            hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, need_repack = workload
            export_json_graph_utils(physical_graph_nx, path_ctx.graph_fn)

            # 3. 决定资源配置
            num_cores, bin_list = determine_resource_config(args, path_params, path_ctx, need_repack)
            
            # 4. 使用资源配置创建调度器元素
            scheduler_elements = create_scheduler_elements_with_config(
                args, path_params, path_ctx, workload, num_cores, bin_list
            )
            task_spec, rsc_list, msg_dispatcher, a_data_pipe, w_data_pipe, \
                scheduler_list, monitor_list, trace_path, sim_step = scheduler_elements
            
            # 5. 构建仿真环境参数
            num_periods, warmup, quantumSize, event_range, event_iter_dict = build_simulation_env(
                args, workload, sim_step
            )

            # 6. 执行装箱算法
            bin_list_save_path, num_cores, glb_p_list, hyper_p = perform_bin_packing(
                args, glb_p_list, num_cores, bin_list, hyper_p,
                sim_step, path_para_dict, para_scan_group1,
                event_iter_dict, quantumSize, num_periods,
                cfg_para_dict, physical_graph_nx, need_repack,
                plot_path_para, path_ctx, 
                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,
                )
            assert bin_list_save_path is not None, "Error: Failed to perform bin packing"

    # 5. generate schedule parameters
    G, pid2name = instantiate_mygraph_from_json(
        path_ctx.graph_fn,
        time_norm_factor=time_norm_factor
        )
    
    # 如果指定了 bin_list 路径则加载，否则使用生成的 bin_list
    # bin_list = load_pickle(args.load_bin_list_path)
    # # 重新计算 hyper_p
    # f_gcd = np.gcd.reduce([freq for task, freq in G.logical_graph.nodes(data="freq")])
    # hyper_p = 1/f_gcd

    print_title("Benchmark Setup Finished")
    
    return G, pid2name, bin_list, args.policy, path_ctx, hyper_p

def print_title(title):
    print("\n" + "="*25)
    print(f" {title}")
    print("="*25 + "\n")
