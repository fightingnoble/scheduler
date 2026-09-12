
from task.task_cfg import export_json_graph_utils
from sim_main import (build_paths_and_ctx, build_workload_and_criticality,
                     init_sched_components, preprocess_args,
                     apply_forced_num_cores, generate_bin_paths, Bin_list_print,
                     render_bin_pack_plots, dump_and_check)
from global_var import *
from .approach_initiator import instantiate_mygraph_from_json

def run_benchmark_setup_pipeline(
    args, path_ctx, path_params, need_repack, hyper_p,
    bin_list, num_cores,
    fix_core_map=None,
    scale_factor=None,
    ):
    """
    Executes the main setup steps for the benchmark: workload generation,
    criticality assignment, resource configuration, creation of scheduler elements,
    simulation environment setup, and bin packing.

    Returns:
        Updated (hyper_p, bin_list, num_cores, rsc_map_w)
    """
    from contextlib import redirect_stdout
    with open(path_ctx.get_log_path(), 'w') as f:
        with redirect_stdout(f):
            # 2. 生成 workload 并设置 criticality
            workload = build_workload_and_criticality(args, fix_core_map=fix_core_map, scale_factor=scale_factor)
            hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w = workload
            export_json_graph_utils(physical_graph_nx, path_ctx.graph_fn)

            # 3. 根据case类型以及是否强制指定核心数，更新num_cores和bin_list
            # num_cores, bin_list = determine_resource_config(args, path_params, path_ctx, need_repack, bin_list)

            # 4-5. 初始化调度组件（global_sched 统一接口公用组件 + 仿真环境），返回执行句柄
            pack = init_sched_components(args, path_params, path_ctx, workload, num_cores, bin_list)

            # 6. 执行装箱算法（句柄封装了公用组件；backup/fallback 在 perform_bin_packing 内部）
            bin_list, max_core_num, glb_p_list, hyper_p, repack_success = pack(
                glb_p_list, bin_list, hyper_p, physical_graph_nx, need_repack
            )

            # 7. 应用资源约束（仅 non-repack 时应用，repack 不改变资源数量）
            if not need_repack:
                if args.num_cores is not None:
                    num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
                else:
                    num_cores = max_core_num

            # 8. 生成 dump 路径（解包 path_params 取路径相关成员）
            cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
                bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root, case_pth = path_params
            if need_repack:
                extra_suffix = f"_ov_{args.exec_t_comp_ratioB:.2f}_repack(T)"
            else:
                extra_suffix = ""
            # 同步更新 trace_path_para 中的 num_cores，确保路径一致性
            path_ctx.num_cores = num_cores
            trace_path_para['num_cores'] = num_cores
            bin_list_save_path, routing_table_save_path = generate_bin_paths(
                path_para_dict, path_ctx, num_cores, "packing save path",
                extra_suffix
            )

            # 9. 打印和绘制（sim_step/num_periods 取自执行句柄）
            Bin_list_print(bin_list, glb_p_list, pack.sim_step)
            if args.plot:
                render_bin_pack_plots(args, bin_list, glb_p_list, pack.sim_step, hyper_p, pack.num_periods, plot_path_para, path_ctx)

            # 10. Dump
            dump_and_check(bin_list_save_path, bin_list)
    return hyper_p, bin_list, num_cores, rsc_map_w

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

    # 检查是否需要重新装箱（repack 模式）
    need_repack = False  # 默认不repack
    if (args.exec_t_comp_ratioB != -1 and args.exec_t_comp_ratioA != args.exec_t_comp_ratioB):
        need_repack = True
    
    # step 0-1-2
    # Run extracted setup pipeline
    args.quantile = args.exec_t_comp_ratioA
    num_cores = args.num_cores
    bin_list = []
    hyper_p, bin_list, num_cores, phase1_rsc_map = run_benchmark_setup_pipeline(
        args, path_ctx, path_params, False, None, bin_list, num_cores
    )
    if need_repack:
        # Repack: recalculate per-task time slices (Step 1) with ratioB.
        # For cyc-S (num_bins=-1): bypass bin packing, only recalculate deadlines.
        # For reserv (num_bins>=2): run perform_bin_packing with pre_defined mapping;
        #   if ResourceInsufficientError (ratioB > ratioA), fallback to Phase 1 layout.
        # Extract fix_core_map from Phase 1 rsc_map_w
        # rsc_map_w[node] = (cores, latency, constr)
        fix_core_map = {node: cores for node, (cores, _, _) in phase1_rsc_map.items()}

        # Compute scale_factor: ratioB/ratioA creates tail margin when ratioB < ratioA
        # This scales down the latency, making windows shorter
        scale_factor = args.exec_t_comp_ratioB / args.exec_t_comp_ratioA

        args.quantile = args.exec_t_comp_ratioB
        hyper_p, bin_list, num_cores, repack_rsc_map = run_benchmark_setup_pipeline(
            args, path_ctx, path_params, True, hyper_p, bin_list, num_cores,
            fix_core_map=fix_core_map,
            scale_factor=scale_factor,
        )

    # 5. generate schedule parameters
    G, pid2name = instantiate_mygraph_from_json(
        path_ctx.graph_fn,
        time_norm_factor=time_norm_factor
        )

    print_title("Benchmark Setup Finished")
    
    return G, pid2name, bin_list, args.policy, path_ctx, hyper_p

def print_title(title):
    print("\n" + "="*25)
    print(f" {title}")
    print("="*25 + "\n")
