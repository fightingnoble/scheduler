import os, re
from typing import Dict
import warnings
from task.task_cfg import gen_workloads, export_json_graph_utils
from task.task_cfg import affinity_cfg
from sched.global_sched import push_task_into_bins_new
from task.task_agent import TaskInt
from task.spec import Spec
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt
from sched.bin_list_utils import get_task_layout_compact, get_task_layout_compact1bin, Bin_list_print
from model.resource_agent import Resource_model_int
from sched.scheduler_agent import Scheduler
from sched.placement import core_mapping_1d
from sched.monitor_agent import Monitor
from model.task_queue_agent import TaskQueue
from utils import dump_and_check, load_pickle, update_df, check_parents_path, build_path_old, get_case_path_str
from global_var import *
from utils import core_distr, time_cnt, vectorized_core_allocation
from paths import PathContext
import numpy as np
import argparse

# ============================================================================
# Global switch for Fixcore Repack mode
# When True: bypass greedy bin-packing in repack, keep Phase 1 layout,
#            ERT/DDL are already updated by deduce_cfg2 with fix_core_map
# When False: use original greedy bin-packing (push_task_into_bins_new)
# ============================================================================
USE_FIXCORE_REPACK = True


def generate_bin_paths(path_para_dict, path_ctx: PathContext, num_cores, check_hints, extra_suffix=""):
    """
    统一的二进制路径生成函数，处理所有算法分支中的重复路径生成代码
    
    Args:
        path_para_dict: 路径参数字典
        path_ctx: PathContext 实例
        num_cores: 核心数
        check_hints: 检查说明
        extra_suffix: 额外的后缀（如 repack 的 "_ov_0.80_repack(T)"）
    
    Returns:
        tuple: (bin_list_save_path, routing_table_save_path)
    """
    # 使用旧方法生成路径（需要将 extra_suffix 也加入旧格式参数）
    old_fmt_params = {**path_para_dict, "num_cores": num_cores}
    if extra_suffix:
        old_fmt_params["i_file_suffix"] = f"{old_fmt_params.get('i_file_suffix', '')}{extra_suffix}"
        old_fmt_params["file_suffix"] = f"{old_fmt_params.get('file_suffix', '')}{extra_suffix}"
    old_bin_list_save_path = bin_save_fmt.format(**old_fmt_params)
    old_routing_table_save_path = routing_table_save_fmt.format(**old_fmt_params)

    # 使用新方法生成路径：直接改 PathContext
    if extra_suffix:
        path_ctx.file_suffix = f"{path_ctx.file_suffix}{extra_suffix}"
        path_ctx.i_file_suffix = f"{path_ctx.i_file_suffix}{extra_suffix}"
    new_bin_list_save_path = path_ctx.get_bin_list_path()
    new_routing_table_save_path = path_ctx.get_routing_table_path()
    
    # 比较路径
    compare_paths(old_bin_list_save_path, new_bin_list_save_path, f"bin_list_save_path ({check_hints})")
    compare_paths(old_routing_table_save_path, new_routing_table_save_path, f"routing_table_save_path ({check_hints})")
    
    # 返回新路径
    return new_bin_list_save_path, new_routing_table_save_path


def compare_paths(old_path, new_path, path_type=""):
    """
    比较新旧路径，如果不一致则报错
    """
    if old_path != new_path:
        print(f"❌ 路径不匹配 ({path_type}):")
        print(f"   旧路径: {old_path}")
        print(f"   新路径: {new_path}")
        raise AssertionError(f"路径不匹配: {path_type}")
    else:
        print(f"✅ 路径匹配 ({path_type}): {old_path}")




def extract_pid2_bin_id(bin_list):
    pid2_bin_id = {}
    for _bin in bin_list:
        for pid in _bin.index_occupy_by_id().keys():
            if pid in pid2_bin_id:
                raise AssertionError(f"duplicated pid {pid}")
            pid2_bin_id[pid] = _bin.id
        _bin.clear()
    return pid2_bin_id


def apply_forced_num_cores(bin_list, estimated_num_cores, target):
    """
    统一处理强制核数逻辑：
    - 若未启用强制或无法从 trace 解析则保持估算值
    - 单 bin：直接扩容
    - 多 bin：按原 bin size 比例进行增量分配（沿用 core_distr）
    """
    if target == estimated_num_cores:
        return estimated_num_cores

    vectorized_core_allocation(bin_list, target)

    print(f"Force the num of Core {estimated_num_cores} -> {target}, the over subcription ratio is {target/estimated_num_cores}")
    return target


def render_bin_pack_plots(args, bin_list, glb_p_list, sim_step, hyper_p, num_periods, plot_path_para, path_ctx: PathContext):
    pid2name = {_p.pid: _p.task.name for _p in glb_p_list}
    num_cores = sum(b.num_resources for b in bin_list)

    # 旧法
    plot_path_para = dict(plot_path_para)
    plot_path_para.update({"num_cores": num_cores})
    old_plot_path_cyclic = plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "cyclic"})
    old_plot_path_full   = plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "full"})

    # PathContext
    path_ctx.case = "new_task_bin_pack"
    new_plot_path_cyclic = path_ctx.get_plot_path("cyclic")
    new_plot_path_full   = path_ctx.get_plot_path("full")

    # 比较路径
    compare_paths(old_plot_path_cyclic, new_plot_path_cyclic, "plot_path (cyclic)")
    compare_paths(old_plot_path_full,   new_plot_path_full,   "plot_path (full)")

    # f"{plot_root}/new_task_bin_pack_cyclic_{num_cores}{args.file_suffix}.pdf"
    get_task_layout_compact(
        bin_list, pid2name, save=True, time_step=sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=False, plot_legend=True,
        format=args.plt_fmt, txt_size=40, tick_dens=2,
        plot_start=hyper_p*(num_periods-1), plot_end=hyper_p*num_periods,
        save_path=new_plot_path_cyclic
    )
    # f"{plot_root}/new_task_bin_pack_full_{num_cores}{args.file_suffix}.pdf"
    get_task_layout_compact1bin(
        bin_list, pid2name, save=True, time_step=sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False,
        format=args.plt_fmt, txt_size=40, tick_dens=4, plot_start=0,
        save_path=new_plot_path_full
    )

def render_runtime_full_plot(args, actual_sched_record, glb_p_list, sim_step, hyper_p, num_periods, case_pth, plot_path_para, path_ctx: PathContext):
    pid2name = {_p.pid: _p.task.name for _p in glb_p_list}
    if not args.jitter_sim_en:
        old_plot_path = plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": case_pth, "plt_size": "full"})
    else:
        old_plot_path = plt_fn_w_seed_fmt.format(**plot_path_para, **{"case": case_pth, "plt_size": "full"})

    path_ctx.case = case_pth
    new_plot_path = path_ctx.get_plot_path("full", with_seed=args.jitter_sim_en)
    compare_paths(old_plot_path, new_plot_path, f"plot_path ({case_pth})")

    get_task_layout_compact1bin(
        actual_sched_record, pid2name, save=True, time_step=sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False,
        format=args.plt_fmt, txt_size=40, tick_dens=4, plot_start=0,
        save_path=new_plot_path
    )



# ======================== API Functions for approach_setup ========================

def build_paths_and_ctx(args):
    """
    构建路径参数和 PathContext，对应 main 中的 "build_paths" 段
    返回: (path_params, path_ctx)
    """
    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root = build_path_old(args)
    case_pth = get_case_path_str(args)

    # 创建 PathContext 实例
    path_ctx = PathContext(
        root_dir=args.root_dir,
        case=case_pth,
        num_bins=args.num_bins,
        aux_scale_factor=args.aux_scale_factor,
        e2e_latency=args.e2e_latency,

        file_suffix=args.file_suffix,
        i_file_suffix=args.i_file_suffix,
        force_suffix=args.force_suffix,
        
        exec_t_comp_ratioA=args.exec_t_comp_ratioA,
        lateness_mode=args.lateness_mode,

        num_cores=args.num_cores,
        exec_t_comp_ratioB=args.exec_t_comp_ratioB,
        
        seed=args.seed,
        jitter=args.jitter_sim_en
    )
    
    # 使用新方法生成 CSV 路径并比较
    old_csv_root = csv_xlxs_root
    new_csv_root = path_ctx.csv_root
    compare_paths(old_csv_root, new_csv_root, "csv_root")
    csv_xlxs_root = new_csv_root

    path_params = (cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict,
                   bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root, case_pth)
    return path_params, path_ctx


def build_workload_and_criticality(args, fix_core_map: Dict[str, int] = None, scale_factor: float = None):
    """
    生成 workload 并设置 criticality，对应 main 中的 "workload settings" 段
    返回: (hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w)
    """
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w = gen_workloads(args, fix_core_map=fix_core_map, scale_factor=scale_factor)

    # assert all the process has hard deadline
    if args.lateness_mode == "all_hard":
        for _p in glb_p_list:
            _p.task.criticality = "hard"
            _p.task.chain_criticality = "hard"
    elif args.lateness_mode == "all_soft":
        for _p in glb_p_list:
            _p.task.criticality = "soft"
            _p.task.chain_criticality = "soft"
    elif args.lateness_mode == "ignore":
        pass

    return hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w


def determine_resource_config(args, path_params, path_ctx, need_repack, bin_list):
    """
    决定如何获取 num_cores 和 bin_list 的配置

    .. deprecated::
        此函数已废弃（2026-02）。
        资源配置逻辑已简化并移到 setup_benchmark 外层。
        保留此函数仅为向后兼容，不应再调用。

    Args:
        args: 命令行参数
        path_params: 路径参数元组
        path_ctx: PathContext 实例
        need_repack: 是否需要重新装箱
        bin_list: Bin 列表

    Returns:
        tuple: (num_cores, bin_list) 或 None（如果配置失败）
    """
    import warnings
    warnings.warn(
        "determine_resource_config() is deprecated and will be removed in a future version. "
        "Resource configuration logic has been moved to setup_benchmark().",
        DeprecationWarning,
        stacklevel=2
    )
    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root, case_pth = path_params
    
    # for repack: load packing relation ship from cache file
    # for dynamic statge of two stage case: read both the num_cores and bin_list from the cache file
    # for ablation study: force the num_cores, and change the size of bins in bin_list
    # procedure: get_bin, force_bin, force core
    # if args.test_case in two_stage_case_coll:
    #     prepared = prepare_induced_env_if_needed(
    #         path_para_dict, path_ctx,
    #         trace_path_para, plot_path_para,
    #     )
    #     if prepared is None:
    #         raise ValueError("Failed to load bin_list from cache file")
    #     matched_num_cores, bin_list = prepared
    #     if args.num_cores is not None:
    #         num_cores = apply_forced_num_cores(bin_list, matched_num_cores, args.num_cores)
    #     else:
    #         num_cores = matched_num_cores
    #     if args.e2e_var_sim_en:
    #         # check max number of bins
    #         num_bins = check_max_bin_num(args, args.num_bins, bin_path_format, path_ctx)
    #         # suitable for the case with variable number of bins
    #         bin_list = extend_dummy_bins(bin_list, num_bins)
    # elif need_repack:
    #     prepared = prepare_induced_env_if_needed(
    #         path_para_dict, path_ctx,
    #         trace_path_para, plot_path_para,
    #     )
    #     if prepared is None:
    #         raise ValueError("Failed to load bin_list from cache file")
    #     matched_num_cores, bin_list = prepared
    #     num_cores = args.num_cores
    # else:
    #     num_cores = args.num_cores
    #     bin_list = []

    # procedure: get_bin, force core

    # if specify the num_cores, use the forced num_cores
    # for repack: load packing relation ship from cache file
    # for dynamic stage of two stage case: read both the num_cores and bin_list from the cache file

    if args.test_case in two_stage_case_coll or need_repack:
        if args.num_cores is not None:
            num_cores = apply_forced_num_cores(bin_list, args.num_cores, args.num_cores)
        else:
            num_cores = sum(_bin.num_resources for _bin in bin_list)
    else:
        num_cores = args.num_cores
        bin_list = []

    return num_cores, bin_list

def create_scheduler_elements_with_config(args, path_params, path_ctx, workload, num_cores, bin_list):
    """
    使用给定的资源配置创建调度器元素
    
    Args:
        args: 命令行参数
        path_params: 路径参数元组
        path_ctx: PathContext 实例
        workload: 工作负载元组
        num_cores: 核心数
        bin_list: bin列表
        
    Returns:
        tuple: (task_spec, rsc_list, msg_dispatcher, a_data_pipe, w_data_pipe, 
                scheduler_list, monitor_list, trace_path, sim_step)
    """
    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root, case_pth = path_params
    
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, _ = workload

    scheduler_args = {
        "exec_t_comp_ratioB": args.exec_t_comp_ratioB,
        "barrier_en": not args.barrier_dis, 
        "forbid_miss": args.forbid_miss,
        "progress_aware": True if args.test_case in two_stage_case_coll else False,
        "allow_realloc": args.allow_realloc,
    }
    
    # 计算仿真步长
    sim_step = elim_nume_error(1e-6 * args.timestepxus)
    
    task_spec, rsc_list, msg_dispatcher, \
        a_data_pipe, w_data_pipe, scheduler_list, \
            monitor_list, trace_path = create_common_scheduler_elements(
                args, trace_path_para, case_pth, 
                hyper_p, glb_p_list, scheduler_args, 
                sim_step, 
                bin_list if bin_list else [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")], 
                path_ctx
                )


    return (task_spec, rsc_list, msg_dispatcher, a_data_pipe, w_data_pipe, 
            scheduler_list, monitor_list, trace_path, sim_step)


def build_simulation_env(args, workload, sim_step):
    """
    构建仿真环境参数，对应 main 中仿真参数设置部分
    返回: (num_periods, warmup, quantumSize, event_range, event_iter_dict)
    """
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, _ = workload
    
    num_periods = args.n_p
    warmup = not args.warmup_dis
    quantumSize = sim_step * args.quantumSize
    event_range = hyper_p * (num_periods + warmup)
    np.random.seed(args.seed)
    
    # simulation of driving dynamics
    # get event generators: arrival time, deadline, load
    jitter_para_dict = dict(jitter_sim_en=args.jitter_sim_en, jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    event_iter_dict = TaskInt.get_event_generator(glb_p_list, hyper_p, num_periods, warmup, **jitter_para_dict)

    return num_periods, warmup, quantumSize, event_range, event_iter_dict

def perform_bin_packing(args, glb_p_list, num_cores, bin_list, hyper_p,
                       sim_step, path_para_dict, para_scan_group1,
                       event_iter_dict, quantumSize, num_periods,
                       cfg_para_dict, physical_graph_nx, need_repack,
                       plot_path_para, path_ctx: PathContext,
                       scheduler_list, monitor_list,
                       msg_dispatcher,
                       a_data_pipe, w_data_pipe
                       ):
    """
    执行 bin-packing 算法，返回装箱结果。

    内部处理两种情况：
    1. Phase 1 (need_repack=False): 使用 coleasing_alloc_cluster 进行空间分区+装箱
    2. Repack (need_repack=True): 使用 push_task_into_bins_new + pre_defined 模式
       - 内部创建备份，失败时自动 fallback 到 Phase 1 布局
       - 适用于 cyc-S (num_bins=-1) 和 reserv (num_bins>=2)

    Returns:
        tuple: (bin_list, max_core_num, glb_p_list, hyper_p, repack_success)
            - bin_list: 装箱后的 bin 列表（可能是 fallback）
            - max_core_num: 装箱计算的资源需求
            - glb_p_list: 进程列表
            - hyper_p: 超参数
            - repack_success: repack 是否成功（Phase 1 为 True）
    """
    # --- 入口参数检查 ---
    if not hasattr(args, "binpack_cfg") or args.binpack_cfg is None:
        raise ValueError("args.binpack_cfg is not initialized")
    if "algorithm" not in args.binpack_cfg:
        raise KeyError("args.binpack_cfg missing required key: 'algorithm'")
    # ------------------

    # Prepare binpack_cfg with var_dist_map and quantile
    def prepare_binpack_cfg(cfg, quantile, p_list, graph):
        from sched.binpack_config import BinPackConfig
        new_cfg_dict = dict(cfg)
        new_cfg_dict['var_dist_map'] = {p.task.name: graph.nodes[p.task.name]['var_dist'] for p in p_list}
        new_cfg_dict['quantile'] = quantile
        return BinPackConfig(new_cfg_dict)

    for _p in glb_p_list:
        _p.task.criticality = "hard"
        _p.task.chain_criticality = "hard"
    
    print("sim_step: ", sim_step)
    # 保持同一 list 对象，避免与 scheduler_list 等引用脱节
    if args.binpack_cfg["algorithm"] == "scratch":
        binpack_cfg_scratch = prepare_binpack_cfg(args.binpack_cfg, args.exec_t_comp_ratioB, glb_p_list, physical_graph_nx)
        bin_list = push_task_into_bins_new(
            bin_list,
            glb_p_list, affinity_cfg, event_iter_dict,
            num_cores, args.quantum_check_en, quantumSize,
            sim_step, hyper_p, args.exec_t_comp_ratioB,

            scheduler_list, monitor_list,
            msg_dispatcher,
            a_data_pipe, w_data_pipe,

            num_periods, binpack_cfg=binpack_cfg_scratch,
            verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
            warmup=True, drain=True,
            )
        max_core_num = sum(b.num_resources for b in bin_list)
        repack_success = True  # scratch algorithm always succeeds (no repack mode)
    
    elif args.binpack_cfg["algorithm"] == "guided":
        from sched.global_sched import coleasing_alloc_cluster
        from sched.pre_alloc_new import ResourceInsufficientError

        if not need_repack:
            # ========== Phase 1: full bin packing ==========
            split_ratio = args.exec_t_comp_ratioA
            binpack_cfg_guided = prepare_binpack_cfg(args.binpack_cfg, split_ratio, glb_p_list, physical_graph_nx)
            print("="* 20 + "Bin-split mode: Cluster-based allocation" + "="* 20 + "\n")
            max_core_num, pid2_bin_id, bin_size_list = coleasing_alloc_cluster(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                None, args.quantum_check_en, quantumSize,
                sim_step, hyper_p, split_ratio,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=binpack_cfg_guided,
                job_graph=physical_graph_nx,
                n_partition = args.num_bins if args.num_bins != -1 else 9999,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True,
                )
            repack_success = True  # Phase 1 always succeeds

        else:
            # ========== Repack: Fixcore mode (bypass greedy bin-packing) ==========
            if USE_FIXCORE_REPACK:
                # Fixcore 模式：ERT/DDL 已在 deduce_cfg2 中更新
                # bin_list 保持 Phase 1 布局不变，不调用贪心装箱
                max_core_num = sum(b.num_resources for b in bin_list)
                repack_success = True
            else:
                # 原始模式：使用贪心装箱
                # 内部创建备份
                import copy
                bin_list_backup = copy.deepcopy(bin_list)
                phase1_pids = set()
                for _b in bin_list_backup:
                    phase1_pids.update(_b.index_occupy_by_id().keys())

                strategy_name = "cyc-S" if args.num_bins == -1 else "reserv"

                # === DIAGNOSTIC: Repack execution tracking ===
                import sys
                sys.stderr.write(f"\n{'='*60}\n")
                sys.stderr.write(f"[REPACK DIAGNOSTIC] Starting repack for {strategy_name}\n")
                sys.stderr.write(f"  - ratioA={args.exec_t_comp_ratioA}, ratioB={args.exec_t_comp_ratioB}\n")
                sys.stderr.write(f"  - num_bins={args.num_bins}, num_cores={num_cores}\n")
                sys.stderr.write(f"  - Phase 1 tasks: {len(phase1_pids)}\n")
                sys.stderr.write(f"  - glb_p_list size: {len(glb_p_list)}\n")
                sys.stderr.write(f"{'='*60}\n\n")
                sys.stderr.flush()

                try:
                    # 获取 Phase 1 的 bin 分配
                    pid2_bin_id = extract_pid2_bin_id(bin_list)
                    max_core_num = sum(b.num_resources for b in bin_list)

                    print("=" * 20 + f" Repack mode ({strategy_name}): Redistribute slack (ratioB={args.exec_t_comp_ratioB})" + "=" * 20 + "\n")

                    # 构造局部 binpack 配置，使用 pre_defined 模式
                    binpack_cfg_local = prepare_binpack_cfg(args.binpack_cfg, args.exec_t_comp_ratioB, glb_p_list, physical_graph_nx)
                    binpack_cfg_local["mapping"] = pid2_bin_id
                    binpack_cfg_local["bin_sel_mod"] = "pre_defined"
                    binpack_cfg_local["affinity_en"] = False
                    binpack_cfg_local["affinity_level"] = 0

                    # 尝试 repack
                    bin_list = push_task_into_bins_new(
                        bin_list,
                        glb_p_list, affinity_cfg, event_iter_dict,
                        num_cores, args.quantum_check_en, quantumSize,
                        sim_step, hyper_p, args.exec_t_comp_ratioB,

                        scheduler_list, monitor_list,
                        msg_dispatcher,
                        a_data_pipe, w_data_pipe,

                        num_periods, binpack_cfg=binpack_cfg_local,
                        verbose=True, DEBUG_FG=False,
                        warmup=True, drain=True,
                        )

                    # 完整性检查：验证所有 Phase 1 任务仍然被放置
                    repack_pids = set()
                    for _b in bin_list:
                        repack_pids.update(_b.index_occupy_by_id().keys())
                    missing = phase1_pids - repack_pids
                    if missing:
                        # === DIAGNOSTIC: Detailed missing task info ===
                        print(f"\n[REPACK DIAGNOSTIC] Incomplete placement detected:")
                        print(f"  - Phase 1 tasks: {len(phase1_pids)}")
                        print(f"  - Repack placed: {len(repack_pids)}")
                        print(f"  - Missing tasks: {len(missing)}")
                        print(f"  - Missing PIDs (first 10): {sorted(list(missing))[:10]}")
                        raise RuntimeError(
                            f"Repack incomplete: {len(missing)} tasks not placed "
                            f"(missing PIDs: {sorted(list(missing))[:5]}...)")

                    # === DIAGNOSTIC: Success summary ===
                    print(f"\n{'='*60}")
                    print(f"[REPACK DIAGNOSTIC] SUCCESS")
                    print(f"  - Strategy: {strategy_name}")
                    print(f"  - ratioB: {args.exec_t_comp_ratioB}")
                    print(f"  - Tasks placed: {len(repack_pids)}")
                    print(f"{'='*60}\n")
                    repack_success = True

                except (ResourceInsufficientError, RuntimeError) as e:
                    # Fallback: 恢复 Phase 1 布局
                    bin_list = bin_list_backup
                    max_core_num = sum(b.num_resources for b in bin_list)
                    # === DIAGNOSTIC: Failure summary ===
                    print(f"\n{'='*60}")
                    print(f"[REPACK DIAGNOSTIC] FAILED - FALLBACK TO PHASE 1")
                    print(f"  - Strategy: {strategy_name}")
                    print(f"  - Error: {e}")
                    print(f"  - Restored Phase 1 layout with {len(phase1_pids)} tasks")
                    print(f"{'='*60}\n")
                    repack_success = False

    else:
        raise NotImplementedError(f"binpack algorithm {args.binpack_cfg['algorithm']} is not implemented")

    # 返回装箱结果（不包含资源约束和 dump）
    return bin_list, max_core_num, glb_p_list, hyper_p, repack_success

def init_sched_components(args, path_params, path_ctx, workload, num_cores, bin_list):
    """初始化 global_sched 系列方法的公用组件 + 仿真环境，返回执行句柄。

    封装 step 4-5：创建统一接口所需的公用组件（Scheduler/Monitor/msg_dispatcher/
    a_data_pipe/w_data_pipe）+ 仿真环境，绑定到闭包。调用句柄即执行 perform_bin_packing。

    句柄属性：
        .sim_step / .num_periods — 供 step 9 打印/绘制使用
    句柄调用签名：
        pack(glb_p_list, bin_list, hyper_p, physical_graph_nx, need_repack)
            -> perform_bin_packing 结果 (bin_list, max_core_num, glb_p_list, hyper_p, repack_success)
    """
    # step 4: 调度器元素（global_sched 统一接口的公用组件）
    sched_elements = create_scheduler_elements_with_config(
        args, path_params, path_ctx, workload, num_cores, bin_list
    )
    _, _, msg_dispatcher, a_data_pipe, w_data_pipe, \
        scheduler_list, monitor_list, _, sim_step = sched_elements
    # step 5: 仿真环境
    num_periods, _, quantumSize, _, event_iter_dict = build_simulation_env(args, workload, sim_step)
    # path_params 解包（绑定闭包）
    cfg_para_dict, para_scan_group1, _, path_para_dict, _, _, plot_path_para, _, _ = path_params

    def pack(glb_p_list, bin_list, hyper_p, physical_graph_nx, need_repack):
        return perform_bin_packing(
            args, glb_p_list, num_cores, bin_list, hyper_p,
            sim_step, path_para_dict, para_scan_group1,
            event_iter_dict, quantumSize, num_periods,
            cfg_para_dict, physical_graph_nx, need_repack,
            plot_path_para, path_ctx,
            scheduler_list, monitor_list, msg_dispatcher, a_data_pipe, w_data_pipe,
        )
    pack.sim_step = sim_step
    pack.num_periods = num_periods
    return pack


def preprocess_args(args):
    """
    预处理 args，包括强制后缀、enforce_wc、参数断言和特殊 case 检查。
    """
    if args.num_cores is not None:
        args.force_num_cores = True
    else:
        args.force_num_cores = False
    
    if args.force_num_cores and args.aux_scale_factor != 9:
        args.force_suffix = "force_"
    else:
        args.force_suffix = ""

    enforce_wc = args.seed == -1 and args.jitter_sim_en
    args.jitter_sim_para.update({"enforce_wc": enforce_wc})
    args.exec_var_para.update({"enforce_wc": enforce_wc})
    assert args.aux_scale_factor <= 9, "aux_scale_factor should be less than or equal to 9"

    if args.test_case in one_stage_case_coll:
        is_induced = args.binpack_cfg.pop("core_size", None)
        if is_induced == "induced":
            warnings.warn("Incorrect config: One-stage case does not support induced core size; removed automatically")
    elif args.test_case in two_stage_case_coll:
        args.binpack_cfg["core_size"] = "induced"
    else:
        # bin_pack
        if args.binpack_cfg["algorithm"] == "repack":
            args.binpack_cfg["core_size"] = "induced"
        else:
            args.binpack_cfg["core_size"] = "specified"

def create_common_scheduler_elements(args, trace_path_para, case_pth, hyper_p, glb_p_list, scheduler_args, sim_step, bin_list, path_ctx: PathContext):
    # ======================== path settings ================
    trace_path = get_trace_path(args, trace_path_para, case_pth, path_ctx)
    scheduler_args.update({"trace_path": trace_path})
    task_spec = Spec(0.1, [1 for _ in glb_p_list]) 
    # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
    exec_para_dict = dict(exec_var_en=args.exec_var_en, exec_var_para=args.exec_var_para, seed=args.seed)
    rsc_list = [Resource_model_int(size=sched_tab.num_resources, **exec_para_dict) for sched_tab in bin_list]
    # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
    # msg_pipe = Message()
    msg_dispatcher = MsgDispatcher(len(bin_list))
    a_data_pipe = DataPipe("activation", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    w_data_pipe = DataPipe("weight", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    scheduler_list = [Scheduler(bin_list[idx], args.e2e_latency, hyper_p, glb_p_list, 
                                        res_cfg=rsc_list[idx], **scheduler_args) for idx in range(len(bin_list))]
    monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]
    return task_spec,rsc_list,msg_dispatcher,a_data_pipe,w_data_pipe,scheduler_list,monitor_list, trace_path

def get_trace_path(args, trace_path_para, case_pth, path_ctx: PathContext):
    # 使用旧方法生成路径
    if args.jitter_sim_en:
        old_trace_path = trace_fn_w_seed_fmt.format(**trace_path_para, **{"case": case_pth})
    else:
        old_trace_path = trace_fn_wo_seed_fmt.format(**trace_path_para, **{"case": case_pth})
    
    path_ctx.case = case_pth
    new_trace_path = path_ctx.get_trace_path()
    
    # 比较路径
    compare_paths(old_trace_path, new_trace_path, f"trace_path ({case_pth})")
    return new_trace_path


