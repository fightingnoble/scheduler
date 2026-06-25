"""sim_main_unused.py — moved dead code from sim_main.py (B2-MOVE-002).
Contents: others() (infrequent cases split from main) + main() (superseded entry,
replaced by main_approach.py) + if __name__ block.
Moved, not rewritten. Recover: git checkout archive/test_pipeline-20260612 -- sim_main.py
"""

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
from allocator_agent import glb_sched, cyclic_sched
from model.event_gen.e2e_latency import discrete_event_sim
from model.task_queue_agent import TaskQueue
from utils import dump_and_check, load_pickle, update_df, check_parents_path, build_path_old, get_case_path_str
from global_var import *
from utils import core_distr, time_cnt, vectorized_core_allocation
from paths import PathContext
import numpy as np
import argparse

def others(args, glb_p_list, num_cores, bin_list, hyper_p,
                       sim_step, path_para_dict, para_scan_group1,
                       event_iter_dict, quantumSize, num_periods,
                       cfg_para_dict, physical_graph_nx, need_repack,
                       plot_path_para, path_ctx: PathContext, 
                       scheduler_list, monitor_list,
                       msg_dispatcher,
                       a_data_pipe, w_data_pipe
                       ):
    """
    执行 bin-packing 算法，生成调度表并保存
    """
    for _p in glb_p_list:
        _p.task.criticality = "hard"
        _p.task.chain_criticality = "hard"
    
    print("sim_step: ", sim_step)
    # 保持同一 list 对象，避免与 scheduler_list 等引用脱节
      
    if args.binpack_cfg["algorithm"] == "mem_plan": 
        from sched.global_sched import test_mem_planner
        bin_list = test_mem_planner(
            bin_list,
            glb_p_list, affinity_cfg, event_iter_dict,
            num_cores, args.quantum_check_en, quantumSize, 
            sim_step, hyper_p, args.exec_t_comp_ratioB,

            scheduler_list, monitor_list,
            msg_dispatcher,
            a_data_pipe, w_data_pipe,

            num_periods, binpack_cfg=args.binpack_cfg,
            verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
            warmup=True, drain=True, 
            )
        return None, num_cores, glb_p_list, hyper_p

    elif args.binpack_cfg["algorithm"] == "full":
        from sched.global_sched import single_turn_solver
        from task.task_cfg import task_graph_srcs, task_graph_sinks
        pid2_bin_id, bin_size_list = single_turn_solver(
            bin_list,
            glb_p_list, affinity_cfg, event_iter_dict,
            num_cores, args.quantum_check_en, quantumSize, 
            sim_step, hyper_p, args.exec_t_comp_ratioB,

            scheduler_list, monitor_list,
            msg_dispatcher,
            a_data_pipe, w_data_pipe,

            num_periods, binpack_cfg=args.binpack_cfg,
            job_graph=physical_graph_nx, src_nodes=task_graph_srcs, end_nodes=task_graph_sinks, 
            n_partition = args.num_bins if args.num_bins != -1 else 9999,
            verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
            warmup=True, drain=True, 
            )

    else:
        raise NotImplementedError(f"binpack algorithm {args.binpack_cfg['algorithm']} is not implemented")

    # ensure_csv(csv_path_and_fn, cfg_para_dict, para_scan_group1)

    if need_repack:
        extra_suffix = f"_ov_{args.exec_t_comp_ratioB:.2f}_repack(T)"
    else:
        extra_suffix = ""
    bin_list_save_path, routing_table_save_path = generate_bin_paths(
        path_para_dict, path_ctx, num_cores, "packing save path", 
        extra_suffix
    )
    # 同步更新 trace_path_para 中的 num_cores，确保路径一致性
    trace_path_para['num_cores'] = num_cores
    Bin_list_print(bin_list, glb_p_list, sim_step)
    if args.plot:
        render_bin_pack_plots(args, bin_list, glb_p_list, sim_step, hyper_p, num_periods, plot_path_para, path_ctx)
    

    # select a period to save 
    assert num_periods >= 1
    bin_list2save = []
    # for _sched_tab in bin_list:
    dump_and_check(bin_list_save_path, bin_list)
    # dump_and_check(routing_table_save_path, scheduler_list[0].detail_alloc_info)
    return bin_list_save_path, num_cores, glb_p_list, hyper_p



@time_cnt("main")
# @pyinstr_profiler("main")
def main(args: argparse.Namespace):
    # ======================== porcess arguments ========================
    preprocess_args(args)
    
    # ======================== build_paths ========================
    path_params, path_ctx = build_paths_and_ctx(args)
    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root, case_pth = path_params
    csv_path_and_fn = os.path.join(csv_xlxs_root, 'coalescing_req_cores.csv')
    check_parents_path(csv_path_and_fn)

    # ======================== workload settings ========================
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, need_repack = build_workload_and_criticality(args)
    workload = (hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, need_repack)
    export_json_graph_utils(physical_graph_nx, path_ctx.graph_fn)

    # ======================== build simulation ================

    num_cores, bin_list = determine_resource_config(args, path_params, path_ctx, need_repack)
    # 使用资源配置创建调度器元素
    scheduler_elements = create_scheduler_elements_with_config(
        args, path_params, path_ctx, workload, num_cores, bin_list
    )
    task_spec, rsc_list, msg_dispatcher, a_data_pipe, w_data_pipe, \
        scheduler_list, monitor_list, trace_path, sim_step = scheduler_elements

    num_periods, warmup, quantumSize, event_range, event_iter_dict = build_simulation_env(
        args, workload, sim_step
    )

    # ======================== select test case ========================
    # compile time reservation
    if args.test_case in [case_name_bp_input,] :
        bin_list_save_path, num_cores, glb_p_list, hyper_p, _ = perform_bin_packing(
            args, glb_p_list, num_cores, bin_list, hyper_p,
             sim_step, path_para_dict, para_scan_group1,
            event_iter_dict, quantumSize, num_periods,
            cfg_para_dict, physical_graph_nx, need_repack,
            plot_path_para, path_ctx,
            scheduler_list, monitor_list,
            msg_dispatcher,
            a_data_pipe, w_data_pipe,
            )
        if bin_list_save_path is None:
            return

    # runtime scheduling
    elif args.test_case in two_stage_case_coll + one_stage_case_coll:

        # seen in the item 21. load_var and thread fork in assumption.md
        if args.load_var_sim_en: 
            for var_item, var_param in args.load_var_sim_para.items():
                dyn_obj_iter = discrete_event_sim(np.arange(var_param["maxsize"], dtype=int), 1, var_param["period"], event_range, args.seed)
                dyn_obj_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
                var_param["stream"] = dyn_obj_stream
                var_param["iter"] = dyn_obj_iter

        if args.e2e_var_sim_en:
            ddl_update_iter = discrete_event_sim(args.e2e_var_sim_para['event_list'], 1, args.e2e_var_sim_para["period"], event_range, args.seed)
            ddl_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
        else:
            ddl_update_iter = None
            ddl_stream = None


        if args.test_case in two_stage_case_coll:
            sensor_pipe = TriggerPipe(len(bin_list))
            cores = [bin.num_resources for bin in bin_list]
            core_map = core_mapping_1d(cores)
            for _sched in scheduler_list:
                _sched.core_map = core_map[_sched._SchedTab.id]

            print("sim_step: ", sim_step)
            cyclic_sched(task_spec, affinity_cfg, 
                    scheduler_list, monitor_list,
                    event_iter_dict,
                    ddl_update_iter, ddl_stream,
                    args.load_var_sim_para if args.load_var_sim_en else None,
                    rsc_list, 
                    num_cores, 
                    glb_p_list,
                    sim_step, hyper_p, num_periods, 
                    msg_dispatcher,
                    sensor_pipe,
                    a_data_pipe, w_data_pipe, 
                    path_ctx,
                    args.verbose, warmup=True, drain=True, 
                    case=args.test_case,)

            tot_cores = 0
            n_switch = 0
            weighted_avg_cumulative_time = 0
            for _sched in scheduler_list:
                partition_id = _sched._SchedTab.id
                bin_size = _sched._SchedTab.num_resources
                tot_cores += bin_size
                print(f"(Partition {partition_id}) number of context switch {_sched.barrier.number_of_asserts}")
                print(f"(Partition {partition_id}) cumulative context switch {elim_nume_error(_sched.barrier.cumulative_time)}")
                weighted_avg_cumulative_time += _sched.barrier.cumulative_time * num_cores
                n_switch += _sched.barrier.number_of_asserts
            print(f"number of context switch {n_switch}")
            weighted_avg_cumulative_time /= tot_cores
            print(f"cumulative context switch {elim_nume_error(weighted_avg_cumulative_time)}")


        elif args.test_case in [case_name_glb_input,]:

            assert "core_size" not in args.binpack_cfg or args.binpack_cfg["core_size"] != "induced"

            print("sim_step: ", sim_step)
            glb_sched(task_spec, affinity_cfg, 
                    scheduler_list, monitor_list,
                    event_iter_dict,
                    ddl_update_iter, ddl_stream,
                    args.load_var_sim_para if args.load_var_sim_en else None,
                    rsc_list, 
                    num_cores, 
                    glb_p_list,
                    sim_step, hyper_p, num_periods, 
                    msg_dispatcher,
                    a_data_pipe, w_data_pipe,
                    args.quantum_check_en, quantumSize, 
                    args.verbose, warmup=True, drain=True, 
                    lateness_mode=args.lateness_mode)
            
            print("number of context switch {}".format(scheduler_list[0].barrier.number_of_asserts))
            print("cumulative context switch {}".format(scheduler_list[0].barrier.cumulative_time))

        # print&save for runtime scheduling result
        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        for _SchedTab in actual_sched_record:
            _SchedTab.print_alloc_detail(pid2name, sim_step)

        if args.plot:
            render_runtime_full_plot(args, actual_sched_record, glb_p_list, sim_step, hyper_p, num_periods, case_pth, plot_path_para, path_ctx)

        # save trace_list to trace_file
        # save_chunk(trace_path.replace(".pkl", ".h5"), trace_list, True)
        dump_and_check(trace_path, trace_list)


if __name__ == "__main__":
    from utils import input_parser
    args = input_parser() 
    print(args)
    main(args)

