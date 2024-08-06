import os, re
from task.task_cfg import create_init_p_list, gen_workloads
from task.task_cfg import affinity_cfg
from task.task_cfg import init_affinity
from sched.global_sched import push_task_into_bins_new, coleasing_alloc_1bin, naive_iso
from task.task_agent import TaskInt
from task.spec import Spec
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt, load_bin_list
from sched.scheduling_table import get_task_layout_compact, get_task_layout_sparse, get_task_layout_compact1bin, Bin_list_print
from model.resource_agent import Resource_model_int
from sched.scheduler_agent import Scheduler
from sched.placement import core_mapping_1d
from sched.monitor_agent import Monitor
from allocator_agent import glb_sched, cyclic_sched
from model.event_gen.e2e_latency import discrete_event_sim
from model.task_queue_agent import TaskQueue
from utils import dump_and_check, load_pickle, update_df, check_parents_path, args_postprocess, get_case_path_str
from global_var import *
from utils import core_distr, save_chunk, load_h5_file, time_cnt, pyinstr_profiler

@time_cnt("main")
# @pyinstr_profiler("main")
def main():
    import numpy as np 
    from utils import input_parser
    args = input_parser() 
    print(args)

    # ======================== porcess arguments ========================
    num_cores = args.num_cores
    assert args.aux_scale_factor <= 9, "aux_scale_factor should be less than or equal to 9"

    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root = args_postprocess(args)
    case_pth = get_case_path_str(args)

    # ======================== workload settings ========================
    hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args)

    # generate the process list
    glb_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
    init_affinity(glb_p_list, mode='job', job_graph_nx=physical_graph_nx, verbose=args.verbose)
    
    # ======================== scheduler settings ================
    scheduler_args = {
        "exec_t_comp_ratioB": args.exec_t_comp_ratioB,
        "barrier_en": not args.barrier_dis, 
        "forbid_miss": args.forbid_miss,
        "progress_aware": args.progress_aware,
        "allow_realloc": args.allow_realloc,
    }

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

    # ======================== simlation settings ========================
    num_periods = args.n_p
    warmup = not args.warmup_dis
    sim_step = elim_nume_error(1e-6 * args.timestepxus) # min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32
    quantumSize = sim_step*args.quantumSize
    event_range = hyper_p * (num_periods+warmup)
    np.random.seed(args.seed)
    
    # simulation of driving dynamics
    # get event generators: arrival time, deadline, load
    jitter_para_dict = dict(jitter_sim_en=args.jitter_sim_en, jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    event_iter_dict = TaskInt.get_event_generator(glb_n_task_dict, hyper_p, num_periods, warmup, **jitter_para_dict)

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
        # check max number of bins
        num_bins = check_max_bin_num(args, args.num_bins, bin_path_format)
    else:
        ddl_update_iter = None
        ddl_stream = None
        num_bins = args.num_bins

    # ======================== select test case ========================
    if args.test_case == "all":
        args.test_all = True

    # compile time reservation 
    if args.test_all or args.test_case in [case_name_bp_input,] :
        for _p in glb_p_list:
            _p.task.criticality = "hard"
        
        bin_list = [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")]        
        task_spec, rsc_list, msg_dispatcher, \
            a_data_pipe, w_data_pipe, scheduler_list, \
                monitor_list, trace_path = create_common_scheduler_elements(
                    args, trace_path_para, case_pth, 
                    hyper_p, glb_p_list, scheduler_args, 
                    sim_step, bin_list
                    )        

        print("sim_step: ", sim_step)
        bin_list.clear()
        if args.binpack_cfg["algorithm"] == "reside":
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **para_scan_group2)
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **para_scan_group2)
            bin_list = push_task_into_bins_new(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=args.binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )
        elif args.binpack_cfg["algorithm"] == "mem_plan": 
            from sched.global_sched import test_mem_planner
            bin_list = test_mem_planner(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=args.binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )
            return
        elif args.binpack_cfg["algorithm"] == "coalescing":
            max_core_layout = coleasing_alloc_1bin(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=args.binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )
            filename = os.path.join(csv_xlxs_root, 'coalescing_req_cores.csv')
            check_parents_path(filename)
            import pandas as pd
            if not os.path.exists(filename):
                pd.DataFrame(columns=list(cfg_para_dict.keys())+list(para_scan_group1.keys())+["num_cores"]).to_csv(filename, index=False)
            
            num_cores = sum(max_core_layout[1].values())
            if args.force_num_cores and args.aux_scale_factor!= 9:
                # read core number from the bin name with TP=9
                _cfg_n_t = cfg_root_fmt.format(**cfg_para_dict, **{**para_scan_group1, "aux_scale_factor": 9})
                _path_para_dict = {"root_dir": args.root_dir, "cfg_n": _cfg_n_t, 
                                   "i_file_suffix": args.i_file_suffix,
                                   "force_suffix": ""}
                folder, files, match = get_core_num_from_trace_name(_path_para_dict)
                if not match:
                    print(f"!!! Warning: no bin_list file "+
                        bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"})
                        +f" in {folder}:{files} !!!")
                    return
                args.num_cores = int(match.group(1)) 
                
                if args.num_cores > num_cores:
                    # just modify the size of the bin
                    bin_list[0].num_resources = args.num_cores
                    print(f"Force the num of Core {num_cores} -> {args.num_cores}, the over subcription ratio is {over_sub_ratio}")
                    num_cores = args.num_cores
                elif args.num_cores < num_cores: 
                    print(f"Forced specified num of Core should be larger than the estimated num of cores {num_cores} > {args.num_cores}")
                    import sys; sys.exit(1)
            plot_path_para.update({"num_cores": num_cores})
            # Load the dataframe                        
            # df = pd.read_csv(filename)
            # df = update_df(df, {**cfg_para_dict, **para_scan_group1}, 
            #                {"num_cores": num_cores})
            # df.to_csv(filename, index=False)

            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
        
        elif args.binpack_cfg["algorithm"] == "bin_split":
            from sched.global_sched import coleasing_alloc_cluster
            from task.task_cfg import task_graph_srcs, task_graph_sinks
            pid2_bin_id, bin_size_list = coleasing_alloc_cluster(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=args.binpack_cfg,
                job_graph=physical_graph_nx, src_nodes=task_graph_srcs, end_nodes=task_graph_sinks, 
                n_partition = args.num_bins if args.num_bins != -1 else 9999,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )
            filename = os.path.join(csv_xlxs_root, 'coalescing_req_cores.csv')
            check_parents_path(filename)
            import pandas as pd
            # if not os.path.exists(filename):
            #     pd.DataFrame(columns=list(cfg_para_dict.keys())+list(para_scan_group1.keys())+["num_bins","num_cores"]).to_csv(filename, index=False)
            
            num_cores = sum(bin_size_list.values())
            if args.force_num_cores and args.aux_scale_factor!= 9:
                # read core number from the bin name with TP=9
                _cfg_n_t = cfg_root_fmt.format(**cfg_para_dict, **{**para_scan_group1, "aux_scale_factor": 9})
                _path_para_dict = {"root_dir": args.root_dir, "cfg_n": _cfg_n_t, 
                                   "i_file_suffix": args.i_file_suffix, 
                                   "force_suffix": ""}
                folder, files, match = get_core_num_from_trace_name(_path_para_dict)
                if not match:
                    print(f"!!! Warning: no bin_list file "+
                        bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"})
                        +f" in {folder}:{files} !!!")
                    return
                args.num_cores = int(match.group(1)) 

                if args.num_cores > num_cores:
                    # culculate the over subcription ratio
                    over_sub_ratio =  args.num_cores / num_cores
                    # score is thier size in the bin
                    score_dict = {_bin.id:_bin.num_resources for _bin in bin_list} 
                    rsc_map = {**score_dict}
                    curr_aval_rsc = args.num_cores - num_cores
                    core_distr(rsc_map, score_dict, curr_aval_rsc)
                    assert sum(rsc_map.values()) == args.num_cores
                    # set the size of the bin
                    for _bin in bin_list:
                        _bin.num_resources = rsc_map[_bin.id]
                    print(f"Force the num of Core {num_cores} -> {args.num_cores}, the over subcription ratio is {over_sub_ratio}")
                    num_cores = args.num_cores
                elif args.num_cores < num_cores: 
                    print(f"Forced specified num of Core should be larger than the estimated num of cores {num_cores} > {args.num_cores}")
                    import sys; sys.exit(1)

            plot_path_para.update({"num_cores": num_cores})
            # Load the dataframe                        
            # df = pd.read_csv(filename)
            # df = update_df(df, {**cfg_para_dict, **para_scan_group1, "num_bins": args.num_bins}, 
            #                {"num_cores": num_cores})
            # df.to_csv(filename, index=False)

            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})

        elif args.binpack_cfg["algorithm"] == "repack":
            folder, files, match = get_core_num_from_trace_name(path_para_dict)
            if not match:
                print(f"!!! Warning: no bin_list file "+
                      bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"})
                      +f" in {folder}:{files} !!!")
                return
            num_cores = int(match.group(1))
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            bin_list = load_pickle(bin_list_save_path)

            # ======================== artifact ======================== 
            
            # assert args.binpack_cfg["algorithm"] == "bin_split"
            assert args.binpack_cfg["slack_sharing"] == False
            # 1. cheat the non-sharing model: 
            #    keep the original exec_t_comp_ratioA in slack distribution and resource estimation
            #    to make sure the repacking step use the same Bin configuration as the original one.
            # 2. backup parameters
            exec_t_comp_ratioA_bk = args.exec_t_comp_ratioA
            # 3. change the exec_t_comp_ratioA to args.exec_t_comp_ratioB in th repacking step
            args.exec_t_comp_ratioA = args.exec_t_comp_ratioB 
            args.binpack_cfg["slack_sharing"] = True
            # reset the ddl and ert
            print("="* 20 + "Redistribute slack:" + "="* 20 + "\n")
            hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args)

            # generate the process list
            glb_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
            init_affinity(glb_p_list, mode='job', job_graph_nx=physical_graph_nx, verbose=args.verbose)

            # get the size and allocated process id
            print("="* 20 + "Bin-assignment:" + "="* 20 + "\n")
            pid2_bin_id = {}
            for _bin in bin_list:
                _bin:SchedulingTableInt
                print(f"{_bin.name}({_bin.id})", list(_bin.index_occupy_by_id().keys()))
                pid_list = _bin.index_occupy_by_id().keys()
                for pid in pid_list:
                    assert pid not in pid2_bin_id
                    pid2_bin_id[pid] = _bin.id
                _bin.clear()

            args.binpack_cfg["mapping"] = pid2_bin_id
            args.binpack_cfg["bin_sel_mod"] = "pre_defined"
            # clear the placement of each bin
            args.binpack_cfg["affinity_en"] = False
            args.binpack_cfg["affinity_level"] = 0

            bin_list = push_task_into_bins_new(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioB,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=args.binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )
            path_para_dict['i_file_suffix'] += f"_ov_{args.exec_t_comp_ratioB:.2f}_repack"
            plot_path_para['file_suffix'] += f"_ov_{args.exec_t_comp_ratioB:.2f}_repack"
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})

        else:
            raise NotImplementedError(f"binpack algorithm {args.binpack_cfg['algorithm']} is not implemented")

        Bin_list_print(bin_list, glb_p_list, sim_step)
        if args.plot:
            pid2name = {_p.pid:_p.task.name for _p in glb_p_list}        
            # f"{plot_root}/new_task_bin_pack_cyclic_{num_cores}{args.file_suffix}.pdf"
            get_task_layout_compact(bin_list, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=False, plot_legend=True, format=args.plt_fmt, 
            txt_size=40, tick_dens=2, plot_start=hyper_p*(num_periods-1), plot_end=hyper_p*num_periods,
            save_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "cyclic"})) 
            
            # f"{plot_root}/new_task_bin_pack_full_{num_cores}{args.file_suffix}.pdf"
            get_task_layout_compact1bin(bin_list, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=args.plt_fmt, 
            txt_size=40, tick_dens=4, plot_start=0,  
            save_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "full"}))
        

        # select a period to save 
        assert num_periods >= 1
        bin_list2save = []
        # for _sched_tab in bin_list:
        dump_and_check(bin_list_save_path, bin_list)
        # dump_and_check(routing_table_save_path, scheduler_list[0].detail_alloc_info)

    # runtime scheduling
    elif args.test_all or args.test_case in two_stage_case_coll + other_case_coll:

        if args.test_all or args.test_case in two_stage_case_coll:
            if args.binpack_cfg["core_size"] == "induced":
                folder, files, match = get_core_num_from_trace_name(path_para_dict)
                if not match:
                    print(f"!!! Warning: no bin_list file "+
                        bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"})
                        +f" in {folder}:{files} !!!")
                    return
                args.num_cores = num_cores = int(match.group(1))
                bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
                routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
                plot_path_para['num_cores'] = num_cores
                trace_path_para['num_cores'] = num_cores
                
            else:
                bin_list_save_path = bin_save_fmt.format(**path_para_dict, **para_scan_group2)
                routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **para_scan_group2)

            bin_list = load_bin_list(bin_list_save_path, num_bins)
            
            task_spec, rsc_list, msg_dispatcher, \
                a_data_pipe, w_data_pipe, scheduler_list, \
                    monitor_list, trace_path = create_common_scheduler_elements(
                        args, trace_path_para, case_pth, 
                        hyper_p, glb_p_list, scheduler_args, 
                        sim_step, bin_list
                        )        

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
                    bin_path_format,
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


        elif args.test_all or args.test_case in [case_name_glb_input,]:
            bin_list = [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")]
            
            task_spec, rsc_list, msg_dispatcher, \
                a_data_pipe, w_data_pipe, scheduler_list, \
                    monitor_list, trace_path = create_common_scheduler_elements(
                        args, trace_path_para, case_pth, 
                        hyper_p, glb_p_list, scheduler_args, 
                        sim_step, bin_list
                        )        

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
        if args.max_core_stat:
            # check path (max_core_stat.pkl) exist
            core_max_dict_checkpoint = os.path.join(cache_dir, f"{case_pth}_max_core_stat.pkl")
            if not os.path.exists(core_max_dict_checkpoint):
                core_max_dict = {"cnt":0}
            else:
                core_max_dict = load_pickle(core_max_dict_checkpoint)
        for _SchedTab in actual_sched_record:
            if args.max_core_stat:
                _SchedTab.print_alloc_detail(pid2name, sim_step, core_max_dict=core_max_dict, max_core_stat=args.max_core_stat)
            else:
                _SchedTab.print_alloc_detail(pid2name, sim_step)
        if args.max_core_stat:
            core_max_dict.update({"cnt":core_max_dict.get("cnt",0)+1})
            dump_and_check(core_max_dict_checkpoint, core_max_dict)
            return

        if args.plot:
            if not args.jitter_sim_en:
                plot_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": case_pth, "plt_size": "full"})
            else:
                plot_path=plt_fn_w_seed_fmt.format(**plot_path_para, **{"case": case_pth, "plt_size": "full"})

            get_task_layout_compact1bin(actual_sched_record, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=args.plt_fmt, 
            txt_size=40, tick_dens=4, plot_start=0, save_path=plot_path)

        # save trace_list to trace_file
        # save_chunk(trace_path.replace(".pkl", ".h5"), trace_list, True)
        dump_and_check(trace_path, trace_list)

def create_common_scheduler_elements(args, trace_path_para, case_pth, hyper_p, glb_p_list, scheduler_args, sim_step, bin_list):
    # ======================== path settings ================
    trace_path = get_trace_path(args, trace_path_para, case_pth)
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

def get_trace_path(args, trace_path_para, case_pth):
    if args.jitter_sim_en:
        trace_path = trace_fn_w_seed_fmt.format(**trace_path_para, **{"case": case_pth})
    else:
        trace_path = trace_fn_wo_seed_fmt.format(**trace_path_para, **{"case": case_pth})
    return trace_path

def get_core_num_from_trace_name(path_para_dict):
    folder = cache_root_fmt.format(**path_para_dict)
    root,dirs,files = os.walk(folder).__next__()
    assert len(dirs) == 0
    bin_list_save_path,  routing_table_save_path = None, None
    for fn in files:
        if match := re.match(bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"}), fn):
            break
    return folder,files,match

def check_max_bin_num(args, num_cores, bin_path_format):
    max_num_bins = 0
    for e2e_latency, aux_scale_factor in args.e2e_var_sim_para['event_list']:            
        bin_list_save_path = bin_path_format.format(aux_scale_factor, e2e_latency, num_cores)
        bin_list = load_pickle(bin_list_save_path)
        max_num_bins = max(max_num_bins, len(bin_list))
    num_bins = max_num_bins
    return num_bins


if __name__ == "__main__":
    main()