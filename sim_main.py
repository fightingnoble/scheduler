import os, re
from task.task_cfg import create_init_p_list, gen_workloads
from task.task_cfg import affinity_cfg
from task.task_cfg import init_affinity
from sched.global_sched import push_task_into_bins_new, coleasing_alloc
from task.task_agent import TaskInt 
from task.task_agent import TaskInt
from task.spec import Spec
from model.message.msg_dispatcher import MsgDispatcher
from model.message.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from sched.scheduler_agent import Scheduler
from sched.scheduler_agent import core_mapping_1d
from sched.monitor_agent import Monitor
from allocator_agent import glb_sched, cyclic_sched
from model.event_gen.e2e_latency import discrete_event_sim
from model.task_queue_agent import TaskQueue
from utils import dump_and_check, load_pickle, update_df
from global_var import *
import flock


def main():
    import numpy as np 
    import pickle
    import json
    from utils import input_parser

    args = input_parser() 
    print(args)

    if args.e2e_var_sim_en: 
        assert args.gen_benchmark == True
    root_dir = args.root_dir
    num_cores = args.num_cores
    num_bins = args.num_bins
    num_periods = args.n_p
    slack_threshold = args.slack_threshold
    warmup = not args.warmup_dis

    cfg_para_dict = {
        "wsc_slack_ratio": args.wsc_slack_ratio, "temporal_rda_ratio": args.temporal_rda_ratio, 
        "lateness_mode": args.lateness_mode
        }
    para_scan_group1 = {"aux_scale_factor": args.aux_scale_factor, "e2e_latency": args.e2e_latency}
    para_scan_group2 = {"num_cores": args.num_cores}

    if not args.gen_benchmark:
        if args.profiling_filename == "profiling/profiling.csv":
            cfg_n = "heavy"
        else:
            cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1] 
        cfg_n += f"_{args.lateness_mode}"
    else:
        cfg_n = cfg_root_fmt.format(**cfg_para_dict, **para_scan_group1)
    path_para_dict = {"root_dir": root_dir, "cfg_n": cfg_n, "i_file_suffix": args.i_file_suffix}
    # remain parameters in group1 unfilled
    cfg_n_format = cfg_root_fmt.format(**cfg_para_dict, **{}.fromkeys(para_scan_group1, r"{}"))

    plot_root = plot_root_fmt.format(**path_para_dict, **para_scan_group2)
    trace_root = trace_root_fmt.format(**path_para_dict, **para_scan_group2)
    bin_path_format = os.path.join('cache', root_dir, cfg_n_format, r"bin_list_{}"+f"{args.i_file_suffix}.pkl")
    # remaining parameters in group2 unfilled
    bin_save_fmt.format(**{**path_para_dict, 'cfg_n': cfg_n_format, "num_cores": r"{}"})
    trace_path_para = {
        "trace_root": trace_root, "num_cores": num_cores, 
        "seed": args.seed, "file_suffix": args.file_suffix, 
        }
    plot_path_para = {"plot_root": plot_root, "num_cores": num_cores, "seed": args.seed, "file_suffix": args.file_suffix}
    csv_xlxs_root = os.path.join(log_dir, root_dir)

    if args.load_var_sim_en:
        if args.load_var_sim_para == {}:
            load_var_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['load_var']
        else:
            load_var_sim_para = args.load_var_sim_para
    if args.e2e_var_sim_en:
        if args.e2e_var_sim_para == {}:
            e2e_var_sim_para = json.load(open(os.path.join(cfg_dir, args.var_sim_cfg), "r"))['e2e_var']
        else:
            e2e_var_sim_para = args.e2e_var_sim_para

    if args.bin_pack_para == {}:
        args.binpack_cfg = binpack_cfg = json.load(open(os.path.join(cfg_dir, args.bin_pack_cfg), "r"))
    else:
        binpack_cfg = args.bin_pack_para

    hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args, slack_threshold)

    # generate the process list
    glb_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
    init_affinity(glb_p_list, mode='job', job_graph_nx=physical_graph_nx, verbose=args.verbose)

    # assert all the process has hard deadline
    if args.lateness_mode == "all_hard":
        for _p in glb_p_list:
            _p.task.criticality = "hard"
    elif args.lateness_mode == "all_soft":
        for _p in glb_p_list:
            _p.task.criticality = "soft"
    elif args.lateness_mode == "ignore":
        pass

    # simlation settings
    sim_step = min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32
    quantumSize = sim_step*args.quantumSize
    np.random.seed(args.seed)
    # from model.message.message_handler import gen_sensor_event
    # event_iter_dict = gen_sensor_event(glb_p_list, hyper_p, num_periods, True, args.jitter_sim_en, args.jitter_sim_para, args.seed)
    jitter_para_dict = dict(jitter_sim_en=args.jitter_sim_en, jitter_sim_para=args.jitter_sim_para, seed=args.seed)
    event_iter_dict = TaskInt.get_event_generator(glb_n_task_dict, hyper_p, num_periods, warmup, **jitter_para_dict)

    event_range = hyper_p * (num_periods+warmup)
    # integrated in to virtual sensor related source operator
    # if the handler find the var scaling factor is greater than 1, it spawns(wake up) x(factor-1) of new threads, 
    # which is marked as "spawned"
    # the "spawned" thread will be terminated as soon as they finish their job
    if args.load_var_sim_en: 
        for var_item, var_param in load_var_sim_para.items():
            dyn_obj_iter = discrete_event_sim(np.arange(load_var_sim_para["maxsize"], dtype=int), 1, var_param["period"], event_range, args.seed)
            dyn_obj_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
            load_var_sim_para[var_item]["stream"] = dyn_obj_stream
            load_var_sim_para[var_item]["iter"] = dyn_obj_iter
    else:
        load_var_sim_para = None

    # integrated in to every message from every sensor source operator
    if args.e2e_var_sim_en:
        ddl_update_iter = discrete_event_sim(e2e_var_sim_para['event_list'], 1, e2e_var_sim_para["period"], event_range, args.seed)
        ddl_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
        # check max number of bins
        max_num_bins = 0
        for e2e_latency, aux_scale_factor in e2e_var_sim_para['event_list']:            
            bin_list_save_path = bin_path_format.format(aux_scale_factor, e2e_latency, num_cores)
            bin_list = load_pickle(bin_list_save_path)
            max_num_bins = max(max_num_bins, len(bin_list))
        num_bins = max_num_bins
    else:
        ddl_update_iter = None
        ddl_stream = None

    if args.test_case == "all":
        args.test_all = True

    elif args.test_case == "bin_pack_new" or args.test_all:
        for _p in glb_p_list:
            _p.task.criticality = "hard"
        
        bin_list = [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")]
        # from message_agent import Message
        
        task_spec = Spec(0.1, [1 for _ in glb_p_list]) 
        # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
        rsc_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list))
        w_data_pipe = DataPipe("weight", len(bin_list))
        scheduler_list = [Scheduler(_SchedTab, args.e2e_latency, hyper_p, glb_p_list, barrier_en=not args.barrier_dis) for _SchedTab in bin_list]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]

        print("sim_step: ", sim_step)
        bin_list.clear()
        if binpack_cfg["algorithm"] == "reside":
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **para_scan_group2)
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **para_scan_group2)
            bin_list = push_task_into_bins_new(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.temporal_rda_ratio,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )

        elif binpack_cfg["algorithm"] == "coalescing":
            max_core_layout = coleasing_alloc(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.temporal_rda_ratio,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )
            filename = os.path.join(csv_xlxs_root, 'coalescing_req_cores.csv')
            import pandas as pd
            if not os.path.exists(filename):
                pd.DataFrame(columns=list(cfg_para_dict.keys())+list(para_scan_group1.keys())+["num_cores"]).to_csv(filename, index=False)
            # Load the dataframe

                        
            df = pd.read_csv(filename)
            num_cores = sum(max_core_layout[1].values())
            df = update_df(df, {**cfg_para_dict, **para_scan_group1}, 
                           {"num_cores": num_cores})
            df.to_csv(filename, index=False)

            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
        else:
            raise NotImplementedError(f"binpack algorithm {binpack_cfg['algorithm']} is not implemented")

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        from sched.scheduling_table import get_task_layout_compact, get_task_layout_sparse
        
        if args.plot:
            # f"{plot_root}/new_task_bin_pack_cyclic_{num_cores}{args.file_suffix}.pdf"
            get_task_layout_compact(bin_list, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
            txt_size=40, tick_dens=2, 
            save_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "cyclic"})) 
            
            # f"{plot_root}/new_task_bin_pack_full_{num_cores}{args.file_suffix}.pdf"
            get_task_layout_compact(bin_list, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
            txt_size=40, tick_dens=4, plot_start=0, 
            save_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "full"}))
        

        # select a period to save 
        assert num_periods >= 1
        bin_list2save = []
        # for _sched_tab in bin_list:
        dump_and_check(bin_list_save_path, bin_list)
        # dump_and_check(routing_table_save_path, scheduler_list[0].detail_alloc_info)

    elif args.test_case == "dynamic" or args.test_all or args.test_case == "cyclic":
        if binpack_cfg["algorithm"] == "coalescing":
            folder = cache_root_fmt.format(**path_para_dict)
            root,dirs,files = os.walk(folder).__next__()
            assert len(dirs) == 0
            bin_list_save_path,  routing_table_save_path = None, None
            for fn in files:
                if match := re.match(bin_fn_fmt.format(**path_para_dict, **{"num_cores": r"(\d*)"}), fn):
                    break
            if not match:
                print(f"!!! Warning: no bin_list file in {folder} !!!")
                return
            num_cores = int(match.group(1))
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
        else:
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **para_scan_group2)
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **para_scan_group2)
        bin_list = load_pickle(bin_list_save_path)
        num_bins = len(bin_list) if num_bins == -1 else num_bins
        for bin_id in range(num_bins):
            if bin_id >= len(bin_list):
                bin_list.append(SchedulingTableInt(0, bin_id, 0, f'dummy_bin_{bin_id}'))
        cores = [bin.num_resources for bin in bin_list]
        core_map = core_mapping_1d(cores)
        
        task_spec = Spec(0.1, [1 for _ in glb_p_list]) 
        # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
        rsc_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        sensor_pipe = TriggerPipe(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list))
        w_data_pipe = DataPipe("weight", len(bin_list))
        scheduler_list = [Scheduler(_SchedTab, args.e2e_latency, hyper_p, glb_p_list, 
                                    barrier_en=not args.barrier_dis, 
                                    ) for _SchedTab in bin_list]
        for _sched in scheduler_list:
            _sched.core_map = core_map[_sched._SchedTab.id]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]        

        print("sim_step: ", sim_step)
        cyclic_sched(task_spec, affinity_cfg, 
                scheduler_list, monitor_list,
                event_iter_dict,
                ddl_update_iter, ddl_stream,
                load_var_sim_para,
                rsc_list, 
                num_cores, 
                glb_p_list,
                sim_step, hyper_p, num_periods, 
                msg_dispatcher,
                sensor_pipe,
                a_data_pipe, w_data_pipe, 
                bin_path_format,
                args.verbose, warmup=True, drain=True, 
                cyclic=args.test_case == "cyclic",)

        tot_cores = 0
        n_switch = 0
        weighted_avg_cumulative_time = 0
        for _sched in scheduler_list:
            partition_id = _sched._SchedTab.id
            bin_size = _sched._SchedTab.num_resources
            tot_cores += bin_size
            print(f"(Partition {partition_id}) number of context switch {_sched.barrier.number_of_asserts}")
            print(f"(Partition {partition_id}) cumulative context switch {_sched.barrier.cumulative_time}")
            weighted_avg_cumulative_time += _sched.barrier.cumulative_time * num_cores
            n_switch += _sched.barrier.number_of_asserts
        print(f"number of context switch {n_switch}")
        weighted_avg_cumulative_time /= tot_cores
        print(f"cumulative context switch {weighted_avg_cumulative_time}")

        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        if args.max_core_stat:
            # check path (max_core_stat.pkl) exist
            if os.path.exists(f"{trace_root}/max_core_stat.pkl"):
                # delete the file
                os.remove(f"{trace_root}/max_core_stat.pkl")
            if not os.path.exists(f"cache/dyn_max_core_stat.pkl"):
                core_max_dict = {"cnt":0}
            else:
                core_max_dict = load_pickle(f"cache/dyn_max_core_stat.pkl")
        for _SchedTab in actual_sched_record:
            if args.max_core_stat:
                _SchedTab.print_alloc_detail(pid2name, sim_step, core_max_dict=core_max_dict, max_core_stat=args.max_core_stat)
            else:
                _SchedTab.print_alloc_detail(pid2name, sim_step)
        if args.max_core_stat:
            core_max_dict.update({"cnt":core_max_dict.get("cnt",0)+1})
            dump_and_check(f"cache/dyn_max_core_stat.pkl", core_max_dict)
            return

        if args.plot:
            if args.jitter_sim_en:
                # "{plot_root}/seed_{args.seed}/cyclic_full_{num_cores}{args.file_suffix}.pdf"
                plot_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "cyclic", "plt_size": "full"})
            else:
                # f"{plot_root}/dyn_full_{num_cores}{args.file_suffix}.pdf"
                plot_path=plt_fn_w_seed_fmt.format(**plot_path_para, **{"case": "dyn", "plt_size": "full"})

            from sched.scheduling_table import get_task_layout_compact, get_task_layout_sparse
            get_task_layout_compact(actual_sched_record, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
            txt_size=40, tick_dens=4, plot_start=0, save_path=plot_path)

        # "case": "cyclic" if args.test_case == "cyclic" else "dynamic"
        if args.jitter_sim_en:
            # trace_path = f"{trace_root}/cyclic_e2e_trace_{num_cores}"
            trace_path = trace_fn_w_seed_fmt.format(**trace_path_para, **{"case": args.test_case})
        else:
            # trace_path = f"{trace_root}/dynamic_e2e_trace_{num_cores}"
            trace_path = trace_fn_wo_seed_fmt.format(**trace_path_para, **{"case": args.test_case})

        dump_and_check(trace_path, trace_list) 

    elif args.test_case == "glb_dynamic" or args.test_all:
        bin_list = [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")]
        # from message_agent import Message
        
        task_spec = Spec(0.1, [1 for _ in glb_p_list]) 
        # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
        rsc_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list))
        w_data_pipe = DataPipe("weight", len(bin_list))
        scheduler_list = [Scheduler(_SchedTab, args.e2e_latency, hyper_p, glb_p_list, barrier_en=not args.barrier_dis) for _SchedTab in bin_list]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]

        print("sim_step: ", sim_step)
        glb_sched(task_spec, affinity_cfg, 
                scheduler_list, monitor_list,
                event_iter_dict,
                ddl_update_iter, ddl_stream,
                load_var_sim_para,
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

        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        if args.max_core_stat:
            # check path (max_core_stat.pkl) exist
            if os.path.exists(f"{trace_root}/max_core_stat.pkl"):
                # delete the file
                os.remove(f"{trace_root}/max_core_stat.pkl")
            if not os.path.exists(f"cache/glb_max_core_stat.pkl"):
                core_max_dict = {"cnt":0}
            else:
                core_max_dict = load_pickle(f"cache/glb_max_core_stat.pkl")
        for _SchedTab in actual_sched_record:
            if args.max_core_stat:
                _SchedTab.print_alloc_detail(pid2name, sim_step, core_max_dict=core_max_dict, max_core_stat=args.max_core_stat)
            else:
                _SchedTab.print_alloc_detail(pid2name, sim_step)
        if args.max_core_stat:
            core_max_dict.update({"cnt":core_max_dict.get("cnt",0)+1})
            dump_and_check(f"cache/glb_max_core_stat.pkl", core_max_dict)
            return

        if args.plot:
            from sched.scheduling_table import get_task_layout_compact
            file_name = f"{plot_root}/seed_{args.seed}/glb_dyn_full_{num_cores}{args.file_suffix}.pdf" if not args.barrier_dis else f"{plot_root}/glb_dyn_full_{num_cores}_ideal{args.file_suffix}.pdf"
            get_task_layout_compact(actual_sched_record, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
            txt_size=40, tick_dens=4, plot_start=0, save_path=file_name)

        # f"{trace_root}/glb_dyn_e2e_trace_{num_cores}"
        if args.jitter_sim_en:
            trace_path = trace_fn_w_seed_fmt.format(**trace_path_para, **{"case": "glb_dyn"})
        else:
            trace_path = trace_fn_wo_seed_fmt.format(**trace_path_para, **{"case": "glb_dyn"})
        
        # save trace_list to trace_file
        dump_and_check(trace_path, trace_list)
    elif args.test_case == "switch_const":
        pass

if __name__ == "__main__":
    main()