import os, re
from task.task_cfg import create_init_p_list, gen_workloads
from task.task_cfg import affinity_cfg
from task.task_cfg import init_affinity
from sched.global_sched import push_task_into_bins_new, coleasing_alloc_1bin
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
from utils import dump_and_check, load_pickle, update_df, check_parents_path, args_postprocess
from global_var import *


def main():
    import numpy as np 
    from utils import input_parser
    args = input_parser() 
    print(args)

    num_cores = args.num_cores
    num_bins = args.num_bins
    num_periods = args.n_p
    warmup = not args.warmup_dis

    cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, trace_root, \
    bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root = args_postprocess(args)

    # hyper_p, glb_n_task_dict, physical_graph_nx = gen_workloads(args)    
    from task.task_cfg import load_taskattrib, deduce_cfg2, gen_taskint_from_cfg, \
        creat_logical_graph, creat_physical_graph, init_depen, deduce_eq_wsc
    from task.load_cfg.loadA import task_graph_srcs, task_graph_ops, task_graph_sinks, sink_attr, src_attr
    from task.load_cfg import load_chain
    taskattr_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=args.verbose) 
    hyper_p = 1/f_gcd
    assert args.aux_scale_factor >= 0
    if args.aux_scale_factor != 1:
        for node, taskattr in taskattr_dict.items():
            # scale up the thread scaling factor
            if taskattr.timing_flag == "realtime":
                taskattr.thread_scaling_factor *= args.aux_scale_factor

    print(f"Ops per second of Workload: {sum([(v.flops*v.var_factor*v.thread_scaling_factor*v.freq) for n,v in taskattr_dict.items()]):.2f} T")
    logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)


    if args.binpack_cfg["algorithm"] == "coalescing":
        # temporal_abs_en = True
        algorithm = 'gurobi'
        wsc_slack_ratio = 1 - args.exec_t_comp_ratioA
    else:
        # temporal_abs_en = False
        algorithm = 'avg'
        wsc_slack_ratio = args.wsc_slack_ratio
    deduce_cfg2(taskattr_dict, f_gcd, hyper_p, logical_graph_nx, task_graph_srcs, 
                task_graph_sinks, sink_attr, src_attr, args.slack_threshold, args.e2e_latency, 
                args.exec_t_comp_ratioA, args.jitter_t_comp_ratio, 
                wsc_slack_ratio, algorithm, args.timestepxus)
    if args.binpack_cfg["algorithm"] == "coalescing":
        print("deduced_eq_wsc:", deduce_eq_wsc(logical_graph_nx, task_graph_srcs, task_graph_sinks, src_attr, args.jitter_t_comp_ratio))
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)
    physical_graph_nx = creat_physical_graph(logical_graph_nx, int(f_gcd), taskattr_dict=taskattr_dict)

    # ***************** artifiact: chain split *******************************
    hyper_p = 0.1
    op_filter = load_chain.task_graph_ops
    sink_filter = load_chain.task_graph_sinks
    src_filter = load_chain.task_graph_srcs
    # filter the process and the node in physical graph, logic graph by the function
    # x.split('_')[-1] == "0" and "_".join(x.split('_')[0:-2]) in op_filter
    # remove the exceptions
    filter_fn = lambda x: x.split('_')[-1] == "0" and "_".join(x.split('_')[0:-2]) in op_filter
    glb_n_task_dict = {k:v for k,v in glb_n_task_dict.items() if filter_fn(k)}
    logical_graph_nx = logical_graph_nx.subgraph(list(glb_n_task_dict.keys())+list(src_filter.keys())+list(sink_filter.keys()))
    physical_graph_nx = physical_graph_nx.subgraph(list(glb_n_task_dict.keys())+list(src_filter.keys())+list(sink_filter.keys()))
    # ***************** artifiact：chain split *******************************

    init_depen(glb_n_task_dict, physical_graph_nx, verbose=args.verbose)

    # generate the process list
    glb_p_list = create_init_p_list(glb_n_task_dict, args.verbose)
    init_affinity(glb_p_list, mode='job', job_graph_nx=physical_graph_nx, verbose=args.verbose)
    process_dict_tmp = {p.pid:p for p in glb_p_list}
    # _p1, _p2 shares the deadline, 
    # ddl1 = _p1.task.exp_comp_t / speed_down_rate + 5e-5 + 1/30*jitter_rate
    # ddl2 = _p2.task.exp_comp_t / speed_down_rate + 5e-5 
    # ddl1 + ddl2 keeps the same
    # given speed_down_rate

    # ***************** artifiact: statistic *******************************
    # args.jitter_sim_para, a truncnorm dist
    # args.exec_var_para, a truncexpon dist
    from model.event_gen.e2e_latency import get_truncexpon_param, get_truncnorm_para
    from scipy.stats import truncnorm, truncexpon
    loc, scale, myclip_b, a, b = get_truncnorm_para(1, {"scale": 0.5})
    scope, loc, scale, b = get_truncexpon_param(1, {"scale": 0.5}, lamda_exp=15)
    jitter_cdf = lambda x: truncnorm.cdf(x, a, b, loc=loc, scale=scale)
    exec_cdf = lambda x: truncexpon.cdf(x, b, loc=loc, scale=scale)

    _p0, _p1 = process_dict_tmp[0], process_dict_tmp[1]
    pair_list = []
    for speed_down_rate1 in np.arange(0.18,0.37,0.01):
        # slack = _p1.task.ddl - (_p1.process_dict[1].task.exp_comp_t / (1-speed_down_rate) + 5e-5)
        # jitter_rate = (_p1.task.ddl + slack- 5e-5 - _p2.task.exp_comp_t / (1-speed_down_rate)) * 30
        speed_down_rate1 = round(speed_down_rate1, 2)
        jitter_rate = _p0.task.ddl + _p1.task.ddl - (_p0.task.exp_comp_t / (1-speed_down_rate1) + _p1.task.exp_comp_t / (1-speed_down_rate1))
        jitter_rate = jitter_rate * 30
        if jitter_rate <0 or jitter_rate > 0.5:
            continue
        pair_list.append((speed_down_rate1, jitter_rate))
    print(pair_list)
    cum_p_sharing = jitter_cdf(pair_list[0][1]) * exec_cdf(pair_list[0][0]) + \
            sum([jitter_cdf(pair_list[i][1]) * (exec_cdf(pair_list[i][0]) - exec_cdf(pair_list[i-1][0])) for i in range(1, len(pair_list))])
    cum_p_coalescing = jitter_cdf(0.2) * exec_cdf(0.3) 
    print("inner chain sharing improves confidence level at least from {:.2f}% to {:.2f}%".format(cum_p_coalescing*100, cum_p_sharing*100))
    # ***************** artifiact: statistic *******************************

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
    sim_step = elim_nume_error(1e-6 * args.timestepxus) # min([glb_n_task_dict[task].exp_comp_t for task in glb_n_task_dict])/32
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
        for var_item, var_param in args.load_var_sim_para.items():
            dyn_obj_iter = discrete_event_sim(np.arange(var_param["maxsize"], dtype=int), 1, var_param["period"], event_range, args.seed)
            dyn_obj_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
            var_param["stream"] = dyn_obj_stream
            var_param["iter"] = dyn_obj_iter

    # integrated in to every message from every sensor source operator
    if args.e2e_var_sim_en:
        ddl_update_iter = discrete_event_sim(args.e2e_var_sim_para['event_list'], 1, args.e2e_var_sim_para["period"], event_range, args.seed)
        ddl_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
        # check max number of bins
        max_num_bins = 0
        for e2e_latency, aux_scale_factor in args.e2e_var_sim_para['event_list']:            
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
        exec_para_dict = dict(exec_var_en=args.exec_var_en, exec_var_para=args.exec_var_para, seed=args.seed)
        rsc_list = [Resource_model_int(size=sched_tab.num_resources, **exec_para_dict) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
        w_data_pipe = DataPipe("weight", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
        scheduler_list = [Scheduler(_SchedTab, args.e2e_latency, hyper_p, glb_p_list, barrier_en=not args.barrier_dis) for _SchedTab in bin_list]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]

        print("sim_step: ", sim_step)
        bin_list.clear()
        if args.binpack_cfg["algorithm"] == "reside":
            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **para_scan_group2)
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **para_scan_group2)
            bin_list = push_task_into_bins_new(
                bin_list,
                glb_p_list, affinity_cfg, event_iter_dict,
                num_cores, args.quantum_check_en, quantumSize, 
                sim_step, hyper_p, args.wsc_slack_ratio, args.exec_t_comp_ratioA,

                scheduler_list, monitor_list,
                msg_dispatcher,
                a_data_pipe, w_data_pipe,

                num_periods, binpack_cfg=args.binpack_cfg,
                verbose=True, DEBUG_FG=False, # args.verbose, args.DEBUG,
                warmup=True, drain=True, 
                )

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
            
            # Load the dataframe                        
            df = pd.read_csv(filename)
            num_cores = sum(max_core_layout[1].values())
            df = update_df(df, {**cfg_para_dict, **para_scan_group1}, 
                           {"num_cores": num_cores})
            df.to_csv(filename, index=False)

            bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
            routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
        else:
            raise NotImplementedError(f"binpack algorithm {args.binpack_cfg['algorithm']} is not implemented")

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        from sched.scheduling_table import get_task_layout_compact, get_task_layout_sparse
        
        for _SchedTab in bin_list:
                _SchedTab.print_alloc_detail(pid2name, sim_step)
        if args.plot:
            # f"{plot_root}/new_task_bin_pack_cyclic_{num_cores}{args.file_suffix}.pdf"
            get_task_layout_compact(bin_list, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
            txt_size=40, tick_dens=2, plot_start=hyper_p*(num_periods-1), plot_end=hyper_p*num_periods,
            save_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "cyclic"})) 
            
            # f"{plot_root}/new_task_bin_pack_full_{num_cores}{args.file_suffix}.pdf"
            get_task_layout_compact(bin_list, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, plot_start=0, warmup=False, drain=True, plot_legend=False, format=["svg","pdf"], 
            txt_size=40, tick_dens=4, 
            save_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "new_task_bin_pack", "plt_size": "full"}))
        

        # select a period to save 
        assert num_periods >= 1
        bin_list2save = []
        # for _sched_tab in bin_list:
        dump_and_check(bin_list_save_path, bin_list)
        # dump_and_check(routing_table_save_path, scheduler_list[0].detail_alloc_info)

    elif args.test_case == "dynamic" or args.test_all or args.test_case == "cyclic":
        if args.binpack_cfg["algorithm"] == "coalescing":
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
        exec_para_dict = dict(exec_var_en=args.exec_var_en, exec_var_para=args.exec_var_para, seed=args.seed)
        rsc_list = [Resource_model_int(size=sched_tab.num_resources, **exec_para_dict) for sched_tab in bin_list]
# curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        sensor_pipe = TriggerPipe(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
        w_data_pipe = DataPipe("weight", len(bin_list), jitter_sim_para=args.jitter_sim_para, seed=args.seed)
        scheduler_list = [Scheduler(bin_list[idx], args.e2e_latency, hyper_p, glb_p_list, 
                                    barrier_en=not args.barrier_dis, res_cfg=rsc_list[idx],
                                    exec_t_comp_ratioB = args.exec_t_comp_ratioB,
                                    ) for idx in range(len(bin_list))]
        for _sched in scheduler_list:
            _sched.core_map = core_map[_sched._SchedTab.id]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]        

        print("sim_step: ", sim_step)
        cyclic_sched(task_spec, affinity_cfg, 
                scheduler_list, monitor_list,
                event_iter_dict,
                ddl_update_iter, ddl_stream,
                args.load_var_sim_para,
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

        case_pth = "cyclic" if args.test_case == "cyclic" else "dyn"
        if args.plot:
            if not args.jitter_sim_en:
                # "{plot_root}/seed_{args.seed}/cyclic_full_{num_cores}{args.file_suffix}.pdf"
                plot_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": case_pth, "plt_size": "full"})
            else:
                # f"{plot_root}/dyn_full_{num_cores}{args.file_suffix}.pdf"
                plot_path=plt_fn_w_seed_fmt.format(**plot_path_para, **{"case": case_pth, "plt_size": "full"})

            from sched.scheduling_table import get_task_layout_compact, get_task_layout_sparse
            get_task_layout_compact(actual_sched_record, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=False, drain=True, plot_legend=False, format=["svg","pdf"], 
            txt_size=40, tick_dens=4, plot_start=0, save_path=plot_path)

        if args.jitter_sim_en:
            # trace_path = f"{trace_root}/cyclic_e2e_trace_{num_cores}"
            trace_path = trace_fn_w_seed_fmt.format(**trace_path_para, **{"case": case_pth})
        else:
            # trace_path = f"{trace_root}/dynamic_e2e_trace_{num_cores}"
            trace_path = trace_fn_wo_seed_fmt.format(**trace_path_para, **{"case": case_pth})

        dump_and_check(trace_path, trace_list) 

    elif args.test_case == "glb_dynamic" or args.test_all:
        bin_list = [SchedulingTableInt(num_cores, 1, 0, "bin_glb_dynamic")]
        # from message_agent import Message
        
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
                                    barrier_en=not args.barrier_dis, res_cfg=rsc_list[idx]
                                    ) for idx in range(len(bin_list))]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]

        print("sim_step: ", sim_step)
        glb_sched(task_spec, affinity_cfg, 
                scheduler_list, monitor_list,
                event_iter_dict,
                ddl_update_iter, ddl_stream,
                args.load_var_sim_para,
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

            # f"{plot_root}/seed_{args.seed}/glb_dyn_full_{num_cores}{args.file_suffix}.pdf"
            # f"{plot_root}/glb_dyn_full_{num_cores}_ideal{args.file_suffix}.pdf"
            if not args.jitter_sim_en:
                plot_path=plt_fn_wo_seed_fmt.format(**plot_path_para, **{"case": "glb_dyn", "plt_size": "full"})
            elif args.barrier_dis:
                plt_fn_wo_seed_fmt.format(**{**plot_path_para, "num_cores": f"{num_cores}_ideal", 
                                             **{"case": "glb_dyn", "plt_size": "full"}})
            else:
                plot_path=plt_fn_w_seed_fmt.format(**plot_path_para, **{"case": "glb_dyn", "plt_size": "full"})

            get_task_layout_compact(actual_sched_record, pid2name, save= True, time_step= sim_step,
            hyper_p=hyper_p, n_p=num_periods, warmup=False, drain=True, plot_legend=False, format=["svg","pdf"], 
            txt_size=40, tick_dens=4, plot_start=0, save_path=plot_path)

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
