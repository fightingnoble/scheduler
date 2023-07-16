import os
from task.task_cfg import create_init_p_list, gen_workloads
from task.task_cfg import affinity_cfg
from task.task_cfg import init_affinity
from global_sched import push_task_into_bins, push_task_into_bins_new
from task.task_agent import TaskInt 
from task.task_agent import TaskInt
from task.spec import Spec
from model.msg_dispatcher import MsgDispatcher
from model.data_pipe import DataPipe, TriggerPipe
from sched.scheduling_table import SchedulingTableInt
from model.resource_agent import Resource_model_int
from scheduler_agent import Scheduler
from sched.monitor_agent import Monitor
from allocator_agent import glb_sched, cyclic_sched
from model.event_gen.e2e_latency import dyn_obj_sim, e2e_var_sim
from model.task_queue_agent import TaskQueue
from utils import dump_and_check

def main():
    import numpy as np 
    import pickle
    from global_var import trace_list
    import json
    from utils import input_parser

    args = input_parser() 
    print(args)
    if not args.gen_benchmark:
        if args.profiling_filename == "profiling.csv":
            cfg_n = "heavy"
        else:
            cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
    else:
        cfg_n = f"x{args.aux_scale_factor}_{args.e2e_latency}s_rda-{(args.wsc_slack_ratio-args.temporal_rda_ratio):.2%}(T)_{args.temporal_rda_ratio:.2%}(S)"
    cfg_n += f"_{args.lateness_mode}"
    root_dir = args.root_dir
    num_cores = args.num_cores
    num_periods = args.n_p
    slack_threshold = args.slack_threshold
    warmup = not args.warmup_dis
    if args.load_var_sim_en:
        if args.load_var_sim_para == {}:
            load_var_sim_para = json.load(open(args.load_var_para_file, "r"))
        else:
            load_var_sim_para = args.load_var_sim_para

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
    bin_list_save_path = f"cache/{root_dir}/{cfg_n}/bin_list_{num_cores}{args.i_file_suffix}.pkl"
    routing_table_save_path = f"cache/{root_dir}/{cfg_n}/routing_table_{num_cores}{args.i_file_suffix}.pkl"
    np.random.seed(args.seed)
    # from model.message_handler import gen_sensor_event
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
            dyn_obj_iter = dyn_obj_sim(var_param, 1, var_param["period"], event_range, args.seed)
            dyn_obj_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
            load_var_sim_para[var_item]["stream"] = dyn_obj_stream
            load_var_sim_para[var_item]["iter"] = dyn_obj_iter
    else:
        load_var_sim_para = None

    # integrated in to every message from every sensor source operator
    if args.e2e_var_sim_en:
        ddl_update_iter = e2e_var_sim(args.e2e_latency, args.e2e_var_sim_para, 1, args.e2e_var_sim_para["period"], event_range, args.seed)
        ddl_stream = TaskQueue(sort_f=lambda x: x[0], descending=False)
    else:
        ddl_update_iter = None
        ddl_stream = None

    if args.test_case == "all":
        args.test_all = True

    elif args.test_case == "bin_pack_new" or args.test_all:
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
        bin_list = push_task_into_bins_new(
            bin_list,
            glb_p_list, affinity_cfg, event_iter_dict,
            num_cores, args.quantum_check_en, quantumSize, 
            sim_step, hyper_p, args.spatial_rda_ratio, args.temporal_rda_ratio,

            scheduler_list, monitor_list,
            msg_dispatcher,
            a_data_pipe, w_data_pipe,

            num_periods, args.verbose, 
            warmup=True, drain=True
            )
        
        from sched.scheduling_table import get_task_layout_compact
        get_task_layout_compact(bin_list, glb_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=False, plot_legend=True, format=["svg","pdf"], 
        txt_size=40, tick_dens=2, save_path=f"plot/{root_dir}/{cfg_n}/{num_cores}/new_task_bin_pack_cyclic_{num_cores}{args.file_suffix}.pdf") 

        get_task_layout_compact(bin_list, glb_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path=f"plot/{root_dir}/{cfg_n}/{num_cores}/new_task_bin_pack_full_{num_cores}{args.file_suffix}.pdf")

        # select a period to save 
        assert num_periods >= 1
        bin_list2save = []
        # for _sched_tab in bin_list:
        dump_and_check(bin_list_save_path, bin_list)
        dump_and_check(routing_table_save_path, scheduler_list[0].detail_alloc_info)

    elif args.test_case == "dynamic" or args.test_all:
        try:
            # load the bin_list and the init_p_list
            with open(bin_list_save_path, "rb") as f:
                bin_list = pickle.load(f)
            # with open(f"init_p_list_{num_cores}{args.file_suffix}.pkl", "rb") as f:
            #     init_p_list = pickle.load(f)
        except:
            print(f"{bin_list_save_path} not found")
            # print(f"{save_path} not found")
            # bin_list, _ = push_task_into_bins(glb_p_list, affinity_cfg, num_cores, args.quantum_check_en, quantumSize, sim_step, hyper_p, 1, args.verbose, warmup=True, drain=True)

        # from message_agent import Message
        
        task_spec = Spec(0.1, [1 for _ in glb_p_list]) 
        # process_dict_list = [{pid:init_p_list[pid] for pid in _SchedTab.index_occupy_by_id()} for _SchedTab in bin_list]
        rsc_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # curr_cfg_list = [Resource_model_int(size=sched_tab.num_resources) for sched_tab in bin_list]
        # msg_pipe = Message()
        msg_dispatcher = MsgDispatcher(len(bin_list))
        sensor_pipe = TriggerPipe(len(bin_list))
        a_data_pipe = DataPipe("activation", len(bin_list))
        w_data_pipe = DataPipe("weight", len(bin_list))
        scheduler_list = [Scheduler(_SchedTab, args.e2e_latency, hyper_p, glb_p_list, barrier_en=not args.barrier_dis) for _SchedTab in bin_list]
        monitor_list = [Monitor(_SchedTab.num_resources, int(3*hyper_p/sim_step), id=_SchedTab.id, name=_SchedTab.name) for _SchedTab in bin_list]
        # with open(routing_table_save_path, "rb") as f:
        #     routing_table = pickle.load(f)
        
        # adjust the bin_list, refer to the routing_table, physical_graph
        

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
                args.verbose, warmup=True, drain=True)

        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        for _SchedTab in actual_sched_record:
            _SchedTab.print_alloc_detail(pid2name, sim_step)

        from sched.scheduling_table import get_task_layout_compact
        get_task_layout_compact(actual_sched_record, glb_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path=f"plot/{root_dir}/{cfg_n}/{num_cores}/dyn_full_{num_cores}{args.file_suffix}.pdf")

        trace_path = f"trace/{root_dir}/{cfg_n}/dynamic_e2e_trace_{num_cores}{args.file_suffix}.pkl"
        # save trace_list to trace_file
        with open(trace_path, "wb") as f:
            pickle.dump(trace_list, f)
        
        try:
            with open(trace_path, "rb") as f:
                trace_list = pickle.load(f)
            print("trace saved successfully")
        except:
            print("trace file not found")
            exit(0)            

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
                args.verbose, warmup=True, drain=True)
        
        print("number of context switch {}".format(scheduler_list[0].barrier.number_of_asserts))
        print("cumulative context switch {}".format(scheduler_list[0].barrier.cumulative_time))

        actual_sched_record = [monitor.trace_recoder for monitor in monitor_list]

        pid2name = {_p.pid:_p.task.name for _p in glb_p_list}
        print("=====================================\n")
        print("bin_pack_result:")
        print("=====================================\n")
        for _SchedTab in actual_sched_record:
            _SchedTab.print_alloc_detail(pid2name, sim_step)

        from sched.scheduling_table import get_task_layout_compact
        file_name = f"plot/{root_dir}/{cfg_n}/{num_cores}/glb_dyn_full_{num_cores}{args.file_suffix}.pdf" if not args.barrier_dis else f"plot/{root_dir}/{cfg_n}/{num_cores}/glb_dyn_full_{num_cores}_ideal{args.file_suffix}.pdf"
        get_task_layout_compact(actual_sched_record, glb_p_list, save= True, time_step= sim_step,
        hyper_p=hyper_p, n_p=num_periods, warmup=True, drain=True, plot_legend=False, format=["svg","pdf"], 
        txt_size=40, tick_dens=4, plot_start=0, save_path=file_name)

        trace_path = f"trace/{root_dir}/{cfg_n}/dyn_glb_e2e_trace_{num_cores}{args.file_suffix}.pkl"
        # save trace_list to trace_file
        dir_path = os.path.dirname(trace_path)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(trace_path, "wb") as f:
            pickle.dump(trace_list, f)
        try:
            with open(trace_path, "rb") as f:
                trace_list = pickle.load(f)
            print("trace saved successfully")
        except:
            print("trace file not found")
            exit(0)

if __name__ == "__main__":
    main()