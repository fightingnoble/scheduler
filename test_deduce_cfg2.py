"""test_deduce_cfg2.py — external smoke test for slack_estim.deduce_cfg2.

Moved out of sched/slack_estim.py (B6-SPLIT-003, 2026-06-29). Was the module __main__ self-test;
now an explicit external test entry. Run: python test_deduce_cfg2.py [--profiling_filename ...]
Recovery: git checkout archive/test_pipeline-20260612 -- sched/slack_estim.py
"""
from sched.slack_estim import deduce_cfg2, update_taskattr_dict


def main():
    import argparse
    import numpy as np 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--profiling_filename", type=str, default="profiling/profiling.csv", help="profiling filename") 
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    # parser.add_argument("--freq", type=float, default=10, help="frequency")
    parser.add_argument("--exec_t_comp_ratioA", default=0.05, type=float, help="temporal ratio")
    parser.add_argument("--wsc_slack_ratio", default=0.8, type=float, help="wsc slack ratio")
    parser.add_argument("--slack_threshold", default=5e-4, type=float, help="slack threshold")
    parser.add_argument("--aux_scale_factor", default=1, type=int, help="aux scale factor")
    args = parser.parse_args() 

    from task.task_cfg import task_graph_srcs, task_graph_sinks, creat_logical_graph, task_graph_ops
    from task.task_cfg import load_taskattrib, gen_taskint_from_cfg
    taskattr_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=args.verbose) 
    hyper_p = 1/f_gcd
    if args.aux_scale_factor != 1:
        for node, taskattr in taskattr_dict.items():
            # scale up the thread scaling factor
            if taskattr.timing_flag == "realtime":
                taskattr.thread_scaling_factor *= args.aux_scale_factor
    logical_graph_nx = creat_logical_graph(task_graph_srcs, task_graph_ops, task_graph_sinks)
    slack_threshold = args.slack_threshold
    ert, ddl, rsc_map_w = deduce_cfg2(taskattr_dict, 
               logical_graph_nx, task_graph_srcs, task_graph_sinks, 0.99,
               slack_threshold, verbose=True)
    
    update_taskattr_dict(ert, ddl, rsc_map_w, taskattr_dict, f_gcd, hyper_p, 
               logical_graph_nx, verbose=False)
    glb_n_task_dict = gen_taskint_from_cfg(taskattr_dict, f_gcd)

    for node, taskint in glb_n_task_dict.items():
        print(node, taskint)
        print()


if __name__ == "__main__":
    main()
