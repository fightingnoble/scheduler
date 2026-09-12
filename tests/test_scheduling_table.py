"""test_scheduling_table.py — 从 sched/scheduling_table.py 外移的 __main__ 调试样例（B9/REQ-002, E0007 批准）。

原样保留调试样例行为，含 lwb case 的既有失败（aeap_insert broadcast ValueError，本批不修）。
来源基线：archive/test_pipeline-20260612；旧行为对照入口: python -m sched.scheduling_table --test_case <case>
"""
from typing import List

from sched.scheduling_table import SchedulingTableInt
from task.task_agent import TaskInt


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.') 
    parser.add_argument('--test_case', type=str, default="no constrants", help='test case name')
    args = parser.parse_args()
    # create a task set
    # first branch: have free cores and free slots at beginning
    # t1 [15, 19] 25 rsc and 2 slot
    # t2 [3, 12] 24 rsc and 7 slot
    # second branch: select a interval with enough free cores and free slots
    # t3 [0, 20] 10 rsc and 4 slot
    # thrird branch: evently distribute the resources in the expected interval
    # t4 [0, 7] 10 rsc and 4 slot
    # last branch: As soon as possible
    # t5 [0, 16] 10 rsc and 8 slot
    # t6 [9, 20] 9 rsc and 9 slot

    N_task = 6
    t1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                ERT=15, ddl=19, period=30, exp_comp_t=2, 
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t2 = TaskInt(task_name="task2", task_id=2, task_flag="moveable", timing_flag="deadline",
                ERT=3, ddl=12, period=30, exp_comp_t=7,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t3 = TaskInt(task_name="task3", task_id=3, task_flag="moveable", timing_flag="deadline", 
                ERT=0, ddl=20, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t4 = TaskInt(task_name="task4", task_id=4, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=7, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t5 = TaskInt(task_name="task5", task_id=5, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=16, period=30, exp_comp_t=8,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t6 = TaskInt(task_name="task6", task_id=6, task_flag="moveable", timing_flag="deadline",
                ERT=9, ddl=20, period=30, exp_comp_t=9,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)

    task_list:List[TaskInt] = [None for i in range(10)]
    alloc_info = [None for i in range(10)]
    require_rsc_size = [0 for i in range(10)]
    task_list[0:N_task] = [t1, t2, t3, t4, t5, t6]
    require_rsc_size[0:N_task] = [25, 24, 10, 10, 10, 9]

    if args.test_case == "no constrants":
        pass
    elif args.test_case == "upb":
        for i in range(N_task):
            task_list[i].parallel_mode = "upb"
            task_list[i].core_max = int(require_rsc_size[i] * 1.2)
    elif args.test_case == "list":
        for i in range(N_task):
            task_list[i].parallel_mode = "list"
            task_list[i].core_list = [i for i in range(0, require_rsc_size[i], 4)]
            task_list[i].core_list.append(require_rsc_size[i])
            task_list[i].core_list.append(int(require_rsc_size[i] * 1.5))
    elif args.test_case == "lwb":
        for i in range(N_task):
            task_list[i].parallel_mode = "lwb"
            task_list[i].core_min = int(require_rsc_size[i] * 0.8)



    # create a scheduling table
    scheduling_table = SchedulingTableInt(30, 20)
    # allocate resources

    # alloc_info[0] = scheduling_table.insert_task(t1, 25, t1.ERT, t1.ddl, t1.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[1] = scheduling_table.insert_task(t2, 24, t2.ERT, t2.ddl, t2.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[2] = scheduling_table.insert_task(t3, 10, t3.ERT, t3.ddl, t3.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[3] = scheduling_table.insert_task(t4, 10, t4.ERT, t4.ddl, t4.exp_comp_t, verbose=True)
    # alloc_info[4] = scheduling_table.print_scheduling_table()
    # alloc_info[5] = scheduling_table.insert_task(t5, 10, t5.ERT, t5.ddl, t5.exp_comp_t, verbose=True)
    # # print the scheduling table
    # scheduling_table.print_scheduling_table()

    pid2name = []
    pid = 0
    for task in task_list[0:N_task]: 
        # for r, d in zip(task.get_release_event(event_range), task.get_deadline_event(event_range)):
        r = task.get_release_time()
        d = task.get_deadline_time()
        p = task.make_process(r, d, pid)
        pid += 1
        pid2name.append(p)

    for i in range(N_task):
        print(f"task {i} allocation\n")
        alloc_info[i] = scheduling_table.insert_task(pid2name[i], require_rsc_size[i], 
                                                     pid2name[i].release_time, pid2name[i].deadline, 
                                                     pid2name[i].exp_comp_t, verbose=True)
        print("="*20)
        scheduling_table.print_scheduling_table()
    
    print("occupy by id:", scheduling_table.index_occupy_by_id())
    
    # release resources
    for i in range(N_task):
        print("before release")
        scheduling_table.print_scheduling_table()
        if alloc_info[i] is not None:
            print(f"task {pid2name[i].pid}({i}) release:")
            print([*alloc_info[i]])
            scheduling_table.release(pid2name[i], *alloc_info[i][1:], verbose=True)
        print("after release")
        scheduling_table.print_scheduling_table()


if __name__ == "__main__":
    main()
