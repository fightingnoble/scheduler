## 20230323
msg_dispatcher.py
- add class MsgDispatcher, utilize queue as a mailbox similar to the multiprocessing queue; 

allocator_agent.py: 
- distinguish the local task list and the global process index; 
- distinguish the trigger signal of jobs in the task chain and data availability check; 
- check the data dependency by checking memory data availability; add mechanism of update & timeout for input dependencies; 
- now input data availability is checked before entering the ready stage; \n\tmessage broadcasting and syn mechanism; 
- sim the event trigger 

profiling.csv: 
- add trigger mode item; 

buffer.py: 
- divide the buffer into three parts, and add mux to select them, data put and get select the partitions according to the label-weight-input-output

task_agent.py
- add data dependency checking
- add trigger_mode supports
- add sim_trigger supports
- refine the dependency from the bool flag as a dict

task_cfg.py
- Added a new function creat_logical_graph to create a logical graph from srcs, ops, and sinks.
- Added a new function creat_physical_graph to create a physical graph from the logical graph and the profiling data.
- modify the init_depen, and graph plot function according to the above changes

change_log.md
- used for recording the changes 

golden_trace.txt
SOTA reference scheduling trace, considered as the right result.

## 20230325
- Clean up the code in allocator_agent.py
- Rename the affinity setting as affinirt_cfg, and beautify the layout of the task_graph plot.
- Extract the scheduler step from sched_step.
- Integrate scheduler_step to Scheduler
- Clean up the input parameters of cyclic_sched, sched_step
- move the test code of bin_pack and dynamic from task_cfg.py to allocator_agent.py
- move some functions into scheudler_agent.py

## 20230327
scheduling_table.py
- extract print_alloc_detail from push_task_into_bins.
- optimize p_name and time print feature in print_scheduling_table
resource_agent.py
- add pid2name conversion support.
Monitor_agent.py
- add trace_recoder to monitor
allocator_agent.py
- pass monitor into scheduler_step
create glb_sched.py

## 20230328
allocator_agent.py
- Plot the dynamic scheduling trace, recorded by monitor. 

monitor_agent.py scheduling_table.py
- add hyper_cycle, avoid slow down by pre-extend the size of the recorder of hyper_cycle each time. 

task_agent.py 
- add attibution curr_start_time to process

scheduler_agent.py
- Fix typos "descending"s!
- Fix bug: "alloc_core"s are not integers at runtime/add ceil operation and type checks.
- Fix bug: the budget is covered rather than replenish
    - We discuss this issue in two scenarios:
        1. with data arriving one time: allocate the resources according to the budget
        2. with data arriving late: allocate the resources following the "EDF", and estimate the resources at runtime
    - read planned_rsc_size from budget_recoder[_p.pid][1] other than curr_cfg.rsc_map[_p.pid].
- Add test case: simulate sensor data arrival time varies by injecting noise to self.task.period, self.task.i_offset

profiling.csv
- modifiy the type, 'soft' or 'hard' of some tasks.

test and update golden trace