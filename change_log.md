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
- move the test code from task_cfg.py to allocator_agent.py