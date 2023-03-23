## 20230323
msg_dispatcher.py
- add class MsgDispatcher, utilize queue as a mailbox similar to the multiprocessing queue; 

allocator_agent.py: 
- distinguish the local task list and the global process index; 
- distinguish the trigger signal of jobs in task chain and data availability check; 
- check the data dependency by checking memory data availability; add mechanism of update & timeout for input dependencies; 
- now input data availability is checked before entering ready stage; \n\tmessage broadcasting and syn machinism; 
- sim the event trigger 

profiling.csv: 
- add trigger mode item; 

buffer.py: 
- divide the buffer into three parts, and add mux to select them, data put and get select the partitions according to the label-weight-input-output

task_agent.py
- add data dependency checking
- add trigger_mode supports
- add sim_trigger supports
- refine the dependency from bool flag as a dict

task_cfg.py
- Added a new function creat_logical_graph to create a logical graph from srcs, ops, and sinks.
- Added a new function creat_physical_graph to create a physical graph from the logical graph and the profiling data.
- modify the init_depen, and graph plot function according to the above changes

change_log.md
- used for recording the changes 

golden_trace.txt
- SOTA referece scheduling trace, considered as a right result.

