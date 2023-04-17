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
- add hyper_cycle, avoid slowing down by pre-extend the size of the recorder of hyper_cycle each time. 

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
- modify the type, 'soft' or 'hard' of some tasks.

test and update the golden trace

## 20230404
- replace `np.allclose` by `math.isclose`;
- fix typos.
global_sched.py, task_cfg.py

add parent folder to log, cache, and plot
- task_cfg.py, allocator_agent.py

add "_{num_cores}" suffix to saved files
- allocator_agent.py

scheduling_table.py 
- explore plotting with Bokeh and Plotly.

scheduler_agent.py, allocator_agent.py, barrier.py monitor.py
- add support to the global dynamic scheduler in Planaria. 
- a cyclic simulation, a glb_dynamic_step function;
- add a placeholder when the resources are occupied by the re-allocation procedure.

global_var.py
- try to add some parameters

## 20230406
fix bug: 
- steering is triggered too frequently; 
- in glb_dyn scheduler, 
    - the required cores are 0.
    - the preemptable checking is unused

- parameterize some test options

## 20230410
- add the parallel constraint
    - resources allocation with parallel constraints (scheduler_agent.py) 
    - profiling.csv define the constraints,
    - task_cfg.py read the constraints.
    - add core_max, core_min, core_list, parallel_mode to task and process classes
- move function `rsc_req_estm` as an element of ProcessInt class

- fix some small issues for bin size exploring in the next step.

## 20230412

- fix bug: 
 - `lru` losses its capacity after `withdraw` operation
 - `parallel_cfg:dict` default value changes from None -> {}
 - pop empty `lru` after withdrawing
 - distinguish the suffix of input files and output files

- print info:
    - title line printing (resource_agent.py, scheduling_agent.py)
    - filter some unimportant messages (scheduling_agent.py)

- new bin packing mapper:
    - pre-alloc.py
    - global_sched.py

- allocator_agent.py
    - new parameter n_p

- task_cfg.py
    - remove close loop in affinity cfg


## 20230413

- extract message processing functions
 - message_handler.py
 - scheduler_agent.py 

- barrier block time statistic
    - scheduler_agent.py 

- test glb_dynamic dynamic bin_pack with
    > 256 core, -0.2=a, 0.2=b, 0.6=loc and 1=var 
    and glb_dynamic method misses some tasks.


## 20230417
integrate context into the stream flow
    - trace_example.py: example of generated trace log
    - task_agent.py: support of extracting the context
    - Context_message.py: The context class, a `dict` class, marks the processor, sensor, 
        and data node, and the properties of these nodes. 
    - buffer.py: integrate the context into the Data. 
    - data_pipe.py: simulate the communication backbone. 
    - scheduler_agent.py, allocator_agent.py: algorithm
    - remove the control message path, and move the handler to msg_dispatcher.py
    - the streaming context trace is saved to trace_list and trace_file, 
        which is defined in global_var.py.

Fix bug:
    - task_cfg.py, clb_sched.py: end time error, curr_t * timestep

New feature:
    - clear() in TaskQueue.