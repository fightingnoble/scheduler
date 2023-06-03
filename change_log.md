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
  > and glb_dynamic method misses some tasks.
  >

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

## 20230417-pm14

document clustering,
    - model, sched
    - old, unused
    - task

## 20230418

Change the sensor simulation from periodic mode to event-driven mode.
    - message_handeler.py, scheduler_agent.py: new message_trigger handler;
    - Context_message.py: replace trigger_time by event_time ingestion_time
    - profiling.csv, task_agent.py: task trigger_mode setting.

## 20230419

add additional information, and NetworkX building for end-to-end analysis

## 20230425

task 的remburst==0的时候还剩下大量的rem_flops_budget

破案：完成前的最后一个slot，remburst有概率小于零。

```
    for _p in running_queue:
        if _p.remburst > 0:
            _p.rem_flop_budget[bin_id] -= res_cfg.rsc_map[_p.pid] * timestep * FLOPS_PER_CORE
```

## 20230429

Try to figure why each configuration executes different number of jobs:
    add job analyzer;
    generate trigger event globally (event_iter_dict, message_trigger_event_new @ allocator_agent.py);
    add folder path cheking to layout plot(get_task_layout_compact @ scheduling_table.py)

## 20230525

some file path, remove lifetime of the weight
mark the part that dyn_sched should debug

## 20230526

watermark strategy 写完:
    event_time of operator is initialized as -1 and the data is inf
    对于data trigger以及 event trigger两种方式：
    1. data trigger: 从最早的事件开始往后找，默认所有的数据周期相同
    2. event trigger: 从给定的event timestamp 开始往前找，以data的period为搜索步长（对于event 频率大于数据和小于数据的情况都适用）
    event time 现在在matched_pair中取最大的
graph_scaling: 修复scaling node 并行度的时候数据流不正确的问题
_p.event_time的设置时刻，执行完才设置。在此之前先缓存在ctx里面

## 20230601

错误进入active状态依然是问题（一方面导致rem_flops 多次启动出错，另一方面是导致生成多份ctx 在msg_cache中）
    为了防止生成多份ctx
    1. 仅允许msg_cache被设置一次：
    ``       if len(_p.msg_cache) == 0:             _p.build_ctx()   ``
    2. 保证其他队列执行任务时（新的任务不是继续中断的任务）基于新的ctx：
    （假设第一次执行在A上完成后会msg_cache.pop(0)，仅仅采用1，会导致其他队列中的任务，访问ctx的时候出现msg_cache为空的错误）
    a. 通过msg_dispatcher 发布消息，当任务完成的时候广播complete消息
    b. partition收到消息之后，过滤消息，确认active，ready，throttle队列中有目标任务之后移除这些任务

    为了防止rem_flops 多次启动出错``           if _p.remburst == 0:                 _p.remburst += _p.task.flops   ``
对于状态切换时，指示属性的设置进行修补：
    _p.released
    _p.ready
    _p.set_state

## 20230601 ++

```python
# if issue the task to runnning list
        for _p in issue_list:
            running_queue.put(_p)
            ready_queue.remove(_p)   # <- pop()
```

这个地方在glb_dyn 里面改了，但是在dynamic 里面却没有改

发现一个隐患，Taskqueue默认是降序排列，因此在顺序访问和实例化的时候应该注意顺序是否和预期匹配

```python
# detect the data trigger
        for key in _p.pred_data:
            stream = stream_dict[key]
            if len(stream)>0:
                # get the minimum event time
                if stream[0].ctx.get_timestamp() < min_event_time_t:
                    min_event_time_t = stream[0].ctx.get_timestamp()
                    period = stream[0].period
```

这段代码搜索最小值的时候，队列实际上是降序的，第一个是最大值

## 20230604
Issue: trigger 不能正常触发多个位于不用队列中的副本：

    ![trigger issue across multiple-queues](/home/zhangchg/git_repo/scheduler/doc/imgs/trigger_issue.png)

  Debug: trigger信号不能触发ctx切换的副本？为什么广播data可以？

    对data 做了广播，但是却共享了消息队列，每个副本都将收到的msg放到了该队列里面（也就是说，一个队列里一个消息实际上有好多份）
    trigger 没有广播，也没有复制，所以只有最早满足条件的可以进入激活状态
  
  方案：解决在每个partition中单独设置，dependency的queue
