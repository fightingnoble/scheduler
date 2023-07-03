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
    ``    if len(_p.msg_cache) == 0:             _p.build_ctx()``
    2. 保证其他队列执行任务时（新的任务不是继续中断的任务）基于新的ctx：
    （假设第一次执行在A上完成后会msg_cache.pop(0)，仅仅采用1，会导致其他队列中的任务，访问ctx的时候出现msg_cache为空的错误）
    a. 通过msg_dispatcher 发布消息，当任务完成的时候广播complete消息
    b. partition收到消息之后，过滤消息，确认active，ready，throttle队列中有目标任务之后移除这些任务

    为了防止rem_flops 多次启动出错``        if _p.remburst == 0:                 _p.remburst += _p.task.flops``
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

## 20230605

fix bug: chk_release input parameters miss match

```
bin_event_flg = WatermarkStrategy.chk_release(curr_t, inactive_list, active_list, bin_event_flg, bin_name)
```

## 20230606

fix bug:

1. glb_sched has no output

```
    def sim_trigger(self, time=None, time_step=1e-6, pred_ctrl:Dict[int, Dict]=None, event_triggers:List[Tuple]=None):
        """
        if the task is activated, return True
        """
        ...
        if event_triggers is None:
            event_triggers = self.event_triggers
```

glb_sched: Planning_2 is matched in two queues:
a lost fix of the bug commited at 20230601 ++

> 发现一个隐患，Taskqueue默认是降序排列，因此在顺序访问和实例化的时候应该注意顺序是否和预期匹配
> 这段代码搜索最小值的时候，队列实际上是降序的，第一个是最大值

## 20230607

detect the lateness of the current chunk of the task rather than the whole task:

- use chunk_s to replace _p.release_time,
- use _p.rem_flop_budget to replace _p.totburst

fix scheduler trigger condition and reallocation condition:
The cases are classified in to 5 condition
preemption, ressignment-in-turn (o3/order), curveup, replenishment
TODO: refine the monitor condition, (monitor the ready tasks and the head of the queue)

add cfg constraint to the dynamic scheduler, (TODO: add constraints to the static global scheduler)

fix issue:
the constraints used in timeline generation and rectifying at scheduling table generation is not aligned

## 20230608

原因在于前面的任务出现miss，没有想complete一样将其他分区内激活的备份踢出

    ![trigger issue across multiple-queues2](/home/zhangchg/git_repo/scheduler/doc/imgs/multi-copies_run.png)

解决了拐角的问题：

    ![L-shape-space](/home/zhangchg/git_repo/scheduler/doc/imgs/L-shape-space.png)

## 20230609

有一些任务在加了20% 扰动之后会出现miss：
如果资源分配的时候出现了starving的情况 那么一定会late; hard deadline一定会miss，后续任务也无法执行。
1. 取消了所有任务hard deadline约束，只限制最后一个任务为hard ddl，control，traffic light
2. 现在soft miss 标记为 violate constraint
TODO:探索task之间的slack sharing
related files: profiling.csv, allocator_agent.py, scheduler_agent.py

优化了整体的测试流程：
related files: allocation.py, one_click.py

The layout of the cores are optimized, but still unsatisfied!!
scheduling_table.py, assumptions.md

Fix bug: 
1. chunk_e + 1 ---> chunk_e
```python
if chunk_s < n_slot < chunk_e:
    # case 1: newest assigned budget is still available                    
    #   tries to finish the remaining work assigned by the configuration chunk until the now
    assert chunk_e == curr_cfg.slot_e + 1
    planned_flops = sum(_p.rem_flop_budget.values())
    req_rsc_size = math.ceil(planned_flops/(chunk_e + 1 - n_slot)/timestep /FLOPS_PER_CORE) 
```
2. add a condition: previous chunk is late, and newest assigned budget is still available but not enough
3. add error tolerance to the flop comparison
```python
planned_flops = sum([v for v in _p.rem_flop_budget.values() if v > flop_error_tol_abs])
	.....
else:
    if round(planned_flops, flop_error_tol_bit) > round(chunk_flops, flop_error_tol_bit):
        # case 2: previous chunk is late
        #   newest assigned budget is not enough
        #   newest assigned budget is still available but not enough
        req_rsc_size = math.ceil(planned_flops/(chunk_e + 1 - n_slot)/timestep /FLOPS_PER_CORE)
    else:
        # newest assigned budget is still available                    
        # tries to finish the remaining work assigned by the configuration chunk until the now
        req_rsc_size = chunk_alloc
```
基本上调通了，在< =257的情况里存在planning 预设3个core，超过了runtime ==2的约束
TODO: 1. 优化core的temporal分配，2. 优化task的layout (spatial)


## 20230614
Apply the size constraint based on parallelism cfg files : 
  Rewrite insert_task in SchedulingTable class
    测试了无约束情况，和upb和list两种类约束
    测试完task_insert的所有case，画了示意图保留了测试结果
    Related files: scheduling_table.py, doc/verification/bin_insert_demo.csv, doc/verification/bin_insert_demo.drawio
  add method for parallel constraints checking in TaskInt class
    Related files: task_agent.py


调整表格：
  更正了利用率的计算方式，等价core，
  更正了组合路径的计算公式，MAX 或者 SUM
  添加了Parallel type Parallel range
  校准Tread 和 throughtput的分解
  标明了列的单位，在代码中更新了相关的索引名称
  related files: task_cfg.py, 

更新slack预分配机制：
  slcak 分为spatial slack 和 temporal slack
    - spatial slack: 任务的spatial slack是指任务的最大可用资源和最小可用资源之间的差值（为一些关键任务预留一些冗余资源，这些的任务对冗余资源保留最高优先级的使用权）
    - temporal slack: 任务的temporal slack是指任务的最大可用时间和最小可用时间之间的差值, 给一些任务预分配更多的资源，以得到跟多的时间slack，然后把这些slack均分给所有的任务以应对调度导致的轻微的抖动（若干个slot）
  根据task graph 和预设的slack比例，计算出每个任务的ERT 和相对的ddl（redist_ert_dll， estim_release_dll_time）
  related files: scheduling_table.py, assumptions.md, task_cfg.py, pre_alloc.py

  fix a bug that the bin_list can not be plotted correctly during the debug process. 
  ```
  def push_task_into_bins_new(
        bin_list: List[SchedulingTableInt], 
      ......
  ```
  ```
        bin_list.clear()
        bin_list = push_task_into_bins_new(
            bin_list,
            ......
  ```
  related files: allocator_agent.py, pre_alloc.py

  更新了batch脚本的默认值

## 20230615
fix bug: some tasks are lost after preemption, and the scheduling table can not converge as the iteration number increases
  The process moved to preempt_list are not put back to the ready_queue
  update scheudling mechanism: 
    - Partial allocation is not allowed, i.e., the task is allocated to the whole cores or none.
    - Each task only try once; the tasks already allocated are skipped;
    - the tasks are preempted are given an extra opportunities.
   TODO: Allow partial allocation and add the logic to ensure the task have allocated enough resource, otherwise, we should not issue the task or compensate the resource latter. 

## 20220619
1. extract affinity initialization into a function: init_affinity
```python
    # initialize task affinity list
    thread_n = int(exe_k)
    affinity_tgt_n_list = affinity_cfg[task_n]        
    affinity_tgt_n_list = [n+'_'+str(thread_n) for n in affinity_tgt_n_list]
    task.affinity_n = affinity_tgt_n_list
    # initialize dependency list
```
related files: task_cfg.py

2. extend process name with thread parallel number
```
  for exe_k in range(task_attr["Throuput factor (Spat.)"]):
  ......
  node_name = node_n+"_"+str(copy_j)+"_"+str(exe_k)
```
related files: task_cfg.py

3. Un-finished: when estimating the ddl and ert, tackle the chain with mixed critical level tasks seperatly. 
related files: task_cfg.py

4. add a extra dispatch mode for graph node scalling:
   根据上下游节点的频率和并行度讨论图变换的模式。
   对于上游频率高于下游的情况：采用间隔均匀采点模式（等间距分割）
   下游高于上游情况：目前提供两种模式接口（interleave，repea），但是采用的是等间距分割模式
```python
    # j == i+freq_A*t
    assert dispatch_mode in ['interleave', 'repeat']
    if dispatch_mode == 'interleave':
        if A_data_idx == B_data_idx % freq_A:
            G.add_edge(A_node, B_node)
    elif dispatch_mode == 'repeat':
        if A_data_idx == int(B_data_idx / freq_B * freq_A):
            G.add_edge(A_node, B_node)
```
related filele:;  graph_scaling.py

5. failed task with enough preemption candidate will directly pop out the searching progress.
```python
  if state: 
      break
  elif fail_info is not None:
      if strategy == "first_fit": 
          break
      elif strategy == "best_fit":
          fail_info_list.append(fail_info)
```

6. optimize the log output and task_graph plot
related files: log_analyse.py, task_cfg.py, wartermark_strategy.py, scheduler_agent.py, message_handler.py

7. solve a bug: task with same priority compete for the same resource in an endless loop
    add an addtional priority level
    ```python
        score_fn = lambda x: (fn_crit(x), fn_task_flag(x), x not in preemptable_list)
        threshold_score = (np.inf, 1, True)
        filtered_ready_queue = [_p for _p in ready_queue.queue if score_fn(_p) < threshold_score]
        sorted_queue = sorted(filtered_ready_queue+preemptable_list, key=score_fn,)
    ```
related files: scheduler_agent.py

8. code clean up
related files: scheduling_table.py, task_agent.py

9. fix bug: when push or pop tasks from issue_list, rsc_recoder report key error
    adjust the position of the push and pop operation
    ```python
    rsc_recoder.pop(_p_2b_preempt.pid)
    ```
    fix bug: wrong condition for partial preemption (illegal fail_info is not cleaned), the condition should be:
    ```python
    if not partial_preempt_en or timestep*FLOPS_PER_CORE - total_FLOPS_occupied > 1e-2*timestep*FLOPS_PER_CORE:
    fail_info = None
    ```
10. scale thread of the aux task 4x, change spatial tread parallelism of the steering control as 24x

## 20230623 
 optimize the shell script for batch run, add option to select the config file
 related files: run/static_optm_test.sh, compare_dyn.sh, run/compare_glb.sh, run/one_click.sh, allocator_agent.py

## 20230625
 1. integrate calculation of hyper_p in to load_taskint;
 2. The sensor event generator was modified as well as the deadline update logic
    related files: task_agent.py, scheduler_agent.py, wartermark_strategy.py
 3. The timeline drawing was modified: the release time and the deadline is represented by arrows
 4. A heavy set of tasks was added
    related files: task_cfg.py, allocator_agent.py, profiling.csv, profiling_medium.csv

## 20230628
fix bug: 
  the copy triggered on the unintetional partition is reported missing deadline, 
  even through it is already finished on the target partition. 
  resean: 
    the "compelete" message sent to the msg queue is not handled immediately.
  solution:
    extract read_msg_queue into a function, 
    and move the msg queue reading logic before the completeness and deadline checking 
  related files: scheduler_agent.py

regulate the cache and trace saving path 
  related files: allocator_agent.py

add lateness_mode option to assert all the process as hard/soft/mixed deadline
  related files: allocator_agent.py

## 20230629
slack and resources estimation performed by excel tool now is integrated into the code
  graph_breakdown decompose the graph into chains
  slack_estim distribute the slack to the nodes in the chain, and estimate the resource requirement
  an additional parallelism constraint used at compile time is add to the profiling csv files, 
  read by task_cfg.py and set as the property of the task
  related files: task_cfg.py, task_agent.py, slack_estim.py, graph_breakdown.py

## 20230702
add some interface for parameterized workload generation
  related files: task_cfg.py, task_agent.py
  use jitter_max to represent the max percentage of the jitter
  improve the __str__ function of the task class
  Crate new data type decorated by @dataclass: 
    record the task properties that are not related to the task graph transformation.
    The manual edit Excel + load_taskint -> gen_workloads(i.e., load_taskattrib + duduce_cfg + gen_taskint_from_cfg)
  fix bug:
    some properties are set after the task graph transformation, which results in the absence of the properties in the tasks generated during the transformation.  
  move estim_release_dll_time to slack_estim.py

Statistic the e2e latency from the trace files:
  the e2e latency analysis now adhere to the setting of the timestamp matching: 
    the event time is equal to the max value of the matched timestamps.
    The e2e latency is calculated sensor by sensor; 
    every path from the sink node to the source node is considered as an e2e latency sample.
    sink node: node generated fedback to event at this moment
    source node: where the sensor data is generated.
    draw the e2e latency distribution of the sensor data

Add a coroutine to statistic the throughput.
  related file: throughput_cnt.py

## 20230703

optimize the code layout
related file: allocator_agent.py

Benchmark generation and comparison with the manually generated benchmark is completed
  fix bug: 
    the only the realtime task (aux task) is scaled, filtered by the timing_flag attribute
  ```python
      if taskattr.timing_flag == "realtime":
          taskattr.thread_scaling_factor *= args.aux_scale_factor
  ```
Report: doc/heavy_scaling.md, doc/medium_scaling.md, slack_estim.py

Tested benchmark generation and simulation is completed. 

  Gen_workloads function is integrated with workload generation and graph transformation
    creat_physical_graph now support to generate the physical graph either from the taskattr_dict or from the cfg file
    related file: task_cfg.py
  optimize the shell:
    Change the way to execute the shell script:
      sh xxx -> ./xxx
    Add p_fn and PY_ARGS to the python script:
      pass more arguments to the python script from the shell script
      support slect the main file of the python script
    ```
    p_fn=${5:-"allocator_agent.py"}
    PY_ARGS=${@:6}
    ```
    Select cfg dir:
      Generated ver: 
        The cfg dir is parsed by the python script, and then passed to the shell script
      Predefined ver:
        The the cfg files are selected by keywords, light/medium/heavy
    related file: run/one_click_gen_bm.sh, run/one_click_fx_bm.sh, sim_main.py
      compare_dyn.sh, compare_glb.sh, static_optm_test.sh
  Optimize the log cleaning: replace shell cmd with python script
    related file: file_path_prepare.py

