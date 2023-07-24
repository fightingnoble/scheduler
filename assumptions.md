## parameters ranges
等价算力：200
最稳妥的cores：315
最少：可能303

## Basic Assumptions
1. Buffer size is very large
2. Every generated results are broad casted to all downstream processes
3. lifetime defination:
   ```
           # metric_fn = lambda x: (x.event_time + x.life_time - curr_t, buffer.sort_fn(x))
           metric_fn = lambda x: (x.processed_done_time + x.life_time - curr_t, buffer.sort_fn(x))
   ```
4. event_time of operator is initialized as -1 and the data is inf
5. # _p.event_time = min_event_time

   _p.event_time = max([matched_pair[key].ctx.get_timestamp() for key in matched_pair])
6. budget is response to pace the execution:
   provide load reference: rem_flop_budget
   provide resource reference: budget_recoder
   TODO marks:
7. check again: has multiple choises, and one choise is adopted temporally.

指示属性：
_p.release_time = curr_t
_p.released = True
_p.ready_time = curr_t
_p.ready = True
_p.set_state("ready")

_p.is_starving


preemption: (new task entering the ready queue) a job is suspend and the resource is taken over by other task(s);

ressignment-in-turn: (new task entering the ready queue and ready queue is not empty and old task free cores) old job(s) finished and new job(s) take over the free resource;

curveup: (free cores exist and No other new tasks can colocate with the current running  tasks) free resources are taken over by the running tasks greedly, even through they can catch up their deadlines;

replenishment: (free cores exist and some task are starving) free resources are replenished to the starving running tasks.

8. layout & placement

Currently, we only consider 1D layout, with a huristic algorithm: 
reallocating the position from the original base position, i.e., cum_pos, 
looking left and right, and select the leftmost position from left_pos, then, rightmost position from right_pos. 
the task decrease the size is handled at first. 

9. preallocation

这段代码是一个插入式调度算法，用于为每个任务分配足够的核心资源。它使用了一个自定义的贪心的装箱算法（Naive Bin Packing Algorithm）。

该代码块中的insert_task函数接受一个任务（task）以及所需的资源大小（req_rsc_size），时间槽的起始和结束时间（time_slot_s和time_slot_e），以及期望的时间槽数量（expected_slot_num）。函数的目标是将任务插入到调度表中的合适时间间隔中（使用First-Fit策略），并分配足够的资源。

首先，函数检查是否存在足够的可用资源来满足任务需求，并且时间槽的数量足够。如果满足条件，则分配资源并返回成功标志（True）、开始时间槽（time_slot_s）、分配的资源大小（req_rsc_size）和分配的时间槽数量（expected_slot_num）。

如果无法直接满足任务的需求，代码会将可用资源划分为不同的间隔（interval）。然后，它会遍历每个间隔，检查是否满足任务的需求条件。如果找到一个间隔，满足时间槽数量要求且具有足够的资源，那么就在该间隔中分配资源，并返回成功标志、分配的开始时间槽、分配的资源大小和分配的时间槽数量。

如果无法找到合适的间隔来满足任务的需求，代码将会对间隔进行重新分配。重新分配的原则是尽快满足任务的需求。代码会计算当前的资源分配情况（C）、可用资源（A）和任务需求（R）。如果可用资源不足以满足任务的需求，代码会尽可能地将资源分配到间隔中，直到达到任务需求。如果可用资源足够满足任务需求，代码会尽量将资源均匀地分配到间隔中。

最后，代码会根据重新分配的结果，分配资源给任务，并返回成功标志、分配的开始时间槽、分配的资源大小和分配的时间槽数量。

首先过滤掉资源数量小于core_min的位置，这样后面只需要考虑，core_max 和 core_list 的约束

slcak 分为spatial slack 和 temporal slack
   - spatial slack: 任务的spatial slack是指任务的最大可用资源和最小可用资源之间的差值（为一些关键任务预留一些冗余资源，这些的任务对冗余资源保留最高优先级的使用权）
   - temporal slack: 任务的temporal slack是指任务的最大可用时间和最小可用时间之间的差值, 给一些任务预分配更多的资源，以得到跟多的时间slack，然后把这些slack均分给所有的任务以应对调度导致的轻微的抖动（若干个slot）
在运行时，应对轻微抖动可能导致少量slot的late，具体表现为，time budget 减少。面对少量且频繁抖动，相对于实时的调整core的分配方案，在预分配阶段在时间上保留少量的的冗余（若干slot）显得更加划算——具体表现为：
exp_comp_t 略微高于 （ops/分配的算力），但是这种冗余反应为端到端的时间增长。那么就只能让一些任务的计算时间更短一些，同样是用资源换取时间。
对于那些只有一种配置的任务，或者已经在预分配阶段就达到上限的任务，只在时间上保留冗余。
在pre-allocation 阶段，ERT作为开始时间的下限，我们取release time作为下游任务mapping的最早时间，
这样下有任务的mapping结果取决于上游任务的mapping效果。
timestamp+ERT+DDL则作为下游任务的最晚时间，以约束上游任务的最差mapping结果。

realtime 不用给冗余 现在都留了
```python
            ddl[node] = ert[node] + comp_time[node] *1e7 / (1 - temporal_rda_ratio)/ 1e7
```
mechanism: 
   Partial allocation is not allowed, i.e., the task is allocated to the whole cores or none.
   Each task only try once; the tasks already allocated are skipped;
   the tasks are preempted are given an extra opportunities.
   TODO: Allow partial allocation and add the logic to ensure the task have allocated enough resource, otherwise, we should not issue the task or compensate the resource latter. 

sort the bin according to the feature
   - free: slot_s, avil_unit, preemption: slot_s, avil_unit


10. compared items
A. Context switching
 - planned 
 - preempted

B. deadline assignment 
   - shared
   - fixed

11. graph scalling
   根据上下游节点的频率和并行度讨论图变换的模式。
   对于上游频率高于下游的情况：采用间隔均匀采点模式（等间距分割）
   下游高于上游情况：目前提供两种模式接口（interleave，repea），但是采用的是等间距分割模式

12. exp_comp_t
   this property is now only used for calculating sim_step, injecting jitter
   now, we use the estimated slack as the exp_comp_t, which is smaller than relative deadline, i.e., ddl.
   
13. 修改update_ctx 使 event_time, e2e_ddl, dyn_obj_num 随着数据流更新
  当多个流汇聚的时候应该选择event_time更新对应的那个流，也就是event_time最迟的流
  对于event_trigger的任务，以上三个属性应该在cache_trigger时候更新
  否则在cache_upstream的时候更新

14. workload variation infomation format：
   {
      "var_item":{
         {
            "src_name": ["task_name1", "task_name2", ...]
            "tgt_name": ["task_nameA", "task_nameB", ...]
            "typical":10, 
            "maxsize":30,
            "period":100,
         }
      }
   }
   The packet injected to the ctx is in the following format:
   {
      var_item: {
            "typical": var_param["typical"],
            "tgt_name": var_param["tgt_name"],
            "size": dyn_obj_num
      }
   }
15. handler should also determine the adjustment of the scheduling table:
   1. e2e var: 
      1. for deadline-driven task, 
         a. e2e latency requirement is redistributed to each tasks:
            DDL is recomputed in proportion to the e2e_var, 
            and the ERT is also updated along each chains in a cascading manner.
         b. Budget: the trunk of the tasks should 

16. In our scheduling algorithm, 
      we should confirm that in which condition the context switching overhead can / cannot be ignored:  
      1. if when and which the next task is executed can not be known in advance, 
         the context switching overhead cannot be ignored.
      Case: 
      1. planned context switching:
         the task initialization can be performed in advance, i.e., 
         transfering the weight and initialize other states, before the previous task is finished.
         In this case, the context switching overhead can be ignored.
      2. preempted context switching:
         As the resources allocation is not performed until the previous task is finished, 
         and all the states should be saved and reinitialized, 
         as soon as the new allocation scheme is determined.
         In this case, the context switching overhead cannot be ignored.

17. overhead of our method: 
   1. fragment cores in each partition, but the switch overhead in each partition is smaller than global scheduling.
   2. as the number unexpected context switching increases, 
      the overhead of the context switching increases.


18. load_var and thread fork

  properties:
    ```
      self.n_fork = 0
      self.fork_pid_list = []
      self.fork_pid_candi = [] 
      self.is_fork_inst = False
      self.parent_pid = None
      self.var_scale_factor = 1
      self.load_var = None  
      self.fork_p_inst = []

    ```
    `fork_pid_base = 1000`
    
   inject ctx from in message_trigger_event_new
   init process pool at init time (task_cfg.py)
   extract workload variation from the context (handle_process_load_var)

   在ready 之前fork 需要考虑fork的进程的依赖，以及budget的设置
   在ready设置，需要考虑进程的throttle以及恢复

   copy budget for the forked process
   fork the task (p_fork) @ ready:
      copy the process, rename the process and change the process id
      set the `parent_pid, is_fork_inst, pid` for the new process
      set `n_fork, new_pid, fork_pid_list` for the parent process
   add process index to process_dict
      
   set ready state for the forked process

   check complete:
      both `n_fork == 0` and totburst
      sorted(running_queue.queue, key=lambda x: x.is_fork_inst, reverse=True) check the forked process first

   terminate forked process (kill_fork) @ miss and complete:
      set the property of process with `parent_pid`
      append pid to fork_pid_candi
      remove the pid from fork_pid_list
      minus n_fork by 1
      delete the process
      

parameter scan
   e2e_var_sim_en
   jitter_sim_en
   load_var_sim_en
   e2e_var_sim_en
   seed, 
   aux_scale_factor
   e2e_latency
   


Hardware paremeters:
   num_cores

event_parameters:
   jitter_sim_en
   jitter_sim_para
   load_var_sim_en
   load_var_sim_para
   load_var_para_file
   e2e_var_sim_en
   e2e_var_sim_para

Simulation parameters:
   seed
   warmup_dis
   n_p

Benchmark params
   aux_scale_factor
   e2e_latency
   profiling_filename

Scheduler parameters
   wsc_slack_ratio
   temporal_rda_ratio
   lateness_mode
   temporal_rda_ratio
   test_case
   barrier_dis