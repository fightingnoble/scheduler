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

应对轻微抖动，轻微抖动可能导致少量slot的late，具体表现为，time budget 减少。相对于调整core数量而言，保留少量的（若干slot）时间上的冗余更加划算。具体表现为，exp_comp_t 略微高于 ops/算力，但是这种冗余反应为端到端的时间增长。那么就只能让一些任务的计算时间更短一些，同样是用资源换取时间。
对于那些只有一种配置的任务，或者已经在预分配阶段就达到上限的任务，只在时间上保留冗余。