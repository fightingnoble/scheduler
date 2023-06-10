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

