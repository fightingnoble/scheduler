1. Buffer size is very large
2. Every generated results are broad casted to all downstream processes
3. lifetime defination:
    ```
            # metric_fn = lambda x: (x.event_time + x.life_time - curr_t, buffer.sort_fn(x))
            metric_fn = lambda x: (x.processed_done_time + x.life_time - curr_t, buffer.sort_fn(x))
    ```
4. event_time of operator is initialized as -1 and the data is inf
5.             # _p.event_time = min_event_time
            _p.event_time = max([matched_pair[key].ctx.get_timestamp() for key in matched_pair])
6. budget is response to pace the execution:
    provide load reference: rem_flop_budget
    provide resource reference: budget_recoder
TODO marks:
1. check again: has multiple choises, and one choise is adopted temporally.