## End-to-End Latency Decomposition using Critical Path Analysis

To accurately attribute end-to-end latency in complex Directed Acyclic Graphs (DAGs) with parallel execution paths, we employ a critical path tracking algorithm. This method ensures that latency components (e.g., `compute`, `realloc`) are not double-counted at merge points, providing a decomposition where the sum of components equals the wall-clock critical path latency.

### Data Structures

For each task instance `i`, we maintain two primary state objects:
1.  **`task_curr_stat[i]`**: A dictionary storing the execution time (`compute`), reallocation overhead (`realloc`), and reallocation count (`realloc_num`) intrinsic to task `i` itself.
2.  **`task_pred_stat[i]`**: A nested dictionary mapping each direct predecessor `p` of task `i` to `p`'s final critical path statistics (`path_stat`).

The `path_stat` propagated for any completed task `j` is a dictionary:
```
{
  'compute': float,     // Critical path compute time from source to j
  'realloc': float,     // Critical path realloc overhead from source to j
  'realloc_num': int,   // Sum of reallocations on the critical path
  'e2e_lat': float      // Finish time of j relative to chain's start (offset)
}
```

### Propagation Logic

The algorithm proceeds as follows upon the completion of a task `i` at time `finish_t`:

1.  **Select Critical Path Predecessor**:
    - Retrieve the set of predecessor statistics `task_pred_stat[i]`.
    - Identify the critical predecessor `p*` by selecting the one with the maximum `e2e_lat` (latest finish time relative to offset).
    - If `i` is a source node (no predecessors), a zero-valued conceptual predecessor statistic is used, with its `e2e_lat` initialized to `start_time - offset`.

2.  **Calculate Path Statistics for `i`**:
    - The critical path statistics for `i`, denoted `path_stat[i]`, are computed by adding `i`'s intrinsic statistics (`task_curr_stat[i]`) to the statistics of its critical predecessor `p*`.
    
    \[
    \text{path\_stat}[i].\text{compute} = \text{path\_stat}[p^*].\text{compute} + \text{task\_curr\_stat}[i].\text{compute}
    \]
    \[
    \text{path\_stat}[i].\text{realloc} = \text{path\_stat}[p^*].\text{realloc} + \text{task\_curr\_stat}[i].\text{realloc}
    \]
    
    - The `e2e_lat` for `i` is its relative finish time:
    
    \[
    \text{path\_stat}[i].\text{e2e\_lat} = \text{finish\_t} - \text{offset}
    \]

3.  **Propagate to Successors**:
    - For each successor `s` of `i`, the calculated `path_stat[i]` is stored in `task_pred_stat[s][i]`. When successor `s` eventually completes, it will use this stored information to determine its own critical path among all its predecessors.

This recursive process ensures that at any merge point, only the longest path's accumulated latency is carried forward, correctly modeling the end-to-end critical path latency and its components.
