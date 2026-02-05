# A Two-Phase Guided Hybrid Allocation Algorithm for Real-Time Systems

## 1. Abstract

This document describes a two-phase hybrid static-dynamic allocation algorithm, termed the "Guided Hybrid Allocator," designed to schedule real-time, end-to-end (E2E) task graphs on multi-core processors. The primary challenge lies in meeting strict E2E latency constraints under significant timing variability from both system-level jitter and workload-dependent execution times.

Purely static allocation schemes are often too pessimistic, leading to severe resource under-utilization. Conversely, purely dynamic schedulers offer high utilization but suffer from unpredictable overheads (e.g., kernel launch latencies) that can jeopardize real-time guarantees.

The Guided Hybrid Allocator addresses this trade-off by decoupling the allocation problem into two distinct phases:
1.  **Phase 1: Spatial Partitioning (Task-to-Bin Clustering):** Tasks are clustered into resource partitions (bins) based on graph topology and affinity. The resource capacity (number of cores) for each bin is sized conservatively using a high-quantile estimate of task execution times (\(q_A\)) to ensure sufficient processing power even under high variability.
2.  **Phase 2: Temporal Scheduling (Intra-Bin Time Window Assignment):** Within each bin, static time windows are assigned to tasks. These windows are calculated using a more aggressive, lower-quantile estimate of execution times (\(q_B\)). This allows for tighter scheduling.

This approach provides the predictability of static time slots to hide scheduling overheads, while the conservatively sized resource partitions offer a dynamic execution fallback to maintain robustness against timing variations, thus combining the benefits of both static and dynamic strategies.

## 2. Problem Formulation and Notations

We model the system as follows:

*   **Task Graph:** A directed acyclic graph (DAG) \(G = (V, E)\), where \(V\) is the set of tasks (nodes) and \(E\) is the set of communication dependencies (edges).
*   **Task (\(v_i \in V\)):** Each task is characterized by:
    *   A period \(T_i\).
    *   An end-to-end deadline \(D_i\) relative to its release.
    *   An execution time, which is a random variable \(C_i\). The q-th quantile of this execution time is denoted by \(C_i(q)\).
*   **End-to-End Chains (\(\tau_k\)):** A path through the graph from a source node to a sink node, representing an E2E dataflow pipeline. Each chain \(\tau_k\) has an overall latency constraint \(L_k\).
*   **Processing Resources:** A set of \(P\) homogeneous processing cores.
*   **Statistical Guarantee:** The objective is to find a schedule such that for each chain \(\tau_k\), the probability of its actual E2E latency \(L_{k,actual}\) exceeding its constraint \(L_k\) is below a specified threshold \(\epsilon\).
    \[
    P(L_{k,actual} > L_k) \le \epsilon
    \]

## 3. The Guided Hybrid Allocation Algorithm

The algorithm operates in two main phases, driven by two distinct quantile levels, \(q_A\) and \(q_B\), where \(0 \ll q_B < q_A < 1\). The level \(q_A\) represents a conservative ("almost worst-case") estimate, while \(q_B\) represents a more optimistic or typical-case estimate.

### Phase 1: Spatial Partitioning (Split)

This phase corresponds to the `coleasing_alloc_cluster` function in the implementation.

*   **Objective:** To group tasks into a set of \(N\) partitions or "bins" \(\{B_1, B_2, ..., B_N\}\). This mapping aims to co-locate tasks with high affinity or strong connectivity to minimize inter-partition dependencies and enable effective resource sharing.
*   **Mechanism:**
    1.  A graph clustering algorithm is applied to the task graph \(G\). The algorithm considers task dependencies and affinity constraints.
    2.  For each potential cluster (bin), the total required processing power is estimated. This estimation is based on the **conservative execution time quantile \(q_A\)** for each task. The required number of cores for a bin \(B_j\) is calculated to ensure that the collective workload of its assigned tasks, even at a high-percentile execution time, can be accommodated.
        \[
        \text{Cores}(B_j) = f\left(\sum_{v_i \in B_j} \frac{C_i(q_A)}{T_i}\right)
        \]
        where \(f\) is a resource allocation function (e.g., ceiling function).
*   **Output:**
    *   A task-to-bin mapping: \(M: V \to \{B_1, ..., B_N\}\).
    *   A list of bins, each with a determined number of processor cores. This is the `bin_list` in its initial state.

### Phase 2: Temporal Scheduling (Repack)

This phase corresponds to the `push_task_into_bins_new` function when operating in a pre-defined mapping mode.

*   **Objective:** Within each bin \(B_j\), assign a static time window (offset and duration) to each task \(v_i \in B_j\). This provides a predictable execution slot to hide system overheads like kernel launch latency.
*   **Mechanism:**
    1.  The algorithm takes the `bin_list` from Phase 1, which defines the partitions and their core counts.
    2.  For each bin, it performs scheduling. The time slots for tasks are calculated based on the **aggressive execution time quantile \(q_B\)**. This allows for a much denser packing of tasks within the timeline of the bin.
    3.  The slack (unallocated time) within each bin's hyper-period is then distributed among tasks to further enhance robustness.
*   **Execution Model:** The resulting schedule operates on a hybrid model:
    *   **Static Path (Happy Path):** If a task instance becomes ready and its assigned time window is active, it is dispatched immediately to a core within its bin. The launch overhead is effectively hidden because the system could prepare for this dispatch in advance.
    *   **Dynamic Fallback:** If a task instance misses its window (e.g., due to input data arriving late, or a preceding task running longer than its \(C_i(q_B)\) allocation), it is placed into a dynamic scheduling queue for its bin. It will then be scheduled on any available core within that bin. The key is that the bin was originally sized using the conservative \(q_A\), so there is inherent resource slack available to handle such overruns and deviations from the optimistic schedule.
*   **Output:** The final `bin_list`, where each bin contains a detailed static timeline of task allocations for a full hyper-period.

## 4. Rationale and Benefits

*   **Combines Predictability and Efficiency:** The static time windows (Phase 2) provide predictability and hide overheads. The dynamic fallback, enabled by the conservative resource sizing (Phase 1), provides efficiency and robustness to variation.
*   **Hierarchical Resource Management:** The problem is decoupled into a high-level spatial partitioning problem and a lower-level temporal scheduling problem. This reduces the complexity of the overall search space.
*   **Tunable Conservatism:** The two quantiles, \(q_A\) and \(q_B\), serve as explicit knobs for tuning the trade-off between resource reservation and scheduling efficiency. This is highly beneficial for ablation studies:
    *   **Split-Only (`only_split`):** Running only Phase 1 provides a purely cluster-based dynamic scheduling result.
    *   **Repack with Varying \(q_B\):** By fixing the result of Phase 1 and re-running Phase 2 with different values of \(q_B\), one can precisely evaluate the impact of temporal scheduling aggressiveness on E2E latency.

## 5. Implementation Mapping

*   **Algorithm Driver:** `sim_main.py`
    *   The `binpack_cfg["algorithm"] == "guided"` case orchestrates this entire flow.
    *   The `need_repack` flag, determined during workload generation based on whether \(q_A > q_B\), controls whether the flow proceeds to Phase 2.
*   **Phase 1 (Split):** Implemented in `sched.global_sched.coleasing_alloc_cluster`.
*   **Phase 2 (Repack):** Implemented in `sched.global_sched.push_task_into_bins_new`, using the `pre_defined` mapping mode.
*   **External Driver:** `scripts/repack_sweep.py` provides an example of how to conduct ablation studies by first running a "split" phase and then iteratively running "repack" phases for a list of different \(q_B\) values.

