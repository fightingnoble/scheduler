# Task Model Overview

This document provides an overview of the task model module (`task/`) in the scheduler simulator.

## Module Structure

```
task/
├── task_agent.py      # Core task and process classes
├── task_cfg.py        # Task configuration loading and graph generation
├── graph_scaling.py   # Node relationship scaling for parallelism
├── graph_breakdown.py # DAG decomposition into chains
├── spec.py            # Specification class for e2e latency
├── throughput_cnt.py  # Throughput counting utilities
├── depraved_ref.py    # Deprecated reference code
└── load_cfg/
    ├── loadA.py       # Default task graph configuration
    └── load_chain.py  # Chain-based load configuration
```

## Class Hierarchy

```
TaskAttr (dataclass)
    └── TaskIntAttr (dataclass)
            - flops, io_time, task_flag, pre_assigned_resource_flag
            - Derived: num_exec, no_stall_latency, min_tot_rsc, max_tot_rsc
            - Derived: flops_ModelSum, flops_ModelSumMax, equiv_core, util

TaskBase
    ├── timing properties (ERT, ddl, period, i_offset)
    ├── dependency tracking (pred_ctrl, succ_data, pred_data, succ_ctrl)
    ├── parallel config (core_max, core_min, core_list, parallel_mode)
    └── statistics (missed_deadline_count, completion_count, etc.)
    │
    └── TaskInt
            - flops, affinity, task_flag, pre_assigned_resource_flag
            - Resource allocation: allocated_resource, required_resource_size
            - pre_assigned_resource: DDL_reservation / RT_reservation

ProcessBase (runtime task instance)
    ├── Reference to parent task
    ├── Runtime state (release_time, deadline, pid, state)
    ├── Execution tracking (cpu_time, totcpu, remburst, cumulative_executed_time)
    ├── Dependency validation (pred_ctrl, pred_data with valid flags)
    ├── Context management (msg_cache for ContextMsg)
    └── Fork support (var_scale_factor, fork_pid_list)
    │
    └── ProcessInt
            - Resource allocation tracking (allocated_resource)
            - Resource estimation methods
            - Parallel constraints (core_max, core_min, core_list, parallel_mode)
```

## TaskState Enum

```python
class TaskState(Enum):
    terminated = 0  # Task finished
    suspend = 1     # Inactive, waiting for activation
    active = 2      # Active, not yet ready to execute
    running = 3     # Currently executing
    throttled = 4   # Ready but no budget
    wait = 5        # Waiting for I/O
    preempted = 6   # Preempted by other tasks
    ready = 7       # Ready to execute
```

## Task Lifecycle

```
                    ┌─────────────────────────────────────────────────┐
                    │                 Configuration                   │
                    │  load_taskattrib() / load_taskint()             │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │              Task Creation                      │
                    │  TaskInt() with timing, resource constraints    │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │           Frequency Division                    │
                    │  task.freq_division(factor, hyper_p, mode)      │
                    │  Creates sub-tasks for temporal partitioning    │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │            Process Generation                   │
                    │  task.make_process(release_t, deadline, pid)    │
                    │  ProcessInt created for simulation runtime      │
                    └─────────────────────┬───────────────────────────┘
                                          │
                                          ▼
     ┌────────────────────────────────────────────────────────────────────────┐
     │                         Process State Machine                          │
     │                                                                        │
     │   suspend ──(release)──► active ──(deps ready)──► ready ──(sched)──►  │
     │                                                                        │
     │                           running ◄───────────────────────────────────  │
     │                              │                                        │
     │                 ┌────────────┼────────────┐                          │
     │                 │            │            │                          │
     │                 ▼            ▼            ▼                          │
     │            terminated   preempted    throttled                       │
     │                 │            │            │                          │
     │                 │            └──────► ready (rescheduled)            │
     │                 │                       │                            │
     │                 └───────────────────────┘                            │
     └────────────────────────────────────────────────────────────────────────┘
```

## Configuration Loading Flow

```
gen_workloads(args)
    │
    ├── 1. load_taskattrib(profiling_filename, mode)
    │       ├── Read CSV profiling data
    │       ├── Create TaskIntAttr for each task
    │       └── Calculate derived properties (flops_ModelSum, equiv_core)
    │
    ├── 2. creat_logical_graph(srcs, ops, sinks, src_attr, sink_attr, taskattr_dict)
    │       ├── Create nx.DiGraph with nodes: src, op, sink
    │       ├── Add control edges (src -> op)
    │       ├── Add data edges (op -> op, op -> sink)
    │       └── Propagate chain_criticality from sinks
    │
    ├── 3. init_var_dist(args, logical_graph_nx)
    │       └── Initialize variation distributions for execution time
    │
    ├── 4. deduce_cfg2(taskattr_dict, logical_graph_nx, ...)
    │       ├── Calculate ERT (Earliest Release Time)
    │       ├── Calculate deadline per task
    │       └── Return rsc_map_w (resource mapping)
    │
    ├── 5. creat_physical_graph(logical_graph_nx, f_gcd, taskattr_dict, mode)
    │       ├── Apply thread_scaling_factor (spatial parallelism)
    │       ├── Apply freq_division_factor (temporal partitioning)
    │       ├── Build node relationships via build_node_relationship()
    │       └── Add edge attributes (reDistPattn: one2one/downscaling/upscaling)
    │
    ├── 6. gen_taskint_from_cfg(taskattr_dict, f_gcd)
    │       └── Create TaskInt instances from TaskIntAttr
    │
    ├── 7. init_depen(glb_n_task_dict, physical_graph_nx)
    │       ├── Initialize pred_ctrl, pred_data with valid=False
    │       ├── Initialize succ_ctrl, succ_data
    │       └── Add event_queue for dependency tracking
    │
    └── 8. create_init_p_list(glb_n_task_dict)
            └── Create initial ProcessInt instances
```

## Key Data Structures

### Task Graphs

| Graph | Type | Description |
|-------|------|-------------|
| Logical Graph | nx.DiGraph | DAG of operators with control/data edges |
| Physical Graph | nx.DiGraph | Expanded graph with thread/freq divisions |
| Job Graph | nx.DiGraph | Task instances for scheduling |

### Edge Types

| Type | Direction | Description |
|------|-----------|-------------|
| control | src -> op | Sensor trigger dependency |
| data | op -> op/sink | Data flow dependency |

### Redistribution Patterns (reDistPattn)

| Pattern | Condition | Description |
|---------|-----------|-------------|
| one2one | freq_A == freq_B | 1:1 data mapping |
| downscaling | freq_A > freq_B | Multiple inputs to one output |
| upscaling | freq_A < freq_B | One input to multiple outputs |

## Scheduler Interaction

### Resource Estimation (ProcessInt)

```python
# Quantile-based resource sizing
req_rsc_size, got_latency, got_constr = _p.rsc_req_estm_quantile(
    slack, FLOPS_PER_CORE, binpack_cfg, constr, max_size
)

# Time-slot based estimation
time_slot_s, time_slot_e, req_rsc_size = _p.rsc_req_estm(
    n_slot, timestep, FLOPS_PER_CORE, time_slot_s, time_slot_e, mode
)

# Apply parallel constraints
req_rsc_size, applied_constraint = _p.get_available_cfg(
    req_rsc_size, curr_aval_rsc
)
```

### Dependency Checking (ProcessBase)

```python
# Check if all predecessors completed
if _p.check_depends(event_cache, trigger_cache, event_triggers):
    # All dependencies satisfied, task can execute
    _p.build_ctx()  # Build context message
```

### State Transitions (ProcessBase)

```python
# Release: task becomes active
_p.release_util(curr_t, active_list)

# Ready: dependencies satisfied
_p.ready_util(curr_t, ready_queue)

# Throttle: budget exhausted
_p.throttle_util(throttle_list, curr_t, mode)
```

## Configuration Files

### Profiling CSV Format

| Column | Description |
|--------|-------------|
| Task (chain) names | Task identifier |
| Flops on path (G) | Compute operations (GFLOPs) |
| Freq. | Task frequency (Hz) |
| T release (ms) | Release time offset |
| DDL (ms) | Absolute deadline |
| Thread factor (Spat.) | Spatial parallelism copies |
| Throuput factor (Spat.) | Temporal division factor |
| Parallel_type | upb/lwb/Range/list |
| Parallel_range | Core constraint values |
| Timing_flag | DDL/realtime |
| Trigger_mode | Event/periodic |

### Load Configuration (load_cfg/)

```python
# loadA.py - Default task graph
task_graph_srcs = {
    "surr_view_camera_pub": ["ImageBB"],
    "IMU_pub": ["Steering_speed"],
}
task_graph_ops = {
    "ImageBB": ["MultiCameraFusion"],
    "MultiCameraFusion": ["Pure_camera_path_head"],
}
task_graph_sinks = {
    "Sink_control": [],
}
```

## Parallelism Scaling (graph_scaling.py)

The `build_node_relationship()` function handles data relationships when tasks are scaled:

```python
# Example: freq_A=3, freq_B=1, S_A=3, S_B=1
# Creates 3 instances of A, 1 instance of B
# Maps data based on frequency ratios
G = build_node_relationship(G, freq_A, freq_B, S_A, S_B, "A", "B", "interleave")
```

Modes:
- `interleave`: Round-robin data distribution
- `repeat`: Block-based data distribution

## Context Management

Each ProcessInt maintains `msg_cache` for ContextMsg objects:

```python
# Build context at dependency satisfaction
ctx = _p.build_ctx()

# Update with upstream data
_p.update_ctx("upstream", buffer=buffer, glb_n_task_dict=glb_n_task_dict)

# Update with trigger info
_p.update_ctx("trigger")
```

## Statistics Tracking

TaskBase tracks per-task statistics:
- `missed_deadline_count`: Deadline violations
- `completion_count`: Successful completions
- `cum_trunAroundTime`: Sum of turnaround times
- `context_switch_count`: Context switches
- `preemption_count`: Preemptions
- `migration_count`: Core migrations
- `throttle_count`: Throttling events
