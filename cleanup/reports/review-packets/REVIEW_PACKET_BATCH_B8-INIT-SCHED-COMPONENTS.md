# REVIEW_PACKET_BATCH_B8-INIT-SCHED-COMPONENTS

Batch: `B8-INIT-SCHED-COMPONENTS` — extract `approach_setup.py` step 4-5 (sched-component init) into a reusable function `init_sched_components` returning an execution handle (closure + function attributes).

## Background & decision (2026-07-01)

- B7 investigation: `approach_setup:36-61` unconditionally creates 5 legacy runtime components (Scheduler/Monitor/msg_dispatcher/a_data_pipe/w_data_pipe) + sim env, passed to perform_bin_packing.
- User prior: these are the **shared components of the unified global_sched method interface** (some methods simulation-based, some input-based; interface unified; this property kept intact). Not meaningless legacy.
- Decision: encapsulate step 4-5 init as a reusable function returning an **execution handle** (avoid caller unpacking ~10 component variables + passing 20 args to perform_bin_packing).
- Handle form: **closure + function attributes** (option A, user-confirmed). `pack(...)` executes perform_bin_packing; `pack.sim_step`/`pack.num_periods` expose env constants for step 9 print/plot.

## Design

### `init_sched_components(args, path_params, path_ctx, workload, num_cores, bin_list)` → pack handle

```python
def init_sched_components(args, path_params, path_ctx, workload, num_cores, bin_list):
    """初始化 global_sched 系列方法的公用组件 + 仿真环境，返回执行句柄。
    调用句柄即执行 perform_bin_packing；句柄属性 .sim_step/.num_periods 供打印/绘制。"""
    # step 4
    sched_elements = create_scheduler_elements_with_config(args, path_params, path_ctx, workload, num_cores, bin_list)
    _, _, msg_dispatcher, a_data_pipe, w_data_pipe, \
        scheduler_list, monitor_list, _, sim_step = sched_elements
    # step 5
    num_periods, _, quantumSize, _, event_iter_dict = build_simulation_env(args, workload, sim_step)
    cfg_para_dict, para_scan_group1, _, path_para_dict, _, _, plot_path_para, _, _ = path_params

    def pack(glb_p_list, bin_list, hyper_p, physical_graph_nx, need_repack):
        return perform_bin_packing(
            args, glb_p_list, num_cores, bin_list, hyper_p,
            sim_step, path_para_dict, para_scan_group1,
            event_iter_dict, quantumSize, num_periods,
            cfg_para_dict, physical_graph_nx, need_repack,
            plot_path_para, path_ctx,
            scheduler_list, monitor_list, msg_dispatcher, a_data_pipe, w_data_pipe)
    pack.sim_step = sim_step
    pack.num_periods = num_periods
    return pack
```

Binding: 15 components/params captured in closure; caller passes only 5 scheduling-specific args (`glb_p_list, bin_list, hyper_p, physical_graph_nx, need_repack`).
Discarded (unused downstream): task_spec, rsc_list, trace_path, warmup, event_range.

### `run_benchmark_setup_pipeline` refactor (approach_setup.py)

step 4-5 (~15 lines + unpacks) → `pack = init_sched_components(...)`; step 6 (20-arg call) → `pack(glb_p_list, bin_list, hyper_p, physical_graph_nx, need_repack)`; step 9 uses `pack.sim_step`/`pack.num_periods`. path_params unpack retained for step 8/9. Pure extract+recompose, **no logic change**.

## 1. MUST REVIEW

### B8-EXTRACT — add init_sched_components + refactor run_benchmark_setup_pipeline

```text
NEW  sim_main.py::init_sched_components  (closure handle, ~20 lines)
EDIT approach_setup.py::run_benchmark_setup_pipeline  (step 4-6 collapse to 2 lines)
EDIT approach_setup.py imports  (add init_sched_components from sim_main)
Reason: encapsulate unified-interface shared-component init as reusable handle.
Risk: LOW. Pure extract-and-recompose; perform_bin_packing/create_scheduler_elements_with_config/build_simulation_env unchanged. Closure binds exact same values the inline code passed.
Regression gate: import probe + 3 --help (closure must correctly forward all 20 perform_bin_packing args).
Recovery: git checkout archive/test_pipeline-20260612 -- approach_setup.py sim_main.py
Recommended decision: approve
```

## 2. SAFE SUMMARY

- New function: 1 (init_sched_components, closure handle).
- Refactored: 1 (run_benchmark_setup_pipeline).
- Logic change: NONE (extract+recompose; perform_bin_packing receives identical args).
- External interface: init_sched_components is new export; existing sim_main API unchanged.
- Regression gate: mandatory.

## 3. Regression gate (gurobi)

1. `import sim_main; from approach_setup import run_benchmark_setup_pipeline, setup_benchmark` OK
2. `main_approach.py --help` PASS; `motiv_exp_runner --help` PASS; `abla_exp_runner --help` PASS
3. (optional deeper) dry_run motiv case1 to confirm pack() forwards args correctly

If any fails → `git checkout archive/test_pipeline-20260612 -- approach_setup.py sim_main.py`.

## Note on commit policy

Per "不要随意的提交": after execution + gate PASS, do NOT auto-commit.

## Postscript (2026-07-01)

User explicitly authorized continuing and committing B8. The initial PyCapsule blocker was traced to WSL Gurobi HostID mismatch, not license expiry. After creating bond0 with MAC `00:15:5d:80:30:e7`, `gurobipy` model creation passed and motiv case1 (`--num_hp 3 --case1_ratios 0.7`) completed with rc=0. This validates the full `init_sched_components -> pack -> perform_bin_packing -> coleasing_alloc_cluster -> gurobi_split_solver` path.
