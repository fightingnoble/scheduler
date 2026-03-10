# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Environment Setup

```bash
conda activate gurobi   # MUST run before any code execution or testing
```

## Quick Code Location

| What you need | Where to look |
|---------------|---------------|
| Main entry point | `main_approach.py:main()` |
| Benchmark setup pipeline | `approach_setup.py:setup_benchmark()` |
| Bin-packing (Phase 1 split) | `sched/global_sched.py:coleasing_alloc_cluster()` |
| Bin-packing (Phase 2 repack) | `sched/global_sched.py:push_task_into_bins_new()` |
| Per-task deadline calculation (Step 1) | `sched/slack_estim.py:deduce_cfg2()` |
| Event-driven simulation | `approach_sim.py:run_simulation()` |
| Statistics collection | `approach_collector.py:StatisticsCollector` |
| Task representation | `task/task_agent.py:ProcessInt` |
| Bin-packing config | `sched/binpack_config.py:BinPackConfig` |
| Run experiment (motiv) | `scripts/motiv_exp_runner.py` |
| Run experiment (ablation) | `scripts/abla_exp_runner.py` |
| Shared experiment utilities | `scripts/exp_common.py` (ParamTemplate, run_main_approach_inproc) |
| Resource allocation (runtime) | `allocator_agent.py` (glb_sched, cyclic_sched) |
| Path resolution | `paths.py` |

### Deprecated (do not use)
- `sched/scheduler_agent.py` — replaced by `approach_sim.py`
- `sched/monitor_agent.py` — replaced by `approach_collector.py`
- `unused_fun.py`, `old/`, `ref/` — dead code

---

## Project Overview

**Real-time task scheduler simulator** for multi-core embedded systems (autonomous vehicles). Evaluates static, dynamic, and hybrid scheduling algorithms under execution time variability.

**Core problem**: Schedule a DAG of tasks with E2E latency constraints and stochastic execution times. Trade-off: utilization vs. miss rate vs. scheduling overhead.

### Scheduling Strategies

| Strategy | Reservation | Isolation | Dynamic sharing | Config |
|----------|:-----------:|:---------:|:---------------:|--------|
| `cyc` | √ | √ | × | `num_bins=-1`, no repack |
| `glb` | × | × | √ | `num_bins=1`, no repack |
| `pglb` | × | √ | √ | `num_bins>1`, no repack |
| `reserv` | √ | √ | √ | `num_bins>=2`, with repack |
| `cyc-S` | √ | √ | √ (time only) | `num_bins=-1`, with repack |

---

## Architecture

### End-to-End Flow

```
main_approach.py::main()
    │
    └── approach_setup.py::setup_benchmark()
            │
            ├── [non-repack] run_benchmark_setup_pipeline(need_repack=False)
            │       ├── Step 0: build_workload_and_criticality()  → DAG, ProcessInt list
            │       ├── Step 1: deduce_cfg2()                     → per-task deadline, resource sizing
            │       ├── Step 2: coleasing_alloc_cluster()         → bin_list (spatial partition)
            │       └── apply_forced_num_cores()                  → enforce num_cores
            │
            ├── [repack, if ratioB != -1 and ratioA > ratioB]
            │       run_benchmark_setup_pipeline(need_repack=True)
            │       └── Step 3: push_task_into_bins_new()         → time windows within bins
            │
            └── approach_sim.py::run_simulation()                 → Step 4: event-driven sim
```

### Step Summary

| Step | Function | Controlled by | All strategies need? |
|------|----------|---------------|----------------------|
| 0 | `build_workload_and_criticality` | profiling CSV + args | **Yes** |
| 1 | `deduce_cfg2` | `exec_t_comp_ratioA` | **Yes** (incl. glb) |
| 2 | `coleasing_alloc_cluster` | `num_bins` | No (skipped if `num_bins=1`) |
| 3 | `push_task_into_bins_new` | `exec_t_comp_ratioB` | No (repack only) |
| 4 | `run_simulation` | `policy` | **Yes** |

### Key Parameters

| Parameter | Meaning | Typical values |
|-----------|---------|----------------|
| `exec_t_comp_ratioA` | Conservative quantile — resource sizing (Step 1) | 0.99, 0.7 |
| `exec_t_comp_ratioB` | Aggressive quantile — time windows (Step 3) | 0.5–0.99; `-1` = no repack |
| `num_bins` | Spatial partitions | -1 (auto/max), 1 (glb), >1 (pglb/reserv) |
| `num_cores` | Forced core count (optional) | None or integer |
| `policy` | Simulation scheduling policy | `cyc`, `glb`, `pglb`, `reserv` |
| `e2e_latency` | End-to-end latency constraint | 0.1 s |

### Key Constraints

1. **All strategies need Step 1** — `glb` still needs `deduce_cfg2` for per-task deadlines
2. **Resource constraint only in non-repack** — repack changes time windows only, not resource count
3. **`num_cores` = `sum(b.num_resources for b in bin_list)`** after constraint applied
4. **Dump paths use constrained `num_cores`** — coupling between `apply_forced_num_cores` and dump
5. **`test_case` is fixed to `'bin_pack_new'`** in `main_approach.py` — scheduling behavior is controlled by `policy` parameter, NOT `test_case`

> **Common pitfalls**: see `doc/spec/readme.md §4` — covers: glb needs Step 1, reserv dual mechanism, Exp 2/3 resource control via load intensity NOT ratioA, cyc-S requires repack.

### BinPackConfig Key Fields

`BinPackConfig` (`sched/binpack_config.py`) wraps a JSON (e.g., `Bp_guided.json`). Key fields:

| Field | Meaning |
|-------|---------|
| `algorithm` | `"guided"` (two-phase) or `"scratch"` (dynamic only) |
| `quantile` | Resource sizing quantile (set from `exec_t_comp_ratioA`) |
| `var_dist_map` | Execution time variation distributions per task |
| `mapping` / `bin_sel_mod` | `"pre_defined"` (repack uses Phase 1 result) or `"search"` |

Full design: `doc/spec/cfg/binpack_config_design.md`

---

## Documentation Hierarchy

Documents are read top-down. Lower levels must align with upper levels. On conflict, upper level wins.

```
CLAUDE.md                           ← you are here (entry point for Claude)
    │
    ├── doc/spec/readme.md          ← doc/spec/ index, cross-doc clarifications
    │       │
    │       ├── doc/spec/key_COT.md              ⭐⭐⭐ Academic logic (HIGHEST)
    │       │       └── core arguments, mechanism decoupling, ablation motivation
    │       │
    │       ├── doc/spec/e2e_sched_sim_flow.md   ⭐⭐  Design spec
    │       │       └── step definitions, param definitions, function responsibilities
    │       │
    │       ├── doc/spec/test_plan.md             ⭐    Implementation details
    │       │       └── experiment params, plotting APIs, script alignment
    │       │
    │       ├── doc/spec/algorithm/
    │       │       ├── guided_hybrid_allocation_algorithm.md  ← two-phase bin-packing
    │       │       ├── chain_slack_assignment_algorithm.md    ← Step 1 (deduce_cfg2)
    │       │       └── binpack_solver_spec.md
    │       │
    │       ├── doc/spec/stat/
    │       │       ├── tdigest_system_spec.md           ← T-Digest algo
    │       │       ├── statistics_collection_spec.md    ← StatisticsCollector arch
    │       │       └── collector.md                     ← E2E latency decomposition (authoritative)
    │       │
    │       ├── doc/spec/sim/
    │       │       ├── approach_sim_spec.md             ← event-driven sim spec
    │       │       └── simulation_numerical_design.md
    │       │
    │       ├── doc/spec/model/
    │       │       └── latency_model.md                ← latency model spec
    │       │
    │       ├── doc/spec/exp_design/                    ← plotting API for motiv/ablation
    │       │       ├── motiv_case_api_README.md
    │       │       ├── motiv_case_api_quick_ref.md
    │       │       ├── motiv_case_api_usage.md
    │       │       └── motiv_case_plotting_examples.md
    │       │
    │       └── doc/spec/cfg/binpack_config_design.md   ← BinPackConfig design
    │
    ├── doc/guide/                  ← codebase exploration docs (module overviews)
    │       ├── architecture_overview.md
    │       ├── exp_scripts_overview.md
    │       ├── sched_core_overview.md
    │       ├── simulation_flow_overview.md
    │       ├── collector_overview.md
    │       ├── task_model_overview.md
    │       └── deprecated_code.md
    │
    └── doc/dev/                    ← development logs & internal guides
            ├── change_log_{YYYY}.md
            ├── claude_revise.md
            └── code_cleanup_2026.md
```

**Rule**: spec docs reference each other with relative paths from `doc/spec/`. E.g., `./algorithm/chain_slack_assignment_algorithm.md`.

---

## Experiment Scripts

### Motivation Experiments (already validated)

```bash
python -m scripts.motiv_exp_runner --case 1 --output_dir motiv_exp_results --num_hp 100
python -m scripts.motiv_exp_runner --case 2 --output_dir motiv_exp_results --num_hp 100
python -m scripts.motiv_exp_runner --case 3 --output_dir motiv_exp_results --case3_num_periods 200
# Or run all:
bash scripts/run_motiv_exps.sh [output_dir]
```

### Ablation Experiments (scripts implemented, pending validation)

```bash
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
# Or run all:
bash scripts/run_abla_exps.sh [output_dir] [num_hp]
```

Common flags: `--dry_run` (print args only), `--use_plot_cache` (skip sim, replot from JSON cache)

### Ablation Experiment Design

> **Academic rationale**: see `doc/spec/key_COT.md` for mechanism decoupling logic and ablation motivation.
> **Plotting API**: see `doc/spec/exp_design/` for `StatisticsCollector.plot_motiv_case1/2()` usage and examples.

Speed-reference (full per-experiment details below):

| Exp | Comparison | Mechanism tested | Key scan params |
|-----|------------|-----------------|-----------------||
| 1 | cyc-S vs cyc | Reservation (serial) | `ratioB ∈ [0.5,0.99]`; cyc reference at `ratioA ∈ [0.5,0.7,0.99]` on Y-axis |
| 2 | pglb vs glb | Isolation | `num_bins ∈ [1,2,4,8]`; load: `tiles ∈ [200,400]`, `chains ∈ [1,4]`, `load_factor ∈ [0.5,1.0]` |
| 3 | reserv vs pglb | Reservation (parallel) | `ratioB ∈ [0.5,0.99]` + `num_bins ∈ [1,2,4,8,-1]`; same load as Exp 2 |

#### Ablation Exp 1 — cyc-S vs cyc

**Goal**: Show soft reservation (repack) beats hard isolation at same resource budget.

| Item | Value |
|------|-------|
| Fixed `ratioA` | `0.7` |
| Fixed `num_bins` | `-1` (max partitions) |
| Resource | **forced `num_cores`** to match cyc budget (same for both) |
| Scan (cyc-S) | `exec_t_comp_ratioB ∈ [0.5,0.6,0.7,0.8,0.9,0.99]`, `policy='reserv'` |
| Reference (cyc) | `exec_t_comp_ratioA ∈ [0.5,0.7,0.99]`, `policy='cyc'`, no repack |
| Plot | `plot_motiv_case1()` + `_plot_satisfy_projection()` (cyc projected as horizontal lines) |
| Metrics | `idle_mean_ratio`, `miss_mean_ratio`, `miss_mean_count` |
| Expected | miss rate monotonically ↓ as `ratioB` more aggressive; utilization mostly unchanged |

#### Ablation Exp 2 — pglb vs glb

**Goal**: Show spatial partitioning reduces realloc overhead.

| Item | Value |
|------|-------|
| Fixed `ratioA` | `0.7`, fixed `ratioB` = `-1` (no repack) |
| Resource control | **Load intensity** (`tiles`, `chains`, `load_factor`) — NOT `ratioA` |
| `num_bins=1` | → glb baseline, `policy='glb'` |
| `num_bins>1` | → pglb, `policy='pglb'` |
| Scan `num_bins` | `[1, 2, 4, 8]` |
| Load sweep | `tiles ∈ [200,400]`, `chains ∈ [1,4]`, `load_factor ∈ [0.5,1.0]` |
| Plot | `_plot_switching_overhead()`, `plot_motiv_case2()` |
| Metrics | `get_realloc_info()['realloc_mean_count']`, `get_realloc_info()['realloc_mean_ratio']` |
| Expected | realloc_ratio ↓ as `num_bins` ↑; realloc_count roughly unchanged |

#### Ablation Exp 3 — reserv vs pglb

**Goal**: Show reservation+partitioning keeps trade-off stable as scale grows.

| Item | Value |
|------|-------|
| Fixed `ratioA` | `0.7` |
| pglb baseline | Step 0-1-2 only, `policy='pglb'`, `num_bins ∈ [1,2,4,8]`, no repack |
| Scan (reserv) | `exec_t_comp_ratioB ∈ [0.5,0.6,0.7,0.8,0.9,0.99]`, `policy='reserv'` |
| Scan `num_bins` | `[1, 2, 4, 8, -1]` for reserv |
| Load sweep | same as Exp 2; `--case3_fixed_strength` for single fixed load point |
| Plot | `_plot_switching_metrics()` (per num_bins), `plot_motiv_case2()` |
| Metrics | `realloc_mean_count`, `realloc_mean_ratio`, `miss_mean_count` |
| Expected | Unlike Exp 1: as `ratioB` more aggressive → realloc count/ratio first ↓ then ↑; miss rate first ↑ then ↓ |

> **tiles range**: `abla_exp_runner.py` defaults `tiles=[200,400]`. `key_COT.md` mentions `[300,500]` as academic framing; **script defaults are authoritative for running experiments**.
> **Common pitfalls**: see `doc/spec/readme.md §4` — covers: glb needs Step 1, reserv dual mechanism, Exp 2/3 resource control via load intensity NOT ratioA, cyc-S requires repack.

---

## Current Progress

| Component | Status |
|-----------|--------|
| Motivation experiments (3 cases) | ✅ Validated, reproducible via `run_motiv_exps.sh` |
| Ablation experiment scripts | ✅ Implemented (`abla_exp_runner.py`) |
| Ablation experiment validation | ⏳ Pending — run with `--dry_run` first to verify args |
| End-to-end comparison experiments | 📋 Designed in `test_plan.md` §端到端的比较 |

---

## Adding New Experiments

1. Import from `scripts/exp_common.py`: `ParamTemplate`, `run_main_approach_inproc`, `_PHYSICAL_CORES`, `mapping_args`, `runtime_args`, `specific_args`
2. Create a Runner class with `run()`, `_run_simulations()`, `_generate_report()`
3. Use `ProcessPoolExecutor` with `max_workers=min(_PHYSICAL_CORES, len(tasks))`
4. Save results to JSON for `--use_plot_cache` support
5. Use `StatisticsCollector.plot_motiv_case1/2()` for standard plots

## Change Logging

Record all code changes in `doc/dev/change_log_{YYYY}.md`:
- File path + line numbers changed
- Brief description

## Style for spec docs

- Design philosophy > code design > specific code
- Keep concise; use relative links for cross-references (e.g., `./algorithm/chain_slack_assignment_algorithm.md`)
- Conflicts: upper-level doc wins; update lower-level to align
