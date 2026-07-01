# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Environment Setup

```bash
conda activate gurobi   # MUST run before any code execution or testing
```

**非交互 shell（子进程/脚本/`python -c`）**：`conda activate` 不生效，用绝对路径
`/home/zhangchg/miniconda3/envs/gurobi/bin/python`（base python 缺 scipy/networkx/gurobipy）。

## Quick Code Location

| What you need | Where to look |
|---------------|---------------|
| Main entry point | `main_approach.py:main()` |
| Benchmark setup pipeline | `approach_setup.py:setup_benchmark()` |
| Bin-packing core (both phases) | `sim_main.py:perform_bin_packing()` — handles Phase 1, repack, backup/fallback |
| Slack allocation formula | `sched/packing_solver/chain_slack_assign.py:287` — `ideal_cores = ceil(flops_rem/(slack_rem*FLOPS_PER_CORE))` |
| Bin-packing (Phase 1 split) | `sched/global_sched_alloc.py:coleasing_alloc_cluster()` (B6: split from global_sched.py) |
| Bin-packing (Phase 2 repack) | `sched/global_sched_repack.py:push_task_into_bins_new()` (B6: split from global_sched.py) |
| Per-task deadline calculation (Step 1) | `sched/slack_estim.py:deduce_cfg2()` |
| Event-driven simulation | `approach_sim.py:run_simulation()` |
| Statistics collection | `approach_collector.py:StatisticsCollector` |
| **Algorithm 2 runtime overhead** | `approach_def.py:Acc_p.sched()` + `approach_collector.py:record_sched_overhead()` |
| Task representation | `task/task_agent.py:ProcessInt` |
| Bin-packing config | `sched/binpack_config.py:BinPackConfig` |
| Fixcore repack switch | `sim_main.py:33` — `USE_FIXCORE_REPACK = True` |
| Fixcore latency calc | `sched/slack_estim.py:393` — `_fixcore_slack_estim()` |
| Run experiment (motiv) | `scripts/motiv_exp_runner.py` |
| Run experiment (ablation) | `scripts/abla_exp_runner.py` |
| Repack debug diagnostics | `sim_main.py:508-583`, `approach_setup.py:127-133` (stderr output) |
| Repack debug history | `doc/dev/ablation_dev.md` §五 Repack 执行验证 |
| Debug test script | `test_repack_diagnostic.py` (standalone repack verification) |
| Shared experiment utilities | `scripts/exp_common.py` (ParamTemplate, run_main_approach_inproc) |
| Resource allocation (runtime, archived) | `old/allocator_agent.py` — old sim-chain, moved B2 |
| Path resolution | `paths.py` |

### Deprecated (do not use)
- `sched/scheduler_agent.py` — 仿真循环被 `approach_sim.py` 替代；但 `Scheduler` 类 + 工具函数仍活（repack 路径用），**文件级不可删**
- `sched/monitor_agent.py` — 仿真循环被替代；但 `get_target_bin_id`/`get_rsc_2b_released` 仍活（`pre_alloc_new` 用），**文件级不可删**
- `unused_fun.py`, `old/`, `ref/`, `unused/` — dead code（`old/`=历史版本, `unused/`=独立未完成）

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
            ├── [Phase 1] run_benchmark_setup_pipeline(need_repack=False)
            │       ├── args.quantile = ratioA
            │       ├── Step 0: build_workload_and_criticality()  → DAG, ProcessInt list
            │       ├── Step 1: deduce_cfg2(quantile=ratioA)      → per-task deadline, resource sizing
            │       ├── Step 2: coleasing_alloc_cluster()         → bin_list (spatial partition)
            │       └── apply_forced_num_cores()                  → enforce num_cores
            │
            ├── [repack, if ratioB != -1 and ratioA != ratioB]
            │       args.quantile = ratioB
            │       run_benchmark_setup_pipeline(need_repack=True)
            │       ├── Step 0-1 re-run: deduce_cfg2(quantile=ratioB)  → recalc deadlines
            │       └── Step 2/3: attempt perform_bin_packing with deepcopy fallback
            │               - cyc-S (num_bins=-1): bypass — spatial rearrangement meaningless
            │               - reserv (num_bins>=2): try repack; on failure, restore Phase 1 layout
            │
            └── approach_sim.py::run_simulation()                 → Step 4: event-driven sim
```

### Step Summary

| Step | Function | Controlled by | All strategies need? |
|------|----------|---------------|----------------------|
| 0 | `build_workload_and_criticality` | profiling CSV + args | **Yes** |
| 1 | `deduce_cfg2` | `args.quantile` (= ratioA or ratioB) | **Yes** (incl. glb); re-run with ratioB during repack |
| 2 | `coleasing_alloc_cluster` | `num_bins` | No (skipped if `num_bins=1`); **bypassed during repack** |
| 3 | `push_task_into_bins_new` | `exec_t_comp_ratioB` | reserv only; fallback to Phase 1 on incomplete placement |
| 4 | `run_simulation` | `policy` | **Yes** |

### Key Parameters

| Parameter | Meaning | Typical values |
|-----------|---------|----------------|
| `exec_t_comp_ratioA` | Conservative quantile — resource sizing (Phase 1 Step 1) | 0.99, 0.7 |
| `exec_t_comp_ratioB` | Repack quantile — recalculates task deadlines via Step 1 re-run; bin layout unchanged | 0.5–0.99; `-1` = no repack |
| `num_bins` | Spatial partitions | -1 (auto/max), 1 (glb), >1 (pglb/reserv) |
| `num_cores` | Forced core count (optional) | None or integer |
| `policy` | Simulation scheduling policy | `cyc`, `glb`, `pglb`, `reserv` |
| `e2e_latency` | End-to-end latency constraint | 0.1 s |
| `args.quantile` | Active quantile for `deduce_cfg2` — set to ratioA in Phase 1, ratioB in repack | Derived from ratioA/B |

### Key Constraints

1. **All strategies need Step 1** — `glb` still needs `deduce_cfg2` for per-task deadlines
2. **Repack re-runs Step 0-1 with `args.quantile = ratioB`** — recalculates task deadlines/FLOPS with the aggressive quantile; bin `num_resources` is loaded from existing `bin_list` and remains unchanged
3. **Repack uses `pre_defined` mode** — `push_task_into_bins_new` with `bin_sel_mod="pre_defined"` preserves Phase 1 spatial layout; both cyc-S and reserv use this path (unified since 2026-03)
4. **Repack failure is algorithm incompatibility** — `extract_pid2_bin_id()` clears scheduling_table; greedy cannot reproduce ILP's temporal allocation → fallback to Phase 1 layout
5. **Quantile changes don't affect window positions** — same `var_factor` pattern means `flops_dict[node]/flops_rem` stays constant; see `chain_slack_assign.py:287`
6. **Repack trigger condition: `ratioB != -1 and ratioA != ratioB`** — triggers whenever ratioB differs from ratioA (not just when ratioA > ratioB)
7. **Resource constraint only in non-repack** — repack does not change resource count
8. **`num_cores` = `sum(b.num_resources for b in bin_list)`** after constraint applied
9. **Dump paths use constrained `num_cores`** — coupling between `apply_forced_num_cores` and dump
10. **`test_case` is fixed to `'bin_pack_new'`** in `main_approach.py` — scheduling behavior is controlled by `policy` parameter, NOT `test_case`
11. **Fixcore Repack bypass** — `USE_FIXCORE_REPACK=True` in `sim_main.py` skips greedy bin-packing; ERT/DDL updated by `deduce_cfg2` with `fix_core_map` and `scale_factor=ratioB/ratioA`
12. **`gen_workloads` returns 5 values** — `(hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w)`; `rsc_map_w[node]=(cores, latency, constr)` used to extract Phase 1 cores for fixcore repack
13. **Old ratio mechanism** — `slack_comp(slack, abs, r) = (slack-abs)*(1-r)` and `cal_lat(lat, abs, r) = lat/(1-r)+abs` are exact inverses → ERT/DDL invariant to ratio changes (only cores change)
14. **ratioB ≤ ratioA required** — when `ratioB > ratioA`, `scale_factor = ratioB/ratioA > 1` makes deadlines looser, violating `ddl[pred] ≤ sink_ert` constraint in `init_topo_time_attr()` (slack_estim.py:355)

> **Common pitfalls**: see `doc/spec/readme.md §4` — covers: glb needs Step 1, reserv dual mechanism, Exp 2/3 resource control via load intensity NOT ratioA, cyc-S requires repack.
>
> **Python pitfall**: `policy in ["cyc" or "cyc-S"]` evaluates to `["cyc"]` (truthy short-circuit). Correct: `policy in ["cyc", "cyc-S"]` — see `approach_sched.py:303`.

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
    │       │       ├── collector.md                     ← E2E latency decomposition (authoritative)
    │       │       └── runtime_overhead_spec.md         ← Algorithm 2 runtime overhead (reviewer Q3)
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

### Ablation Experiments (validated)

```bash
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
# Optional: --case1_cycS_policy reserv|cyc|pglb (default: reserv)
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
| Scan (cyc-S) | `exec_t_comp_ratioB ∈ [0.5,0.6,0.7]`, `policy='reserv'` (ratioB ≤ ratioA required) |
| Reference (cyc) | `exec_t_comp_ratioA ∈ [0.5,0.7,0.99]`, `policy='cyc'`, no repack |
| Plot | `plot_motiv_case1()` + `_plot_satisfy_projection()` (cyc projected as horizontal lines) |
| Plot customization | `show_realloc=False` to hide orange bars; `cyc_ref` passed via `data_points[0]['cyc_ref']` contains `{ratio, miss_mean_count, idle_mean_ratio}` |
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
| Ablation experiment validation | ✅ All 3 cases validated (Case1: 12/12, Case2: 32/32, Case3: 196/224) |
| Ablation experiment plotting | ✅ Unified style — `case{N}_overhead.pdf` + `case{N}_tradeoff.pdf` |
| End-to-end comparison experiments | 📋 Designed in `test_plan.md` §端到端的比较 |

---

## Adding New Experiments

1. Import from `scripts/exp_common.py`: `ParamTemplate`, `run_main_approach_inproc`, `_PHYSICAL_CORES`, `mapping_args`, `runtime_args`, `specific_args`
2. Create a Runner class with `run()`, `_run_simulations()`, `_generate_report()`
3. Use `ProcessPoolExecutor` with `max_workers=min(_PHYSICAL_CORES, len(tasks))`
4. Save results to JSON for `--use_plot_cache` support
5. Use `StatisticsCollector.plot_motiv_case1/2()` for standard plots
6. **Do not modify** `plot_motiv_case1/2()` in `approach_collector.py` — shared with motiv experiments
7. For ablation-specific plots, use `ABLA_COLORS` dict in `abla_exp_runner.py` (unified with motiv color scheme: exec=C0, realloc=C1, wait=C2, miss_bar=C3, miss_line=C4, idle=C7)

## 代码清理（legacy-prune）

清理工作在 audit worktree `~/git_repo/scheduler-audit-20260612/`（基于 `test_pipeline`）进行，**不碰主仓库**。

| 约定 | 说明 |
|------|------|
| 全局状态入口 | `CLEANUP_STATUS.md`（当前状态+下一步）；`FILE_ADJUSTMENT_RECORD.md`（更改历史+recovery）。每次动作后强制更新两者 |
| 幽灵文件陷阱 | 分析**必须在 audit worktree 做**。主仓库有未跟踪旧文件（`bin_ops.old.py`/`message_handler_old.py` 等）会污染死代码判定 |
| 死活判定用 AST | `from X import *` 造成假定义位置。用 `inspect.getsourcefile(fn)` 或 AST 调用图，**不要靠 grep 反查定义** |
| byte-identical 验证 | `git diff -- <f> \| awk '/^\+[^+]/{i++}/^-[^-]/{d++}'` → 删除操作应为 `i=0`（纯删除，0 插入） |
| 归档语义 | `old/`=历史版本（有新版本取代）；`unused/`=独立未完成功能。符号级→`*_old.py`/`*_unused.py`。详见 `.claude/skills/legacy-prune/references/cleanup-policy.md` |
| 回归门 | 每次清理后跑：6 行 import 探针 + `main_approach.py`/`motiv_exp_runner`/`abla_exp_runner` `--help`（gurobi 环境） |
| 搬迁函数陷阱 | 函数签名默认参数（如 `def f(cfg=default_binpack_cfg)`）在定义时求值，依赖**模块级常量**（夹在 import 块与 def 之间）。搬迁脚本只搬 import+函数会漏常量 → import 探针抓 `NameError`。md5 验搬迁无损，**不证搬迁完整** |
| `--help` 退出码假阳性 | 不要用 `cmd >/dev/null 2>&1 && echo ✓`（for 循环+变量传播会假阳性）。显式 `"$GP" "$cmd" --help >/tmp/o 2>&1; rc=$?` 判 `rc==0` |

## Troubleshooting

| Problem | Likely cause | Where to look |
|---------|-------------|---------------|
| Resource insufficient exception | `num_cores` constraint too tight | `apply_forced_num_cores()` |
| All results identical | repack not triggered or cache stale | Check `exec_t_comp_ratioB` value (must != -1) |
| Parameter not passed through | `binpack_cfg` update missed | `utils.py:build_path_old()` |
| Statistics data missing | `forward_hyperperiod()` not called | Simulation main loop — hyperperiod boundary |
| Repack incomplete (tasks not placed) | Greedy algorithm cannot reproduce ILP solution | `push_task_into_bins_new` — algorithm incompatibility; fallback handles this |
| Repack falls back to Phase 1 | Expected behavior — `extract_pid2_bin_id` clears scheduling_table | Check log for "Repack failed" message; fallback is intentional |
| `TypeError: cannot pickle 'PyCapsule'` | Gurobi license expired | Check `gurobi` env license; renew if needed |
| `KeyError: 'acc_pN'` in simulation | `num_bins` exceeds actual task groups | `coleasing_alloc_cluster()` produces fewer bins; worker has try/except guard |
| `KeyError: 'miss_mean_count'` | Case 2 nests it in `stats['utilization']`, Case 3 flattens to top-level | `_case2_worker` vs `_case3_worker` data structure difference |
| Diagnostic `print()` invisible | `run_benchmark_setup_pipeline` uses `redirect_stdout` to log file | Use `sys.stderr.write()` for terminal-visible diagnostics |

## Change Logging

Record all code changes in `doc/dev/change_log_{YYYY}.md`:
- File path + line numbers changed
- Brief description

## Style for spec docs

- Design philosophy > code design > specific code
- Keep concise; use relative links for cross-references (e.g., `./algorithm/chain_slack_assignment_algorithm.md`)
- Conflicts: upper-level doc wins; update lower-level to align

## MCP Tools Notes

- **zai-mcp image tools** (`analyze_image`, `analyze_data_visualization`, etc.) only support `.jpg`, `.jpeg`, `.png` — **not PDF**. Convert PDF plots to PNG first if image analysis is needed.
