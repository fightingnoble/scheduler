
● Write(CLAUDE.md)
  ⎿  Added 165 lines, removed 133 lines
        1  # CLAUDE.md
        2
        3 -This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
        3 +This file provides guidance to Claude Code when working with this repository.
        4
        5  ## Environment Setup
        6
        7  ```bash
        8 -conda activate gurobi
        8 +conda activate gurobi   # MUST run before any code execution or testing
        9  ```
       10
       11 -This is required before running or testing any code in this project.
       11 +## Quick Code Location
       12
       13 -## Project Overview
       13 +| What you need | Where to look |
       14 +|---------------|---------------|
       15 +| Main entry point | `main_approach.py:main()` |
       16 +| Benchmark setup pipeline | `approach_setup.py:setup_benchmark()` |
       17 +| Bin-packing (Phase 1 split) | `sched/global_sched.py:coleasing_alloc_cluster()` |
       18 +| Bin-packing (Phase 2 repack) | `sched/global_sched.py:push_task_into_bins_new()` |
       19 +| Per-task deadline calculation (Step 1) | `sched/slack_estim.py:deduce_cfg2()` |
       20 +| Event-driven simulation | `approach_sim.py:run_simulation()` |
       21 +| Statistics collection | `approach_collector.py:StatisticsCollector` |
       22 +| Task representation | `task/task_agent.py:ProcessInt` |
       23 +| Bin-packing config | `sched/binpack_config.py:BinPackConfig` |
       24 +| Run experiment (motiv) | `scripts/motiv_exp_runner.py` |
       25 +| Run experiment (ablation) | `scripts/abla_exp_runner.py` |
       26 +| Shared experiment utilities | `scripts/exp_common.py` (ParamTemplate, run_main_approach_inproc) |
       27 +| Path resolution | `paths.py` |
       28
       15 -This is a **real-time task scheduler simulator** for multi-core embedded systems (e.g., autonomous vehicles). It evaluates static, d
          -ynamic, and hybrid scheduling algorithms under execution time variability.
       29 +### Deprecated (do not use)
       30 +- `sched/scheduler_agent.py` — replaced by `approach_sim.py`
       31 +- `sched/monitor_agent.py` — replaced by `approach_collector.py`
       32 +- `unused_fun.py`, `old/`, `ref/` — dead code
       33
       17 -### Core Problem
       34 +---
       35
       19 -Schedule a DAG of tasks with:
       20 -- **E2E latency constraints** per chain
       21 -- **Execution time variability** (stochastic, different quantiles)
       22 -- Trade-offs between: utilization, miss rate, and scheduling overhead
       36 +## Project Overview
       37
       24 -### Scheduling Strategies
       38 +**Real-time task scheduler simulator** for multi-core embedded systems (autonomous vehicles). Evaluates static, dynamic, and hybrid
          +scheduling algorithms under execution time variability.
       39
       26 -| Strategy | Description | Config |
       27 -|----------|-------------|--------|
       28 -| `cyc` | Pure static cyclic scheduling | `num_bins=-1`, no repack |
       29 -| `glb` | Pure dynamic gang scheduling | `num_bins=1`, no repack |
       30 -| `reserv` | Hybrid: spatial partition + temporal scheduling | `num_bins >= 2`, with repack (abl. exp. scans `num_bins ∈ [1,2,4,8,-1
          -]`) |
       31 -| `pglb` | Partitioned dynamic scheduling | `num_bins>1`, no repack |
       32 -| `cyc-S` | Hybrid with soft reservation | `num_bins=-1`, with repack (soft reserve) |
       40 +**Core problem**: Schedule a DAG of tasks with E2E latency constraints and stochastic execution times. Trade-off: utilization vs. mi
          +ss rate vs. scheduling overhead.
       41
       34 -## Architecture
       42 +### Scheduling Strategies
       43
       36 -### Entry Points
       44 +| Strategy | Reservation | Isolation | Dynamic sharing | Config |
       45 +|----------|:-----------:|:---------:|:---------------:|--------|
       46 +| `cyc` | √ | √ | × | `num_bins=-1`, no repack |
       47 +| `glb` | × | × | √ | `num_bins=1`, no repack |
       48 +| `pglb` | × | √ | √ | `num_bins>1`, no repack |
       49 +| `reserv` | √ | √ | √ | `num_bins>=2`, with repack |
       50 +| `cyc-S` | √ | √ | √ (time only) | `num_bins=-1`, with repack |
       51
       38 -- **`main_approach.py`**: Main entry point for running benchmarks and experiments
       39 -- **`approach_setup.py`**: `setup_benchmark()` - Main benchmark setup pipeline
       40 -- **`sim_main.py`**: `perform_bin_packing()` - Core bin-packing algorithm
       41 -- **`approach_sim.py`**: `run_simulation()` - Event-driven simulation runtime
       42 -- **`scripts/motiv_exp_runner.py`**: Motivation experiments runner
       43 -- **`scripts/abla_exp_runner.py`**: Ablation experiments runner
       52 +---
       53
       45 -### Key Flow: Benchmark Setup
       54 +## Architecture
       55
       56 +### End-to-End Flow
       57 +
       58  ```
       48 -setup_benchmark (approach_setup.py)
       59 +main_approach.py::main()
       60      │
       50 -    └── run_benchmark_setup_pipeline
       51 -            ├── build_workload_and_criticality    # Step 0-1
       52 -            ├── create_scheduler_elements_with_config
       53 -            ├── build_simulation_env
       54 -            ├── perform_bin_packing               # Step 2 (split)
       55 -            │   └── returns (bin_list, max_core_num, ...)
       56 -            ├── apply_forced_num_cores            # Resource constraint (non-repack only)
       57 -            └── dump                               # Save bin_list
       61 +    └── approach_setup.py::setup_benchmark()
       62 +            │
       63 +            ├── [non-repack] run_benchmark_setup_pipeline(need_repack=False)
       64 +            │       ├── Step 0: build_workload_and_criticality()  → DAG, ProcessInt list
       65 +            │       ├── Step 1: deduce_cfg2()                     → per-task deadline, resource sizing
       66 +            │       ├── Step 2: coleasing_alloc_cluster()         → bin_list (spatial partition)
       67 +            │       └── apply_forced_num_cores()                  → enforce num_cores
       68 +            │
       69 +            ├── [repack, if ratioB != -1 and ratioA > ratioB]
       70 +            │       run_benchmark_setup_pipeline(need_repack=True)
       71 +            │       └── Step 3: push_task_into_bins_new()         → time windows within bins
       72 +            │
       73 +            └── approach_sim.py::run_simulation()                 → Step 4: event-driven sim
       74  ```
       75
       60 -### Key Flow: Guided Hybrid Allocation
       76 +### Step Summary
       77
       62 -Two-phase algorithm (`doc/spec/guided_hybrid_allocation_algorithm.md`):
       78 +| Step | Function | Controlled by | All strategies need? |
       79 +|------|----------|---------------|----------------------|
       80 +| 0 | `build_workload_and_criticality` | profiling CSV + args | **Yes** |
       81 +| 1 | `deduce_cfg2` | `exec_t_comp_ratioA` | **Yes** (incl. glb) |
       82 +| 2 | `coleasing_alloc_cluster` | `num_bins` | No (skipped if `num_bins=1`) |
       83 +| 3 | `push_task_into_bins_new` | `exec_t_comp_ratioB` | No (repack only) |
       84 +| 4 | `run_simulation` | `policy` | **Yes** |
       85
       64 -1. **Phase 1 - Spatial Partitioning (Split)**: `coleasing_alloc_cluster()`
       65 -   - Uses conservative quantile `exec_t_comp_ratioA` (e.g., 0.99)
       66 -   - Clusters tasks into bins, sizes bin resources
       67 -   - Output: `bin_list` with task-to-bin mapping
       86 +### Key Parameters
       87
       69 -2. **Phase 2 - Temporal Scheduling (Repack)**: `push_task_into_bins_new()`
       70 -   - Uses aggressive quantile `exec_t_comp_ratioB` (e.g., 0.80)
       71 -   - Assigns static time windows within each bin
       72 -   - Enables "soft reservation" - dynamic fallback when windows are missed
       88 +| Parameter | Meaning | Typical values |
       89 +|-----------|---------|----------------|
       90 +| `exec_t_comp_ratioA` | Conservative quantile — resource sizing (Step 1) | 0.99, 0.7 |
       91 +| `exec_t_comp_ratioB` | Aggressive quantile — time windows (Step 3) | 0.5–0.99; `-1` = no repack |
       92 +| `num_bins` | Spatial partitions | -1 (auto/max), 1 (glb), >1 (pglb/reserv) |
       93 +| `num_cores` | Forced core count (optional) | None or integer |
       94 +| `policy` | Simulation scheduling policy | `cyc`, `glb`, `pglb`, `reserv` |
       95 +| `e2e_latency` | End-to-end latency constraint | 0.1 s |
       96
       74 -### Key Modules
       97 +### Key Constraints
       98
       76 -| Module | Purpose |
       77 -|--------|---------|
       78 -| `sched/global_sched.py` | Core bin-packing algorithms (`coleasing_alloc_cluster`, `push_task_into_bins_new`) |
       79 -| `task/task_agent.py` | Task representation (ProcessInt) |
       80 -| `allocator_agent.py` | Resource allocation (glb_sched, cyclic_sched) |
       81 -| `approach_collector.py` | StatisticsCollector for experiment data |
       82 -| `sched/scheduler_agent.py` | Deprecated Runtime scheduler implementation |
       83 -| `sched/monitor_agent.py` | Deprecated runtime Monitoring and statistics collection |
       84 -| `scripts/exp_common.py` | Shared components for experiment runners (ParamTemplate, run_main_approach_inproc, plotting helpers) |
       85 -| `paths.py` | Centralized path resolution (see `path_migration_guide.md`) |
       99 +1. **All strategies need Step 1** — `glb` still needs `deduce_cfg2` for per-task deadlines
      100 +2. **Resource constraint only in non-repack** — repack changes time windows only, not resource count
      101 +3. **`num_cores` = `sum(b.num_resources for b in bin_list)`** after constraint applied
      102 +4. **Dump paths use constrained `num_cores`** — coupling between `apply_forced_num_cores` and dump
      103
       87 -### Configuration
      104 +---
      105
       89 -- **`BinPackConfig`** (`sched/binpack_config.py`): Configuration wrapper for bin-packing parameters
       90 -- Loaded from JSON files (e.g., `Bp_guided.json`)
       91 -- Key parameters: `algorithm`, `quantile`, `var_dist_map`, `mapping`, `bin_sel_mod`
      106 +## Documentation Hierarchy
      107
       93 -### Policy-Based Control
      108 +Documents are read top-down. Lower levels must align with upper levels. On conflict, upper level wins.
      109
       95 -The `test_case` parameter is fixed to `'bin_pack_new'` in `main_approach.py`. Scheduling behavior is controlled by the `policy` para
          -meter (`cyc`, `glb`, `pglb`, `reserv`).
      110 +```
      111 +CLAUDE.md                           ← you are here (entry point for Claude)
      112 +    │
      113 +    ├── doc/spec/readme.md          ← doc/spec/ index, cross-doc clarifications
      114 +    │       │
      115 +    │       ├── doc/spec/key_COT.md              ⭐⭐⭐ Academic logic (HIGHEST)
      116 +    │       │       └── core arguments, mechanism decoupling, ablation motivation
      117 +    │       │
      118 +    │       ├── doc/spec/e2e_sched_sim_flow.md   ⭐⭐  Design spec
      119 +    │       │       └── step definitions, param definitions, function responsibilities
      120 +    │       │
      121 +    │       ├── doc/spec/test_plan.md             ⭐    Implementation details
      122 +    │       │       └── experiment params, plotting APIs, script alignment
      123 +    │       │
      124 +    │       ├── doc/spec/algorithm/
      125 +    │       │       ├── guided_hybrid_allocation_algorithm.md  ← two-phase bin-packing
      126 +    │       │       ├── chain_slack_assignment_algorithm.md    ← Step 1 (deduce_cfg2)
      127 +    │       │       └── binpack_solver_spec.md
      128 +    │       │
      129 +    │       ├── doc/spec/stat/
      130 +    │       │       ├── tdigest_system_spec.md           ← T-Digest algo
      131 +    │       │       ├── statistics_collection_spec.md    ← StatisticsCollector arch
      132 +    │       │       └── collector.md                     ← E2E latency decomposition (authoritative)
      133 +    │       │
      134 +    │       ├── doc/spec/sim/
      135 +    │       │       ├── approach_sim_spec.md             ← event-driven sim spec
      136 +    │       │       └── simulation_numerical_design.md
      137 +    │       │
      138 +    │       └── doc/spec/cfg/binpack_config_design.md   ← BinPackConfig design
      139 +    │
      140 +    └── doc/guide/                  ← codebase exploration docs (module overviews)
      141 +            ├── architecture_overview.md
      142 +            ├── exp_scripts_overview.md
      143 +            ├── sched_core_overview.md
      144 +            ├── simulation_flow_overview.md
      145 +            ├── collector_overview.md
      146 +            ├── task_model_overview.md
      147 +            └── deprecated_code.md
      148 +```
      149
       97 -## Important Parameters
      150 +**Rule**: spec docs reference each other with relative paths from `doc/spec/`. E.g., `./algorithm/chain_slack_assignment_algorithm.m
          +d`.
      151
       99 -| Parameter | Meaning | Typical Values |
      100 -|-----------|---------|----------------|
      101 -| `exec_t_comp_ratioA` | Conservative quantile for Phase 1 (resource sizing) | 0.99, 0.7 |
      102 -| `exec_t_comp_ratioB` | Aggressive quantile for Phase 2 (time windows) | 0.80, 0.5 (repack only) |
      103 -| `num_bins` | Spatial partitions: -1=auto-search (max number of bin), 1=skip clustering (glb), >1=partition | -1 (cyc, cyc-S, reser
          -v), 1 (glb), >=2 (pglb, reserv) |
      104 -| `num_cores` | Forced core count (optional) | None or specific number |
      105 -| `e2e_latency` | End-to-end latency constraint | 0.1 (seconds) |
      152 +---
      153
      107 -## Documentation in `doc/spec/`
      154 +## Experiment Scripts
      155
      109 -When modifying code, reference these specifications:
      156 +### Motivation Experiments (already validated)
      157
      111 -### Document Hierarchy and Priority
      158 +```bash
      159 +python -m scripts.motiv_exp_runner --case 1 --output_dir motiv_exp_results --num_hp 100
      160 +python -m scripts.motiv_exp_runner --case 2 --output_dir motiv_exp_results --num_hp 100
      161 +python -m scripts.motiv_exp_runner --case 3 --output_dir motiv_exp_results --case3_num_periods 200
      162 +# Or run all:
      163 +bash scripts/run_motiv_exps.sh [output_dir]
      164 +```
      165
      113 -| Document | Priority | Role | Content |
      114 -|----------|----------|------|--------|
      115 -| **`key_COT.md`** | ⭐⭐⭐ | **Academic Logic (Highest)** | Core arguments, mechanism decoupling, ablation experiment high-level id
          -eas |
      116 -| `e2e_sched_sim_flow.md` | ⭐⭐ | **Design Specification** | Step definitions, parameter definitions, execution paths, function res
          -ponsibilities |
      117 -| `test_plan.md` | ⭐ | **Implementation Details** | Detailed experiment parameters, plotting methods, script parameters |
      166 +### Ablation Experiments (scripts implemented, pending validation)
      167
      119 -### Key Principle
      168 +```bash
      169 +python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
      170 +python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
      171 +python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
      172 +# Or run all:
      173 +bash scripts/run_abla_exps.sh [output_dir] [num_hp]
      174 +```
      175
      121 -- **Top-down alignment**: e2e and test_plan must align with key_COT
      122 -- **Missing details**: If key_COT lacks necessary technical details, supplement in e2e
      123 -- **Conflict resolution**: In case of conflict, defer to key_COT.md (academic expression is authoritative)
      176 +Common flags: `--dry_run` (print args only), `--use_plot_cache` (skip sim, replot from JSON cache)
      177
      125 -### Other Specifications
      178 +### Ablation Experiment Design
      179
      127 -| Document | Purpose |
      128 -|----------|---------|
      129 -| `guided_hybrid_allocation_algorithm.md` | Two-phase allocation algorithm |
      130 -| `chain_slack_assignment_algorithm.md` | Slack assignment for time windows |
      131 -| `stat/collector.md` | **Authoritative**: E2E latency decomposition via critical path analysis |
      132 -| `stat/tdigest_system_spec.md` | T-Digest streaming histogram system |
      133 -| `stat/statistics_collection_spec.md` | StatisticsCollector and simulation integration |
      180 +| Exp | Comparison | Mechanism tested | Key scan params |
      181 +|-----|------------|-----------------|-----------------|
      182 +| 1 | cyc-S vs cyc | Reservation (serial) | `ratioB ∈ [0.5,0.99]`; cyc reference at `ratioA ∈ [0.5,0.7,0.99]` on Y-axis |
      183 +| 2 | pglb vs glb | Isolation | `num_bins ∈ [1,2,4,8]`; load: `tiles ∈ [200,400]`, `chains ∈ [1,4]`, `load_factor ∈ [0.5,1.0]` |
      184 +| 3 | reserv vs pglb | Reservation (parallel) | `ratioB ∈ [0.5,0.99]` + `num_bins ∈ [1,2,4,8,-1]`; same load as Exp 2 |
      185
      135 -**Stat doc hierarchy**: `stat/tdigest_system_spec.md` (low-level algo) → `stat/statistics_collection_spec.md` (business integration)
          - → `stat/collector.md` (extension guide)
      186 +---
      187
      137 -## Change Logging
      188 +## Current Progress
      189
      139 -When modifying code, record changes in `doc/change_log_{YYYY}.md` with:
      140 -- Modified filenames and line numbers
      141 -- Brief description of changes
      190 +| Component | Status |
      191 +|-----------|--------|
      192 +| Motivation experiments (3 cases) | ✅ Validated, reproducible via `run_motiv_exps.sh` |
      193 +| Ablation experiment scripts | ✅ Implemented (`abla_exp_runner.py`) |
      194 +| Ablation experiment validation | ⏳ Pending — run with `--dry_run` first to verify args |
      195 +| End-to-end comparison experiments | 📋 Designed in `test_plan.md` §端到端的比较 |
      196
      143 -## Experiment Scripts
      197 +---
      198
      145 -- **`scripts/motiv_exp_runner.py`**: Run motivation experiments (Case 1/2/3)
      146 -- **`scripts/abla_exp_runner.py`**: Run ablation experiments
      147 -- **`scripts/exp_common.py`**: Shared module - import `ParamTemplate`, `run_main_approach_inproc`, `_PHYSICAL_CORES`, and base args
          -from here when creating new experiment runners
      199 +## Adding New Experiments
      200
      149 -```
      150 -  python -m scripts.motiv_exp_runner --case 1 --output_dir motiv_exp_results --num_hp 100
      151 -  python -m scripts.motiv_exp_runner --case 2 --output_dir motiv_exp_results --num_hp 100
      152 -  python -m scripts.motiv_exp_runner --case 3 --output_dir motiv_exp_results --case3_num_periods 200
      201 +1. Import from `scripts/exp_common.py`: `ParamTemplate`, `run_main_approach_inproc`, `_PHYSICAL_CORES`, `mapping_args`, `runtime_arg
          +s`, `specific_args`
      202 +2. Create a Runner class with `run()`, `_run_simulations()`, `_generate_report()`
      203 +3. Use `ProcessPoolExecutor` with `max_workers=min(_PHYSICAL_CORES, len(tasks))`
      204 +4. Save results to JSON for `--use_plot_cache` support
      205 +5. Use `StatisticsCollector.plot_motiv_case1/2()` for standard plots
      206
      154 -  python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
      155 -  python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
      156 -  python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
      207 +## Change Logging
      208
      158 -```
      159 -- Use `--use_plot_cache` to regenerate plots without re-running simulation
      209 +Record all code changes in `doc/dev/change_log_{YYYY}.md`:
      210 +- File path + line numbers changed
      211 +- Brief description
      212
      161 -## Key Constraints
      213 +## Style for spec docs
      214
      163 -1. **Resource constraint only applies in non-repack phase** - Repack doesn't change resource allocation
      164 -2. **`num_cores` always equals `sum(b.num_resources for b in bin_list)`** after constraint
      165 -3. **Dump paths use constrained `num_cores`** - Resource constraint and dump are coupled in the outer layer
      166 -
      167 -## Common Misconceptions
      168 -
      169 -- **glb DOES need Step 1** - Even though glb doesn't use bin_split or repack, it requires Step 1 (deduce_cfg2) for per-task deadline
          - calculation via `exec_t_comp_ratioA`
      170 -- **reserv has dual mechanisms** - Combines spatial isolation (`num_bins >= 2`) AND temporal reservation (repack), so ablation exper
          -iments scan both `exec_t_comp_ratioB` AND `num_bins`
      171 -- **Ablation Exp 2 resource control** - Uses load intensity (tiles/chains/load_factor), NOT `exec_t_comp_ratioA` for controlling res
          -ources
      172 -
      173 -## Ablation Experiment Parameters
      174 -
      175 -| Experiment | Comparison | Key Scan Parameters |
      176 -|------------|------------|---------------------|
      177 -| Exp 1 | cyc-S vs cyc | `exec_t_comp_ratioB ∈ [0.5, 0.99]`; reference lines: cyc at `exec_t_comp_ratioA ∈ [0.5, 0.7, 0.99]` project
          -ed on Y-axis |
      178 -| Exp 2 | pglb vs glb | `num_bins`, load intensity: `tiles ∈ [300,500]`, `chains ∈ [1,4]`, `load_factor ∈ [0.5,1.0]` |
      179 -| Exp 3 | reserv vs pglb | `exec_t_comp_ratioB ∈ [0.5, 0.99]`, `num_bins ∈ [1, 2, 4, 8, -1]`, same load intensity as Exp 2 |
      180 -
      181 -## Style for spec doc
      182 -
      183 -- **Content priority**: spec文档起到纲领的作用，在内容呈现上的优先级上：设计思想 > 代码设计思路 > 具体代码(Design philosophy > Code
          -design > Specific code)
      184 -- **Keep docs concise**: Design concepts upfront, implementation details at the end
      185 -- **Cross-reference**: Use relative links (e.g., `./stat/collector.md`) for internal references
      215 +- Design philosophy > code design > specific code
      216 +- Keep concise; use relative links for cross-references (e.g., `./algorithm/chain_slack_assignment_algorithm.md`)
      217 +- Conflicts: upper-level doc wins; update lower-level to align

● 现在修复 architecture_overview.md 中的路径错误（doc/dev/ → doc/guide/），并创建两个常用 skill：

● Update(doc/guide/architecture_overview.md)
  ⎿  Added 1 line, removed 1 line
      46  │   │   ├── key_COT.md        # 学术逻辑（最高优先级）
      47  │   │   ├── e2e_sched_sim_flow.md
      48  │   │   └── test_plan.md
      49 -│   └── dev/                  # 开发文档
      49 +│   └── guide/                # 代码库理解指南 (doc/guide/)
      50  │
      51  ├── model/                    # 系统模型
      52  │   ├── barrier_agent.py      # 屏障代理

● Update(doc/guide/architecture_overview.md)
  ⎿  Added 2 lines, removed 2 lines
      136  | `doc/spec/key_COT.md` | ⭐⭐⭐ | 学术逻辑 |
      137  | `doc/spec/e2e_sched_sim_flow.md` | ⭐⭐ | 设计规范 |
      138  | `doc/spec/test_plan.md` | ⭐ | 实现细节 |
      139 -| `doc/dev/*.md` | 参考 | 模块概览 |
      139 +| `doc/guide/*.md` | 参考 | 模块概览 |
      140
      141  ---
      142 -*生成时间: 2026-03-10*
      142 +*生成时间: 2026-03-11*

● 现在创建常用 skill 定义：

● Read 1 file (ctrl+o to expand)

● 现在创建项目专用 skill：

● Bash(mkdir -p /home/zhangchg/.claude/skills/run-experiment && mkdir -p /home/zhangchg/.claude/skills/spec-writer && mkdir -p
      /home/zhangchg/.claude/skills/feature-de…)
  ⎿  Done

● 现在创建这三个 skill 文件：

● Write(~/.claude/skills/run-experiment/skill.md)
  ⎿  Wrote 32 lines to ../../.claude/skills/run-experiment/skill.md
       1 # /run-experiment
       2
       3 Run scheduler experiments with proper setup and validation.
       4
       5 ## Steps
       6
       7 1. Verify `conda activate gurobi` is active
       8 2. Determine experiment type (motiv or ablation) and case number from user intent
       9 3. Suggest `--dry_run` first if parameters are non-default
      10 4. Run the experiment command
     … +22 lines (ctrl+o to expand)

● Write(~/.claude/skills/spec-writer/skill.md)
  ⎿  Wrote 31 lines to ../../.claude/skills/spec-writer/skill.md
       1 # /spec-writer
       2
       3 Update or create specification documents in doc/spec/ following the project hierarchy.
       4
       5 ## Document Hierarchy (top = highest priority)
       6
       7 ```
       8 doc/spec/key_COT.md              ← academic logic (never break alignment with this)
       9 doc/spec/e2e_sched_sim_flow.md   ← design spec (step defs, param defs, function roles)
      10 doc/spec/test_plan.md            ← implementation details (script params, plot APIs)
     … +21 lines (ctrl+o to expand)

● Write(~/.claude/skills/feature-dev/skill.md)
  ⎿  Wrote 45 lines to ../../.claude/skills/feature-dev/skill.md
       1 # /feature-dev
       2
       3 Guide for developing new features or fixing bugs in this scheduler codebase.
       4
       5 ## Workflow
       6
       7 1. **Read before modifying** — always read the relevant files first
       8 2. **Check spec docs** — find the relevant spec in `doc/spec/` before coding
       9 3. **Locate code** — use the quick lookup table in CLAUDE.md
      10 4. **Implement** — follow existing patterns, avoid over-engineering
     … +35 lines (ctrl+o to expand)

● 所有改动已完成。以下是汇总：

  ---
  完成的改动

  1. 修复引用路径（3处）

  ┌───────────────────────────────────┬───────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
  │               文件                │                  原路径                   │                         修正路径                         │
  ├───────────────────────────────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ doc/spec/e2e_sched_sim_flow.md:20 │ doc/chain_slack_assignment_algorithm.md   │ doc/spec/algorithm/chain_slack_assignment_algorithm.md   │
  ├───────────────────────────────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ doc/spec/test_plan.md:29          │ doc/chain_slack_assignment_algorithm.md   │ doc/spec/algorithm/chain_slack_assignment_algorithm.md   │
  ├───────────────────────────────────┼───────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
  │ doc/spec/test_plan.md:84          │ doc/guided_hybrid_allocation_algorithm.md │ doc/spec/algorithm/guided_hybrid_allocation_algorithm.md │
  └───────────────────────────────────┴───────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

  注：CLAUDE.md 中原有的 doc/spec/guided_hybrid_allocation_algorithm.md 引用路径本身正确，但文件已移到 doc/spec/algorithm/，在重写中已更新。

  ---
  2. 重写 CLAUDE.md

  新增/改善的内容：

  - Quick Code Location 速查表 — 常见需求直接映射到文件+函数名
  - Deprecated 列表 — 明确标出不应使用的旧模块
  - Step 汇总表 — Step 0-4，哪些策略需要，由哪个参数控制
  - 完整文档层次树 — 带缩进的树状结构，包含 doc/guide/ 层级
  - Cross-reference 规则 — 明确 spec 文档之间用相对路径引用
  - Current Progress 进度表 — 当前实验进展状态
  - Adding New Experiments 模板 — 下次开发 runner 的标准步骤

  ---
  3. 创建 3 个 Skills

  ┌─────────────────┬────────────────────────────────────────────────────────────────┐
  │      Skill      │                              用途                              │
  ├─────────────────┼────────────────────────────────────────────────────────────────┤
  │ /run-experiment │ 运行 motiv/ablation 实验，包含命令模板和常用 flags             │
  ├─────────────────┼────────────────────────────────────────────────────────────────┤
  │ /spec-writer    │ 编写/更新 spec 文档，包含层级规则和正确的 cross-reference 路径 │
  ├─────────────────┼────────────────────────────────────────────────────────────────┤
  │ /feature-dev    │ 开发新功能，包含代码定位速查和常见陷阱                         │
  └─────────────────┴────────────────────────────────────────────────────────────────┘

  ---
  待你确认的一个问题

  doc/guide/architecture_overview.md 中还残留了一个旧路径 deprecated_code.md 的引用 —— 该文件实际在
  doc/guide/deprecated_code.md，相对路径是正确的（同目录），无需修改。