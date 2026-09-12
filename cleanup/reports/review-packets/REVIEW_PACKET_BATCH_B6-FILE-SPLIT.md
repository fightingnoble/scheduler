# REVIEW_PACKET_BATCH_B6-FILE-SPLIT

Batch: `B6-FILE-SPLIT` — split impure `sched/` files by purpose (physical split, **move not rewrite**).
Strategy: `move-reference` extended — relocate functions byte-identical to new purpose-specific files; original file kept as a **re-export shell** (variant B) so external imports need zero change.

## Background & decision context

- User observed some `sched/` files mix unrelated purposes, making classification hard. Requested physical split.
- **Core tension acknowledged & accepted by user (2026-06-27)**: splitting a file = delete original + create new files → permanently increases `test_pipeline` merge/cherry-pick conflict cost (original filename gone). User chose **option A (physical split)**, accepting this cost (readability > mergeability for these files).
- Variant **B (re-export shell)** chosen for global_sched & scheduling_table: original filename retained as a thin re-export layer → external `from sched.global_sched import X` keeps working, zero external import churn, while new files make the structure clear.
- `slack_estim.py`: **not split** (user: iii) — `plot_timeline_graph` is called by `deduce_cfg2` (coupled), low payoff.

Split rule (extends "delete vs rewrite"): functions move **byte-identical**; logic/interfaces unchanged; only `import` statements adjusted (the permitted companion). New re-export shells are the only new code, and they contain only import-forwarding (no logic).

## AST evidence (intra-file call graphs)

- `global_sched.py`: repack group (`push_task_into_bins_new`→`push_step_new`) and alloc group (`coleasing_alloc_cluster`→`coleasing_alloc_1bin`/`gurobi_split_solver`/`update_bp_result2_schedtab`/`rename_bins_and_relable_assignments`/`build_greedy_obj`/`build_search_obj`) have **zero mutual calls** → clean split.
- `scheduling_table.py`: event group (`init_event`/`parse_event_msg`/`filter_msg`/`process_tab_event`/`event_handle_migrate_*`/`check_legality`) is **not called by any class method** → self-contained, clean split.

## 1. MUST REVIEW

### Decision B6-SPLIT-001 — split `global_sched.py` (variant B: re-export shell)

```text
Source: sched/global_sched.py (1057 lines, 10 top-level funcs)
Proposed:
  NEW sched/global_sched_alloc.py  (Step 2 预分配, 7 funcs):
    coleasing_alloc_cluster, coleasing_alloc_1bin, gurobi_split_solver,
    update_bp_result2_schedtab, rename_bins_and_relable_assignments,
    build_greedy_obj, build_search_obj
  NEW sched/global_sched_repack.py (Step 3 repack, 2 funcs):
    push_task_into_bins_new, push_step_new
  sched/global_sched.py → RE-EXPORT SHELL (variant B):
    explicit re-export of the 9 public symbols (no `import *`, to avoid np/math pollution);
    test_mem_planner — DEAD function (only ref is sim_main_old.py:53, the moved dead main).
    Kept in shell re-export for now (zero-risk backward compat); candidate for archive in a later dead-code batch.
Reason: file mixes Step2 (spatial partition) and Step3 (temporal repack) — two independent purposes.
Risk: LOW. AST proves zero mutual calls between groups. External importers = only sim_main.py:6,475,
       both satisfied by the shell → zero external change.
Companion import (per-group, deduplicated):
  alloc needs: ClusterGurobiSolverSemi2D, DiGraph, OrderedDict, PosTableInt, bin_iter_list,
               defaultdict, get_chains, get_initlist_and_biniter, math, new_bin, np, reduce, slack_comp,
               DataPipe, Monitor, MsgDispatcher, Scheduler
  repack needs: Buffer, Resource_model_int, TaskQueue, WatermarkStrategy, check_complete, check_miss,
                data_pipe_read, glb_alloc_new2, message_trigger_event_new, pendingToReady,
                DataPipe, Monitor, MsgDispatcher, Scheduler, get_initlist_and_biniter, new_bin
  (init_event import is dead — not called by any top-level fn — discard on split)
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py; rm the 2 new files
Recommended decision: approve
```

### Decision B6-SPLIT-002 — split `scheduling_table.py` (variant B: body retained + re-export)

```text
Source: sched/scheduling_table.py (1080 lines)
Proposed:
  NEW sched/scheduling_table_event.py (7 funcs, self-contained event group):
    init_event, parse_event_msg, filter_msg, process_tab_event,
    event_handle_migrate_from, event_handle_migrate_to, check_legality
  sched/scheduling_table.py → RETAIN BODY + append re-export of the 7 event fns:
    SchedulingTableInt class, dense_to_sparse (called by to_sparse_dict), new_bin (called by _new_bin),
    extend_dummy_bins, BinGenSelInt/BinSelInt/BinGenInt, calc_free_spaces, get_freespace_features stay.
Reason: data-structure file mixes core table + event handling (independent concern).
Risk: LOW. Event group not called by any class method (AST-verified). External importer of event fns:
       global_sched.py:34 `from sched.scheduling_table import init_event` — but init_event is a dead import
       in global_sched (not called) → will be discarded in B6-SPLIT-001 anyway. No live external breakage.
Recovery: git checkout archive/test_pipeline-20260612 -- sched/scheduling_table.py; rm scheduling_table_event.py
Recommended decision: approve
```

### Decision B6-SPLIT-003 — relocate slack_estim parasitic fns + delete dead test()

```text
Source: sched/slack_estim.py (561 lines)
Proposed (user choice r + g-1 + delete test):
  MOVE EstimCoreNums4Process + EstimCoreNums4Task → sched/sched_fn.py (their sole consumer; co-locate by use).
       Both are self-contained (deps: math / Dict only; zero dependence on slack_estim main-line fns).
       No circular import: slack_estim does NOT import sched_fn (one-way edge sched_fn→slack_estim; move removes that edge).
  DELETE sched_fn.py:27 `from sched.slack_estim import EstimCoreNums4Process` (fn now local to sched_fn).
  DELETE slack_estim.py::test() + `if __name__ == "__main__": test()` — dead self-test entry (0 external callers;
       superseded by main_approach flow; print-only, no assertions). NOTE: test() DID call deduce_cfg2/update_taskattr_dict,
       so it is a runnable smoke test, just obsolete (params hardcoded, no asserts, only prints).
  KEEP plot_timeline_graph in slack_estim main-line (called by deduce_cfg2:466 as debug viz; user: 绘图不变).
Reason: slack_estim mixes Step1 static pre-alloc (main line) with runtime core estimation (parasitic, used only by
        dead sim-loop sched_fn). User insight: "EstimCoreNums4Process 似乎算 sched_fn" — correct; relocate to consumer.
Risk: LOW. EstimCoreNums4Process/Task self-contained (AST-verified). EstimCoreNums4Task has 0 callers (dead).
       sched_fn.py:27 is top-level import but deleting it is safe once fn is local. No circular import.
       test() deletion safe (0 external callers, only __main__ entry).
Recovery: git checkout archive/test_pipeline-20260612 -- sched/slack_estim.py sched/sched_fn.py
Recommended decision: approve
```

## 2. SAFE SUMMARY

- Files split: 3 (global_sched, scheduling_table, slack_estim). Files created: 3 (global_sched_alloc, global_sched_repack, scheduling_table_event).
- Re-export shells: 2 (global_sched full shell; scheduling_table body + appended re-export).
- slack_estim: NOT split into new file — instead relocate its parasitic runtime fns into their consumer (sched_fn), delete dead test().
- External import changes: ZERO (variant B shells absorb global_sched/scheduling_table; slack_estim relocation handled by local move + delete one import line).
- Byte-identical guarantee: all moved functions unchanged; only new code is re-export shells (import-forwarding only).
- Smoke/regression gate: NOT YET RUN — mandatory after execution.

## 3. NO NEED TO REVIEW

- AST intra-file call graphs (this session): see `FILE_ADJUSTMENT_RECORD.md` 2026-06-27 B6 entry.
- Per-group import dependency analysis (above).
- `cleanup/move-ledger.csv` rows B6-SPLIT-001/002 (added with this packet).

## Response format

```text
approve all
approve B6-SPLIT-001 B6-SPLIT-002 B6-SPLIT-003
reject B6-SPLIT-002
pause batch
```

## Regression gate (post-execution, mandatory, gurobi env)

1. Import probe — all OK: `sched.global_sched`, `sched.global_sched_alloc`, `sched.global_sched_repack`, `sched.scheduling_table`, `sched.scheduling_table_event`, `sched.slack_estim`, `sched.sched_fn`, `approach_sim`, `approach_setup`, `main_approach`
2. Re-export check: `from sched.global_sched import coleasing_alloc_cluster, push_task_into_bins_new` works (shell)
3. `main_approach.py --help` PASS; `scripts.motiv_exp_runner --help` PASS; `scripts.abla_exp_runner --help` PASS

If any fails → restore from `archive/test_pipeline-20260612` and record in `FILE_ADJUSTMENT_RECORD.md`.

## Note on commit policy

Per user constraint "不要随意的提交": after execution + gate PASS, do NOT auto-commit. Report results and wait for explicit commit instruction.
