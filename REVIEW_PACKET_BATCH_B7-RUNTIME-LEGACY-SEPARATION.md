# REVIEW_PACKET_BATCH_B7-RUNTIME-LEGACY-SEPARATION

Batch: `B7-RUNTIME-LEGACY-SEPARATION` — physically relocate the deprecated runtime-scheduling module cluster into `sched/runtime_legacy/`, with re-export shells at original positions (variant B, like B6).

## Background & decision (2026-07-01)

- C-class cleanup investigation revealed the deprecated runtime-scheduling cluster (scheduler_agent / sched_fn / state_trans / sched_utils) is **not purely dead**: repack's retained path (`push_step_new`, to be re-enabled after fix) borrows live symbols from it, and `create_common_scheduler_elements` (active repack) instantiates `Scheduler` (as container) and `Monitor`.
- User decision: respect "类内先不管" (no internal symbol deletion), but **physically separate the whole deprecated cluster** into `sched/runtime_legacy/` because "后续老的 scheduler 类别可能会另有用途". Internal contents kept intact (no symbol deletion); only import paths adjusted as a necessary companion of the move.
- `monitor_agent.py` is **excluded** — `Monitor` class is live (instantiated by active repack path) + 2 helpers live. It stays in `sched/`.

## Live-borrowed symbols (must remain reachable via shells)

Active path borrows these from the 4 files (verified by call-graph closure + grep):

- `scheduler_agent.py`: `Scheduler` class (instantiated by `create_common_scheduler_elements`); class methods `updateRunningQueue`/`get_queues`/`get_buffer`/`get_state` (used by `push_step_new`)
- `state_trans.py`: `check_miss` / `check_complete` / `pendingToReady` / `release_rsc` (used by `push_step_new`)
- `sched_utils.py`: `data_pipe_read` (used by `push_step_new`)
- `sched_fn.py`: no live borrowers (entirely dead), but moves together as the cluster's import hub

External importers (all satisfied by shells, zero change):
- `global_sched_alloc.py:30-31`, `global_sched_repack.py:29-30`: `from sched.scheduler_agent import Scheduler, check_miss, check_complete, data_pipe_read, pendingToReady`
- `sim_main.py:14`, `test_mem_planner.py:12`: `from sched.scheduler_agent import Scheduler`
- `placement.py:4`, `state_trans.py:4`, `sched_utils.py:4`: TYPE_CHECKING `from sched.scheduler_agent import Scheduler` (runtime no-op)

## 1. MUST REVIEW

### B7-MOVE — relocate 4 files to sched/runtime_legacy/

```text
git mv sched/scheduler_agent.py  -> sched/runtime_legacy/scheduler_agent.py
git mv sched/sched_fn.py         -> sched/runtime_legacy/sched_fn.py
git mv sched/state_trans.py      -> sched/runtime_legacy/state_trans.py
git mv sched/sched_utils.py      -> sched/runtime_legacy/sched_utils.py
new  sched/runtime_legacy/__init__.py   (empty package marker)
Reason: physical separation of deprecated runtime-scheduling cluster (user intent: future reuse).
Risk: LOW. Files move as a group; internal logic/symbols untouched.
```

### B7-IMPORT-FIX — 3 intra-cluster import paths (necessary companion of move)

```text
runtime_legacy/scheduler_agent.py:18
  from sched.sched_fn import *          -> from sched.runtime_legacy.sched_fn import *
runtime_legacy/sched_fn.py:27
  from sched.sched_utils import *       -> from sched.runtime_legacy.sched_utils import *
runtime_legacy/sched_fn.py:28
  from sched.state_trans import *       -> from sched.runtime_legacy.state_trans import *
Reason: after move, absolute paths `from sched.X` would resolve to the shell (which re-exports
        back to runtime_legacy) → circular-load risk with `import *` losing symbols. Direct
        intra-cluster paths avoid the cycle. This is a path-only change (no symbol/logic change),
        the permitted companion of a move.
Other intra-file imports UNCHANGED (sched.scheduling_table / sched.monitor_agent / sched.placement /
        model.* / task.* / global_var — these modules are NOT moved, paths still valid).
```

### B7-SHELL — 4 re-export shells at original sched/ positions

```text
sched/scheduler_agent.py  ->  from sched.runtime_legacy.scheduler_agent import *
sched/sched_fn.py         ->  from sched.runtime_legacy.sched_fn import *
sched/state_trans.py      ->  from sched.runtime_legacy.state_trans import *
sched/sched_utils.py      ->  from sched.runtime_legacy.sched_utils import *
Reason: zero external import churn (variant B, same pattern as B6 global_sched shell).
        import * chosen (consistent with these files' existing import-* style; borrowers take
        specific symbols explicitly, so namespace pollution is harmless).
```

## 2. SAFE SUMMARY

- Files moved: 4 (git mv, history preserved). New package: `sched/runtime_legacy/` + `__init__.py`.
- Import-path fixes: 3 (intra-cluster only, path-only).
- Re-export shells: 4 (import * forward).
- Internal symbol deletion: ZERO (user: 内部不再删).
- External import changes: ZERO (shells absorb all).
- monitor_agent.py: NOT moved (live).
- Regression gate: mandatory post-execution.

## 3. Regression gate (gurobi env, mandatory)

1. Shell import: `import sched.scheduler_agent, sched.sched_fn, sched.state_trans, sched.sched_utils` — all OK
2. Symbol forward: `from sched.scheduler_agent import Scheduler, check_miss, check_complete, data_pipe_read, pendingToReady` — OK
3. `main_approach.py --help` PASS; `scripts.motiv_exp_runner --help` PASS; `scripts.abla_exp_runner --help` PASS

If any fails → `git checkout archive/test_pipeline-20260612 -- sched/scheduler_agent.py sched/sched_fn.py sched/state_trans.py sched/sched_utils.py` and remove runtime_legacy/.

## Note on commit policy

Per "不要随意的提交": after execution + gate PASS, do NOT auto-commit. Report and wait for explicit instruction.

## Recovery

```bash
git checkout archive/test_pipeline-20260612 -- sched/scheduler_agent.py sched/sched_fn.py sched/state_trans.py sched/sched_utils.py
rm -rf sched/runtime_legacy/
```
