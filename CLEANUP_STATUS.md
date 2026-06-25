# Cleanup status

Last updated: 2026-06-16 (B2-MOVE-DEAD-SIM-CHAIN executed + committed to audit branch; gate PASS)

This is the canonical global status file for the scheduler cleanup work. It supersedes `PHASE1_STATUS_FOR_NEXT_AGENT.md` as the main entry point for future agents.

Use this file to understand the current cleanup state, approved decisions, executed batches, protected areas, and next actions. Use `FILE_ADJUSTMENT_RECORD.md` for the global action/change history.

## Mandatory update rule

After every cleanup action, preflight, environment verification, status-maintenance step, or batch execution, update both global records before handing off:

- `CLEANUP_STATUS.md`: update the current global action state, decision status, tracked diff, validation state, and recommended next action.
- `FILE_ADJUSTMENT_RECORD.md`: append a chronological action-history entry. If the action did not change source/tracked files, explicitly record that it was a no-source-change or preflight/status-only action.

Per-batch packets and ledgers are still required when applicable, but they do not replace these two global records.

## Scope

- Source repo: `/home/zhangchg/git_repo/scheduler`
- Active cleanup worktree: `/home/zhangchg/git_repo/scheduler-audit-20260612`
- Integration worktree: `/home/zhangchg/git_repo/scheduler-integration-20260612`
- Source branch snapshot: `test_pipeline @ d5bfda5`
- Archive tag: `archive/test_pipeline-20260612`
- Archive branch: `archive/test_pipeline-20260612-branch`
- Mainline basis for now: `master`

Rules still in force:

- Do not operate directly on `/home/zhangchg/git_repo/scheduler`.
- Work only in audit/integration worktrees.
- Ignore untracked files unless the user explicitly brings them into scope.
- Focus on git tracked / git cache state.
- Do not execute old `P1-REMOVE-*` or `P1-CACHE-*` actions.

## Current audit branch state

The B0 source changes, Phase 1 analysis artifacts (`cleanup/`), review packets, and the two global status files were committed to the audit branch on 2026-06-16 (first commit beyond `test_pipeline`; see `FILE_ADJUSTMENT_RECORD.md`).

Current uncommitted audit-worktree changes are B2 inventory/status artifacts only:

- `REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/b2-runtime-test-inventory.csv`
- updated `CLEANUP_STATUS.md`
- updated `FILE_ADJUSTMENT_RECORD.md`

No runtime source file was changed by this B2 inventory/report step.

Committed B0 source changes (recovery: `git checkout archive/test_pipeline-20260612 -- <path>`):

- `B0-SHARED-NODE` (P1-NODE-001, archive-only): removed empty `package.json` + `package-lock.json`.
- `B0-SHARED-DEP` (P1-DEP-001): updated `requirement.txt` (+8 deps).

## Executed batches

### B0-SHARED-NODE

- Decision: `P1-NODE-001`
- Status: executed and accepted by user on 2026-06-13.
- Changed files:
  - removed `package.json`
  - removed `package-lock.json`
- Strategy: `archive-only`
- Recovery:

```bash
git checkout archive/test_pipeline-20260612 -- package.json package-lock.json
```

Validation:

- `scripts/motiv_exp_runner.py --help`: PASS in `gurobi`
- `scripts/abla_exp_runner.py --help`: PASS in `gurobi`
- `main_approach.py --help`: PASS in `gurobi`

### B0-SHARED-DEP

- Decision: `P1-DEP-001`
- Status: executed and accepted by user on 2026-06-13.
- Changed file:
  - updated `requirement.txt`
- Added dependencies:
  - `gurobipy`
  - `h5py`
  - `networkx`
  - `psutil`
  - `pyinstrument`
  - `pytest`
  - `tdigest`
  - `tqdm`
- Kept review dependencies:
  - `pyyaml`
  - `plotly`
  - `bokeh`
- Recovery:

```bash
git checkout archive/test_pipeline-20260612 -- requirement.txt
```

Validation:

- `scripts/motiv_exp_runner.py --help`: PASS in `gurobi`
- `scripts/abla_exp_runner.py --help`: PASS in `gurobi`
- `main_approach.py --help`: PASS in `gurobi`
- `import mapper.mem_planner`: PASS after the user installed `tqdm`

## Completed preflights

### B1-REVIEW-TRIAGE-PREFLIGHT

Status: preflight complete. No cleanup executed.

Included decisions:

- `P1-DEP-002`
- `P1-REVIEW-DOCS-001`
- `P1-REVIEW-RUNTIME-001`
- `P1-REUSE-001`
- `P1-SYMBOL-001`

Result:

- `P1-DEP-002`: keep `bokeh`, `plotly`, and `pyyaml` under review.
- `P1-REVIEW-DOCS-001`: protected docs are KEEP, not cleanup candidates.
- `P1-REVIEW-RUNTIME-001`: runtime-adjacent files require targeted tests before cleanup.
- `P1-REUSE-001`: reuse candidates are preserved by default.
- `P1-SYMBOL-001`: symbol candidates are record-only and do not authorize symbol deletion.

Report files:

- `REVIEW_PACKET_BATCH_B1-REVIEW-TRIAGE-PREFLIGHT.md`
- `cleanup/reports/batch-B1-REVIEW-TRIAGE-preflight.md`
- `cleanup/reports/b1-review-triage-actions.csv`

## B2-RUNTIME-TEST-INVENTORY (in progress)

### Understanding phase — COMPLETE (2026-06-16)

Goal: separate NEW runtime (KEEP) from OLD/dead before any symbol-level action.

Key finding (corrects `CLAUDE.md` / `deprecated_code.md`):

- `sched/scheduler_agent.py` and `sched/monitor_agent.py` are marked "deprecated, replaced by `approach_sim.py`/`approach_collector.py`". This is TRUE only for the **simulation-loop** role. They are still **LIVE** dependencies of the config-generation path:
  - `monitor_agent.get_target_bin_id`, `monitor_agent.get_rsc_2b_released` ← imported by active `sched/pre_alloc_new.py` (repack path).
  - `scheduler_agent` Scheduler class + sim-loop helpers (check_miss/check_complete/data_pipe_read/pendingToReady/load_sched_tab) ← imported by `sched/global_sched.py` + `allocator_agent.py`.
  - => **File-level deletion breaks the active repack path. Only symbol-level dead/live split is safe.**

Boundary (verified by import-chain tracing + static grep):

- NEW (active, KEEP): `main_approach.py`; `approach_setup.py`; `sim_main.py::perform_bin_packing`; `approach_sim.py` + `approach_def/sched/initiator/Eq/collector.py`; `sched/global_sched.py` (coleasing_alloc_cluster, push_task_into_bins_new); `sched/pre_alloc_new.py`.
- `approach_sim.py` (NEW sim backend) imports NONE of scheduler_agent/monitor_agent/allocator_agent.
- MIXED (need symbol-level split): `scheduler_agent.py`, `monitor_agent.py`, `allocator_agent.py`.
- 23 runtime-adjacent REVIEW candidates are NOT in the active path: 6 package `__init__.py` (real packages → KEEP), 13 standalone analyze/run scripts (0 importers), 4 standalone plot/test.

### User-approved execution order

1. **B** — symbol-level live/dead split of `scheduler_agent.py` / `monitor_agent.py` / `allocator_agent.py` (the high-value, higher-risk target). For ambiguous symbols, **ask the user directly** (code author) for priors.
2. **A** — the independent scripts (analyze/*, run/*, plot/test pair) via `archive-only` / `move-reference`.

### Symbol-level split — COMPLETE (2026-06-16, AST-based)

Real finding (AST, not text grep — `from sched.sched_fn import *` creates false-definition locations):

- The dead root is the **OLD SIMULATION LOOP SCC**, not `scheduler_agent.py` per se. `Scheduler` class is **LIVE** on the active repack path (`create_common_scheduler_elements` instantiates it → fed into `perform_bin_packing`).
- Dead cluster: `sim_main::main()` + `others()` → `allocator_agent.{glb_sched,cyclic_sched,sched_step,period_boader_display,AllocatorInt}` → `Scheduler` stepping methods → `sched_fn`/`state_trans`/`sched_utils` standalone fns. Reachable only from the superseded `sim_main::main()`.
- `check_miss`/`check_complete`/`pendingToReady` have TWO forms: LIVE standalone fn (`state_trans.py`/`sched_utils.py`) + DEAD `Scheduler` class method (0 `.method()` calls).
- `preprocess_args` (sim_main L692) is ACTIVE (approach_setup.py:106), interleaved between dead `others`/`main` — must not be swept up.

### B2-MOVE-DEAD-SIM-CHAIN — EXECUTED 2026-06-16 (regression gate PASS)

Strategy: **MOVE (move-reference), not delete**. All 3 decisions executed together (001+002 code-coupled via sim_main.py top-level import).

| Decision | Object | Target | Status |
|----------|--------|--------|--------|
| B2-MOVE-001 | `allocator_agent.py` (whole file, 825 lines) | `unused/allocator_agent.py` | ✅ executed |
| B2-MOVE-002 | `sim_main.py::others()`+`::main()`+`if __name__` (241 lines) | `sim_main_unused.py` (273 lines) | ✅ executed (pure deletion, byte-identical surviving code) |
| B2-MOVE-003 | `scripts/repack_sweep.py` (whole file) | `scripts/old/repack_sweep.py` | ✅ executed |

Companion import cleanup done: deleted `sim_main.py` top-level L17 (allocator_agent) + L18 (discrete_event_sim) — both served only the moved functions.

Regression gate (gurobi env) — ALL PASS: 6-line import probe OK; `main_approach.py`/`motiv_exp_runner`/`abla_exp_runner` `--help` all PASS; `allocator_agent` no longer importable from root (expected).

Deferred (not done this batch):
- `Scheduler` class-internal dead methods (user: 先不管)
- `sched_fn`/`state_trans`/`sched_utils` dead standalone fns (import * tangle — next batch)



Do not treat `scheduler_agent.py` / `monitor_agent.py` as REMOVABLE at file level. Any packet proposing file-level deletion of these is a violation; only symbol-level dead-function/method removal is acceptable.

### Inventory packet and lightweight validation — COMPLETE (2026-06-16)

Added B2 reports:

- `REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/b2-runtime-test-inventory.csv`

Additional B2 findings:

- Current runtime path remains `main_approach.py -> approach_setup.py -> sim_main.py::perform_bin_packing() -> approach_initiator.py -> approach_sim.py::run_simulation() -> approach_collector.py`.
- `scripts/test_alloc_lat.py` is blocked for direct execution because it hardcodes `/home/zhangchg/git_repo/scheduler` into `sys.path`, which would bypass the audit worktree.
- `test_event_update.py` and `test_mapping.py` need unresolved import triage for `approach_plot` before execution.
- No test deletion is authorized.

Validation:

- WSL startup probe: PASS.
- `main_approach.py --help`: PASS in `gurobi`.
- `python -m scripts.motiv_exp_runner --help`: PASS in `gurobi`.
- `python -m scripts.abla_exp_runner --help`: PASS in `gurobi`.
- Targeted imports for current runtime modules: PASS in `gurobi`.



### 2026-06-15 status-file role cleanup

- Created `CLEANUP_STATUS.md` as the canonical global status file.
- Downgraded `PHASE1_STATUS_FOR_NEXT_AGENT.md` to a compatibility pointer.
- Clarified that `FILE_ADJUSTMENT_RECORD.md` records the global action/change history.

### 2026-06-15 mandatory update rule

- Added the rule that every future action must update both `CLEANUP_STATUS.md` and `FILE_ADJUSTMENT_RECORD.md`.
- This applies to executions, preflights, environment checks, and status-only maintenance.
- Confirmed `PHASE1_STATUS_FOR_NEXT_AGENT.md` remains only a compatibility pointer to `CLEANUP_STATUS.md`.

### 2026-06-16 commit B0 + Phase 1 artifacts to audit branch

- Snapshotted the accumulated audit worktree state (B0 source changes + Phase 1 analysis + status files) into the first commit on the audit branch beyond `test_pipeline`.
- Follows the mandatory update rule; both global records updated as part of the commit.
- No source change in the commit action itself; `test_pipeline` and the integration worktree untouched.

## Decision status

Executed and accepted:

- `P1-NODE-001`
- `P1-DEP-001`

Active constraints:

- `P1-GIT-001`: use `master` as current mainline basis.
- `P1-GIT-002`: untracked files are out of scope.
- `P1-KEEP-001`: current KEEP seed is accepted.
- `P1-REUSE-001`: preserve reuse candidates.
- `P1-SYMBOL-001`: symbol candidates are record-only.

Keep under review:

- `P1-DEP-002`: `bokeh`, `plotly`, `pyyaml`.
- `P1-REVIEW-DOCS-001`: docs not covered by explicit KEEP/protected policy.
- `P1-REVIEW-RUNTIME-001`: runtime-adjacent files until targeted tests exist.
- B2 runtime/test inventory follow-ups:
  - `B2-RUNTIME-LEGACY-001`: keep `sched/scheduler_agent.py` until old imports are split.
  - `B2-RUNTIME-LEGACY-002`: keep `sched/monitor_agent.py` until old imports are split.
  - `B2-RUNTIME-TEST-004`: do not run `scripts/test_alloc_lat.py` until its hardcoded original-worktree path is fixed or isolated.

Not approved:

- `P1-CACHE-001`
- `P1-CACHE-002`
- all `P1-REMOVE-*` batches

## Protected paths

Do not delete or bulk-move these areas:

- `.claude/`
- `claude_talk/`
- `doc/spec/`
- `doc/guide/`
- `doc/dev/`
- tests covered by rejected `P1-REMOVE-B19-TESTS`

## Test environment

Repository checks must use the documented Conda environment:

```bash
conda activate gurobi
```

Preferred command shape from Windows/Codex:

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help'
```

System `python3` outside this environment is not valid for this repo.

## Recommended next action

B2-MOVE-DEAD-SIM-CHAIN **EXECUTED** (3 moves, gate PASS). Changes are uncommitted in the audit worktree — consider committing to the audit branch to secure them (recovery commands in `FILE_ADJUSTMENT_RECORD.md`).

Next options:
1. **Commit B2 results** to the audit branch (secures the 3 moves + artifacts; recommended before next batch).
2. **Deferred sim-loop cleanup**: `Scheduler` class-internal dead methods + `sched_fn`/`state_trans`/`sched_utils` dead standalone fns (the `import *` tangle — needs import-cleanup coordination).
3. **Batch A**: independent scripts (`analyze/*`, `run/*`, `appoach_plot6.py`+`approach_util33.py`, `test_core_allocation.py`, `test_repack_diagnostic.py`) via move-reference.
4. `B2-DOC-INVENTORY`, `B2-OPTIONAL-DEPS-DECISION`.

Do not start cleanup execution from old `P1-REMOVE-*` or `P1-CACHE-*` decisions.

## File roles

- `CLEANUP_STATUS.md`: current global status and next-action entry point.
- `FILE_ADJUSTMENT_RECORD.md`: chronological global action/change history; include recovery commands for actual file changes.
- `REVIEW_PACKET_BATCH_*.md`: per-batch review details.
- `cleanup/reports/*`: machine-readable or supporting audit reports.
