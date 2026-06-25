# File adjustment record

This file records the chronological global action/change history: what happened, why it happened, validation, and how to recover it when files changed. For the current global cleanup state and next actions, use `CLEANUP_STATUS.md`.

Rule: after every cleanup action, preflight, environment verification, status-maintenance step, or batch execution, append an entry here. If the action did not change source/tracked files, explicitly record that it was a no-source-change or preflight/status-only action.

## 2026-06-15 status file role cleanup

Action type: status-maintenance

Reason:

- `PHASE1_STATUS_FOR_NEXT_AGENT.md` had grown beyond its original Phase 1 handoff role and was being used as a global state file.
- The user requested a clearer global status file and a clearer change-history file.

Changed files:

- Created `CLEANUP_STATUS.md` as the canonical global cleanup status.
- Replaced `PHASE1_STATUS_FOR_NEXT_AGENT.md` with a short compatibility pointer to `CLEANUP_STATUS.md`.
- Updated this file's role statement.

Scope kept out:

- No source code was changed.
- No cleanup batch was executed.
- No operation was performed on `/home/zhangchg/git_repo/scheduler`.

Recovery:

- Restore the previous status-file layout from the audit worktree or archive if needed.

## 2026-06-15 mandatory global record update rule

Action type: status-maintenance

Reason:

- The user requested an explicit rule that every later action must update both the global action state and the global change history.

Changed files:

- Updated `CLEANUP_STATUS.md` with the mandatory update rule.
- Updated `FILE_ADJUSTMENT_RECORD.md` with the same rule and this history entry.
- Restored `PHASE1_STATUS_FOR_NEXT_AGENT.md` as a short compatibility pointer to `CLEANUP_STATUS.md`.

Scope kept out:

- No source code was changed.
- No cleanup batch was executed.
- No operation was performed on `/home/zhangchg/git_repo/scheduler`.

Required future behavior:

- Update `CLEANUP_STATUS.md` after every action.
- Append to `FILE_ADJUSTMENT_RECORD.md` after every action.
- Continue to update batch packets and ledgers where applicable.

## B0-SHARED-NODE

Date: 2026-06-12

Decision executed: `P1-NODE-001`

Strategy: `archive-only`

User review: accepted on 2026-06-13

Changed files:

- Removed tracked `package.json` from the audit worktree. It only contained `{}`.
- Removed tracked `package-lock.json` from the audit worktree. Its `packages` object was empty.

Scope kept out:

- Did not touch untracked `node_modules/`.
- Did not change `.gitignore`.
- Did not run or execute any `P1-CACHE-*` decision.
- Did not run or execute any `P1-REMOVE-*` decision.
- Did not touch `.claude/`, `claude_talk/`, `doc/spec`, `doc/guide`, `doc/dev`, or protected tests.
- Did not change Python source, Python dependencies, docs, tests, CI, or runtime configuration.

Evidence:

- `package.json` had no scripts or dependencies.
- `package-lock.json` had no packages.
- Exact string search found no tracked references to `package.json`, `package-lock.json`, or `node_modules` outside cleanup reports.
- `test_pipeline` had 0 commits after `archive/test_pipeline-20260612` at the time of this batch, so no porting queue entry was needed.

Recovery:

```bash
git checkout archive/test_pipeline-20260612 -- package.json package-lock.json
```

Smoke:

- PASS in the documented `gurobi` Conda environment.
- Checked `scripts/motiv_exp_runner.py --help`, `scripts/abla_exp_runner.py --help`, and `main_approach.py --help`.
- Details: `cleanup/reports/batch-B0-SHARED-NODE-smoke.md`.

Ledgers updated:

- `cleanup/deletion-ledger.csv`
- `cleanup/dependency-ledger.csv`

Ledgers intentionally unchanged:

- `cleanup/move-ledger.csv`: no files were moved.
- `cleanup/gitignore-cache-ledger.csv`: no `git rm --cached` or `.gitignore` action was performed.
- `cleanup/reuse-candidates.csv`: reuse candidates were not touched.
- `cleanup/porting-queue.csv`: no later `test_pipeline` commits were found.

## B0-SHARED-DEP

Date: 2026-06-13

Decision executed: `P1-DEP-001`

Strategy: dependency manifest reconciliation

User review: accepted on 2026-06-13

Changed files:

- Updated `requirement.txt` by appending approved reachable/test dependencies: `gurobipy`, `h5py`, `networkx`, `psutil`, `pyinstrument`, `pytest`, `tdigest`, and `tqdm`.

Scope kept out:

- Did not install packages into the `gurobi` Conda environment.
- Did not remove `pyyaml`, `plotly`, or `bokeh`; those remain `P1-DEP-002` keep-review dependencies.
- Did not change Python source, docs, tests, CI, `.gitignore`, or cache files.
- Did not operate on `/home/zhangchg/git_repo/scheduler`.

Recovery:

```bash
git checkout archive/test_pipeline-20260612 -- requirement.txt
```

Smoke:

- PASS for the three documented `--help` probes in the `gurobi` Conda environment.
- At batch execution time, `mapper.mem_planner` failed because `tqdm` was not installed in the environment.
- After the user installed `tqdm`, `mapper.mem_planner` imports successfully; see `cleanup/reports/batch-B0-SHARED-DEP-smoke.md`.

Ledgers updated:

- `cleanup/dependency-ledger.csv`

## 2026-06-16 commit B0 results + Phase 1 artifacts to audit branch

Action type: status-maintenance (git checkpoint)

Reason:
- Phase 1 analysis and B0 batch execution had accumulated as uncommitted working-tree state in the audit worktree (staged D package.json/package-lock.json, M requirement.txt; untracked cleanup/, review packets, status files).
- Risk of loss across branch switch/reset. User requested securing the work to the audit branch, following the mandatory global-record-update rule.

Changed files:
- No source change in THIS action itself. This commit snapshots the existing B0 source changes (D package.json, D package-lock.json, M requirement.txt) and adds the Phase 1 analysis artifacts (cleanup/), review packets, CLEANUP_STATUS.md, FILE_ADJUSTMENT_RECORD.md, and the PHASE1_STATUS_FOR_NEXT_AGENT.md compatibility pointer to the audit branch.

Scope kept out:
- Did not operate on /home/zhangchg/git_repo/scheduler (main working tree, test_pipeline).
- Did not touch untracked files outside the audit worktree.
- Did not merge test_pipeline or the audit branch into main/master.
- Did not start Phase 3 (integration worktree untouched, still at master base).

Recovery:
- Undo this commit on the audit branch while keeping the changes staged: `git switch audit/minimal-from-test_pipeline-20260612 && git reset --soft d5bfda5`
- B0 source recovery: `git checkout archive/test_pipeline-20260612 -- package.json package-lock.json requirement.txt`

## 2026-06-16 B2-RUNTIME-TEST-INVENTORY — understanding phase

Action type: preflight (analysis / no-source-change)

Reason:
- Started B2-RUNTIME-TEST-INVENTORY. Per skill workflow + user instruction, first step is understanding the runtime new/old code boundary before running targeted import tests.
- Read doc/spec/readme.md, doc/spec/e2e_sched_sim_flow.md, doc/guide/deprecated_code.md, runtime-contract.md; traced import chains from the 3 declared entrypoints; cross-verified with static import grep.

Key finding (corrects doc/error):
- CLAUDE.md and deprecated_code.md mark `sched/scheduler_agent.py` and `sched/monitor_agent.py` as "deprecated, replaced by approach_sim.py / approach_collector.py". This is only TRUE for the simulation-loop role. These files are still LIVE dependencies of the config-generation path:
  - `monitor_agent.get_target_bin_id` and `monitor_agent.get_rsc_2b_released` are imported by `sched/pre_alloc_new.py` (active repack path).
  - `scheduler_agent` Scheduler class + sim-loop helpers (check_miss/check_complete/data_pipe_read/pendingToReady/load_sched_tab) are imported by `sched/global_sched.py` and `allocator_agent.py`.
  - => File-level deletion of either module would break the active repack path. Only symbol-level dead/live separation is safe.

Boundary established:
- NEW (active, KEEP): main_approach.py; approach_setup.py; sim_main.py::perform_bin_packing; approach_sim.py / approach_def / approach_sched / approach_initiator / approach_Eq / approach_collector; sched/global_sched.py (coleasing_alloc_cluster, push_task_into_bins_new); sched/pre_alloc_new.py.
- approach_sim.py (NEW sim backend) imports NONE of scheduler_agent/monitor_agent/allocator_agent.
- MIXED (need symbol-level): scheduler_agent.py, monitor_agent.py, allocator_agent.py (allocator_agent only imported by sim_main.py).
- 23 runtime-adjacent REVIEW candidates: reachability shows they are NOT in the active path. 6 are package __init__.py (real packages in use => KEEP); 13 are standalone analyze/run scripts (0 importers); 4 are standalone plot/test (appoach_plot6.py + approach_util33.py pair, test_core_allocation.py, test_repack_diagnostic.py).

Changed files: none (analysis only).

Scope kept out:
- No source change, no batch execution, no test run yet.
- Did not operate on /home/zhangchg/git_repo/scheduler main working tree.
- test_pipeline untouched.

Next:
- User approved order: B (symbol-level live/dead split of scheduler_agent/monitor_agent/allocator_agent) then A (independent scripts). For ambiguous symbols, ask the user directly (code author) for priors.

## 2026-06-16 B2-RUNTIME-TEST-INVENTORY — report and lightweight validation

Action type: inventory/preflight

Reason:
- Continued B2 after the understanding phase by producing a user-facing review packet and a machine-readable runtime/test inventory.
- Added lightweight validation through the documented WSL `zsh` + `conda activate gurobi` environment without running long experiments.

Changed files:
- Added `REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md`.
- Added `cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md`.
- Added `cleanup/reports/b2-runtime-test-inventory.csv`.
- Updated `CLEANUP_STATUS.md`.
- Updated this file.

Scope kept out:
- No runtime source code was changed.
- No tests were edited or deleted.
- No dependency manifest, CI file, `.gitignore`, README, CLAUDE.md, `.claude/`, `claude_talk/`, `doc/spec/`, `doc/guide/`, or `doc/dev/` file was modified.
- No operation was performed on `/home/zhangchg/git_repo/scheduler`.
- No old `P1-REMOVE-*` or `P1-CACHE-*` decision was executed.

Evidence and conclusion:
- Current runtime path remains `main_approach.py -> approach_setup.py -> sim_main.py::perform_bin_packing() -> approach_initiator.py -> approach_sim.py::run_simulation() -> approach_collector.py`.
- `sim_main.py` is still active as the configuration/bin-packing backend.
- `approach_sim.py` is the current event-driven runtime simulator.
- `sched/scheduler_agent.py` and `sched/monitor_agent.py` remain `KEEP_UNTIL_SPLIT`, not file-level deletion candidates.
- `scripts/test_alloc_lat.py` hardcodes `/home/zhangchg/git_repo/scheduler` in `sys.path`; do not run it until that is fixed or isolated.

Validation:
- WSL startup probe passed.
- In `conda activate gurobi`, `main_approach.py --help` passed.
- In `conda activate gurobi`, `python -m scripts.motiv_exp_runner --help` passed.
- In `conda activate gurobi`, `python -m scripts.abla_exp_runner --help` passed.
- In `conda activate gurobi`, targeted imports passed for current runtime modules including `main_approach`, `approach_setup`, `approach_sim`, `approach_def`, `approach_sched`, `approach_collector`, `sim_main`, `sched.global_sched`, `sched.slack_estim`, `sched.packing_solver.chain_slack_assign`, and `mapper.mem_planner`.

Recovery:
- These are audit/status artifacts only. To discard this B2 report layer: `git restore CLEANUP_STATUS.md FILE_ADJUSTMENT_RECORD.md && rm REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md cleanup/reports/b2-runtime-test-inventory.csv`

## 2026-06-16 B2 symbol-level analysis complete + move packet prepared

Action type: preflight (analysis / no-source-change) — packet + ledger only, execution pending approval

Reason:
- Completed symbol-level live/dead split of the mixed modules (scheduler_agent / monitor_agent / allocator_agent) using AST call-graph analysis (not text grep, because `from sched.sched_fn import *` re-exports create false-definition locations).
- User confirmed priors: (1) the sim chain sim_main::main + allocator_agent.{glb_sched,cyclic_sched,sched_step,period_boader_display} + AllocatorInt is dead; (2) others() is split-off infrequent cases from main, move together; (3) move allocator_agent.py whole; (4) class-internal dead methods of Scheduler deferred (先不管).
- User chose strategy: MOVE (move-reference), not delete. Code segments → same-name `_unused.py`; whole files → `unused/` / `scripts/old/`.

Key technical finding:
- The real dead root is the OLD SIMULATION LOOP SCC, not scheduler_agent.py per se. scheduler_agent.Scheduler is still LIVE on the active repack path (instantiated in create_common_scheduler_elements → fed into perform_bin_packing). What's dead is the timestep loop: sim_main::main/others → allocator_agent → Scheduler stepping methods → sched_fn/state_trans/sched_utils standalone fns.
- check_miss/check_complete/pendingToReady exist in TWO forms: a LIVE standalone function (state_trans.py / sched_utils.py, the real definition) AND a DEAD Scheduler class method (scheduler_agent.py, 0 .method() calls). User memory correct: the standalone fns are used by repack path (via global_sched), the class methods are dead.
- preprocess_args(L692) is ACTIVE (approach_setup.py:106) and interleaved between the two dead functions to be moved — must NOT be swept up.

Packet + ledger (audit artifacts only, no source change):
- Created REVIEW_PACKET_BATCH_B2-MOVE-DEAD-SIM-CHAIN.md with 3 decisions: B2-MOVE-001 (allocator_agent.py → unused/), B2-MOVE-002 (sim_main others+main → sim_main_unused.py), B2-MOVE-003 (scripts/repack_sweep.py → scripts/old/).
- Appended 3 rows to cleanup/move-ledger.csv (status=proposed).
- Out of scope this batch: Scheduler class-internal dead methods; sched_fn/state_trans/sched_utils dead standalone fns (deferred — import * tangle).

Changed files: none in source. Only audit artifacts (packet + move-ledger + this record + CLEANUP_STATUS).

Scope kept out:
- No source moved yet. Execution pending user approval of the 3 decisions.
- test_pipeline untouched.

Recovery (if executed and needs undo):
- git checkout archive/test_pipeline-20260612 -- allocator_agent.py scripts/repack_sweep.py sim_main.py

Next:
- Awaiting user approval (approve all / per-decision / pause). On approval: execute the 3 moves + companion import cleanup, then run the regression gate (6-line import probe + 3× --help).

## 2026-06-16 B2-MOVE-DEAD-SIM-CHAIN — EXECUTED (3 moves, regression gate PASS)

Action type: execution (move-reference)

Reason:
- User approved option 2: execute B2-MOVE-001 + B2-MOVE-002 + B2-MOVE-003 together (001 and 002 are code-coupled: sim_main.py:17 top-level `from allocator_agent import ...` exists only to serve the dead main/others, so allocator_agent cannot move alone without also moving main).
- Strategy: MOVE (not delete). Dead sim-loop code relocated to unused/ and *_unused.py.

Executed changes:
- B2-MOVE-001: `git mv allocator_agent.py unused/allocator_agent.py` (whole file, 825 lines).
- B2-MOVE-002: sliced `sim_main.py::others()` (L606-691) + `::main()` (L726-870, incl. decorator comment L725) + `if __name__` block (L952-957) into new `sim_main_unused.py` (273 lines). sim_main.py 957→716 lines, pure deletion (241 lines removed, no rewrite). Deleted top-level imports L17 (`from allocator_agent import glb_sched, cyclic_sched`) and L18 (`from model.event_gen.e2e_latency import discrete_event_sim`) — both only served the moved functions.
- B2-MOVE-003: `git mv scripts/repack_sweep.py scripts/old/repack_sweep.py` (whole file).

Regression gate (gurobi env) — ALL PASS:
- import probe: sched.global_sched, sched.scheduler_agent, sched.monitor_agent, approach_sim, approach_setup, main_approach → all OK
- main_approach.py --help → PASS
- scripts.motiv_exp_runner --help → PASS
- scripts.abla_exp_runner --help → PASS
- allocator_agent no longer importable from root (expected, moved to unused/)

Scope kept out:
- Scheduler class-internal dead methods: NOT touched (deferred per user).
- sched_fn/state_trans/sched_utils dead standalone fns: NOT touched (deferred — import * tangle).
- No operation on /home/zhangchg/git_repo/scheduler main working tree (test_pipeline). All done in audit worktree.
- Surviving sim_main.py code is byte-identical to original (pure deletion, no rewrite) — verified by git diff showing 241 deletions, 0 insertions in sim_main.py.

Recovery:
- git checkout archive/test_pipeline-20260612 -- allocator_agent.py scripts/repack_sweep.py sim_main.py
- rm unused/allocator_agent.py scripts/old/repack_sweep.py sim_main_unused.py

## 2026-06-16 commit B2-MOVE-DEAD-SIM-CHAIN results to audit branch

Action type: status-maintenance (git checkpoint)

Reason:
- B2-MOVE-DEAD-SIM-CHAIN executed (3 moves, regression gate PASS) but uncommitted. Securing to the audit branch to avoid loss across branch switch/reset, per the mandatory record-update rule.

Changed files (this commit snapshots existing B2 work):
- No new source change in this action. Commits the executed moves: allocator_agent.py → unused/ (rename), sim_main.py (-241 lines, others+main+__name__ removed), sim_main_unused.py (new, 273 lines), scripts/repack_sweep.py → scripts/old/ (rename).
- Plus audit artifacts: CLEANUP_STATUS.md, FILE_ADJUSTMENT_RECORD.md, cleanup/move-ledger.csv (3 rows → executed), REVIEW_PACKET_BATCH_B2-MOVE-DEAD-SIM-CHAIN.md, REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md, cleanup/reports/b2-runtime-test-inventory.*.

Scope kept out:
- Did not operate on /home/zhangchg/git_repo/scheduler (test_pipeline). All in audit worktree.
- Did not merge into main/master. Phase 3 not started.

Recovery:
- Undo commit keeping changes staged: `git switch audit/minimal-from-test_pipeline-20260612 && git reset --soft d9bf216`
- Restore moved files: `git checkout archive/test_pipeline-20260612 -- allocator_agent.py scripts/repack_sweep.py sim_main.py`
