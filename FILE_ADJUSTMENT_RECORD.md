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

## 2026-06-16 B2-MOVE-DEAD-SIM-FNS analysis — MAJOR CORRECTION + packet ready

Action type: preflight (analysis / no-source-change)

Reason:
- Analyzed the deferred sim-loop standalone functions (sched_fn/state_trans/sched_utils). Prepared move packet.

⚠️ MAJOR ANALYSIS CORRECTION (important for trust):
- First-pass reverse analysis (calling-context heuristic) WRONGLY concluded all 23 functions dead.
- Forward-reachability BFS from confirmed-live roots corrected it: push_task_into_bins_new (repack core) → push_step_new (global_sched.py:131) → check_miss/check_complete/data_pipe_read/pendingToReady → release_rsc is LIVE.
- Without checking global_sched.py:131 (the push_step_new call site), 5 live functions would have been wrongly moved, breaking the repack path. Lesson: liveness MUST be proven by forward reachability from entrypoints, not by "all my callers look dead" heuristics — because a dead-looking caller (push_step_new) can itself be called by a live function (push_task_into_bins_new).

Accurate dead/alive list (forward reachability):
- LIVE (KEEP, 5): state_trans.{check_miss, check_complete, release_rsc, pendingToReady}, sched_utils.data_pipe_read
- DEAD (move, 18): sched_fn (all 6), state_trans {throttleToReady, check_throttle, pendingToReady_cbs}, sched_utils {9 fns except data_pipe_read}

Packet + plan:
- Created REVIEW_PACKET_BATCH_B2-MOVE-DEAD-SIM-FNS.md (decisions B2-MOVE-004/005/006).
- Move 18 dead fns → sched/<module>_unused.py (3 files). Keep shell (imports + TYPE_CHECKING) + live fns.
- Companion import cleanup: NONE — global_sched named imports all live; import * self-adapts.
- Scheduler class-internal dead methods: out of scope (先不管).
- Execution method: AST-driven function-name slicing (not line-number) to guarantee byte-identical survivors.

Changed files: none in source. Only audit artifact (this packet).

Scope kept out:
- No source moved yet. Execution pending user confirmation of the corrected 5-function LIVE list (given the analysis error, explicit confirmation requested before the interleaved cut).
- test_pipeline untouched.

Next:
- Awaiting user confirmation of LIVE list {check_miss, check_complete, release_rsc, pendingToReady, data_pipe_read}, then execute B2-MOVE-004/005/006 + regression gate.

## 2026-06-16 bin_packing 函数与文档现状分析

Action type: preflight (analysis / no-source-change)

Reason:
- User: bin_packing 部分写过好几版，函数在文件中组织非常乱，先整理和分析。
- Did AST reachability closure from perform_bin_packing (not text grep — same import-* caveat as B2). Produced structured inventory report.

Key findings:
- 3 generations stacked across 5 files (3431 lines), ~40% dead.
- LIVE path (guided/scratch): sim_main::perform_bin_packing → global_sched.{coleasing_alloc_cluster, push_task_into_bins_new} → pre_alloc_new.glb_alloc_new2 + bin_ops + gurobi_MP_semi2DClst.
- DEAD whole files: sched/bin_ops.old.py (775 lines, 0 live importers), sched/pre_alloc.py (592 lines, only called by bin_ops.old). These are the v0/v1 generations replaced by _new/_new2.
- DEAD solver variants: packing_solver/gurobi_semi2Dclst_mapping.py + _mapping2.py (GurobiSemi2DClstMapping, 0 live callers). Live one is gurobi_MP_semi2DClst.py.
- BROKEN branch: perform_bin_packing `elif algorithm=="full"` calls single_turn_solver, but its def is commented out (global_sched.py:947 `# def single_turn_solver`). Would ImportError if triggered.
- CONDITIONAL dead: test_mem_planner only in `algorithm=="mem_plan"` branch; default guided never hits it.
- First-round "external caller" check MISJUDGED several fns as dead (coleasing_alloc_1bin, build_greedy_obj, build_search_obj, etc.); reachability closure corrected them — they are indirectly called by live coleasing_alloc_cluster.

Output: cleanup/reports/bin_packing_inventory.md (structured analysis, not a cleanup packet).

Changed files: none (analysis only). Added cleanup/reports/bin_packing_inventory.md + this record + CLEANUP_STATUS update.

Scope kept out:
- No source moved/deleted. No execution. test_pipeline untouched.
- Did not cover chain_slack_assign.py, fit.py, ref_alloc_search.py, test scripts, or the 28 doc files (deferred to follow-up if user wants).

Next:
- Awaiting user decision on整理方向: (1) move bin_ops.old.py + pre_alloc.py → unused/; (2) move dead solver variants → packing_solver/old/; (3) handle single_turn_solver broken branch; (4) test_mem_planner keep?; (5) doc archival.

## 2026-06-16 bin_packing spec 对照 + 文档层深挖

Action type: preflight (analysis / no-source-change)

Reason:
- User: 参考 binpack_solver_spec.md 梳理函数关系；继续深挖文档层。User decisions: test_mem_planner 留着, single_turn_solver 不管.
- Cross-checked spec-described function hierarchy vs actual AST call graph; resolved 3 apparent contradictions.

Key findings:
- spec (binpack_solver_spec.md §2.1) function hierarchy is ACCURATE — all described functions map to live code. Trustworthy as the canonical function map.
- 5 contradictions resolved (import exists ≠ used):
  1. gurobi_semi2Dclst_mapping import in global_sched.py is COMMENTED OUT (L1049/1067) → both mapping files confirmed dead.
  2. pre_alloc.py glb_alloc_new imported by global_sched.py:25 but NEVER called (dead import fossil); only caller is bin_ops.old.py (dead).
  3. sim_main.py does NOT import pre_alloc (earlier grep false positive).
- check_miss/check_complete/pendingToReady/data_pipe_read: spec says push_step_new calls them — TRUE, but real defs are in state_trans.py/sched_utils.py; global_sched gets them via scheduler_agent's import-* re-export (B2 "fake body" pattern). Functionally fine, pollutes definition-location tracking.
- Dead whole files (0 live deps, safe to move): bin_ops.old.py (775L), pre_alloc.py (592L), message_handler_old.py (246L, new finding), gurobi_semi2Dclst_mapping.py + _mapping2.py.
- Dead import fossil: global_sched.py:25 `from sched.pre_alloc import glb_alloc_new` (never called).
- Dead function: global_sched.py naive_iso (0 calls).
- User decisions locked: test_mem_planner KEEP (mem_plan branch); single_turn_solver LEAVE AS-IS (broken branch not touched).

Doc layer findings:
- 3 tiers: spec/(25, keep) / dev/(10, process logs) / guide/(8, navigation).
- guide/deprecated_code.md is INCOMPLETE — only lists bin_ops.old.py; misses pre_alloc.py, message_handler_old.py, dead solver variants. Needs update.
- doc/dev/ process docs (code_cleanup_2026/ablation_dev/repack_debug_instrumentation/claude_revise) candidates for archival to doc/dev/archive/.

Output: cleanup/reports/bin_packing_function_map.md (spec↔code function map + doc analysis).

Changed files: none (analysis only). Added bin_packing_function_map.md + this record + CLEANUP_STATUS update.

Scope kept out: no source/doc change, no execution. test_pipeline untouched.

Next: awaiting user decision on cleanup packet — proposed low-risk moves (bin_ops.old/pre_alloc/message_handler_old/dead solvers → unused/old/) + dead-import cleanup + naive_iso removal + deprecated_code.md update.

## 2026-06-16 B3-BINPACK-DEAD-MOVE packet prepared + pre_alloc_new.py 深度分析

Action type: preflight (analysis + packet / no-source-change)

Reason:
- User: pre_alloc_new.py 函数定义混乱，含义/用途/文档覆盖存疑，深入思考后出 packet.
- Deep-analyzed pre_alloc_new.py internal call graph (AST); cross-checked spec coverage; identified staleness.

pre_alloc_new.py findings:
- 8 top-level symbols. Internal call graph: glb_alloc_new2→allocate_rsc_4_process_new2→{bin_sel, check_and_preemt_alloc, Bp_print_util}; check_and_preemt_alloc→{index_occupy_by_id_chunk_ver, push_into_bin}.
- bin_select_new (L287-328) is DEAD — a wrapper that only calls bin_sel, 0 external callers. Name collision with bin_sel (similar name, overlapping role). spec §2.1 only mentions bin_sel.
- L627-748 (~122 lines) is commented-out test code referencing pre_alloc.py old fns (check_and_preemt_at_queue/get_preempt_candi/bin_select) — fully detached.
- spec §8.1 STALE: §8.1.1 sys.exit(1)→already raise ResourceInsufficientError (L206); §8.1.2 tot_cores=300→already binpack_cfg.get (L158). Suggestions implemented but spec not updated.
- Known issue (NOT fixed this round — rewrite scope): index_occupy_by_id_chunk_ver param named process_sort but called with key=lambda _p:_p.deadline (L453). Callable so runs, but semantic mismatch.

Packet: REVIEW_PACKET_BATCH_B3-BINPACK-DEAD-MOVE.md (11 decisions: 5 whole-file moves + 3 symbol moves + 1 import cleanup + 2 doc updates). Ledger rows B3-MOVE-001~008 + B3-CLEAN-001 added (status=proposed).

Dependencies: B3-MOVE-002 (pre_alloc.py) BINDS B3-CLEAN-001 (delete global_sched.py:25 dead import) — same coupling pattern as B2. Rest independent.

Out of scope (user decisions): test_mem_planner KEEP; single_turn_solver LEAVE AS-IS; param-semantic-mismatch record-only.

Changed files: none in source. Added packet + 9 ledger rows + this record + CLEANUP_STATUS update.

Scope kept out: no source moved. Execution pending approval. test_pipeline untouched.

Next: awaiting user approval (approve all / per-group / pause).

## 2026-06-16 B3 packet double-check vs 顶层 spec (readme/e2e/test_plan)

Action type: verification (no-source-change)

Reason:
- User requested cross-checking B3 packet against doc/spec/readme.md + e2e_sched_sim_flow.md + test_plan.md before approval.

Result: ZERO contradiction. B3 packet validated.
- DEAD symbols to move (bin_ops.old/pre_alloc/message_handler_old/gurobi_semi2Dclst_mapping*/bin_select_new/naive_iso/glb_alloc_new): 0 references in all 3 top-level specs.
- LIVE path retained (coleasing_alloc_cluster/push_task_into_bins_new/perform_bin_packing/apply_forced_num_cores): explicitly described as active in specs (test_plan:11/25/29/92/100, e2e:24/61/109/63/114/123, readme:37).
- User decisions (single_turn_solver LEAVE / test_mem_planner KEEP): also 0 refs in top-level specs → spec only recognizes guided algo (test_plan:27/288); mem_plan/full branches are not spec-sanctioned paths. Consistent with user decisions.
- Granularity note: top-level specs describe at "algorithm name" granularity (coleasing_alloc_cluster / push_task_into_bins_new), not internal impl fns (glb_alloc_new2 etc, those in binpack_solver_spec.md). So moving internal dead fns does not affect top-level specs.

Minor note (out of B3 scope): test_plan:11 misspells `coalesce_alloc_cluster` (missing 's'); code is `coleasing_alloc_cluster`. Spec typo, recorded not fixed.

Changed files: REVIEW_PACKET_BATCH_B3-BINPACK-DEAD-MOVE.md (added §4 Double-check section). No source change.

Scope kept out: no execution. test_pipeline untouched.

Next: B3 packet double-checked, awaiting user approval.

## 2026-06-16 B3 batch 1 executed (4/6) + ghost-file discovery

Action type: execution (move-reference) — PARTIAL + 2 N/A

Reason:
- User approved "分组" execution. Batch 1 = group A (independent whole-file) + bound pair (002+CLEAN-001).

Executed (4, tracked files, regression gate PASS):
- B3-MOVE-002: git mv sched/pre_alloc.py → unused/pre_alloc.py
- B3-MOVE-004: git mv sched/packing_solver/gurobi_semi2Dclst_mapping.py → packing_solver/old/
- B3-MOVE-005: git mv sched/packing_solver/gurobi_semi2Dclst_mapping2.py → packing_solver/old/
- B3-CLEAN-001: deleted sched/global_sched.py:25 `from sched.pre_alloc import glb_alloc_new` (dead import fossil)

Regression gate (gurobi) — ALL PASS:
- import probe: sched.global_sched/scheduler_agent/monitor_agent, approach_sim/approach_setup/main_approach → OK
- main_approach/motiv/abla --help → PASS
- global_sched imports cleanly after dead-import removal.

N/A (2 — important discovery):
- B3-MOVE-001 (bin_ops.old.py) and B3-MOVE-003 (message_handler_old.py) FAILED `git mv: bad source` in audit worktree. Investigation:
  - These files exist in MAIN working tree (~/git_repo/scheduler/) but are UNTRACKED.
  - `git ls-files` returns empty; `git cat-file -e test_pipeline:<path>` → "Not a valid object name" → NOT in test_pipeline.
  - => They are ghost files in the main working tree, never committed. Per P1-GIT-002 (untracked out of scope) and user rule (only touch git-tracked files), NO action. They won't enter clean main anyway (which derives from test_pipeline).
- Methodology lesson: earlier bin_packing analysis ran grep in MAIN working tree, which mixed untracked ghosts (bin_ops.old.py 775L, message_handler_old.py 246L) into the dead-code inventory. The "~2400 lines" estimate was inflated by ~1021 lines of ghost files. True tracked dead code is smaller. Future analysis should run in audit worktree (the cleanup basis), not main.

Ledger: B3-MOVE-001/003 status → N/A; 002/004/005/CLEAN-001 → executed.

Changed source files: sched/global_sched.py (1 line del); 3 renames (pre_alloc.py, 2 mapping files). test_pipeline untouched.

Recovery:
- git checkout archive/test_pipeline-20260612 -- sched/global_sched.py sched/pre_alloc.py sched/packing_solver/gurobi_semi2Dclst_mapping.py sched/packing_solver/gurobi_semi2Dclst_mapping2.py
- then git mv back / rmdir unused & packing_solver/old as needed.

Next: batch 2 = group B symbol-level (B3-MOVE-006 bin_select_new / 007 naive_iso / 008 commented test). Will re-verify line numbers in audit worktree first (analysis was in main tree). Then batch 3 = group D docs.

## 2026-06-16 B3 batch 2 executed (group B symbol-level)

Action type: execution (move-reference, symbol-level slice)

Executed (3, byte-identical surviving code, gate PASS):
- B3-MOVE-006: pre_alloc_new.py::bin_select_new (L287-327, 41 lines) → new sched/pre_alloc_new_unused.py
- B3-MOVE-007: global_sched.py::naive_iso (L280-315, 36 lines) → new sched/global_sched_unused.py
- B3-MOVE-008: pre_alloc_new.py commented test block (L627-749, 123 lines) → sched/pre_alloc_new_unused.py

Line numbers re-verified in audit worktree via AST end_lineno BEFORE slicing (critical: global_sched.py had shifted by -1 vs main-tree analysis due to B3-CLEAN-001 deleting line 25; naive_iso was L280-315 in audit, not L281-317 from main). This avoided a mis-cut.

byte-identical verification (git diff vs test_pipeline, pure deletion):
- pre_alloc_new.py: +0 / -164 (bin_select_new 41 + commented test 123)
- global_sched.py: +0 / -37 (line-25 import 1 [from CLEAN-001] + naive_iso 36). Trailing newline matched to test_pipeline (which has no final newline) to keep diff pure-deletion.

Surviving functions verified present:
- pre_alloc_new.py: glb_alloc_new2, allocate_rsc_4_process_new2, Bp_print_util, bin_sel (L288), push_into_bin, check_and_preemt_alloc, index_occupy_by_id_chunk_ver
- global_sched.py: push_task_into_bins_new, push_step_new, test_mem_planner (L281), coleasing_alloc_1bin, coleasing_alloc_cluster, update_bp_result2_schedtab, gurobi_split_solver, rename_bins_and_relable_assignments, build_greedy_obj, build_search_obj

Regression gate (gurobi) — ALL PASS: 6-line import probe OK; main_approach/motiv/abla --help PASS.

Changed source files: sched/pre_alloc_new.py (M, -164), sched/global_sched.py (M, -36 naive_iso, cumulative with CLEAN-001). New: sched/pre_alloc_new_unused.py (untracked), sched/global_sched_unused.py (untracked). test_pipeline untouched.

Recovery:
- git checkout archive/test_pipeline-20260612 -- sched/pre_alloc_new.py sched/global_sched.py
- rm sched/pre_alloc_new_unused.py sched/global_sched_unused.py

Ledger: B3-MOVE-006/007/008 → executed.

Next: batch 3 (group D docs): B3-DOC-001 (deprecated_code.md 补全) + B3-DOC-002 (spec §8.1 标记已解决). Then commit all of B3.

## 2026-06-16 B3 batch 3 executed (group D docs)

Action type: execution (doc updates, no code change)

Executed (2):
- B3-DOC-001: updated doc/guide/deprecated_code.md:
  - §8 澄清: scheduler_agent/monitor_agent 仿真循环被替代但工具函数仍活（B2 发现），文件级不可删
  - 新增 §10: B3 bin_packing 清理记录表（pre_alloc.py/mapping/bin_select_new/naive_iso/注释测试 的新归档位置）
  - 待确认: bin_ops.old.py 标注为主仓库未跟踪幽灵文件（不在 test_pipeline）
  - 日期 → 2026-06-16
- B3-DOC-002: updated doc/spec/algorithm/binpack_solver_spec.md §8（保护路径，只标注不删）:
  - §8.1.1 sys.exit(1) → 标注✅已解决（raise ResourceInsufficientError, pre_alloc_new.py:206）
  - §8.1.2 tot_cores=300 → 标注✅已解决（binpack_cfg.get, pre_alloc_new.py:158）
  - §8.2 表格 pre_defined 行 ⚠️→✅
  - §8.3 改进建议 1/2 标注✅已落实
  - 保留所有原建议文本（历史），只加✅标注+新代码位置

Regression gate (gurobi): import probe OK; main_approach --help PASS (docs don't affect code, confirmed no .py touched).

Changed files: doc/guide/deprecated_code.md (M), doc/spec/algorithm/binpack_solver_spec.md (M). No source change.

Ledger: B3-DOC-001/002 → executed.

B3 COMPLETE: 7 executed (002/004/005/006/007/008/CLEAN-001) + 2 N/A (001/003 ghost files) + 2 docs. All batches gate PASS. Uncommitted — ready to commit.

## 2026-06-16 归档目录语义重新分类（old vs unused）

Action type: status-maintenance + git mv (reclassification, no logic change)

Reason:
- User defined archive semantics: `old/` = historical versions (superseded by newer live impl); `unused/` = standalone features currently unreferenced (no version relationship).
- Review found prior B2/B3 classification was by location not semantics — most files were in the wrong dir. Reclassified all.

Reclassification (7 moves + 3 renames + 4 empty-dir removals):
- unused/ → old/ (historical versions): allocator_agent.py, pre_alloc.py
- old/ → unused/ (standalone features): repack_sweep.py (→ scripts/unused/), gurobi_semi2Dclst_mapping.py + _mapping2.py (→ packing_solver/unused/)
- *_unused.py → *_old.py (symbol-level historical versions): sim_main_old.py, pre_alloc_new_old.py, global_sched_old.py (naive_iso = old, user-confirmed: predecessor of coleasing)
- Removed empty dirs: model/message/old (误建), unused/ (root, emptied), scripts/old (emptied), sched/packing_solver/old (emptied)

Policy codified: added "Archive directory naming — old/ vs unused/" section to cleanup-policy.md (slim canonical + repo copy). Decision test: "Is there a newer live version of THIS feature?" Yes→old/, No→unused/.

Docs/ledger updated: deprecated_code.md §10 paths; move-ledger target_path for all B2/B3 rows + RECLASSIFIED note.

Regression gate (gurobi): import probe OK (global_sched/pre_alloc_new/approach_sim/main_approach); --help PASS. No logic change (pure git mv + rename + header text).

Final layout:
- old/ (historical): allocator_agent.py, pre_alloc.py
- scripts/unused/: repack_sweep.py
- sched/packing_solver/unused/: gurobi_semi2Dclst_mapping.py, _mapping2.py
- *_old.py: sim_main_old.py, sched/pre_alloc_new_old.py, sched/global_sched_old.py

Changed files: git mv (7) + mv (2 untracked) + header sed (3 _old.py) + move-ledger + deprecated_code.md + cleanup-policy.md (slim+repo). test_pipeline untouched.

Recovery: git checkout archive/test_pipeline-20260612 -- <original path> (paths in ledger source_path column).

Next: B3 + reclassification all done, gate PASS, uncommitted. Ready to commit.

## 2026-06-16 commit B3 + 归档语义重分类 to audit branch

Action type: status-maintenance (git checkpoint)

Reason: B3-BINPACK-DEAD-MOVE (3 batches) + old/unused semantic reclassification all executed and gate-passed but uncommitted. Securing to audit branch.

Snapshot (no new source change in this action):
- B3: pre_alloc.py→old/, mapping×2→packing_solver/unused/, dead-import delete, bin_select_new/naive_iso/注释测试 symbol-level slice to *_old.py; deprecated_code.md §10 + spec §8 doc updates.
- Reclassification: allocator_agent.py unused/→old/, repack_sweep scripts/old/→scripts/unused/, *_unused.py→*_old.py, 4 empty dirs removed; cleanup-policy.md old/unused semantics codified (slim+repo).
- Audit artifacts: CLEANUP_STATUS, FILE_ADJUSTMENT_RECORD, move-ledger, B3 packet, bin_packing reports.

Scope kept out: no main/master merge; test_pipeline untouched; Phase 3 not started.

Recovery:
- Undo commit keeping changes staged: git switch audit/minimal-from-test_pipeline-20260612 && git reset --soft HEAD~1
- Restore moved files: git checkout archive/test_pipeline-20260612 -- <path>
