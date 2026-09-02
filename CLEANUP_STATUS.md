# Cleanup status

Last updated: 2026-09-02 (B9 committed; REQ-002 closed; next: REQ-003 pending user preview)

This is the canonical global status file for the scheduler cleanup work. It supersedes `PHASE1_STATUS_FOR_NEXT_AGENT.md` as the main entry point for future agents.

Use this file to understand the current cleanup state, approved decisions, executed batches, protected areas, and next actions. Use `FILE_ADJUSTMENT_RECORD.md` for the global action/change history.

## Mandatory update rule

After every cleanup action, preflight, environment verification, status-maintenance step, or batch execution, update both global records before handing off:

- `CLEANUP_STATUS.md`: update the current global action state, decision status, tracked diff, validation state, and recommended next action.
- `FILE_ADJUSTMENT_RECORD.md`: append a chronological action-history entry. If the action did not change source/tracked files, explicitly record that it was a no-source-change or preflight/status-only action.

Per-batch packets and ledgers are still required when applicable, but they do not replace these two global records.

## Peer-review gate

`AGENT_DIALOGUE.md` is the coordination channel between Codex and the peer reviewer. Only one request may be active. Codex must receive proposal approval before changing the requested code and result acceptance before starting the next request.

Current coordination state:

- Request: `REQ-002` (closed)
- State: `ACCEPTED`
- Scope: move `sched/scheduling_table.py` `__main__` demo to `test_scheduling_table.py`
- Next writer: proposer/implementer (`codex` in the event log, may open the next request)
- Source changes: implemented and independently verified; not committed
- Ledger result: `B9-MAINOUT` is one 14-field LF-terminated addition; every non-B9 byte remains identical to HEAD

`E0014` closes REQ-002. The moved demo preserves the reviewed behavior, and the final ledger diff contains no unrelated rewrite. Monitoring remains active because the previously agreed refactor discussion still has further candidates; completing this one request is not the global termination condition.

Background monitoring:

- Hidden local watcher: Windows PID `35404`, following only `AGENT_DIALOGUE.md`.
- Watch log: `C:\Users\diyuf\.codex\state\scheduler-agent-dialogue\watch.log`.
- Codex heartbeat: `scheduler-agent`, active every 5 minutes to wake this review task.
- The shell watcher is runtime-only and may stop after Windows/WSL restart; the heartbeat remains the wake mechanism.
- Termination: after all collaboration requests are closed and there is no pending or agreed next refactor, stop PID `35404` if present and delete the `scheduler-agent` heartbeat. A single batch completing is not sufficient.

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

Current audit-worktree state after B8:

- B8 source change: `approach_setup.py` step 4-6 now uses `sim_main.py::init_sched_components(...)` to create the shared global_sched component context and run `perform_bin_packing` through a small pack handle.
- B8 status docs: `REVIEW_PACKET_BATCH_B8-INIT-SCHED-COMPONENTS.md`, `CLEANUP_STATUS.md`, `FILE_ADJUSTMENT_RECORD.md`, and the Gurobi troubleshooting row in `CLAUDE.md`.
- B8 validation: import probe + 3 help commands PASS; motiv case1 (`--num_hp 3 --case1_ratios 0.7`) full run rc=0 after fixing WSL Gurobi HostID via bond0.
- Validation artifacts from `motiv_exp_results_b8verify/` were removed before commit.

No changes were made to `/home/zhangchg/git_repo/scheduler` or `test_pipeline`.

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

### B3-BINPACK-DEAD-MOVE — batch 1 EXECUTED (2026-06-16), batch 2/3 pending

**Batch 1 (group A + bound pair) — DONE, gate PASS:**
- ✅ B3-MOVE-002: pre_alloc.py → unused/
- ✅ B3-MOVE-004/005: gurobi_semi2Dclst_mapping*.py → packing_solver/old/
- ✅ B3-CLEAN-001: deleted global_sched.py:25 dead import
- ❌ B3-MOVE-001/003: **N/A** — bin_ops.old.py & message_handler_old.py are UNTRACKED ghost files in main working tree, NOT in test_pipeline (git cat-file confirmed). Out of scope per P1-GIT-002.

**Ghost-file discovery (important):** earlier bin_packing analysis ran grep in MAIN working tree, mixing untracked ghosts (bin_ops.old.py 775L, message_handler_old.py 246L) into the dead-code inventory. The "~2400 lines" estimate was inflated by ~1021 lines. Future analysis must run in audit worktree (cleanup basis). True tracked bin_packing dead code is smaller.

**Batch 2 (group B symbol-level) — DONE, gate PASS, byte-identical:**
- ✅ B3-MOVE-006: bin_select_new (L287-327, 41 lines) → pre_alloc_new_unused.py
- ✅ B3-MOVE-007: naive_iso (L280-315, 36 lines) → global_sched_unused.py
- ✅ B3-MOVE-008: commented test block (L627-749, 123 lines) → pre_alloc_new_unused.py
- Line numbers re-verified in audit worktree via AST end_lineno BEFORE slicing (naive_iso shifted to L280-315 due to CLEAN-001; main-tree analysis said L281-317 — would have mis-cut).
- byte-identical: pre_alloc_new.py +0/-164; global_sched.py +0/-37 (pure deletion, trailing newline matched test_pipeline).

**Batch 3 (group D docs) — DONE:**
- ✅ B3-DOC-001: deprecated_code.md §8 澄清（scheduler_agent/monitor_agent 部分活）+ 新增 §10 清理记录 + bin_ops.old 幽灵标注
- ✅ B3-DOC-002: spec §8.1.1/8.1.2/8.2/8.3 标注✅已解决（保留原文，保护路径只标注不删）

**B3 COMPLETE**: 7 executed (002/004/005/006/007/008/CLEAN-001) + 2 N/A (001/003 幽灵文件) + 2 docs. 全部回归门 PASS.

### 归档目录语义重分类（2026-06-16）— DONE

用户定义语义：`old/`=历史版本（被新实现取代）；`unused/`=独立功能暂无引用。按此重分类所有 B2/B3 归档：

| 类 | 位置 | 内容 |
|----|------|------|
| **old/**（历史版本） | `old/` | allocator_agent.py, pre_alloc.py |
| **old/**（符号级） | `*_old.py` | sim_main_old.py, sched/pre_alloc_new_old.py, sched/global_sched_old.py |
| **unused/**（独立功能） | `scripts/unused/` | repack_sweep.py |
| **unused/**（独立功能） | `sched/packing_solver/unused/` | gurobi_semi2Dclst_mapping.py, _mapping2.py |

语义约定已写入 `cleanup-policy.md`（slim + repo）。删除空目录：model/message/old、unused/(根)、scripts/old、packing_solver/old。回归门 PASS。改动未 commit。

pre_alloc_new.py 深度分析完成（内部调用图 + spec 覆盖核对）。Packet: `REVIEW_PACKET_BATCH_B3-BINPACK-DEAD-MOVE.md`，11 个决策：

| 组 | 决策 | 对象 | 风险 |
|----|------|------|------|
| A 整文件 | B3-MOVE-001 | `bin_ops.old.py` (775L) → `unused/` | 低（独立） |
| A 整文件 | B3-MOVE-002 | `pre_alloc.py` (592L) → `unused/` | 低（**绑定 B3-CLEAN-001**） |
| A 整文件 | B3-MOVE-003 | `message_handler_old.py` (246L) → `model/message/old/` | 低（独立） |
| A 整文件 | B3-MOVE-004/005 | `gurobi_semi2Dclst_mapping.py` + `_mapping2.py` → `packing_solver/old/` | 低（独立） |
| B 符号级 | B3-MOVE-006 | `pre_alloc_new.py::bin_select_new` (L287-328) | 低（bin_sel 留活） |
| B 符号级 | B3-MOVE-007 | `global_sched.py::naive_iso` (L281-317) | 低 |
| B 符号级 | B3-MOVE-008 | `pre_alloc_new.py` 注释测试 (L627-748) | 低（死注释） |
| C 配套 | B3-CLEAN-001 | 删 `global_sched.py:25` 死 import | 低（绑定 B3-MOVE-002） |
| D 文档 | B3-DOC-001 | 更新 `deprecated_code.md`（补全） | 低 |
| D 文档 | B3-DOC-002 | 更新 `spec §8.1`（标记已解决，保护路径） | 低-中 |

总清理 ~2400+ 行死代码 + 2 处文档同步。不动：test_mem_planner（保留）、single_turn_solver（不管）、参数语义错位（记录）。



Reports: `cleanup/reports/bin_packing_inventory.md` + `cleanup/reports/bin_packing_function_map.md`. Analysis only, no source change.

- spec (`binpack_solver_spec.md §2.1`) function hierarchy is **ACCURATE** — maps to live code. Trustworthy canonical map.
- LIVE (guided/scratch): `sim_main::perform_bin_packing` → `global_sched.{coleasing_alloc_cluster, push_task_into_bins_new}` → `pre_alloc_new.glb_alloc_new2` + `bin_ops` + `gurobi_MP_semi2DClst`.
- DEAD whole files (0 live deps, safe to move): `bin_ops.old.py` (775L), `pre_alloc.py` (592L), `message_handler_old.py` (246L), `gurobi_semi2Dclst_mapping.py` + `_mapping2.py`.
- DEAD import fossil: `global_sched.py:25 from sched.pre_alloc import glb_alloc_new` (imported, never called).
- DEAD function: `global_sched.py::naive_iso` (0 calls).
- 3 contradictions resolved (import ≠ used): mapping import commented out; pre_alloc glb_alloc_new dead-import; sim_main doesn't import pre_alloc.
- User decisions: test_mem_planner **KEEP**; single_turn_solver **LEAVE AS-IS** (broken branch).
- Doc: `guide/deprecated_code.md` INCOMPLETE (misses pre_alloc/message_handler_old/dead solvers); `doc/dev/` process docs candidates for archival.



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

**Coordination gate: `REQ-002` is `ACCEPTED` and closed.** The proposer/implementer may open the next request for one of the previously identified low-risk refactor candidates. Every new proposal and its implementation still require reviewer approval; do not combine the higher-risk `global_sched_repack.py` or `pre_alloc_new.py` refactors with routine cleanup.

**B8-INIT-SCHED-COMPONENTS EXECUTED AND VERIFIED**. The B8 commit is the current checkpoint on the audit branch.

B8 result:
- `sim_main.py::init_sched_components(...)` encapsulates the shared Scheduler/Monitor/msg_dispatcher/DataPipe + simulation-env setup used by the unified `global_sched` interface.
- `approach_setup.py::run_benchmark_setup_pipeline(...)` no longer unpacks and forwards those runtime-adjacent components manually; it calls the pack handle.
- Logic change: none intended. `perform_bin_packing(...)` still receives the same values, just through the closure.
- Gurobi diagnosis corrected: the PyCapsule failure was from WSL HostID mismatch, not license expiry. `gurobi-wsl-fix`/manual `gurobi_fix` creates bond0 with MAC `00:15:5d:80:30:e7`.

Recommended next options:
1. Propose the next narrow, behavior-preserving cleanup/refactor request from the earlier candidate list.
2. Keep higher-risk runtime/repack restructuring in a separate request with targeted equivalence tests.
3. If WSL is fully restarted and Gurobi fails again, run `gurobi_fix` manually or install a systemd service after explicit user approval. Do not auto-run sudo from `.zshrc`.

Do not start cleanup execution from old `P1-REMOVE-*` or `P1-CACHE-*` decisions.

## File roles

- `AGENT_DIALOGUE.md`: append-oriented Codex/reviewer handoff log and review gate.
- `CLEANUP_STATUS.md`: current global status and next-action entry point.
- `FILE_ADJUSTMENT_RECORD.md`: chronological global action/change history; include recovery commands for actual file changes.
- `REVIEW_PACKET_BATCH_*.md`: per-batch review details.
- `cleanup/reports/*`: machine-readable or supporting audit reports.
