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
