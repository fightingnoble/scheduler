# Cleanup status

Last updated: 2026-06-16 (B0 results + Phase 1 analysis + status files committed to audit branch)

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

The B0 source changes, Phase 1 analysis artifacts (`cleanup/`), review packets, and the two global status files were committed to the audit branch on 2026-06-16 (first commit beyond `test_pipeline`; see `FILE_ADJUSTMENT_RECORD.md`). The audit worktree is clean pending the next batch.

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

## Status maintenance history

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

Choose one follow-up batch:

1. `B2-RUNTIME-TEST-INVENTORY`: run targeted import/test checks for runtime-adjacent review files.
2. `B2-DOC-INVENTORY`: reclassify docs into KEEP / REVIEW without deleting.
3. `B2-OPTIONAL-DEPS-DECISION`: decide whether `bokeh`, `plotly`, and `pyyaml` should remain supported optional dependencies or be removed from `requirement.txt`.

Do not start cleanup execution from old `P1-REMOVE-*` or `P1-CACHE-*` decisions.

## File roles

- `CLEANUP_STATUS.md`: current global status and next-action entry point.
- `FILE_ADJUSTMENT_RECORD.md`: chronological global action/change history; include recovery commands for actual file changes.
- `REVIEW_PACKET_BATCH_*.md`: per-batch review details.
- `cleanup/reports/*`: machine-readable or supporting audit reports.
