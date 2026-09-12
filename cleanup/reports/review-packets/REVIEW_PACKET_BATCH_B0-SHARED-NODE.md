# Batch review packet: B0-SHARED-NODE

Status: executed and accepted by user on 2026-06-13.

This batch only executed the approved Node manifest cleanup from `P1-NODE-001`.

## What changed

- Removed tracked `package.json`.
- Removed tracked `package-lock.json`.

Both files were empty project manifests, not active tooling:

- `package.json` contained only `{}`.
- `package-lock.json` had an empty `packages` object.
- No tracked code or docs outside cleanup reports referenced `package.json`, `package-lock.json`, or `node_modules`.

## What did not change

- No cleanup was done in `/home/zhangchg/git_repo/scheduler`.
- No untracked files were touched.
- `node_modules/` remains out of scope.
- `.gitignore` was not changed.
- Python source, Python dependencies, docs, tests, CI, and runtime config were not changed.
- `.claude/`, `claude_talk/`, `doc/spec`, `doc/guide`, `doc/dev`, and protected tests were not touched.
- No `P1-CACHE-*` or `P1-REMOVE-*` action was executed.

## Decision IDs

Executed:

- `P1-NODE-001`

Not executed in this batch:

- `P1-DEP-001`
- `P1-DEP-002`
- `P1-CACHE-001`
- `P1-CACHE-002`
- all `P1-REMOVE-*`

## Ledger updates

- `cleanup/deletion-ledger.csv`: added rows for `package.json` and `package-lock.json`.
- `cleanup/dependency-ledger.csv`: marked the Node manifest row as `EXECUTED_ARCHIVE_ONLY`.
- `cleanup/move-ledger.csv`: unchanged; no move was made.
- `cleanup/gitignore-cache-ledger.csv`: unchanged; no cached file removal or ignore rule was changed.
- `cleanup/reuse-candidates.csv`: unchanged; reuse candidates were not touched.
- `cleanup/porting-queue.csv`: unchanged; `archive/test_pipeline-20260612..test_pipeline` had 0 commits.

## Smoke result

PASS in the documented `gurobi` Conda environment:

- `scripts/motiv_exp_runner.py --help`
- `scripts/abla_exp_runner.py --help`
- `main_approach.py --help`

Details are in `cleanup/reports/batch-B0-SHARED-NODE-smoke.md`.

## Recovery

```bash
git checkout archive/test_pipeline-20260612 -- package.json package-lock.json
```

## User review

Accepted by user on 2026-06-13.

The next sensible batch is a separate preflight for `P1-DEP-001`, focused only on Python dependency and environment reconciliation.
