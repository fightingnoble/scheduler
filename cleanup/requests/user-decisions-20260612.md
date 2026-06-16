# User decisions recorded on 2026-06-12

This file records the user's Phase 1 review decisions. It does not authorize cleanup execution by itself; Phase 2 must still operate batch-by-batch and must not touch the original `test_pipeline` worktree.

## Approved

- `P1-GIT-001`: use `master` as the current mainline basis unless replaced later.
- `P1-GIT-002`: original untracked files are out of scope.
- `P1-KEEP-001`: accept the KEEP seed.
- `P1-DEP-001`: reconcile dependency manifest/environment later.
- `P1-REUSE-001`: keep reuse candidates as reuse candidates.
- `P1-SYMBOL-001`: symbol candidates are record-only.
- `P1-NODE-001`: approved for tracked `package.json` / `package-lock.json` only. Do not touch untracked `node_modules/`.

## Keep review

- `P1-DEP-002`: keep `bokeh`, `plotly`, and `pyyaml` under review.
- `P1-REVIEW-DOCS-001`: documentation policy remains review; important docs should be kept and updated with code changes.
- `P1-REVIEW-RUNTIME-001`: runtime-adjacent files remain review until targeted tests exist.

## Not approved

The user rejected the previous low-risk cleanup classification. Do not execute any `P1-REMOVE-*` batch from the Phase 1 packet without a new, narrower review.

Specific protected areas:

- `.claude/`: necessary Claude skills/config; do not delete.
- `claude_talk/`: important conversation history; do not delete.
- `doc/spec`, `doc/guide`, `doc/dev`: important documentation; keep and update when code changes require it.
- `P1-REMOVE-B19-TESTS`: rejected. These tests include converged unit tests and should stay.

## Scope constraint

Do not directly operate on `/home/zhangchg/git_repo/scheduler`. Use audit/integration worktrees only. Ignore untracked files unless the user explicitly brings them into scope.
