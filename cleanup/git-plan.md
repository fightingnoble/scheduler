# Git plan

## Archive refs
- Annotated tag: `archive/test_pipeline-20260612` -> `d5bfda5000c4a9597cd2ad51986037dc200d6b29`
- Branch: `archive/test_pipeline-20260612-branch` -> `d5bfda5000c4a9597cd2ad51986037dc200d6b29`

## Worktrees
- Source legacy worktree, untouched: `/home/zhangchg/git_repo/scheduler` on `test_pipeline`
- Audit worktree: `/home/zhangchg/git_repo/scheduler-audit-20260612` on `audit/minimal-from-test_pipeline-20260612`
- Integration worktree: `/home/zhangchg/git_repo/scheduler-integration-20260612` on `integrate/minimal-to-main-20260612`

## Important deviation
No local `main` or `origin/main` ref exists. The integration worktree was created from `master`, the available mainline-equivalent branch. Confirm with decision `P1-GIT-001` before Phase 2/3 integration planning.

## Rules
- Never modify, reset, rebase, clean, or commit to `test_pipeline`.
- Never merge `test_pipeline` into mainline, audit, or integration branches.
- Do not merge the audit branch into mainline.
- Phase 2 changes execute only approved decision IDs.
- Recovery uses `git show archive/test_pipeline-20260612:path`, `git restore --source archive/test_pipeline-20260612`, or selected cherry-picks/manual patches.
