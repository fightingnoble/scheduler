# Phase 1 审计包

这份报告是批复后的版本。它只记录第一阶段审计和你的决策，不执行清理。

我没有改 `test_pipeline`，也没有对原始 worktree 做 reset、rebase、clean、merge 或 commit。后续也不会直接操作：

`/home/zhangchg/git_repo/scheduler`

后续只允许在 audit / integration worktree 里工作，并且只关注 git tracked / git cache 内的内容。untracked 文件不纳入本轮清理。

## Git 状态

源分支：

`test_pipeline @ d5bfda5`

归档引用：

- `archive/test_pipeline-20260612`
- `archive/test_pipeline-20260612-branch`

审计 worktree：

`/home/zhangchg/git_repo/scheduler-audit-20260612`

集成 worktree：

`/home/zhangchg/git_repo/scheduler-integration-20260612`

本地没有 `main` 或 `origin/main`。你已经批准 `P1-GIT-001`，所以当前按 `master` 作为 mainline 基准记录。以后如果你给出真正主线分支，我再调整。

## 当前分类

这次只看 tracked 文件，共 237 个。

根据你的批复，我已经撤掉之前的低风险删除判断。现在的分类是：

- `KEEP`：124 个
- `REVIEW`：101 个
- `REUSE_CANDIDATE`：12 个
- `REMOVE_FROM_CLEAN`：0 个

原先 46 个 `REMOVE_FROM_CLEAN` 候选全部标为 `NOT_APPROVED_BY_USER`。这些不会进入 Phase 2 执行。

函数、类和变量只记录，不自动删除。`cleanup/symbol-candidates.csv` 仍然只是参考。

## 测试环境更正

`CLAUDE.md` 写明测试前必须进入：

```bash
conda activate gurobi
```

第一次探测用了系统 `python3`，这不符合仓库要求，所以缺 `tdigest`、`networkx` 的结果已经作废。

按正确环境重新跑后，三个入口的 `--help` 都通过：

- `scripts/motiv_exp_runner.py --help`
- `scripts/abla_exp_runner.py --help`
- `main_approach.py --help`

使用的命令形态是：

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help'
```

所以 `P1-DEP-001` 现在只表示“复核 `requirement.txt` 和 `gurobi` Conda 环境的一致性”，不是 runtime blocker。

## 已批准

`P1-GIT-001`

暂按 `master` 作为 mainline 基准。

`P1-GIT-002`

untracked 文件不进入本轮审计。比如 `node_modules/`、`docs/`、`motiv_exp_results_zh/`、`abla_results_zh/`，都不处理。

`P1-KEEP-001`

接受当前 KEEP 种子集合。后续 Phase 2 不能移除这些文件，除非你给新的、更具体的批复。

`P1-DEP-001`

允许后续复核依赖清单和 Conda 环境的一致性。只复核，不自动删依赖。

`P1-REUSE-001`

12 个文件保留为 `REUSE_CANDIDATE`。它们不进入当前最小运行面，但保留恢复和移植路径。

`P1-SYMBOL-001`

符号级候选只记录，不自动删函数、类或变量。

`P1-NODE-001`

只批准处理 tracked 的 `package.json` 和 `package-lock.json`。`node_modules/` 是 untracked，已经标为 out of scope。

## 保持 review

`P1-DEP-002`

`bokeh`、`plotly`、`pyyaml` 先不删，继续 review。

`P1-REVIEW-DOCS-001`

文档不做粗暴清理。你明确指出这些目录重要：

- `doc/spec`
- `doc/guide`
- `doc/dev`

后续代码改动后，这些文档可能需要整理和更新。

`P1-REVIEW-RUNTIME-001`

运行时附近的文件继续 review。现在只做过入口 `--help`，还没有跑实验或针对性测试。

## 明确不批准

你没有批准之前那些“低风险清理”批次。我已经把所有 `P1-REMOVE-*` 标成 `NOT_APPROVED_BY_USER`。

特别记录：

- `.claude/` 是 Claude 需要的技能和配置，不要删除。
- `claude_talk/` 是重要对话历史，不要删除。
- `P1-REMOVE-B19-TESTS` 不批准。里面有已经收敛到代码的单元测试，需要留着。
- `P1-CACHE-001` 和 `P1-CACHE-002` 也没有批准，暂不执行。

## 后续 Phase 2 约束

下一阶段如果继续，只能做这几类事：

1. 在 audit / integration worktree 里操作，不碰原始 `test_pipeline` worktree。
2. 只处理 git tracked / git cache 内容。
3. 不处理 untracked 文件。
4. 不执行任何 `P1-REMOVE-*` 清理批次。
5. 不删 `.claude/`、`claude_talk/`、`doc/spec`、`doc/guide`、`doc/dev`。
6. 不删测试，尤其不删 `P1-REMOVE-B19-TESTS` 覆盖的测试文件。
7. 符号级清理仍然只是记录，不能执行。

## 记录位置

你的批复已经写入：

- `cleanup/requests/user-decisions-20260612.md`
- `cleanup/requests/decision-map.csv`
- `cleanup/reachable-files.csv`
- `cleanup/remove-from-clean-candidates.csv`
- `cleanup/dependency-ledger.csv`
- `cleanup/gitignore-cache-ledger.csv`
- `cleanup/reports/phase1-summary.json`

当前最重要的人工可读记录是：

`cleanup/requests/user-decisions-20260612.md`
