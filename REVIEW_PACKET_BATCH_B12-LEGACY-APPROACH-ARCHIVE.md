# B12 审查材料：旧 approach 示例归档

## 当前结论

真实 reviewer 已在 E0041 按 A1-A6 独立复测并验收 B12，Codex 已从磁盘核对裁决和守卫 check rc=0。独立复跑新增7项通过（5.31秒），B10/B11/collector为37通过、同样5项原有失败（21.08秒）。只操作 audit worktree；未提交、未推送，B11 未提交内容保持原样。主线基准为 master，本批执行前 HEAD 为 a1d933b。

## 文件调整

| 原位置 | 新位置 | 内容变化 |
| --- | --- | --- |
| appoach_plot6.py | old/appoach_plot6.py | 仅第8行 import 改为 from old.approach_util33 import Acc_p, Sen_p, MyGraph |
| approach_util33.py | old/approach_util33.py | 逐字节不变 |

两个文件完整保留，共370行。旧类的全部方法、算法、注释、拼写和 sys.path 操作不动。utils.py 的 core_distr 仍被当前调度器使用，example/bm4.py 是旧示例数据，两者都留在原位。

没有根兼容入口，旧根模块名不复用。后续运行方式为：

```bash
conda activate gurobi
cd /home/zhangchg/git_repo/scheduler-audit-20260612
PYTHONDONTWRITEBYTECODE=1 python -m old.appoach_plot6
```

从其他目录运行时，需将 PYTHONPATH 设为 audit worktree 的绝对路径。old/ 使用已有的 Python 隐式命名空间，不新增 __init__.py。

## 验证结果

- 迁移前，本地双跑与 reviewer 独立保存的 /tmp/b12_review/run1.out 逐字节相同，stderr为空；基线全文已进入机器报告，不依赖临时目录永久存在。
- 新测试先因归档目标缺失而失败：1 failed、6 deselected。迁移后7项全部通过，2.29秒。
- 验证包括 helper 静默导入、模块/脚本入口各两次独立运行、完整输出与最终状态，以及文件原字节与唯一 import 差异。
- B10 + B11兼容测试 + 既有collector：37 passed、5 failed。五个失败的测试名及异常类型与B11逐项相同，没有新增失败或 skip/xfail。
- main_approach、motiv、abla 三入口 --help 都为0；150份非B12源码/脚本哈希保持原样，含B11七个实现及兼容测试；保护区、utils.py、example/bm4.py、既有测试零diff。
- move-ledger只追加2行×14列，reachable-files只追加3行×10列；原15,435/66,088字节完全保留，包括B11记录，新增均为LF。不新建actions.csv。
- git diff --check 通过。新归档源码原有空白未重新格式化。

五个原有失败：test_basic_functionality 为 AttributeError；test_motiv_exp_specific_stats、test_summary_generation、test_formatted_output、test_edge_cases 为 ZeroDivisionError。它们属于collector旧问题，本批不修。

## 保留的边界

历史根模块名的pickle不承诺兼容。已核对tracked调用者及B11获批的新package，没有发现旧helper的外部使用或pickle写入路径；这不等于证明仓库外没有用户。

旧示例最后图节点为空，pred_t=107、curr_t=112，但state仍为R，running还含R任务。本批保留这一行为，不宣称算法正确，也不借归档修正它。

其余 *_old.py / *_unused.py 已根据账本确认为从活文件切出的符号存档，继续与源文件同目录。空的 unused_fun.py 暂留，不为减少文件数删除占位文件。

## 证据与记录

- 新测试：test_legacy_approach_archive.py。
- 机器报告：cleanup/reports/b12-legacy-approach-archive.json，含独立输出基线、原SHA、150份非B12源码快照、验证结果及账本前缀。
- 原始回归日志：/tmp/scheduler-b12-regression-x28_n281/。
- Phase1的原路径分类行作为历史证据保留；当前归档位置以 B12-MOVE-001/002 和新增目标行判定，不能把原路径行误认为文件仍在根目录。
- 状态与历史：CLEANUP_STATUS.md、FILE_ADJUSTMENT_RECORD.md。

## 恢复方法

以下命令仅供决定撤销本批时使用，本轮没有执行。先确认没有后续修改；哈希或根路径检查不符即停止，不能强行覆盖。

```bash
set -e
cd /home/zhangchg/git_repo/scheduler-audit-20260612
test ! -e appoach_plot6.py
test ! -e approach_util33.py
test "$(sha256sum old/appoach_plot6.py | cut -d' ' -f1)" = "44332d84c5d05f7d520b505c96c80c62a8ec6511c41440db41ca4bc6081d0d81"
test "$(sha256sum old/approach_util33.py | cut -d' ' -f1)" = "89ae0ac387bd44484b1a345e5bd3999cc279acc2f5a705cba6ab6fc285701352"
mv -n -- old/approach_util33.py approach_util33.py
mv -n -- old/appoach_plot6.py appoach_plot6.py
git restore --source=a1d933b1f26efd4d569eb3d8ffb313447294443b --worktree -- appoach_plot6.py
```

上述restore只恢复刚移回、哈希已验证的旧脚本原import。其他文件、B11改动和分支不动。新增测试/报告另按本批差异处理；共享账本和历史以补偿记录说明撤销，不整文件checkout抹去其他agent的记录。
