# B13 审查材料：分离闲置 HDF5 工具

## 当前结论

真实 reviewer 已在 E0045 按 D1-D6 独立复测并验收B13，Codex已从磁盘核对裁决和守卫check rc=0。联合回归58项中53通过、5项原有失败，无error或skip。未提交、未推送；B11/B12已验收内容保留。主线基准master，本批执行前HEAD为a1d933b。

## 本批调整

utils.py仅删除41行：原L13的专用h5py导入，以及原L16-55的注释、CHUNK_SIZE和get_next_chunk_id/save_chunk/load_h5_file三个完整函数。40行函数簇原字节保存在同目录utils_unused.py，新文件只补必要导入。

活utils其余代码逐字节不变，没有清理重复numpy导入，没有改路径构造、Found类、任何既有类内方法或接口。共享check_parents_path仍在utils。新模块只从utils引用这个建目录函数，不构成循环。

分类为unused：这是独立、当前未启用的HDF5 trace方案，不能证明被另一个版本替代。它仍可通过import utils_unused使用；原utils不再导出这四个HDF5符号，也不再直接加载h5py。

h5py仍保留在requirement.txt中，供这项保留能力使用。本批没有修改依赖清单或安装环境。

## 依据和验证

全项目静态扫描只负责定位，实际实施前又做了调用、星号导入、命名空间使用、文档及所有tracked文件类型复核。三函数的外部引用只剩两处历史注释，没有发现活调用；真实reviewer独立确认这一边界。

| 检查 | 结果 |
| --- | --- |
| 新测试RED | 目标utils_unused.py缺失，1 failed / 8 deselected |
| 新测试GREEN | 9 passed，2.82秒 |
| 函数簇 | 原40行SHA完全一致 |
| utils剩余代码 | 0新增/41删除，SHA与实施前计算的纯删除结果一致 |
| 新进程import utils | sys.modules不含h5py |
| 新进程再import utils_unused | 归档功能可用，共享建目录函数身份一致 |
| 五套联合回归 | 53 passed / 5原有failed |
| 三入口help | 全部rc=0 |
| 非B13源码/脚本 | 152份SHA逐项不变 |
| 依赖及保护区 | requirement.txt哈希不变，保护路径零diff |
| 账本 | move-ledger新增1行14列，reachable-files新增2行10列，旧前缀保留 |
| git diff --check | 通过 |

真实HDF5基线保留：24条短缓冲不写文件但会先创建父目录；25条写chunk_0并清空缓冲；force尾部2条写chunk_1，共27条读回，gzip压缩；空文件下个编号为0，首次写入后为1。非法键unrelated的IndexError、OSError/KeyError处理、输出文字均不修正。

五项原有失败仍属于collector：basic_functionality缺task_realloc_curr，其余四项为除零。本批没有修改既有测试或用skip/xfail隐藏问题。

## 证据与后续

- 源码：utils.py、utils_unused.py。
- 新测试：test_unused_hdf5_utils.py。
- 机器基线：cleanup/reports/b13-utils-hdf5-baseline.json，含原40行、原/剩余SHA、真实HDF5行为、152份源码与依赖/账本快照。
- 原始回归日志：/tmp/scheduler-b13-regression-uxov2h3s/。
- 深度优先探索摘要：REVIEW_PACKET_EXPLORATION_DFS.md。137文件清单是B13前的静态快照，不冒充实时清单；本批新增路径和变化由B13报告与账本说明。
- B13已验收；下一步沿utils路径/配置依赖继续只读核对，新的源码批次仍须另提案。不因这组通过就宣称全仓清理完成。

## 恢复

以下命令仅供明确决定撤销B13时使用，本轮没有执行。先确认当前utils正好是本批结果；有后续修改则必须停止，改用局部反向补丁。

```bash
set -e
cd /home/zhangchg/git_repo/scheduler-audit-20260612
test "$(sha256sum utils.py | cut -d' ' -f1)" = "14bcb37deccafb0b390ed8cb724eba4b0f167dee1aaa4e4d4c97b68168542965"
test "$(git show a1d933b1f26efd4d569eb3d8ffb313447294443b:utils.py | sha256sum | cut -d' ' -f1)" = "94c6ac9d52a1a3506fae0ab9a68629cbcd3a5c817f67103431ca7ac7de0c4790"
git restore --source=a1d933b1f26efd4d569eb3d8ffb313447294443b --worktree -- utils.py
```

上述命令只恢复utils原片段/导入，不改B11/B12和其他源码。新增归档模块、测试、报告另按B13差异处理；共享历史用补偿记录，不整文件checkout覆盖双方记录。
