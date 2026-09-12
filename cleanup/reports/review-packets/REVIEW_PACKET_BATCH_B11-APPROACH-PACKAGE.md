# B11 审查材料：approach 实现归入 package

## 当前结论

Codex 已按 REQ-008 / E0033-E0035 的 C1-C8 完成实现和验证，真实 reviewer 在 E0037 独立复测后验收通过。未提交、未推送。执行分支仍是 audit，主线基准是 master；本批迁移前快照为 a1d933b，未操作原 scheduler worktree 源码。

## 改了什么

七个活跃实现整体移入 `approach/`：approach_Eq、approach_def、approach_sched、approach_initiator、approach_collector、approach_sim、approach_setup。文件名保留，七份内容逐字节未变，共 5,181 行。函数体、类内方法、签名、常量、内部 import 均未改。

根目录保留同名兼容入口，所有外部调用者不改。新旧导入得到同一个 module 对象，setter、直接属性赋值和 monkeypatch 不会分成两份状态。两个原有脚本入口仍转发执行原逻辑。新增 package 初始化文件不导入任何子模块。

`approach_util33.py`、`appoach_plot6.py`、旧 runtime、Repack 修复、全部保护区和既有测试不在本批修改范围。

## 验证

| 检查 | 迁移前 | 迁移后 |
| --- | --- | --- |
| B10 真实 Split/Repack 基线及初始化契约 | 3 passed | 3 passed |
| 既有 collector 测试 | 2 passed / 5 failed | 2 passed / 5 failed，名称和异常类型逐项一致 |
| 新增 package 兼容测试 | package 缺失的预期 RED | 32 passed |
| 七个实现 SHA-256 | 已固定 | 七项完全一致 |
| 12-module import probe | reviewer 独立通过 | Codex 通过 |
| main_approach / motiv / abla 的 --help | reviewer 独立通过 | 三项 rc=0 |

使用 `conda activate gurobi`，实际 Python 为 `/home/zhangchg/miniconda3/envs/gurobi/bin/python`，禁用 pytest cache 和 pyc 写入，绘图后端为 Agg。迁移后合跑为 **37 passed / 5 failed，22.30 秒**；不是全套测试全绿。JUnit 结构化比对确认没有新增失败。

reviewer 在 `/tmp/b11_review_accept/` 独立复跑得到相同的 **37 passed / 5 failed，29.45 秒**，并核对七份实现 SHA、兼容入口、两份账本原字节和保护区无改动。Codex 随后确认 E0037 和守卫 check rc=0，没有代替 reviewer 自行批准。

五项原有失败：
- test_basic_functionality：AttributeError，缺少 task_realloc_curr。
- test_motiv_exp_specific_stats、test_summary_generation、test_formatted_output、test_edge_cases：ZeroDivisionError，task_cnt 为零。

上述问题没有在本批修复，也没有改测试以掩盖失败。原测试文件保持不变。两个可能写文件的原演示没有直接运行；兼容测试拦截 runpy 验证转发参数、运行名和 __main__ 身份，原演示内容由文件 SHA 保证未变。

## Pickle 边界

保留九份迁移前真实生成的 pickle 基线，覆盖七个模块的类/函数引用及两个对象实例。它们在迁移后都能读取；新对象的 pickle 往返正常。

**新 pickle 的模块名为 approach.*，不保证能被未迁移的 test_pipeline 读取。** 这项边界由 reviewer C3 明确接受。没有为了跨分支兼容引入自定义 loader、exec 或改写类的 __module__。

## 账本与恢复

`cleanup/move-ledger.csv` 追加七条 14 列记录，`cleanup/reachable-files.csv` 追加九条 10 列记录。追加部分为 LF，两个文件的全部旧字节与 HEAD 相同；没有创建不存在的 cleanup/actions.csv。

精确迁移前源码可从 `git show a1d933b:<原文件名>` 读取。恢复应只反向应用 B11 的路径迁移、兼容入口及其新增测试/报告，不恢复整份共享状态文件，不覆盖其他 agent 的改动，不使用 git reset/clean。

`cleanup/tools/dialogue_guard.py` 的 E0034 例外条目来自 reviewer 的 E0035 恢复动作，是独立协调改动，不算作 B11 源码实现，也没有随本批提交。

## 证据位置

- 机器报告：`cleanup/reports/b11-approach-package-compat.json`，含迁移前 SHA、九份旧 pickle、失败集合和 CSV 前缀校验。
- 新测试：`test_approach_package_compat.py`，32 项。
- 临时原始日志及 JUnit：`/tmp/scheduler-b11-post.2Kwyt8/`。临时目录不是长期唯一依据，关键结果已写入机器报告。
- 全局状态：`CLEANUP_STATUS.md`；动作历史：`FILE_ADJUSTMENT_RECORD.md`。

## 七份兼容入口全文

### approach_Eq.py

```python
"""Compatibility alias for approach.approach_Eq."""

import sys
from approach import approach_Eq as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
```

### approach_collector.py

```python
"""Compatibility entry point for approach.approach_collector."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("approach.approach_collector", run_name="__main__", alter_sys=True)
else:
    import sys
    from approach import approach_collector as _implementation

    sys.modules[__name__] = _implementation
```

### approach_def.py

```python
"""Compatibility alias for approach.approach_def."""

import sys
from approach import approach_def as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
```

### approach_initiator.py

```python
"""Compatibility alias for approach.approach_initiator."""

import sys
from approach import approach_initiator as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
```

### approach_sched.py

```python
"""Compatibility alias for approach.approach_sched."""

import sys
from approach import approach_sched as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
```

### approach_setup.py

```python
"""Compatibility alias for approach.approach_setup."""

import sys
from approach import approach_setup as _implementation

if __name__ != "__main__":
    sys.modules[__name__] = _implementation
```

### approach_sim.py

```python
"""Compatibility entry point for approach.approach_sim."""

if __name__ == "__main__":
    import runpy

    runpy.run_module("approach.approach_sim", run_name="__main__", alter_sys=True)
else:
    import sys
    from approach import approach_sim as _implementation

    sys.modules[__name__] = _implementation
```
