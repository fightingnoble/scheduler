# Repack 诊断辅助代码说明

> 2026-03-17：为调试 repack 执行情况添加的诊断代码

---

## 一、诊断代码位置与作用

### 1. `approach_setup.py:127-133` — Repack 触发点

```python
import sys
sys.stderr.write(f"[SETUP_BENCHMARK] Repack triggered!\n")
sys.stderr.write(f"  ratioA={args.exec_t_comp_ratioA}, ratioB={args.exec_t_comp_ratioB}\n")
sys.stderr.write(f"  num_bins={args.num_bins}\n")
```

**作用**：确认 repack 是否被 `setup_benchmark` 触发。输出到 stderr 是因为 `run_benchmark_setup_pipeline` 内部用 `redirect_stdout` 把 stdout 重定向到日志文件，print 不会显示在终端。

**输出示例**：
```
============================================================
[SETUP_BENCHMARK] Repack triggered!
  ratioA=0.7, ratioB=0.5
  num_bins=-1
============================================================
```

### 2. `sim_main.py:508-517` — Repack 开始执行

```python
sys.stderr.write(f"[REPACK DIAGNOSTIC] Starting repack for {strategy_name}\n")
sys.stderr.write(f"  - ratioA=..., ratioB=...\n")
sys.stderr.write(f"  - num_bins=..., num_cores=...\n")
sys.stderr.write(f"  - Phase 1 tasks: {len(phase1_pids)}\n")
sys.stderr.write(f"  - glb_p_list size: {len(glb_p_list)}\n")
```

**作用**：在 `perform_bin_packing` 内部，确认 repack 分支被进入，打印 Phase 1 备份的任务数和 glb_p_list 大小。同样输出到 stderr。

**输出示例**：
```
============================================================
[REPACK DIAGNOSTIC] Starting repack for cyc-S
  - ratioA=0.7, ratioB=0.5
  - num_bins=-1, num_cores=231
  - Phase 1 tasks: 53
  - glb_p_list size: 53
============================================================
```

### 3. `sim_main.py:555-560` — Repack 不完整放置检测

```python
print(f"[REPACK DIAGNOSTIC] Incomplete placement detected:")
print(f"  - Phase 1 tasks: {len(phase1_pids)}")
print(f"  - Repack placed: {len(repack_pids)}")
print(f"  - Missing tasks: {len(missing)}")
print(f"  - Missing PIDs (first 10): {sorted(list(missing))[:10]}")
```

**作用**：repack 执行完 `push_task_into_bins_new` 后，对比 Phase 1 和 repack 放置的任务集合，报告缺失任务数和 PID。输出到 stdout（被重定向到日志文件）。

### 4. `sim_main.py:565-571` — Repack 成功

```python
print(f"[REPACK DIAGNOSTIC] SUCCESS")
print(f"  - Strategy: {strategy_name}")
print(f"  - ratioB: {args.exec_t_comp_ratioB}")
print(f"  - Tasks placed: {len(repack_pids)}")
```

**作用**：所有 Phase 1 任务都被 repack 成功放置时输出。

### 5. `sim_main.py:578-584` — Repack 失败 fallback

```python
print(f"[REPACK DIAGNOSTIC] FAILED - FALLBACK TO PHASE 1")
print(f"  - Strategy: {strategy_name}")
print(f"  - Error: {e}")
print(f"  - Restored Phase 1 layout with {len(phase1_pids)} tasks")
```

**作用**：repack 抛出 `ResourceInsufficientError` 或 `RuntimeError` 后，fallback 到 Phase 1 布局时输出。

---

## 二、输出通道说明

| 位置 | 输出通道 | 原因 |
|------|---------|------|
| `approach_setup.py` 触发点 | **stderr** | `run_benchmark_setup_pipeline` 用 `redirect_stdout` 把 stdout 重定向到日志文件 |
| `sim_main.py` repack 开始 | **stderr** | 同上，在 `redirect_stdout` 作用域内 |
| `sim_main.py` 完整性检查/成功/失败 | **stdout** (日志文件) | 在 `redirect_stdout` 作用域内，输出到 `path_ctx.get_log_path()` |

**查看诊断输出**：
- stderr 输出直接显示在终端
- stdout 输出需要查看日志文件：`find csv/ -name "*.log" -newer /tmp/timestamp`

---

## 三、VSCode 调试配置

`.vscode/launch.json` 中添加了两个调试配置：

| 配置名 | 策略 | 关键参数 |
|--------|------|----------|
| `cyc-S repack (ratioB=0.5)` | cyc-S | `num_bins=-1, ratioA=0.7, ratioB=0.5` |
| `reserv repack (num_bins=4, ratioB=0.5)` | reserv | `num_bins=4, ratioA=0.7, ratioB=0.5` |

**建议断点位置**：

| 文件 | 行号 | 含义 |
|------|------|------|
| `sim_main.py` | 521 | `extract_pid2_bin_id` — 清空 scheduling_table 前 |
| `sim_main.py` | 534 | `push_task_into_bins_new` — repack 贪心算法入口 |
| `sim_main.py` | 553 | `missing = phase1_pids - repack_pids` — 完整性检查 |
| `sched/pre_alloc_new.py` | 243 | `"No more bin can be created"` 警告处 |

---

## 四、清理

这些诊断代码是临时添加的，确认 repack 行为后可移除。移除范围：

- `approach_setup.py:127-133` — 6 行 stderr 输出
- `sim_main.py:508-517` — 9 行 stderr 输出
- `sim_main.py:555-560` — 5 行 print（Incomplete）
- `sim_main.py:565-571` — 6 行 print（SUCCESS）
- `sim_main.py:578-584` — 6 行 print（FAILED）

移除后不影响 repack 逻辑本身（backup/fallback 机制保留）。
