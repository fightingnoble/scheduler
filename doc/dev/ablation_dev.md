# Ablation Experiment Development Log

> 2026-03-14：Repack 启用调试记录

---

## 〇、问题定位：Repack 放不下的根因分析

### 1. ratioA / ratioB 参数含义回顾

| 参数 | 阶段 | 作用 | 影响 |
|------|------|------|------|
| `ratioA` | Phase 1 (Step 1) | `deduce_cfg2(quantile=ratioA)` → 决定 bin 的 `num_resources` | ratioA 大 → 保守估算 → bin 更大 |
| `ratioB` | Repack (Step 1 重跑) | 重算 ERT/deadline/slack + `rsc_req_estm_quantile(ratioB)` | ratioB 小 → 激进 → 时间窗口更宽 |

**关键约束**：bin 的 `num_resources` 在 Phase 1 由 ratioA 确定，**repack 阶段不变**。

### 2. 关键洞察：quantile 变化不影响窗口相对位置

**代码分析** (`chain_slack_assign.py:287`)：
```python
ideal_cores = int(math.ceil(flops_rem / (slack_rem * FLOPS_PER_CORE)))
```

slack 按 FLOPS 比例分配。关键问题：quantile 变化时，`flops_dict[node] / flops_rem` 是否保持不变？

**验证**：所有任务使用相同的 `var_factor_list` 和 `lambda_ld`（Poisson 参数）

```
var_factor_list: [0.5, 1.0, 1.5, 2.0]
lambda_ld: 1.0
probs (所有任务相同): [0.375, 0.375, 0.1875, 0.0625]

quantile    A(exp=10)   B(exp=20)   B/A
  0.50        10.00      20.00     2.000  ✓
  0.70        10.00      20.00     2.000  ✓
  0.90        15.00      30.00     2.000  ✓
  0.99        20.00      40.00     2.000  ✓
```

**结论**：相同 var_factor 模式下，所有 quantile 的比例恒定 → **窗口相对位置不变**

### 3. 观察：测试结果

| 测试 | ratioA | ratioB | ratioB vs ratioA | 结果 | 未放置任务数 |
|------|:------:|:------:|:----------------:|:----:|:------------:|
| Test 1 | 0.7 | 0.5 | **B < A** (任务变小) | 失败 | **53** |
| Test 2 | 0.7 | 0.99 | **B > A** (任务变大) | 失败 | **53** |

**关键证据**：两个测试未放置任务数**完全相同（53个）**，证明与 ratioB 大小无关。

### 4. 分析：根因是时间维重建失败

| | Phase 1 (`coleasing_alloc_cluster`) | Repack (`push_task_into_bins_new`) |
|---|---|---|
| **空间维** | ILP 优化 bin 分配 | `pre_defined` 模式继承 Phase 1 |
| **时间维** | ILP 优化 scheduling_table | **需要重建**，但贪心算法无法复现 ILP 解 |
| **算法** | Gurobi ILP 全局优化 | 贪心在线模拟，逐时隙处理 |

**53 个任务未放置的原因**：
1. `extract_pid2_bin_id()` 调用 `bin.clear()` 清空 scheduling_table
2. Phase 1 ILP 建立的时间维分配被破坏
3. 贪心算法在线模拟顺序不同于 ILP 全局优化
4. DAG 非源节点的前驱在贪心模拟中未按 ILP 解的时序完成 → 后继永远不进入 ready_queue

### 5. 结论（修正版）

| 问题 | 答案 |
|------|------|
| Repack 放不下的根因 | **时间维重建失败**：`extract_pid2_bin_id` 清空 scheduling_table 后，贪心算法无法复现 ILP 的时间分配 |
| 与 quantile 关系 | **无关**。ratioB=0.5 和 ratioB=0.99 失败任务数完全相同（53个），说明根因是算法不兼容而非资源量 |
| 与窗口位置关系 | **无关**。相同 var_factor 模式下，窗口相对位置不随 quantile 变化 |
| 正确理解 | Phase 1 ILP 同时优化空间+时间；Repack 只继承空间，时间维需重建，但贪心算法做不到 |
| 是否应该报错退出 | **不应该**。Fallback 到 Phase 1 布局完全合法，报错会导致整批实验中断 |
| Fallback 行为 | repack 成功 → 使用 ratioB 优化的布局；repack 失败 → 恢复 Phase 1 的 ratioA 布局 + 打印警告 |

### 6. 处理策略

采用**方案 A（备份恢复）**：

```python
# approach_setup.py
bin_list_backup = copy.deepcopy(bin_list)  # 备份
try:
    perform_bin_packing(...)  # 尝试 repack
    # completeness check
    if missing:
        raise RuntimeError(...)
except (ResourceInsufficientError, RuntimeError):
    bin_list = bin_list_backup  # 恢复 Phase 1 布局
```

---

## 一、代码修改

### 1. `sched/pre_alloc_new.py` — 修复隐藏 bug

**问题**：`effective_max_size` 变量从未定义，被 bypass 掩盖多时。

**修复**（L166-170）：
```python
if bin_sel_mod == "pre_defined":
    effective_max_size = bin_list[pid2bin_id[_p.pid]].num_resources
else:
    effective_max_size = tot_cores
```

### 2. `sim_main.py::perform_bin_packing` — 重构：统一 repack 逻辑

**修改**（L401-565）：
- 移除 `bin_list_backup` 和 `phase1_pids` 参数（备份在内部创建）
- 移除 cyc-S bypass 分支，统一为 `pre_defined` 模式
- 备份和 fallback 逻辑完全封装在函数内部

**内部逻辑**：
| 条件 | 行为 | 备注 |
|------|------|------|
| `need_repack=False` | Phase 1: `coleasing_alloc_cluster` | 空间分区 + 装箱 |
| `need_repack=True` | Repack: `push_task_into_bins_new` + `pre_defined` | 继承 Phase 1 空间分配，只调整时间片 |

**返回值**：5-tuple `(bin_list, max_core_num, glb_p_list, hyper_p, repack_success)`

### 3. `approach_setup.py` — 简化调用

**修改前**（25 行）：准备备份 + 传递参数
**修改后**（15 行）：直接调用，无需准备备份

```python
# 6. 执行装箱算法（backup 和 fallback 逻辑已封装在 perform_bin_packing 内部）
bin_list, max_core_num, glb_p_list, hyper_p, repack_success = perform_bin_packing(
    args, glb_p_list, num_cores, bin_list, hyper_p,
    sim_step, path_para_dict, para_scan_group1,
    event_iter_dict, quantumSize, num_periods,
    cfg_para_dict, physical_graph_nx, need_repack,
    plot_path_para, path_ctx,
    scheduler_list, monitor_list,
    msg_dispatcher,
    a_data_pipe, w_data_pipe,
)
```

---

## 二、问题定位（详细版）

### Repack 失败根因：时间维重建失败

| 测试 | ratioB vs ratioA | 缺失任务数 |
|------|:---:|:---:|
| ratioB=0.5 | B < A（任务变小） | **53** |
| ratioB=0.99 | B > A（任务变大） | **53** |

**关键证据**：两个测试缺失数完全相同 → 失败与 ratioB 大小无关，是结构性问题。

**根因**（修正版）：
1. `extract_pid2_bin_id()` 调用 `bin.clear()` 清空 scheduling_table
2. Phase 1 ILP 建立的时间维分配被破坏
3. 贪心算法在线模拟顺序不同于 ILP 全局优化
4. DAG 非源节点的前驱在贪心模拟中未按 ILP 解的时序完成 → 后继不进入 ready_queue

**与 quantile 无关**：即使 ratioB = ratioA，repack 也会失败（相同 var_factor 模式下窗口相对位置不变）

**处理决策**：不报错退出，fallback 到 Phase 1 布局（本身合法）。

### 问题定位问答表

| 问题 | 答案 |
|------|------|
| Repack 放不下的根因 | **时间维重建失败**：`extract_pid2_bin_id` 清空 scheduling_table 后，贪心算法无法复现 ILP 的时间分配 |
| 与 quantile 关系 | **无关**。相同 var_factor 模式下，窗口相对位置不随 quantile 变化 |
| 证据 | ratioB=0.5 和 ratioB=0.99 失败任务数完全相同（53个），与资源量无关 |
| 是否应该报错退出 | 不应该。Fallback 到 Phase 1 布局合法，报错会导致整批实验中断 |
| pickle 问题 | Gurobi 许可证过期导致的异常传播，许可证更新后自动解决 |

---

## 三、实验结果

### 最终结果

| Case | 成功/总数 | PDF 输出 | 状态 |
|------|:---------:|----------|:----:|
| Case 1 (cyc-S vs cyc) | 12/12 | `case1_motiv1_style.pdf`, `case1_satisfy_projection.pdf` | ✅ |
| Case 2 (pglb vs glb) | 32/32 | `case2_overhead.pdf`, `case2_tradeoff.pdf` | ✅ |
| Case 3 (reserv vs pglb) | 196/224 | `case3_overhead.pdf`, `case3_tradeoff.pdf` | ✅ |

### Case 3 缺失 28 点分析

```
reserv bins=1:  48/48  (缺 0)
reserv bins=2:  48/48  (缺 0)
reserv bins=4:  35/48  (缺 13)
reserv bins=8:  33/48  (缺 15)
                       ------
                 总缺:   28
```

**原因**：`num_bins` 是请求的分区数，但 Phase 1 的 `coleasing_alloc_cluster()` 根据 DAG 拓扑分组。小负载（如 200T-1C）只能分出 2-3 个组，请求 `num_bins=8` 时，后续仿真访问 `acc_p7` 触发 `KeyError`。

**与 repack 无关**——之前 bypass 版本也存在此问题（198/224）。

---

## 四、趋势观察

### Case 1 (cyc-S vs cyc)
- cyc-S miss rate ~7%（ratioB=0.5~0.9），ratioB=0.99 跳升至 13%
- 跳升原因：`deduce_cfg2(ratioB=0.99)` 重算后 deadline 变紧，但 repack fallback 使 bin 布局等价于 Phase 1

### Case 2 (pglb vs glb)
- `realloc_ratio` 随 `num_bins` 增加单调下降（0.131 → 0.009）：核心结论成立
- `miss_ratio` 在高分区数时略升（资源碎片化）

### Case 3 (reserv vs pglb)
- `realloc_ratio` 随 ratioB 增大单调下降
- 预期 U-shape 不明显：repack fallback 导致 ratioB 只影响时间参数，未实现真正的空间重放置

### 关键数据对比（小规模测试）

| 配置 | ReallocCount | ReallocRatio |
|------|:-----------:|:------------:|
| pglb (无 repack) | 32.67 | 0.0583 |
| reserv p50 (repack fallback) | **4.53** | **0.0075** |
| reserv p99 (repack fallback) | **2.00** | **0.0006** |

---

## 五、Repack 执行验证（2026-03-18）


```prompt
Ultrathink:
- 使用superpower 规划修改和测试
- 添加测试用的代码，保证输出辅助信息帮助确认repack的执行情况和成功失败情况统计。
- 启动一个subagent 测试，可以使用python-debug 技能
-  确定repack 真有有开始在执行，而不是一直卡着
- 现在的cyc-S 和 reserv 都能触发repack了 确认现在repack的执行情况。
```
### 验证目的

确认 repack 在 cyc-S 和 reserv 策略下是否真正被触发和执行，而非被静默跳过。

### 诊断方法

在 `sim_main.py:perform_bin_packing` 和 `approach_setup.py:setup_benchmark` 中添加 `sys.stderr.write` 诊断输出（绕过 `redirect_stdout` 日志重定向），跟踪 repack 的触发、执行和结果。

### 测试命令

```bash
python main_approach.py \
  --profiling_filename profiling/profiling_light.csv \
  --gen_benchmark --G_decomp_mode full \
  --exec_t_comp_ratioA 0.7 --exec_t_comp_ratioB 0.5 \
  --e2e_latency 0.1 --aux_scale_factor 1 \
  --test_case cyclic --num_bins -1 \
  --bin_pack_cfg Bp_guided.json --n_p 2 \
  --policy reserv --root_dir test_repack_diag/single_test
```

### 诊断输出

```
[SETUP_BENCHMARK] Repack triggered!
  ratioA=0.7, ratioB=0.5
  num_bins=-1

[REPACK DIAGNOSTIC] Starting repack for cyc-S
  - ratioA=0.7, ratioB=0.5
  - num_bins=-1, num_cores=231
  - Phase 1 tasks: 53
  - glb_p_list size: 53
```

### 测试结论

| 检查项 | 结果 |
|--------|------|
| `setup_benchmark` 中 repack 触发条件 | ✅ `ratioB != -1 and ratioA != ratioB` 正确触发 |
| `perform_bin_packing` 进入 repack 分支 | ✅ `need_repack=True` 进入 `else` 分支 |
| `push_task_into_bins_new` 实际执行 | ✅ 使用 `pre_defined` 模式调用 |
| 完整性检查 | 53 个任务未放置 → 触发 `RuntimeError` |
| Fallback 机制 | ✅ 恢复 `bin_list_backup`（Phase 1 布局） |

### 结论

1. **Repack 确实被触发且执行**，不存在"被静默跳过"的问题
2. **所有 repack 均失败并 fallback**，根因不变：`extract_pid2_bin_id` 清空 `scheduling_table` 后贪心算法无法复现 ILP 时间维分配
3. **ratioB 只影响 `deduce_cfg2` 重算的 deadline**，不改变实际 bin 布局（因为 fallback）
4. 这解释了 Exp 1 非单调现象和 Exp 3 U-shape 不明显的原因

### 诊断代码位置

| 文件 | 行号 | 诊断内容 |
|------|------|----------|
| `approach_setup.py` | L127-133 | `[SETUP_BENCHMARK] Repack triggered!`（stderr） |
| `sim_main.py` | L508-517 | `[REPACK DIAGNOSTIC] Starting repack`（stderr） |
| `sim_main.py` | L554-563 | `[REPACK DIAGNOSTIC] Incomplete placement`（stdout → log） |
| `sim_main.py` | L565-571 | `[REPACK DIAGNOSTIC] SUCCESS`（stdout → log） |
| `sim_main.py` | L575-583 | `[REPACK DIAGNOSTIC] FAILED - FALLBACK`（stderr） |

> **注意**：`run_benchmark_setup_pipeline` 使用 `redirect_stdout` 将 stdout 重定向到日志文件，因此诊断信息使用 `sys.stderr.write` 输出到终端。

---

## 六、遗留问题

1. **Repack 贪心算法放置不完整**：需在 Gurobi 框架内重解时间片分配，而非使用贪心在线模拟
2. **Case 1 ratioB=0.99 跳升**：需验证真正的 repack 成功时是否恢复
3. **num_bins 超出分组数**：需在参数扫描前自动过滤无效配置

---

## 六、运行命令

```bash
# 单个 Case
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100

# 或批量运行
bash scripts/run_abla_exps.sh ./abla_results 100
```
