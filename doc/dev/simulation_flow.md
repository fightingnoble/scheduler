# 仿真主流程详解

> 本文档详细说明从实验入口到仿真执行的完整流程。

---

## 1. 调用链总览

```
runner.py
    │
    └── run_main_approach_inproc() [scripts/exp_common.py:99]
            │
            └── main() [main_approach.py]
                    │
                    ├── setup_benchmark() [approach_setup.py:90]
                    │       ├── build_workload_and_criticality()
                    │       ├── create_scheduler_elements_with_config()
                    │       ├── build_simulation_env()
                    │       ├── perform_bin_packing() [sim_main.py:397]
                    │       └── apply_forced_num_cores() [非 repack 时]
                    │
                    └── run_simulation() [approach_sim.py:43]
```

---

## 2. Phase 1: 基准设置 (setup_benchmark)

**位置**: `approach_setup.py:90`

### 2.1 流程步骤

```
Step 0: build_workload_and_criticality()
    └── 生成任务图、设置 E2E 约束、计算关键性

Step 1: create_scheduler_elements_with_config()
    └── deduce_cfg2() - 计算 per-task deadline
    └── rsc_req_estm_quantile() - 估计资源需求

Step 2: build_simulation_env()
    └── 创建仿真环境、处理器实例

Step 3: perform_bin_packing()
    └── Phase 1: coleasing_alloc_cluster() - 空间分区
    └── Phase 2: push_task_into_bins_new() - 时间调度

Step 4: apply_forced_num_cores() [可选]
    └── 仅在非 repack 模式下应用资源约束
```

### 2.2 关键函数

| 函数 | 文件:行号 | 作用 |
|------|-----------|------|
| `setup_benchmark()` | approach_setup.py:90 | 主入口 |
| `coleasing_alloc_cluster()` | sched/global_sched.py | 空间分区（Split） |
| `push_task_into_bins_new()` | sched/global_sched.py:42 | 时间调度（Repack） |
| `apply_forced_num_cores()` | sim_main.py:91 | 资源约束 |

---

## 3. Phase 2: 仿真执行 (run_simulation)

**位置**: `approach_sim.py:43`

### 3.1 事件驱动循环

```python
while curr_t < stop_cond:
    # 1. 更新运行中任务
    update_run(processors, curr_t)

    # 2. 更新就绪队列
    update_ready(processors, curr_t)

    # 3. 调度决策
    sched(processors, curr_t)

    # 4. 推进时间
    curr_t = next_event_time

    # 5. 周期边界处理
    if curr_t >= next_hp_boundary:
        stats_collector.record_miss(...)
        stats_collector.forward_hyperperiod(T_hp)
```

### 3.2 统计收集点

| 事件 | 调用方法 | 收集内容 |
|------|----------|----------|
| 任务完成 | `record_task_finish()` | 任务级分布 |
| 链完成 | `record_e2e_finish()` | 链级分布 |
| 重分配 | `record_realloc()` | 重分配开销 |
| 闲置 | `record_idle_capacity()` | 闲置容量 |
| 周期边界 | `forward_hyperperiod()` | 系统级分布 |

---

## 4. 参数流向

### 4.1 exec_t_comp_ratioA

```
args.exec_t_comp_ratioA
    → args.quantile
    → coleasing_alloc_cluster(quantile=split_ratio)
    → rsc_req_estm_quantile(quantile=split_ratio)
```

**作用**: Phase 1 资源估计用的保守分位数（如 0.99 = 99th percentile）

### 4.2 exec_t_comp_ratioB

```
args.exec_t_comp_ratioB
    → binpack_cfg['exec_t_comp_ratioB']
    → push_task_into_bins_new(quantile=ratioB)
```

**作用**: Phase 2 时间窗分配用的激进分位数（如 0.80）
- `-1` 表示不执行 repack
- `ratioA > ratioB` 时启用软预留

### 4.3 num_cores

```
args.num_cores
    → apply_forced_num_cores(bin_list, max_core_num, target)
    → 修改 bin_list[i].num_resources
```

**约束**: 仅在 `need_repack=False` 时应用

---

## 5. 执行路径对照表

| 策略 | Steps | need_repack | num_bins | ratioB |
|------|-------|:-----------:|:--------:|:------:|
| cyc | 0-1-2 | False | -1 | -1 |
| glb | 0-1 | False | 1 | -1 |
| pglb | 0-1-2 | False | >1 | -1 |
| reserv | 0-1-2 + repack | True | >=2 | 0.5-0.99 |
| cyc-S | 0-1-2 + repack | True | -1 | 0.5-0.99 |

---

## 6. 关键约束

1. **资源约束仅适用于非 repack 阶段**
   - repack 修改 slack 分配，不修改资源分配

2. **`num_cores` 恒等于 `sum(b.num_resources for b in bin_list)`**
   - 由 `vectorized_core_allocation()` 维护

3. **Dump 路径使用约束后的 `num_cores`**
   - PathContext 与 bin_list 状态一致

---

## 7. 相关文档

- [e2e_sched_sim_flow.md](../spec/e2e_sched_sim_flow.md) - 设计规范
- [configuration_system.md](./configuration_system.md) - 配置系统
- [result_collection.md](./result_collection.md) - 结果收集
