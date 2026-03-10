# 统计收集与仿真集成规范

**一句话核心**：基于 T-Digest 的流式统计收集系统，支持关键路径追踪的延迟分解，实现任务级/分区级/系统级多粒度性能分析。

---

## 1. 设计思想

### 1.1 四个关键设计决策

| 决策 | 问题 | 解决方案 |
|------|------|----------|
| **为什么用 T-Digest？** | 调度延迟呈长尾分布，传统分箱难以准确估计 p99 | 自适应质心，尾部高分辨率 |
| **为什么关键路径追踪？** | DAG 汇聚点会导致延迟重复计算 | 只追踪最长路径（关键路径） |
| **为什么分层存储？** | 不同分析需要不同粒度 | 任务级/分区级/系统级三级存储 |
| **为什么超周期归一化？** | 跨配置比较需要统一基准 | 归一化到 `total_pwr * T_hp` |

### 1.2 核心设计原则

1. **流式统计**：所有分布指标使用 `TDigestStreamingHistogram`，无需存储原始数据
2. **关键路径分析**：延迟分解采用关键路径追踪，避免在 DAG 汇聚点重复计算
3. **超周期归一化**：系统级指标归一化到 `(total_pwr * T_hp)`，实现跨配置可比性

---

## 2. 数据流架构

### 2.1 整体流程

```
仿真事件
    │
    ├── 任务启动 ──────┐
    ├── 任务计算 ──────┼── 记录 API ── 临时累加器 (task_curr_stat)
    ├── 任务完成 ──────┤                    │
    ├── 重分配 ────────┤                    │
    ├── 闲置 ──────────┤                    │
    └── 周期边界 ──────┘                    ▼
                                    周期边界处理
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
              任务级分布            分区级分布            系统级分布
         (dist_per_task_*)      (dist_per_part_*)    (dist_overall_*)
                    │                     │                     │
                    └─────────────────────┴─────────────────────┘
                                          │
                                          ▼
                                    get_summary()
```

### 2.2 关键流程

**任务生命周期**：
```
record_task_start()      # 记录任务到达时间
        │
        ▼
record_compute_progress() # 累积计算时间
        │
        ▼
record_task_finish()     # 任务完成，关键路径传播
        │
        ▼
record_e2e_finish()      # 链完成（如果是 sink）
```

**周期边界处理**：
```
forward_hyperperiod(T_hp)
    ├── 归一化: value / (total_pwr * T_hp)
    ├── 添加到 TDigest 分布
    └── 重置临时累加器
```

---

## 3. 核心概念

### 3.1 端到端延迟分解

在 DAG 结构中，端到端延迟由**关键路径**（最长路径）决定。为准确归因并行执行路径中的延迟，系统采用关键路径追踪算法，确保延迟组件（compute、realloc）在汇聚点不会被重复计算。

```
         ┌── Task A (2s) ──┐
Source ──┤                  ├── Sink
         └── Task B (3s) ──┘    (关键路径: B)
              ↑
           e2e_lat 最大
```

> **详细算法说明**：关键路径追踪的完整数据结构定义和传播逻辑，请参考 [`collector.md`](./collector.md)。

**双结构追踪概览**：

| 结构 | 用途 | 生命周期 |
|------|------|----------|
| `task_curr_stat[i]` | 存储任务 i 自身的执行时间、重分配开销 | 任务执行期间累积 |
| `task_pred_stat[i]` | 存储从各前驱传播来的关键路径统计 | 任务完成时选择最大 e2e_lat 的前驱 |

**传播逻辑**：
1. 任务完成时，从所有前驱中选择 `e2e_lat` 最大的（关键路径）
2. 累加当前任务统计，形成新的路径统计
3. 传播给所有后继

> 完整的数据结构定义（`path_stat` 字段）和递归传播公式，请参考 [`collector.md`](./collector.md)。

### 3.2 任务名称处理

`_get_base_task_name()` 方法去除超周期标号，实现跨周期聚合：

| 输入 | 输出 |
|------|------|
| `task1_0` | `task1` |
| `S1_1` | `S1` |
| `OP2_-1` | `OP2` |

### 3.3 归一化约定

系统级指标归一化到 `(total_pwr * T_hp)`：

```python
norm_factor = self.total_pwr * T_hp

self.dist_overall_idle.add(idle_load / norm_factor)
self.dist_overall_miss.add(miss_load / norm_factor)
self.dist_overall_realloc.add(realloc_cost / norm_factor)
```

---

## 4. 分布存储层次

### 4.1 任务级分布 (原始值)

```python
# Key: base_task_name (去除超周期标号)

# 关键路径计算时间
dist_per_task_exe: Dict[str, TDigestStreamingHistogram]

# 任务完成时间（相对于链起点 offset）
dist_per_task_ft: Dict[str, TDigestStreamingHistogram]

# 关键路径重分配开销
dist_per_task_realloc: Dict[str, TDigestStreamingHistogram]
```

### 4.2 分区级分布 (归一化)

```python
# Key: partition_id (如 "acc_p0", "acc_p1")

# 重分配时间比 (realloc_time / T_hp)
dist_per_part_realloc: Dict[str, TDigestStreamingHistogram]

# 每超周期重分配次数
dist_per_part_realloc_count: Dict[str, TDigestStreamingHistogram]
```

### 4.3 系统级分布 (归一化)

```python
# 归一化: value / (total_pwr * T_hp)

dist_overall_realloc   # 系统重分配成本比
dist_overall_idle      # 闲置算力比
dist_overall_miss      # 错失负载比
dist_overall_used      # 已用算力比
dist_overall_miss_count # 每周期错失任务数
dist_overall_total_load # 每周期总负载
```

---

## 5. 仿真集成流程

### 5.1 集成点概览

| API | 调用位置 | 功能 |
|-----|---------|------|
| `record_task_start(pid, start_t)` | `update_ready` | 记录任务到达时间 |
| `record_task_finish(G, pid, finish_t)` | `update_run` | 记录任务完成，传播关键路径 |
| `record_e2e_finish(G, sink, finish_t)` | `update_ready` | 记录链完成 |
| `record_realloc(part_id, delta_ld, tasks)` | `update_run` (state="R") | 累积重分配开销 |
| `record_realloc_num(part_id, tasks)` | `sched` | 记录重分配次数 |
| `record_compute_progress(task, dt, dload)` | `update_run` | 累积计算时间 |
| `record_period_load_arrival(load)` | `update_ready` | 累积周期负载 |
| `record_idle_capacity(idle_ld, state)` | `update_run` (state="S") | 累积闲置容量 |
| `record_miss(timeout_iter)` | `run_simulation` | 累积超时任务负载 |
| `forward_hyperperiod(T_hp)` | `run_simulation` | 周期边界处理 |

### 5.2 主循环流程

```
run_simulation()
    │
    ├─> 初始化: stats_collector.init_partition_stats()
    │
    └─> 主循环:
            │
            ├─> update_run()
            │       ├─> record_realloc()         [if state=="R"]
            │       ├─> record_compute_progress() [for each task]
            │       ├─> record_task_finish()     [if rem_load <= 0]
            │       └─> record_idle_capacity()   [if state=="S"]
            │
            ├─> update_ready()
            │       ├─> record_e2e_finish()      [for sinks]
            │       ├─> record_period_load_arrival() [for ops]
            │       └─> record_task_start()      [for new ready tasks]
            │
            ├─> sched()
            │       └─> record_realloc_num()     [if realloc triggered]
            │
            └─> 超周期边界:
                    ├─> record_miss()
                    └─> forward_hyperperiod(T_hp)
```

---

## 6. 摘要 API 速查

### 6.1 接口列表

| API | 返回内容 |
|-----|---------|
| `get_summary(num_bins, p_list)` | 完整统计摘要 (histogram + percentiles) |
| `get_utilization_avg_ratio()` | 利用率指标字典 |
| `get_realloc_info()` | 重分配统计 |
| `get_motiv_case1_stats()` | 利用率-可靠性权衡指标 |
| `get_motiv_case2_stats()` | 延迟分解 (exec/realloc/wait ratio) |
| `get_motiv_case3_stats()` | 负载-延迟相关性 (Spearman rho) |

### 6.2 摘要输出结构

```python
{
    'per_task_exe': {
        'task1': {
            'histogram': [(bin_start, bin_end, count), ...],
            'percentiles': {'p50': x, 'p90': y, 'p99': z, ...},
            'total_processed_count': N
        },
        ...
    },
    'per_task_ft': {...},
    'per_task_realloc': {...},
    'per_part_realloc': {...},
    'per_part_realloc_count': {...},
    'overall_realloc': {'histogram': [...], 'percentiles': {...}},
    'overall_idle': {...},
    'overall_miss': {...},
    'overall_used': {...},
    'overall_miss_count': {...},
    'overall_total_load': {...},
}
```

---

## 7. 实现细节

### 7.1 关键路径追踪实现

关键路径追踪的完整实现（数据结构、`record_task_finish` 方法等）请参考 [`collector.md`](./collector.md)。

### 7.2 周期边界处理

```python
def forward_hyperperiod(self, T_hp: float = 1.0):
    """超周期边界处理：归一化并累积到分布。"""
    norm_factor = self.total_pwr * T_hp

    # 系统级分布（归一化后添加）
    self.dist_overall_idle.add(self.idle_load / norm_factor)
    self.dist_overall_miss.add(self.miss_load / norm_factor)
    self.dist_overall_realloc.add(self.realloc_cost / norm_factor)
    self.dist_overall_used.add(self.used_load / norm_factor)
    self.dist_overall_total_load.add(self.total_load / norm_factor)

    # 分区级分布
    for part_id in self.part_realloc_curr:
        realloc_ratio = self.part_realloc_curr[part_id] / T_hp
        self.dist_per_part_realloc[part_id].add(realloc_ratio)

        realloc_num = self.part_realloc_num[part_id]
        self.dist_per_part_realloc_count[part_id].add(realloc_num)

    # 重置累加器
    self._reset_accumulators()
```

---

## 8. 文件位置

| 组件 | 文件 | 关键位置 |
|------|------|----------|
| `StatisticsCollector` 类 | `approach_collector.py` | 类定义 |
| `TDigestStreamingHistogram` | `ref_tdigest.py` | 分布存储 |
| 仿真集成 | `approach_sim.py` | `update_run`, `update_ready` |
| 主仿真循环 | `approach_sim.py` | `run_simulation` |
| 动机实验绘图 | `approach_collector.py` | `plot_motiv_case1/2` |

---

## 9. 添加新统计项

添加新统计指标的详细步骤，请参考 [`collector.md`](./collector.md)，包含完整的 6 步检查清单：

1. 在 `__init__` 中声明数据结构
2. 在数据收集点记录数据
3. 在 `get_summary` 中添加输出键
4. 扩展 `save_state` 序列化
5. 扩展 `load_state` 反序列化
6. 更新 `load_save_check` 验证

---

## 10. 注意事项

1. **样本量要求**：推荐样本数 > 1000，小样本分位数估计可能不可靠
2. **线程安全**：当前实现非线程安全，多线程环境需外部锁
3. **性能影响**：统计收集增加约 5-10% 开销
4. **任务名称格式**：超周期标号应遵循 `_数字` 模式（如 `task1_0`）
