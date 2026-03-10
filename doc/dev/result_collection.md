# 结果收集与绘图

> 本文档说明统计收集系统和绘图流程。

---

## 1. StatisticsCollector 架构

### 1.1 三级分布存储

```
任务级 (dist_per_task_*)
    │ - 原始值（不归一化）
    │ - Key: base_task_name
    │
分区级 (dist_per_part_*)
    │ - 归一化到 T_hp
    │ - Key: partition_id
    │
系统级 (dist_overall_*)
      - 归一化到 total_pwr * T_hp
      - 单一分布
```

### 1.2 核心数据结构

| 分布 | 类型 | 说明 |
|------|------|------|
| `dist_per_task_exe` | Dict[str, TDigest] | 关键路径计算时间 |
| `dist_per_task_ft` | Dict[str, TDigest] | 任务完成时间 |
| `dist_per_task_realloc` | Dict[str, TDigest] | 关键路径重分配开销 |
| `dist_per_part_realloc` | Dict[str, TDigest] | 分区重分配时间比 |
| `dist_overall_idle` | TDigest | 闲置算力比 |
| `dist_overall_miss` | TDigest | 错失负载比 |
| `dist_overall_realloc` | TDigest | 重分配成本比 |

---

## 2. 收集 API

### 2.1 记录方法

| 方法 | 调用时机 | 收集内容 |
|------|----------|----------|
| `record_task_start(pid, start_t)` | update_ready | 任务到达时间 |
| `record_task_finish(G, pid, finish_t)` | update_run | 任务完成 + 关键路径传播 |
| `record_e2e_finish(G, sink, finish_t)` | update_ready | 链完成 |
| `record_realloc(part_id, delta_ld, tasks)` | update_run (state="R") | 重分配开销 |
| `record_compute_progress(task, dt, dload)` | update_run | 计算时间累积 |
| `record_idle_capacity(idle_ld, state)` | update_run (state="S") | 闲置容量 |
| `record_miss(timeout_iter)` | run_simulation | 超时任务负载 |
| `forward_hyperperiod(T_hp)` | 周期边界 | 归一化 + 累积 |

### 2.2 关键路径追踪

```python
# 双结构
task_curr_stat[task] = {'realloc', 'compute', 'realloc_num'}
task_pred_stat[task][pred] = {'realloc', 'compute', 'realloc_num', 'e2e_lat'}

# 传播逻辑
1. 任务完成时，选择 e2e_lat 最大的前驱（关键路径）
2. 累加当前任务统计
3. 传播给所有后继
```

> 详见 [doc/spec/stat/collector.md](../spec/stat/collector.md)

---

## 3. 摘要 API

### 3.1 通用方法

| 方法 | 返回内容 |
|------|----------|
| `get_summary(num_bins, p_list)` | 完整摘要 (histogram + percentiles) |
| `get_utilization_avg_ratio()` | 利用率指标 |
| `get_latency_breakdown_avg_ratio()` | 延迟分解 (exec/realloc/wait) |

### 3.2 实验专用方法

| 方法 | 实验 | 返回内容 |
|------|------|----------|
| `get_motiv_case1_stats()` | Case 1 | `{idle_mean_ratio, miss_mean_ratio, ...}` |
| `get_motiv_case2_stats()` | Case 2 | `{utilization, latency_breakdown, ...}` |
| `get_motiv_case3_stats()` | Case 3 | `{spearman_rho, binned_summary, ...}` |

---

## 4. 绘图 API

### 4.1 静态方法

| 方法 | 位置 | 用途 |
|------|------|------|
| `plot_motiv_case1(data_points, save_path)` | L714-909 | 利用率-可靠性权衡 |
| `plot_motiv_case2(data_points, plot_type, save_path)` | L951-1063 | 可扩展性分析 |

### 4.2 数据格式

**Case 1**:
```python
data_points = [
    {
        'idle_mean_ratio': 0.45,
        'miss_mean_ratio': 0.15,
        'realloc_mean_ratio': 0.0,
        'miss_mean_count': 12.5,
        'label': 'p50'
    },
    ...
]
```

**Case 2**:
```python
data_points = [
    {
        'tiles': 400,
        'load_factor': 1.0,
        'chains': 4,
        'label': '400T-1.0×-4C',
        'utilization': {...},
        'latency_breakdown': {...}
    },
    ...
]
```

---

## 5. 缓存机制

### 5.1 缓存文件

```
{output_dir}/
├── case1_summary.json
├── case2_summary.json
└── case3_summary.json
```

### 5.2 使用方式

```bash
# 正常运行（保存缓存）
python -m scripts.motiv_exp_runner --case 1 --output_dir ./results

# 从缓存重绘（跳过仿真）
python -m scripts.motiv_exp_runner --case 1 --use_plot_cache
```

### 5.3 缓存验证

```python
if len(results) != len(self.ratios):
    print(f"Error: Cache invalid. Expected {len(self.ratios)}, found {len(results)}")
    return
```

---

## 6. 序列化

### 6.1 save_state()

```python
stats_collector.save_state('./stats_collector.json')
```

### 6.2 load_state()

```python
stats_collector = StatisticsCollector.load_state('./stats_collector.json')
```

---

## 7. 相关文档

- [statistics_collection_spec.md](../spec/stat/statistics_collection_spec.md) - 设计规范
- [collector.md](../spec/stat/collector.md) - 关键路径算法
- [tdigest_system_spec.md](../spec/stat/tdigest_system_spec.md) - T-Digest 算法
