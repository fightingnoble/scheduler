# 运行时开销测量规范

**一句话核心**：测量 Algorithm 2（`alloc_fn`）的软件决策延迟，判断其是否在关键路径上。

---

## 1. 问题背景

### 1.1 审稿人问题

> "What is the runtime overhead of Algorithm 1 and Algorithm 2? Will the algorithms be on the critical path?"

### 1.2 算法性质分析

| 算法 | 执行时机 | 是否在关键路径 |
|------|----------|----------------|
| Algorithm 1 (Multi-Chain Slack Distribution) | 离线/编译时 | **否** |
| Algorithm 2 (Dynamic colocation and allocation) | 运行时，事件驱动 | **需测量** |

### 1.3 关键路径定义

```
任务切换流程：
  trigger_cond() → alloc_fn() → swt_lat (硬件切换)
       ↓              ↓              ↓
   不在关键路径   可能在关键路径   必定在关键路径
   (metadata收集)  (仅当realloc=True)  (数据迁移)
```

**关键洞察**：
- `trigger_cond` 的 metadata 收集是异步的，不在关键路径上
- `alloc_fn` 仅当 `realloc=True` 时在关键路径上
- 如果 `alloc_fn << swt_lat`，软件开销被硬件切换"吸收"，**不延长**关键路径
- 如果 `alloc_fn > swt_lat`，软件决策**显著延长**了关键路径

---

## 2. 测量方案

### 2.1 测量位置

**文件**: `approach_def.py:Acc_p.sched()`

```python
def sched(self, curr_t, new_comp, new_ready_list, **kwargs):
    # ...
    if realloc:
        _t0 = time.perf_counter()
    alloc_map_curr = self.alloc_fn(curr_t, realloc)
    if self.stats_collector and realloc:
        _elapsed = time.perf_counter() - _t0
        self.stats_collector.record_sched_overhead(self.id, _elapsed, self.swt_lat)
    # ...
```

### 2.2 为什么只测 `alloc_fn`？

| 组件 | 是否测量 | 原因 |
|------|----------|------|
| `trigger_cond` | 否 | metadata 收集异步，条件判断是 scalar 布尔逻辑，开销可忽略 |
| `alloc_fn` | **是** | 仅当 `realloc=True` 时在关键路径上，是 Algorithm 2 的核心 |
| `update_queue` | 否 | 可与 `swt_lat` 并行执行，不在关键路径上 |

### 2.3 记录内容

| 数据 | 存储方式 | 说明 |
|------|----------|------|
| 绝对时间 (秒) | TDigest | `alloc_fn` 执行的墙钟时间 |
| Ratio | TDigest | `alloc_fn` 时间 / `swt_lat` |
| Max 时间 | float | 精确最大值（TDigest 无法提供） |

---

## 3. 时间单位系统

### 3.1 关键变量

```python
# approach_Eq.py
time_unit = 1e-6  # 1 sim unit = 1 µs

# set_time_unit(timestep, int_slot)
#   int_slot=False: time_unit = timestep, normalize_factor = 1
#   int_slot=True:  time_unit = 1, normalize_factor = timestep
```

### 3.2 `swt_lat` 的单位

**关键结论**：`trasfer_realloc_as_task()` 返回的是**秒**，不是 sim units！

```python
def trasfer_realloc_as_task(BW_DRAM, cap, tile_buffer_size, time_norm_factor=1.0):
    return cap * tile_buffer_size / BW_DRAM / time_norm_factor  # 单位：秒
```

**常见错误**：
```python
# 错误！swt_lat 已经是秒，不需要再乘 time_unit
swt_lat_s = swt_lat * time_unit  # ❌

# 正确
swt_lat_s = swt_lat  # ✓
```

### 3.3 Ratio 计算

```python
ratio = elapsed_s / swt_lat_s  # 量纲：秒/秒 = 无量纲
```

---

## 4. TDigest API 注意事项

### 4.1 `percentile(p)` 参数范围

**关键**：参数 `p` 是 **0-100**，不是 0-1！

```python
# 正确
td.percentile(99)   # P99 的值
td.percentile(50)   # P50 的值

# 错误！
td.percentile(0.99)  # P0.99 的值（几乎是最小值）
```

### 4.2 TDigest 无法提供精确 Max

```python
# 必须单独记录 max
self.sched_overhead_max_s = max(self.sched_overhead_max_s, elapsed_s)
```

### 4.3 AccumulationTree 结构

```
TDigest
  └── C: AccumulationTree（有序字典）
        └── key: float (mean 值，用于排序)
        └── value: Centroid(mean=..., count=...)
```

```python
td.C.min_item()         # → (key, Centroid)
td.C.min_item()[1].mean # → 最小质心的中心值
td.C.max_item()[1].mean # → 最大质心的中心值
```

---

## 5. Python Import 陷阱

### 5.1 问题

```python
# 错误：在模块加载时捕获初始值
from approach_Eq import time_unit  # time_unit = 1 (初始值)

# set_time_unit() 修改了全局变量，但上面的 time_unit 不会更新
```

### 5.2 正确做法

```python
# 方法 1：动态访问
import approach_Eq
approach_Eq.time_unit  # 每次访问时获取当前值

# 方法 2：传递参数
time_unit = get_time_unit()  # 通过函数获取
```

---

## 6. 输出格式

### 6.1 `get_sched_overhead_info()` 返回值

```python
{
    'sched_overhead_count': 270,     # 触发次数
    # 绝对时间 (µs)
    'time_mean_us': 30.59,
    'time_p50_us': 22.04,
    'time_p90_us': 41.70,
    'time_p99_us': 84.10,
    'time_max_us': 151.90,
    # Ratio (软件/硬件)
    'ratio_mean': 0.0444,   # 4.44%
    'ratio_p50': 0.0353,    # 3.53%
    'ratio_p90': 0.0667,    # 6.67%
    'ratio_p99': 0.1346,    # 13.46%
}
```

### 6.2 判断标准

| P99 Ratio | 结论 |
|-----------|------|
| < 50% | 软件开销被硬件切换吸收，**不在关键路径** |
| ≥ 50% | 软件开销可能显著延长关键路径，需关注 |

---

## 7. 实验结论

### 7.1 测量结果（100 hyperperiods）

| 指标 | 绝对时间 (µs) | Ratio (%) |
|------|---------------|-----------|
| mean | 30.6 | 4.44% |
| P50 | 22.0 | 3.53% |
| P90 | 41.7 | 6.67% |
| P99 | 84.1 | 13.46% |
| max | 151.9 | ~24% |

### 7.2 审稿回复要点

> **Algorithm 1**: 离线执行，不在运行时关键路径。
>
> **Algorithm 2**:
> - 平均决策延迟 ~30 µs，P99 ~84 µs
> - 相对于硬件切换时间 (625 µs)，ratio < 15%
> - 相对于 100 ms E2E deadline，ratio < 0.1%
> - **结论：不在关键路径上**

---

## 8. 相关文件

| 文件 | 内容 |
|------|------|
| `approach_def.py:Acc_p.sched()` | 计时逻辑插入点 |
| `approach_collector.py` | `record_sched_overhead()`, `get_sched_overhead_info()` |
| `doc/spec/stat/statistics_collection_spec.md` | StatisticsCollector 架构 |
