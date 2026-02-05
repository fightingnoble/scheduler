## End-to-End Latency Decomposition using Critical Path Analysis

To accurately attribute end-to-end latency in complex Directed Acyclic Graphs (DAGs) with parallel execution paths, we employ a critical path tracking algorithm. This method ensures that latency components (e.g., `compute`, `realloc`) are not double-counted at merge points, providing a decomposition where the sum of components equals the wall-clock critical path latency.

### Data Structures

For each task instance `i`, we maintain two primary state objects:
1.  **`task_curr_stat[i]`**: A dictionary storing the execution time (`compute`), reallocation overhead (`realloc`), and reallocation count (`realloc_num`) intrinsic to task `i` itself.
2.  **`task_pred_stat[i]`**: A nested dictionary mapping each direct predecessor `p` of task `i` to `p`'s final critical path statistics (`path_stat`).

The `path_stat` propagated for any completed task `j` is a dictionary:
```
{
  'compute': float,     // Critical path compute time from source to j
  'realloc': float,     // Critical path realloc overhead from source to j
  'realloc_num': int,   // Sum of reallocations on the critical path
  'e2e_lat': float      // Finish time of j relative to chain's start (offset)
}
```

### Propagation Logic

The algorithm proceeds as follows upon the completion of a task `i` at time `finish_t`:

1.  **Select Critical Path Predecessor**:
    - Retrieve the set of predecessor statistics `task_pred_stat[i]`.
    - Identify the critical predecessor `p*` by selecting the one with the maximum `e2e_lat` (latest finish time relative to offset).
    - If `i` is a source node (no predecessors), a zero-valued conceptual predecessor statistic is used, with its `e2e_lat` initialized to `start_time - offset`.

2.  **Calculate Path Statistics for `i`**:
    - The critical path statistics for `i`, denoted `path_stat[i]`, are computed by adding `i`'s intrinsic statistics (`task_curr_stat[i]`) to the statistics of its critical predecessor `p*`.
    
    \[
    \text{path\_stat}[i].\text{compute} = \text{path\_stat}[p^*].\text{compute} + \text{task\_curr\_stat}[i].\text{compute}
    \]
    \[
    \text{path\_stat}[i].\text{realloc} = \text{path\_stat}[p^*].\text{realloc} + \text{task\_curr\_stat}[i].\text{realloc}
    \]
    
    - The `e2e_lat` for `i` is its relative finish time:
    
    \[
    \text{path\_stat}[i].\text{e2e\_lat} = \text{finish\_t} - \text{offset}
    \]

3.  **Propagate to Successors**:
    - For each successor `s` of `i`, the calculated `path_stat[i]` is stored in `task_pred_stat[s][i]`. When successor `s` eventually completes, it will use this stored information to determine its own critical path among all its predecessors.

This recursive process ensures that at any merge point, only the longest path's accumulated latency is carried forward, correctly modeling the end-to-end critical path latency and its components.

## 添加新的 Collector 统计项目

向 `StatisticsCollector` 类添加新的统计项目（如分布、计数器等）需要遵循以下步骤以确保完整性和一致性：

### 步骤 1：在 `__init__` 中声明数据结构

在 `StatisticsCollector.__init__` 方法中初始化新的统计数据结构。常见类型：
- **TDigestStreamingHistogram**：用于收集分布统计（支持分位数查询）
- **defaultdict**：用于 per-task/per-partition 级别的分组统计
- **float/int**：用于单一累加器

**示例**（添加 per-partition realloc 次数分布）：
```python
self.dist_per_part_realloc_count = defaultdict(
    lambda: TDigestStreamingHistogram(delta=delta, K=K)
)
```

### 步骤 2：在数据收集点记录数据

在相应的记录方法中添加数据写入逻辑：
- **任务级别**：在 `record_task_finish` 或 `record_e2e_finish` 中
- **周期级别**：在 `forward_hyperperiod` 中汇总
- **实时记录**：在 `record_realloc`、`record_compute_progress` 等方法中

**示例**（在 hyperperiod 结束时收集）：
```python
def forward_hyperperiod(self, T_hp: float=1.0):
    for part_id in list(self.part_realloc_curr.keys()):
        realloc_num = self.part_realloc_num.pop(part_id)
        self.dist_per_part_realloc_count[part_id].add(realloc_num)
```

### 步骤 3：在 `get_summary` 中添加输出键

在 `get_summary` 方法中：
1. 在 `summary` 字典初始化时添加新键
2. 遍历数据结构并调用 `.get_summary(num_bins, p_list)` 生成统计摘要

**示例**：
```python
def get_summary(self, num_bins: int = 20, p_list: List[float] = [0.5, 0.9, 0.99, 0.999]) -> Dict:
    summary = {
        # ... 其他键 ...
        'per_part_realloc_count': {},  # 步骤 3.1: 添加键
    }
    
    # 步骤 3.2: 填充数据
    for part_id, dist in self.dist_per_part_realloc_count.items():
        summary['per_part_realloc_count'][part_id] = dist.get_summary(num_bins, p_list)
    
    return summary
```

### 步骤 4：扩展 `save_state` 序列化

在 `save_state` 方法的 `state` 字典中添加新字段的序列化：
- **TDigestStreamingHistogram**：调用 `.to_dict()`
- **defaultdict of TDigest**：字典推导式 `{k: v.to_dict() for k, v in ...}`
- **基本类型**：直接存储

**示例**：
```python
state = {
    # ... 其他字段 ...
    'dist_per_part_realloc_count': {k: v.to_dict() for k, v in self.dist_per_part_realloc_count.items()},
}
```

### 步骤 5：扩展 `load_state` 反序列化

在 `load_state` 类方法中添加对应的加载逻辑：
- 使用 `TDigestStreamingHistogram.from_dict(v)` 恢复 TDigest 对象
- 对于 defaultdict，需要同时提供 lambda factory

**示例**：
```python
collector.dist_per_part_realloc_count = defaultdict(
    lambda: TDigestStreamingHistogram(delta=state['delta'], K=state['K']),
    {k: TDigestStreamingHistogram.from_dict(v) 
     for k, v in state.get('dist_per_part_realloc_count', {}).items()}
)
```

### 步骤 6：更新 `load_save_check` 验证逻辑

在 `load_save_check` 方法中添加新字段的验证，确保序列化/反序列化的正确性：
- **单一分布**：添加到 `overall_dists` 列表
- **分组分布**：添加到相应的 `dist_type` 循环列表

**示例**：
```python
for dist_type in ['dist_per_task_ft', 'dist_per_task_realloc', 
                  'dist_per_part_realloc', 'dist_per_part_realloc_count']:  # 新增
    # 验证逻辑 ...
```

### 检查清单

完成上述步骤后，验证以下内容：
- [ ] 数据结构在 `__init__` 中正确初始化
- [ ] 数据在适当的生命周期点被收集
- [ ] `get_summary()` 可以正确输出统计信息
- [ ] `save_state()` 和 `load_state()` 配对实现
- [ ] `load_save_check()` 测试通过
- [ ] 无 linter 错误

### 常见陷阱

1. **忘记在 load_state 中提供 defaultdict factory**：会导致新键插入时崩溃
2. **save/load 不对称**：save 了但 load 没加载，或键名不一致
3. **忘记更新 load_save_check**：导致测试覆盖不全
4. **数据类型不匹配**：如整数被存为浮点数，需要显式转换
