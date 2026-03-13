# StatisticsCollector 模块概览

本文档描述 `approach_collector.py` 中 `StatisticsCollector` 类的设计和功能。

## 1. 类结构图

```
StatisticsCollector
├── 初始化与配置
│   ├── __init__(delta, K, **kwargs)
│   ├── set_path(key, path)
│   ├── set_task_cnt(task_cnt)
│   └── set_motiv3_mode(mode)
│
├── 数据收集方法 (运行时调用)
│   ├── init_partition_stats(partition_id, cap, base_pwr)
│   ├── record_task_start(process_id, start_time)
│   ├── record_task_finish(G, process_id, finish_t)
│   ├── record_e2e_finish(G, sink_name, finish_t)
│   ├── record_realloc(partition_id, delta_ld, task_list)
│   ├── record_realloc_num(partition_id, task_list)
│   ├── record_compute_progress(task_name, delta_compute_t, delta_load)
│   ├── record_period_load_arrival(load)
│   ├── record_idle_capacity(idle_ld, sys_state)
│   ├── record_miss(timeout_iter)
│   └── forward_hyperperiod(T_hp)
│
├── 统计查询方法 (分析阶段)
│   ├── get_summary(num_bins, p_list) → Dict
│   ├── get_utilization_avg_ratio() → Dict
│   ├── get_realloc_info() → Dict
│   ├── get_latency_breakdown_avg_ratio() → Dict
│   ├── get_spearman_correlation(percentile) → float
│   │
│   ├── [Motiv-Exp专用]
│   │   ├── get_motiv_case1_stats() → Dict
│   │   ├── get_motiv_case2_stats() → Dict
│   │   └── get_motiv_case3_stats(percentile, mode) → Dict
│   │
│   └── [Motiv-Exp-3辅助]
│       ├── get_adaptive_load_latency_summary(p_list) → List[Dict]
│       └── get_raw_load_latency() → List[Tuple]
│
├── 绘图方法 (静态方法)
│   ├── plot_motiv_case1(data_points, save_path, show)
│   ├── plot_motiv_case2(data_points, plot_type, save_path, show, group_order)
│   ├── plot_motiv_case3(collector_base, collector_exp, percentile, ...)
│   ├── plot_load_latency_binned(binned_summary, spearman_rho, ...)
│   └── plot_load_latency_raw(data_groups, save_path, show)
│
└── 持久化方法
    ├── save_state(file_path)
    ├── load_state(file_path)
    └── export_summary(num_bins, p_list, verbose)
```

### 1.1 三级分布存储架构

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

### 1.2 运行时调用时机

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

## 2. 核心数据结构

### 2.1 T-Digest 流式直方图

StatisticsCollector 基于 `TDigestStreamingHistogram` 实现流式统计，支持：
- 长尾分布的高分位数估计
- O(1) 空间复杂度的在线更新
- 分布合并 (reduce 操作)

```python
# 关键分布存储
dist_per_task_exe: Dict[str, TDigest]      # 每任务执行时间分布
dist_per_task_ft: Dict[str, TDigest]       # 每任务完成时间分布
dist_per_task_realloc: Dict[str, TDigest]  # 每任务重调度开销分布

dist_overall_idle: TDigest                 # 整体空闲资源分布
dist_overall_miss: TDigest                 # 整体miss负载分布
dist_overall_used: TDigest                 # 整体已用资源分布
dist_overall_realloc: TDigest              # 整体重调度开销分布
```

### 2.2 关键路径追踪 (延迟分解)

```python
# 任务当前状态缓存 (per-instance)
task_curr_stat: Dict[str, Dict]  # {'realloc': float, 'compute': float, 'realloc_num': int}

# 前驱任务统计传播 (用于关键路径追踪)
task_pred_stat: Dict[str, Dict[str, Dict]]  # task -> pred -> stat

# 链级约束存储
chain_e2e_constraint: Dict[str, float]  # sink_name -> relative deadline
```

### 2.3 周期级缓存 (每个hyperperiod重置)

```python
part_realloc_curr: Dict[str, float]  # 分区级重调度开销累积
hp_idle_curr: float                   # 周期内空闲算力
hp_miss_curr: float                   # 周期内miss负载
hp_used_curr: float                   # 周期内已用算力
hp_total_load_curr: float             # 周期内总负载
hp_worst_e2e_curr: float              # 周期内最差E2E延迟
```

## 3. 统计指标定义

### 3.1 资源利用率指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| `idle_mean_ratio` | 闲置算力占比 | `mean(idle) / (total_pwr * T_hp)` |
| `miss_mean_ratio` | miss任务剩余负载占比 | `mean(miss) / (total_pwr * T_hp)` |
| `realloc_mean_ratio` | 重调度开销占比 | `mean(realloc_cost) / (total_pwr * T_hp)` |
| `used_mean_ratio` | 有效计算占比 | `mean(used) / (total_pwr * T_hp)` |
| `miss_mean_count` | 每周期miss任务数 (归一化) | `mean(miss_count) / task_cnt` |

### 3.2 延迟分解指标

提供两种口径：

**vs_e2e (相对于E2E延迟)**
```python
exec_ratio = exec_mean / e2e_mean
realloc_ratio = realloc_mean / e2e_mean
wait_ratio = max(0, 1 - exec_ratio - realloc_ratio)
```

**vs_constraint (相对于约束，推荐用于比较)**
```python
exec_ratio = exec_mean / constraint
realloc_ratio = realloc_mean / constraint
wait_ratio = max(0, e2e_mean/constraint - exec_ratio - realloc_ratio)
```

### 3.3 相关性指标 (Motiv-Exp-3)

| 指标 | 含义 | 使用场景 |
|------|------|----------|
| `spearman_rho` | Spearman秩相关系数 | 负载-延迟单调相关性 |
| `rmse` | 线性回归RMSE | 趋势线垂直离散度 |

## 4. Motiv-Exp 专用方法

### 4.1 Case 1: 利用率-可靠性权衡

**用途**: 纯静态调度(cyc)的资源利用分析

```python
def get_motiv_case1_stats(self) -> Dict:
    # 返回字段: idle_mean_ratio, miss_mean_ratio, miss_mean_count, realloc_mean_ratio
    # 字段定义见 Section 3.1
```

**绘图**: `plot_motiv_case1()` - 三柱状图+附轴折线
- 主轴(对数): idle/miss/realloc ratio
- 附轴(线性): miss rate

### 4.2 Case 2: 可扩展性分析

**用途**: 纯动态调度(glb)的延迟分解

```python
def get_motiv_case2_stats(self) -> Dict:
    return {
        'utilization': {
            'idle_mean_ratio': float,
            'miss_mean_ratio': float,
            'realloc_mean_ratio': float
        },
        'latency_breakdown': {
            'overall': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
            'first_chain': {...},
            'first_chain_name': str
        },
        'miss_mean_count': float
    }
```

**绘图**: `plot_motiv_case2(plot_type='breakdown'|'utilization')`
- breakdown: 堆叠柱状图 (exec/realloc/wait)
- utilization: 堆叠柱状图 (effective/idle/miss/realloc)

### 4.3 Case 3: 切换行为不确定性

**用途**: 负载-延迟关系分析

```python
def get_motiv_case3_stats(self, percentile=0.99, mode=None) -> Dict:
    return {
        'mode': 'raw' | 'binned',
        'spearman_rho': float,        # Spearman相关系数
        'percentile': float,          # 使用的分位数
        'binned_summary': [...] | None,  # binned模式
        'raw_data': [(load, lat), ...] | None  # raw模式
    }
```

**自适应分箱**:
- warmup阶段: 前100个周期学习负载分布
- 在线分箱: 使用TDigest分位数边界
- 延迟分布: 每箱独立TDigest

**绘图**:
- `plot_motiv_case3()`: 统一入口，自动选择raw/binned
- `plot_load_latency_raw()`: 散点图+趋势线
- `plot_load_latency_binned()`: 分位数曲线+IQR带

## 5. 与实验脚本的集成

### 5.1 调用流程

```
motiv_exp_runner.py
    │
    ├── _case1_worker(payload)
    │       ├── run_main_approach_inproc(run_args) → collector
    │       └── collector.get_motiv_case1_stats() → stats
    │
    ├── MotivExp1Runner._generate_report()
    │       └── StatisticsCollector.plot_motiv_case1(plot_data, save_path)
    │
    └── ...
```

### 5.2 exp_common.py 共享组件

```python
# 参数模板
class ParamTemplate:
    def with_updates(mapping, runtime, specific) -> ParamTemplate
    def to_run_args() -> Dict[str, Any]

# 主程序调用
def run_main_approach_inproc(args_dict, dry_run) -> StatisticsCollector

# 绘图辅助
def group_by_key(data_points, key) -> Dict
def compute_group_means(grouped_data, value_keys) -> Dict
```

### 5.3 典型用法

```python
# 1. 运行仿真获取collector
collector = run_main_approach_inproc(run_args)

# 2. 获取统计
stats = collector.get_motiv_case1_stats()

# 3. 生成图表
StatisticsCollector.plot_motiv_case1(
    data_points=[{
        'label': 'p50',
        'idle_mean_ratio': 0.1,
        'miss_mean_ratio': 0.05,
        'realloc_mean_ratio': 0.0,
        'miss_mean_count': 2.5
    }],
    save_path='output/case1.pdf'
)
```

## 6. 关键路径追踪算法

延迟分解基于关键路径分析 (Critical Path Analysis):

```
1. 任务完成时:
   a. 获取所有前驱的路径统计
   b. 选择 e2e_lat 最大的前驱作为关键路径
   c. 累加当前任务的 compute/realloc
   d. 传播给所有后继

2. Sink节点完成时:
   a. 获取关键路径统计
   b. 记录 exec/realloc 分布
   c. 更新周期最差E2E
```

## 7. 颜色方案 (统一)

| 元素 | 颜色代码 | 用途 |
|------|----------|------|
| Idle | C7 (灰) | 空闲算力 |
| Miss | C3 (红) | 未完成负载(警示) |
| Realloc | C1 (橙) | 重调度开销 |
| Effective | C0 (蓝) | 有效计算 |
| Execution | C0 (蓝) | 执行时间 |
| Waiting | C2 (绿) | 等待时间 |
| 折线 | C4 (紫) | Miss Rate |

## 8. 文件依赖

```
approach_collector.py
├── ref_tdigest.py          # TDigestStreamingHistogram
├── approach_Eq.py          # time_gt, elim_nume_error, cal_cost
├── approach_def.py         # MyGraph (TYPE_CHECKING)
└── numpy, matplotlib
```

## 9. 缓存机制

### 9.1 缓存文件

```
{output_dir}/
├── case1_summary.json
├── case2_summary.json
└── case3_summary.json
```

### 9.2 使用方式

```bash
# 正常运行（保存缓存）
python -m scripts.motiv_exp_runner --case 1 --output_dir ./results

# 从缓存重绘（跳过仿真）
python -m scripts.motiv_exp_runner --case 1 --use_plot_cache
```

### 9.3 缓存验证

```python
if len(results) != len(self.ratios):
    print(f"Error: Cache invalid. Expected {len(self.ratios)}, found {len(results)}")
    return
```

## 10. 序列化

### 10.1 save_state()

```python
stats_collector.save_state('./stats_collector.json')
```

### 10.2 load_state()

```python
stats_collector = StatisticsCollector.load_state('./stats_collector.json')
```

## 11. 相关文档

- [../spec/stat/collector.md](../spec/stat/collector.md) - E2E延迟分解规范
- [../spec/stat/tdigest_system_spec.md](../spec/stat/tdigest_system_spec.md) - T-Digest算法细节
- [../spec/stat/statistics_collection_spec.md](../spec/stat/statistics_collection_spec.md) - 业务集成规范
