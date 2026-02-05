# T-Digest 流式统计系统说明

## 系统概述

本系统使用 T-Digest 算法实现了高效的流式统计收集，支持长尾分布和高分位数估计。系统集成在调度仿真框架中，用于在线计算延迟和重分配开销的分布。

## 核心组件

### 1. TDigestStreamingHistogram (ref_tdigest.py)
- **功能**: 基于 T-Digest 的流式直方图
- **关键方法**:
  - `add(value)`: 添加数据点 (O(log K))
  - `get_histogram_data(num_bins)`: 生成直方图 (O(num_bins))
  - `percentile(p)`: 获取分位数
- **参数**:
  - delta: 精度控制 (默认 0.01)
  - K: 合并参数 (默认 25)

### 2. StatisticsCollector (approach_collector.py)
- **功能**: 管理所有统计指标的收集
- **分布存储**:
  - dist_per_task_ft: 每个任务的完成时间分布
  - dist_per_task_realloc: 每个任务的重分配开销分布
  - dist_per_part_realloc: 每个分区的重分配开销分布
- **接口**:
  - record_task_finish: 记录任务完成时间
  - record_e2e_finish: 记录端到端完成
  - record_realloc: 记录重分配开销
  - forward_hyperperiod: 周期推进处理
  - get_summary: 获取统计摘要 (包括直方图和分位数)
  - print_summary: 打印摘要

## 集成方式

### 1. 在处理器中调用
- **任务完成 (update_run)**:
  ```python
  if rem_load <= 0:
      if self.stats_collector:
          self.stats_collector.record_task_finish(self.G_ptr, node, curr_t)
  ```
- **端到端完成 (update_ready for sink)**:
  ```python
  if node in self.G_ptr.sinks:
      if self.stats_collector:
          self.stats_collector.record_e2e_finish(self.G_ptr, node, curr_t)
      self.G_ptr.mark_finish(node)
  ```
- **重分配更新 (update_run for "R")**:
  ```python
  if self.sys_state == "R":
      # ... 更新 rem_load ...
      if self.stats_collector:
          task_list = list(self.ready.keys())
          self.stats_collector.record_realloc(self.id, delta_load, task_list)
  ```

### 2. 在仿真循环中调用
- **周期推进 (run_simulation)**:
  ```python
  if time_gtq(curr_t, next_hp_boundary) and time_lt(pred_t, next_hp_boundary):
      curr_hp += 1
      # ... duplicate graph ...
      if processors:
          stats_collector = processors[0].stats_collector
          if stats_collector:
              stats_collector.forward_hyperperiod()
  ```
- **仿真结束**:
  ```python
  stats_collector = processors[0].stats_collector
  if stats_collector:
      stats_collector.print_summary()
  ```

## 使用示例

```python
# 获取摘要
summary = stats_collector.get_summary(num_bins=20)

# 访问 per-task 分布
task_dist = summary['per_task_finish_time']['task1']
print(task_dist['percentiles']['p99'])

# 访问整体 e2e 分布
overall = summary['overall_e2e_latency']
print(overall['histogram'])
```

## 性能分析

- **时间效率**: 添加数据点 O(log K)，查询 O(num_bins)
- **内存效率**: O(K) 质心，固定大小
- **精度**: 对长尾分布友好，高分位数准确
- **流式处理**: 支持无限数据流，无需存储原始数据

## 注意事项

- 分位数估计在数据少时可能不准，建议样本数 > 1000
- 直方图是近似，适合可视化
- 如果需要精确总和，使用 scalar 累加 (如 system_realloc_cost)
- 多线程环境下需加锁保护 add 操作

## 扩展指南

- **添加新分布**: 在 __init__ 添加新 defaultdict(TDigestStreamingHistogram)
- **自定义摘要**: 修改 get_summary 中的计算逻辑
- **可视化**: 使用 matplotlib 绘制 histogram 