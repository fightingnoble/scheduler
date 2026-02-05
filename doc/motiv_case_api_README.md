# Motivation实验统计API重构说明

## 概述

根据冻结的test_plan，为`StatisticsCollector`类重新设计了专门的统计接口，为三个Motivation实验提供定制化的数据获取、格式化和可视化功能。

## 设计理念

### 问题
之前的统计接口过于通用，用户需要：
1. 理解底层数据结构
2. 手动提取和组合指标
3. 自己实现格式化输出
4. 分别调用多个函数才能完成一个实验的统计

### 解决方案
为每个case提供**一站式接口**：
- **数据获取**: `get_motiv_caseN_stats()` - 返回结构化的字典，包含该case所需的所有指标
- **格式化输出**: `format_motiv_caseN_output()` - 生成可读的文本报告
- **可视化**: `plot_motiv_case3()` - Case 3专用绘图接口
- **统一导出**: `export_motiv_case_results(case=N)` - 一键导出统计和图表

## 新增接口一览

### Case-Specific Interfaces（核心）

| 实验 | 数据获取 | 格式化 | 可视化 | 统一导出 |
|------|---------|--------|--------|---------|
| **Motiv-Exp-1**<br>纯静态-利用率问题 | `get_motiv_case1_stats()` | `format_motiv_case1_output()` | N/A | ✓ |
| **Motiv-Exp-2**<br>纯动态-延迟开销问题 | `get_motiv_case2_stats()` | `format_motiv_case2_output()` | N/A | ✓ |
| **Motiv-Exp-3**<br>切换-不确定性问题 | `get_motiv_case3_stats()` | `format_motiv_case3_output()` | `plot_motiv_case3()` | ✓ |

### 统一接口

```python
# 统一导出（支持所有case）
collector.export_motiv_case_results(case=1|2|3, save_path, verbose, **kwargs)

# 快捷打印（支持所有case）
collector.print_motiv_case_summary(case=1|2|3, **kwargs)
```

## 接口设计细节

### Motiv-Exp-1: 纯静态调度

**目标**: 证明"利用率-可靠性权衡"

**输入**: 无（自动从collector提取）

**输出结构**:
```python
{
    'idle_mean_ratio': float,      # 闲置算力占比
    'miss_mean_ratio': float,      # miss负载占比
    'miss_mean_count': float,      # miss任务数量
    'realloc_mean_ratio': float    # 切换开销（应为0）
}
```

**使用场景**: 扫描不同预留分位数，观察idle和miss的权衡关系

---

### Motiv-Exp-2: 纯动态调度

**目标**: 证明"可扩展性瓶颈"

**输入**: 无（自动从collector提取）

**输出结构**:
```python
{
    'utilization': {
        'idle_mean_ratio': float,
        'miss_mean_ratio': float,
        'realloc_mean_ratio': float
    },
    'latency_breakdown': {
        'overall': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
        'first_chain': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
        'first_chain_name': str
    },
    'miss_mean_count': float
}
```

**关键特性**:
1. 分离了"统计1"（资源利用率）和"统计2"（延迟分解）
2. 提供overall和first_chain两个视角的延迟分解
3. 延迟分解相对于端到端约束归一化（`vs_constraint`）

**使用场景**: 扫描硬件/任务规模，观察各组件随规模的变化

---

### Motiv-Exp-3: 切换行为的不确定性

**目标**: 证明切换开销破坏负载-延迟相关性

**输入**: 
- `percentile` (默认0.99): 用于计算相关性的分位数
- `mode` (可选): 'raw'或'binned'，默认使用`self.motiv3_mode`

**输出结构**:
```python
{
    'mode': 'raw' | 'binned',
    'spearman_rho': float,           # Spearman相关系数
    'percentile': float,
    'binned_summary': [...] | None,  # binned模式：分箱摘要
    'raw_data_count': int | None     # raw模式：数据点数
}
```

**关键特性**:
1. 支持两种数据模式：
   - `raw`: 保留所有(load, worst_e2e)原始点，适合小数据量
   - `binned`: 自适应分箱，适合大数据量（>1000周期）
2. 自动选择Spearman计算方法（raw或binned）
3. 集成绘图功能，根据模式自动选择散点图或分位数曲线

**使用场景**: 对照实验（禁用/启用切换开销），比较相关系数差异

---

## 使用示例

### 最简单的用法
```python
# 运行仿真后
collector.print_motiv_case_summary(case=1)  # 打印到控制台
```

### 标准用法（导出文件）
```python
# Case 1
collector.export_motiv_case_results(case=1, save_path='./case1.txt')

# Case 2
collector.export_motiv_case_results(case=2, save_path='./case2.txt')

# Case 3（同时生成图表）
collector.export_motiv_case_results(
    case=3, 
    save_path='./case3.txt',
    percentile=0.99,
    fit='wls'
)
```

### 高级用法（自定义处理）
```python
# 获取原始数据
stats = collector.get_motiv_case2_stats()

# 提取特定指标
realloc_ratio = stats['utilization']['realloc_mean_ratio']
wait_ratio = stats['latency_breakdown']['overall']['wait_ratio']

# 自定义判断逻辑
if wait_ratio > 0.5:
    print("警告：等待时间占比超过50%！")

# 自定义格式化
custom_output = f"Realloc: {realloc_ratio:.2%}, Wait: {wait_ratio:.2%}"
```

## 与test_plan的对应关系

### Case 1对应关系
| test_plan描述 | API接口 |
|--------------|---------|
| 闲置算力占比 | `stats['idle_mean_ratio']` |
| miss任务剩余负载占比 | `stats['miss_mean_ratio']` |
| miss任务数量 | `stats['miss_mean_count']` |
| 切换开销占比 | `stats['realloc_mean_ratio']` |
| 获取方式 | `collector.get_motiv_case1_stats()` |

### Case 2对应关系

**统计1 - 资源利用率**:
| test_plan描述 | API接口 |
|--------------|---------|
| 闲置算力占比 | `stats['utilization']['idle_mean_ratio']` |
| miss任务剩余负载占比 | `stats['utilization']['miss_mean_ratio']` |
| 切换开销占比 | `stats['utilization']['realloc_mean_ratio']` |

**统计2 - 延迟分解**:
| test_plan描述 | API接口 |
|--------------|---------|
| 所有链合并的exec占比 | `stats['latency_breakdown']['overall']['exec_ratio']` |
| 所有链合并的realloc占比 | `stats['latency_breakdown']['overall']['realloc_ratio']` |
| 所有链合并的wait占比 | `stats['latency_breakdown']['overall']['wait_ratio']` |
| 第一条链的breakdown | `stats['latency_breakdown']['first_chain']` |
| miss任务数量 | `stats['miss_mean_count']` |
| 获取方式 | `collector.get_motiv_case2_stats()` |

### Case 3对应关系
| test_plan描述 | API接口 |
|--------------|---------|
| Spearman相关系数 | `stats['spearman_rho']` |
| 负载-延迟关系图 | `collector.plot_motiv_case3()` |
| 自适应分箱摘要 | `stats['binned_summary']` (binned模式) |
| 原始数据点数 | `stats['raw_data_count']` (raw模式) |
| 设置数据模式 | `collector.set_motiv3_mode('raw'|'binned')` |
| 获取方式 | `collector.get_motiv_case3_stats(percentile=0.99)` |

## 文件说明

本次重构涉及的文件：

1. **`approach_collector.py`** (核心修改)
   - 新增3个case专用数据获取接口
   - 新增3个case专用格式化接口
   - 新增统一导出接口
   - 新增快捷打印接口

2. **`doc/motiv_case_api_usage.md`** (详细文档)
   - 完整的API使用指南
   - 每个case的详细说明
   - 完整的使用示例
   - 批量扫描脚本示例

3. **`doc/motiv_case_api_quick_ref.md`** (快速参考)
   - API速查表
   - 关键指标说明
   - 快捷命令列表
   - 返回值结构速查

4. **`doc/test_plan.md`** (已冻结)
   - 实验设计文档
   - 统计方式说明
   - 期望结果描述

## 向后兼容性

本次重构**完全向后兼容**：
- 所有原有接口保持不变
- 新增接口为optional，不影响现有代码
- 原有的`get_summary()`, `get_utilization_avg_ratio()`, `get_latency_breakdown_avg_ratio()`等接口继续可用

## 迁移指南

### 从旧接口迁移到新接口

**旧方式（需要多步操作）**:
```python
# 旧：需要手动组合和提取
util = collector.get_utilization_avg_ratio()
breakdown = collector.get_latency_breakdown_avg_ratio()

idle = util['idle_mean_ratio']
miss = util['miss_mean_ratio']
exec_ratio = breakdown['overall_vs_constraint']['exec_ratio']

# 手动格式化
print(f"Idle: {idle:.4f}")
print(f"Miss: {miss:.4f}")
print(f"Exec: {exec_ratio:.4f}")
```

**新方式（一步到位）**:
```python
# 新：一键获取和输出
collector.print_motiv_case_summary(case=2)

# 或者获取数据后自定义处理
stats = collector.get_motiv_case2_stats()
# stats已经包含所有需要的数据，结构清晰
```

## 测试建议

建议测试以下场景：

1. **Case 1**: 扫描exec_t_comp_ratioA ∈ [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
   - 验证idle_ratio和miss_ratio的权衡关系
   - 验证realloc_ratio恒为0

2. **Case 2**: 扫描tiles ∈ [300, 500], chains ∈ [1, 4], load ∈ [0.5, 1.0]
   - 验证realloc_ratio随规模增长
   - 验证wait_ratio随规模增长
   - 验证延迟分解总和>1的情况

3. **Case 3**: 对照实验（禁用/启用切换开销）
   - 验证基线组ρ > 0.85
   - 验证实验组ρ < 0.6
   - 验证Δρ > 0.25
   - 验证raw和binned两种模式

## 下一步

建议的后续工作：

1. **集成到主程序**: 在`main_approach.py`中根据policy自动选择对应case
2. **批量扫描脚本**: 创建shell脚本自动化参数扫描
3. **结果可视化**: 基于导出的统计数据绘制论文图表
4. **CI/CD集成**: 将统计结果作为回归测试的一部分

## 联系与反馈

如有问题或建议，请参考：
- 详细文档: `doc/motiv_case_api_usage.md`
- 快速参考: `doc/motiv_case_api_quick_ref.md`
- 实验计划: `doc/test_plan.md`

