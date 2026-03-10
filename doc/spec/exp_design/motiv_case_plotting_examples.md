# Motivation实验绘图函数使用示例

## 概述

为三个Motivation实验提供了专门的绘图函数，用于可视化实验结果。

## Case 1: 利用率-可靠性权衡图（三柱 + 附轴）

### 函数签名
```python
collector.plot_motiv_case1(data_points=None, save_path=None, show=False)
```

### 功能
绘制 **三柱状图 + 附轴**：
- 主轴（对数轴，百分比）：`idle_mean_ratio`（橙）、`miss_mean_ratio`（绿）、`realloc_mean_ratio`（红）
- 附轴（线性轴，数量）：`miss_mean_count`（蓝线）
适用于对比不同预留分位数下的利用率-可靠性权衡。

### 完整示例

```python
#!/usr/bin/env python3
"""Case 1: 扫描不同预留分位数，绘制权衡曲线"""

# 扫描不同的exec_t_comp_ratioA
exec_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
results = []

for ratio in exec_ratios:
    # 运行仿真
    collector = run_cyc_simulation(exec_t_comp_ratioA=ratio)
    
    # 获取统计
    stats = collector.get_motiv_case1_stats()
    
    # 保存数据点
    results.append({
        'idle_mean_ratio': stats['idle_mean_ratio'],
        'miss_mean_ratio': stats['miss_mean_ratio'],
        'label': f'p{int(ratio*100)}'  # 标签：p50, p60, ...
    })
    
    print(f"p{int(ratio*100)}: idle={stats['idle_mean_ratio']:.4f}, "
          f"miss={stats['miss_mean_ratio']:.4f}")

# 绘制权衡曲线
collector.plot_motiv_case1(
    data_points=results,
    save_path='./results/case1_tradeoff_curve.pdf',
    show=False
)

print("权衡曲线已保存！")
```

### 图表特点
- **三柱并排**：每个预留分位数对应三个柱子（idle/miss/realloc）
- **主轴为对数轴**：以百分比显示，便于同时观察小/大占比
- **附轴为线性轴**：显示`miss_mean_count`，直观观察失败任务数量
- **数值标注**：
  - 柱顶显示百分比（<1% 使用科学计数法），例如`0.37%`或`1.2e-3`
  - 线图节点显示`miss_mean_count`的数值
- **参考线**：主轴添加10%、1%阈值参考
- **期望结果**：
  - Idle随预留分位数增大而上升
  - Miss随预留分位数增大而下降
  - Realloc在纯静态下应接近0

---

## Case 2: 可扩展性分析图

### 函数签名
```python
collector.plot_motiv_case2(data_points=None, plot_type='breakdown', save_path=None, show=False)
```

### 功能
支持两种堆叠柱状图：
1. **`plot_type='breakdown'`**: 延迟分解（exec/realloc/wait占比相对于约束）
2. **`plot_type='utilization'`**: 资源利用率（effective/realloc/idle/miss）

### 示例1: 延迟分解堆叠柱状图

```python
#!/usr/bin/env python3
"""Case 2: 扫描规模，绘制延迟分解图"""

# 扫描不同硬件和任务规模
tile_counts = [300, 500]
chain_counts = [1, 4]
load_factors = [0.5, 1.0]

results = []

for tiles in tile_counts:
    for chains in chain_counts:
        for load_factor in load_factors:
            # 运行仿真
            collector = run_glb_simulation(
                num_tiles=tiles,
                num_chains=chains,
                load_factor=load_factor
            )
            
            # 获取统计
            stats = collector.get_motiv_case2_stats()
            
            # 保存数据点（包含完整的stats字典）
            results.append({
                'label': f'{tiles}T-{chains}C-{load_factor}×',
                **stats  # 展开stats（包含utilization和latency_breakdown）
            })
            
            # 打印关键指标
            overall = stats['latency_breakdown']['overall']
            print(f"{tiles}T-{chains}C-{load_factor}×: "
                  f"exec={overall['exec_ratio']:.3f}, "
                  f"realloc={overall['realloc_ratio']:.3f}, "
                  f"wait={overall['wait_ratio']:.3f}, "
                  f"sum={sum(overall.values()):.3f}")

# 绘制延迟分解堆叠柱状图
collector.plot_motiv_case2(
    data_points=results,
    plot_type='breakdown',
    save_path='./results/case2_latency_breakdown.pdf'
)

print("延迟分解图已保存！")
```

### 示例2: 资源利用率堆叠柱状图

```python
# 使用相同的results数据
collector.plot_motiv_case2(
    data_points=results,
    plot_type='utilization',
    save_path='./results/case2_resource_utilization.pdf'
)

print("资源利用率图已保存！")
```

### 图表特点

**延迟分解图 (breakdown)**:
- **堆叠顺序**（从下到上）：
  1. 蓝色：Execution（执行时间）
  2. 橙色：Scheduling/Realloc（调度开销）
  3. 绿色：Waiting（等待时间）
- **参考线**：红色虚线标记100%约束
- **期望结果**：随规模增加，realloc和wait占比增大，总和可能超过100%

**资源利用率图 (utilization)**:
- **堆叠顺序**（从下到上）：
  1. 蓝色：Effective Utilization（有效利用率）
  2. 橙色：Realloc Overhead（切换开销）
  3. 黄色：Idle（闲置）
  4. 红色：Missed（miss负载）
- **总和**：恰好100%
- **期望结果**：随规模增加，realloc和miss占比增大

---

## Case 3: 负载-延迟关系图

### 函数签名
```python
collector.plot_motiv_case3(percentile=0.99, iqr_band=(0.25, 0.75), 
                          fit='none', save_path=None, show=False)
```

### 功能
根据数据模式自动选择：
- **raw模式**: 散点图 + 可选拟合曲线
- **binned模式**: pXX分位数曲线 + IQR带 + 可选拟合

### 示例1: 对照实验（基线组 vs 实验组）

```python
#!/usr/bin/env python3
"""Case 3: 对照实验，比较有无切换开销的相关性"""

# ====== 基线组：禁用切换开销 ======
print("运行基线组（无切换开销）...")
collector_baseline = run_glb_simulation(
    realloc_overhead_enabled=False,
    num_periods=500  # 较少周期，使用raw模式
)

# 设置raw模式
collector_baseline.set_motiv3_mode('raw')

# 获取统计
stats_base = collector_baseline.get_motiv_case3_stats(percentile=0.99)
rho_base = stats_base['spearman_rho']

# 绘图
collector_baseline.plot_motiv_case3(
    percentile=0.99,
    fit='wls',  # 加权线性拟合
    save_path='./results/case3_baseline_raw.pdf'
)

print(f"基线组 Spearman ρ: {rho_base:.4f}")


# ====== 实验组：启用切换开销 ======
print("运行实验组（有切换开销）...")
collector_exp = run_glb_simulation(
    realloc_overhead_enabled=True,
    num_periods=10000  # 大量周期，使用binned模式
)

# 设置binned模式
collector_exp.set_motiv3_mode('binned')

# 获取统计
stats_exp = collector_exp.get_motiv_case3_stats(percentile=0.99)
rho_exp = stats_exp['spearman_rho']

# 绘图
collector_exp.plot_motiv_case3(
    percentile=0.99,
    iqr_band=(0.25, 0.75),  # 添加IQR带
    fit='lowess',  # LOWESS平滑拟合
    save_path='./results/case3_experiment_binned.pdf'
)

print(f"实验组 Spearman ρ: {rho_exp:.4f}")


# ====== 对比分析 ======
delta_rho = rho_base - rho_exp
print("\n" + "="*60)
print("对照实验结果：")
print(f"  基线组（无开销）Spearman ρ: {rho_base:.4f}")
print(f"  实验组（有开销）Spearman ρ: {rho_exp:.4f}")
print(f"  差异 Δρ: {delta_rho:.4f}")
print("="*60)

# 判断假设是否成立
if rho_base > 0.85 and rho_exp < 0.6 and delta_rho > 0.25:
    print("✓ 假设验证成功：切换开销显著破坏了负载-延迟相关性")
else:
    print("✗ 假设验证失败")
    if rho_base <= 0.85:
        print(f"  - 基线组相关性不够强 ({rho_base:.4f} <= 0.85)")
    if rho_exp >= 0.6:
        print(f"  - 实验组相关性未明显减弱 ({rho_exp:.4f} >= 0.6)")
    if delta_rho <= 0.25:
        print(f"  - 相关性差异不够大 ({delta_rho:.4f} <= 0.25)")
```

### 示例2: 单独使用（当前collector的数据）

```python
# 运行仿真
collector = run_glb_simulation(num_periods=5000)
collector.set_motiv3_mode('binned')

# 直接绘图（使用当前collector的数据）
collector.plot_motiv_case3(
    percentile=0.99,
    iqr_band=(0.25, 0.75),
    fit='wls',
    save_path='./load_latency.pdf',
    show=True  # 显示图形
)
```

### 图表特点

**Raw模式（散点图）**:
- 每个点代表一个周期的(total_load, worst_e2e)
- 可选拟合曲线：OLS线性拟合
- 适合小数据量（<1000个周期）

**Binned模式（分位数曲线）**:
- pXX曲线：每个负载bin的p99延迟
- IQR带：四分位距，显示延迟分布宽度
- 点大小：正比于该bin的样本数
- 可选拟合曲线：WLS加权拟合
- 适合大数据量（≥1000个周期）

### 参数说明

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `percentile` | float | 使用的分位数（binned模式） | 0.99 |
| `iqr_band` | tuple | IQR带范围 (p_low, p_high) | (0.25, 0.75) |
| `fit` | str | 拟合方式：'none', 'wls', 'lowess' | 'none' |
| `save_path` | str | 保存路径 | `self.path['motiv3']` |
| `show` | bool | 是否显示图形 | False |

---

## 批量绘图脚本模板

### 完整的Case 1 + Case 2 + Case 3 绘图流程

```python
#!/usr/bin/env python3
"""完整的Motivation实验绘图流程"""

import os
from pathlib import Path

# 创建输出目录
output_dir = Path('./motiv_exp_results')
output_dir.mkdir(exist_ok=True)

# ========== Case 1: 纯静态调度 ==========
print("Case 1: 扫描预留分位数...")
case1_results = []
for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    collector = run_cyc_simulation(exec_t_comp_ratioA=ratio)
    stats = collector.get_motiv_case1_stats()
    case1_results.append({
        'idle_mean_ratio': stats['idle_mean_ratio'],
        'miss_mean_ratio': stats['miss_mean_ratio'],
        'label': f'p{int(ratio*100)}'
    })

collector.plot_motiv_case1(
    data_points=case1_results,
    save_path=output_dir / 'case1_tradeoff.pdf'
)
print("✓ Case 1 权衡曲线已保存")


# ========== Case 2: 纯动态调度 ==========
print("\nCase 2: 扫描硬件和任务规模...")
case2_results = []
for tiles in [300, 500]:
    for chains in [1, 4]:
        for load_factor in [0.5, 1.0]:
            collector = run_glb_simulation(
                num_tiles=tiles,
                num_chains=chains,
                load_factor=load_factor
            )
            stats = collector.get_motiv_case2_stats()
            case2_results.append({
                'label': f'{tiles}T-{chains}C-{load_factor}×',
                **stats
            })

# 绘制延迟分解图
collector.plot_motiv_case2(
    data_points=case2_results,
    plot_type='breakdown',
    save_path=output_dir / 'case2_breakdown.pdf'
)
print("✓ Case 2 延迟分解图已保存")

# 绘制资源利用率图
collector.plot_motiv_case2(
    data_points=case2_results,
    plot_type='utilization',
    save_path=output_dir / 'case2_utilization.pdf'
)
print("✓ Case 2 资源利用率图已保存")


# ========== Case 3: 切换行为不确定性 ==========
print("\nCase 3: 对照实验...")

# 基线组
collector_base = run_glb_simulation(realloc_overhead_enabled=False, num_periods=500)
collector_base.set_motiv3_mode('raw')
collector_base.plot_motiv_case3(
    fit='wls',
    save_path=output_dir / 'case3_baseline.pdf'
)
rho_base = collector_base.get_motiv_case3_stats()['spearman_rho']
print(f"✓ Case 3 基线组已保存 (ρ={rho_base:.4f})")

# 实验组
collector_exp = run_glb_simulation(realloc_overhead_enabled=True, num_periods=10000)
collector_exp.set_motiv3_mode('binned')
collector_exp.plot_motiv_case3(
    percentile=0.99,
    iqr_band=(0.25, 0.75),
    fit='lowess',
    save_path=output_dir / 'case3_experiment.pdf'
)
rho_exp = collector_exp.get_motiv_case3_stats()['spearman_rho']
print(f"✓ Case 3 实验组已保存 (ρ={rho_exp:.4f})")

print(f"\n所有图表已保存到: {output_dir}")
print(f"Δρ = {rho_base - rho_exp:.4f}")
```

---

## 注意事项

1. **Case 1 和 Case 2 需要多个数据点**:
   - 必须提供`data_points`参数（包含多个配置的统计结果）
   - 单个collector调用这些函数会只显示一个点

2. **Case 3 自动使用当前collector的数据**:
   - 不需要提供`data_points`
   - 直接使用collector内部收集的负载-延迟数据

3. **数据点格式要求**:
   - Case 1: `{'idle_mean_ratio', 'miss_mean_ratio', 'label'(可选)}`
   - Case 2: `{'label', 'utilization', 'latency_breakdown'}` - 使用`**stats`展开最方便
   - Case 3: 无需data_points参数

4. **保存路径**:
   - 所有函数都支持自定义`save_path`
   - 如果路径中的目录不存在，会自动创建

5. **图形显示**:
   - 默认`show=False`（不显示，只保存）
   - 设置`show=True`可在保存后弹窗显示图形

