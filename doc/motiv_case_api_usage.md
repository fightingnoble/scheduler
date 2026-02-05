# Motivation实验统计API使用指南

## 概述

`StatisticsCollector` 为三个Motivation实验提供了专门的统计接口，简化了数据获取和结果展示流程。

## 快速开始

```python
from approach_collector import StatisticsCollector

# 假设已经运行仿真并获得 stats_collector 对象
collector = stats_collector

# 方式1: 快捷打印到控制台
collector.print_motiv_case_summary(case=1)  # Motiv-Exp-1
collector.print_motiv_case_summary(case=2)  # Motiv-Exp-2
collector.print_motiv_case_summary(case=3, percentile=0.99)  # Motiv-Exp-3

# 方式2: 导出到文件并打印
collector.export_motiv_case_results(case=1, save_path='./results/case1.txt')
collector.export_motiv_case_results(case=2, save_path='./results/case2.txt')
collector.export_motiv_case_results(case=3, save_path='./results/case3.txt', 
                                    percentile=0.99, fit='wls')
```

## Motiv-Exp-1: 纯静态调度 - 利用率问题

### 实验目的
证明静态方法存在"利用率-可靠性权衡"：保守预留→低miss但高idle；激进预留→高利用率但高miss。

### API接口

#### 1. 获取统计数据
```python
stats = collector.get_motiv_case1_stats()
# 返回:
# {
#     'idle_mean_ratio': float,      # 闲置算力占比
#     'miss_mean_ratio': float,      # miss任务剩余负载占比
#     'miss_mean_count': float,      # miss任务数量
#     'realloc_mean_ratio': float    # 切换开销占比（静态应为0）
# }
```

#### 2. 格式化输出
```python
output_text = collector.format_motiv_case1_output(stats)
print(output_text)
```

#### 3. 完整导出
```python
# 打印到控制台 + 保存到文件
stats = collector.export_motiv_case_results(
    case=1, 
    save_path='./case1_results.txt',
    verbose=True
)
```

### 示例输出
```
============================================================
Motiv-Exp-1: 纯静态调度 - 利用率问题
============================================================
闲置算力占比 (idle_mean_ratio):       0.2345
Miss任务剩余负载占比 (miss_mean_ratio): 0.0123
Miss任务数量 (miss_mean_count):        1.50
切换开销占比 (realloc_mean_ratio):     0.0000 (应为0)
------------------------------------------------------------
有效利用率:                            0.7532
============================================================
```

### 扫描实验示例
```python
# 扫描不同的预留分位数
exec_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
results = []

for ratio in exec_ratios:
    # 运行仿真（假设有 run_cyc_simulation 函数）
    collector = run_cyc_simulation(exec_t_comp_ratioA=ratio)
    stats = collector.get_motiv_case1_stats()
    results.append({
        'ratio': ratio,
        'idle': stats['idle_mean_ratio'],
        'miss': stats['miss_mean_ratio'],
        'miss_count': stats['miss_mean_count']
    })

# 绘制权衡曲线
import matplotlib.pyplot as plt
plt.plot([r['idle'] for r in results], [r['miss'] for r in results], 'o-')
plt.xlabel('Idle Ratio')
plt.ylabel('Miss Ratio')
plt.title('Utilization-Reliability Tradeoff')
plt.savefig('case1_tradeoff.pdf')
```

---

## Motiv-Exp-2: 纯动态调度 - 延迟开销问题

### 实验目的
说明动态方法存在"可扩展性瓶颈"：随着规模增长，切换开销和排队延迟激增。

### API接口

#### 1. 获取统计数据
```python
stats = collector.get_motiv_case2_stats()
# 返回:
# {
#     'utilization': {
#         'idle_mean_ratio': float,
#         'miss_mean_ratio': float,
#         'realloc_mean_ratio': float
#     },
#     'latency_breakdown': {
#         'overall': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
#         'first_chain': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
#         'first_chain_name': str
#     },
#     'miss_mean_count': float
# }
```

#### 2. 访问具体指标
```python
stats = collector.get_motiv_case2_stats()

# 资源利用率
idle_ratio = stats['utilization']['idle_mean_ratio']
miss_ratio = stats['utilization']['miss_mean_ratio']
realloc_ratio = stats['utilization']['realloc_mean_ratio']
effective_util = 1.0 - idle_ratio - miss_ratio - realloc_ratio

# 延迟分解（Overall）
overall = stats['latency_breakdown']['overall']
exec_ratio = overall['exec_ratio']
realloc_ratio_lat = overall['realloc_ratio']
wait_ratio = overall['wait_ratio']

# 延迟分解（第一条链）
first_chain = stats['latency_breakdown']['first_chain']
chain_name = stats['latency_breakdown']['first_chain_name']
```

#### 3. 格式化输出
```python
output_text = collector.format_motiv_case2_output(stats)
print(output_text)
```

### 示例输出
```
============================================================
Motiv-Exp-2: 纯动态调度 - 延迟开销问题
============================================================
统计1 - 资源利用率分解:
  闲置算力占比 (idle):     0.0523
  Miss负载占比 (miss):     0.1234
  切换开销占比 (realloc):  0.2156
  有效利用率:              0.6087

统计2 - 端到端延迟分解 (相对于约束):
  Overall (所有链合并):
    执行时间占比 (exec):    0.4523
    调度开销占比 (realloc): 0.3156
    等待时间占比 (wait):    0.3521
    总和:                   1.1200 (>1表示超时)

  First Chain (chain_sink):
    执行时间占比 (exec):    0.4800
    调度开销占比 (realloc): 0.3000
    等待时间占比 (wait):    0.3400
    总和:                   1.1200 (>1表示超时)

Miss任务数量 (miss_mean_count): 5.60
============================================================
```

### 扫描实验示例
```python
# 扫描不同规模
tile_counts = [300, 500]
chain_counts = [1, 4]
load_factors = [0.5, 1.0]

results = []
for tiles in tile_counts:
    for chains in chain_counts:
        for load_factor in load_factors:
            collector = run_glb_simulation(
                num_tiles=tiles, 
                num_chains=chains, 
                load_factor=load_factor
            )
            stats = collector.get_motiv_case2_stats()
            
            # 提取关键指标
            util = stats['utilization']
            overall = stats['latency_breakdown']['overall']
            
            results.append({
                'tiles': tiles,
                'chains': chains,
                'load_factor': load_factor,
                'realloc_ratio': util['realloc_mean_ratio'],
                'exec_ratio': overall['exec_ratio'],
                'wait_ratio': overall['wait_ratio'],
                'miss_count': stats['miss_mean_count']
            })

# 保存结果
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('case2_scalability.csv', index=False)
```

---

## Motiv-Exp-3: 切换行为的不确定性

### 实验目的
证明切换开销破坏负载-延迟的单调相关性，worst case不再简单对应peak load。

### API接口

#### 1. 设置数据收集模式
```python
# 对照组（基线）：禁用切换开销，使用raw模式
collector_baseline.set_motiv3_mode('raw')

# 实验组：启用切换开销，使用binned模式（大数据量）
collector_experiment.set_motiv3_mode('binned')
```

#### 2. 获取统计数据
```python
stats = collector.get_motiv_case3_stats(percentile=0.99)
# 返回:
# {
#     'mode': 'raw' | 'binned',
#     'spearman_rho': float,          # Spearman相关系数
#     'percentile': float,            # 使用的分位数
#     'binned_summary': [...] | None, # binned模式的摘要
#     'raw_data_count': int | None    # raw模式的数据点数
# }
```

#### 3. 绘制图表
```python
# 方式1: 使用统一接口（自动根据模式选择）
collector.plot_motiv_case3(
    percentile=0.99, 
    iqr_band=(0.25, 0.75),
    fit='wls',  # 'none', 'wls', 'lowess'
    save_path='./case3_plot.pdf',
    show=False
)

# 方式2: 直接调用具体方法
# Raw模式: 散点图
collector.plot_load_latency_raw(fit='wls', save_path='./raw_plot.pdf')

# Binned模式: 分位数曲线
collector.plot_load_latency_binned(
    percentile=0.99, 
    iqr_band=(0.25, 0.75),
    fit='lowess',
    save_path='./binned_plot.pdf'
)
```

#### 4. 完整导出
```python
# 导出统计 + 绘图
stats = collector.export_motiv_case_results(
    case=3,
    save_path='./case3_results.txt',
    verbose=True,
    percentile=0.99,
    fit='wls',
    iqr_band=(0.25, 0.75),
    show=False
)
```

### 示例输出
```
============================================================
Motiv-Exp-3: 切换行为的不确定性
============================================================
数据模式: binned
Spearman相关系数 (ρ): 0.5234
使用分位数: p99

自适应分箱数量: 10

负载分箱摘要 (前5个和后5个):
    Load Range      |   Count |       p50 |       p90 |       p99
----------------------------------------------------------------------
[1000.0 , 1200.0 )  |     150 |     45.23 |     78.45 |    123.56
[1200.0 , 1400.0 )  |     180 |     52.34 |     89.12 |    145.67
[1400.0 , 1600.0 )  |     200 |     61.23 |    102.34 |    178.90
[1600.0 , 1800.0 )  |     210 |     73.45 |    121.56 |    201.23
[1800.0 , 2000.0 )  |     190 |     89.12 |    145.78 |    234.56
         ...        |     ... |       ... |       ... |       ...
[3800.0 , 4000.0 )  |      80 |    234.56 |    378.90 |    567.12
============================================================
提示: 使用 plot_motiv_case3() 绘制负载-延迟关系图
============================================================
```

### 对照实验完整示例
```python
# 基线组：禁用切换开销
collector_baseline = run_glb_simulation(realloc_overhead_enabled=False)
collector_baseline.set_motiv3_mode('raw')
stats_baseline = collector_baseline.get_motiv_case3_stats(percentile=0.99)
rho_baseline = stats_baseline['spearman_rho']

# 实验组：启用切换开销
collector_experiment = run_glb_simulation(realloc_overhead_enabled=True)
collector_experiment.set_motiv3_mode('binned')
stats_experiment = collector_experiment.get_motiv_case3_stats(percentile=0.99)
rho_experiment = stats_experiment['spearman_rho']

# 计算差异
delta_rho = rho_baseline - rho_experiment

# 判断
print(f"基线组 Spearman ρ: {rho_baseline:.4f}")
print(f"实验组 Spearman ρ: {rho_experiment:.4f}")
print(f"差异 Δρ: {delta_rho:.4f}")

if rho_baseline > 0.85 and rho_experiment < 0.6 and delta_rho > 0.25:
    print("✓ 假设验证成功：切换开销显著破坏了负载-延迟相关性")
else:
    print("✗ 假设验证失败")

# 绘制对比图
collector_baseline.plot_load_latency_raw(
    fit='wls', 
    save_path='./case3_baseline.pdf'
)
collector_experiment.plot_load_latency_binned(
    percentile=0.99, 
    fit='wls', 
    save_path='./case3_experiment.pdf'
)
```

---

## 主程序集成示例

### 在 `main_approach.py` 中使用

```python
from approach_sim import run_simulation
from approach_collector import StatisticsCollector

# ... 前置代码 ...

# 运行仿真
processors, event_t_rt, stats_collector = instantiate_processors(...)
run_simulation(processors, event_t_rt, G, num_hp=100, T_hp=T_hp)

# 根据policy选择对应的case
if policy == 'cyc':
    # Motiv-Exp-1
    stats_collector.export_motiv_case_results(
        case=1, 
        save_path=path_ctx.get_stat_log_path(),
        verbose=args.verbose
    )
    
elif policy == 'glb':
    # Motiv-Exp-2 或 Motiv-Exp-3
    # 可以通过命令行参数控制
    if args.motiv_case == 2:
        stats_collector.export_motiv_case_results(
            case=2,
            save_path=path_ctx.get_stat_log_path(),
            verbose=args.verbose
        )
    elif args.motiv_case == 3:
        stats_collector.set_motiv3_mode(args.motiv3_mode)  # 'raw' or 'binned'
        stats_collector.export_motiv_case_results(
            case=3,
            save_path=path_ctx.get_stat_log_path(),
            verbose=args.verbose,
            percentile=0.99,
            fit='wls'
        )
```

### 批量扫描脚本示例

```python
#!/usr/bin/env python3
"""
Motiv-Exp-1 批量扫描脚本
扫描不同的exec_t_comp_ratioA，绘制权衡曲线
"""
import subprocess
import json
import matplotlib.pyplot as plt

exec_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
results = []

for ratio in exec_ratios:
    # 运行仿真
    cmd = [
        'python', 'main_approach.py',
        '--policy', 'cyc',
        '--exec_t_comp_ratioA', str(ratio),
        '--output_dir', f'./results/case1_ratio_{ratio}',
        '--motiv_case', '1'
    ]
    subprocess.run(cmd, check=True)
    
    # 读取结果（假设输出为JSON）
    with open(f'./results/case1_ratio_{ratio}/stats.json') as f:
        stats = json.load(f)
    
    results.append({
        'ratio': ratio,
        'idle': stats['idle_mean_ratio'],
        'miss': stats['miss_mean_ratio'],
        'miss_count': stats['miss_mean_count']
    })

# 绘制权衡曲线
plt.figure(figsize=(8, 6))
plt.plot([r['idle'] for r in results], 
         [r['miss'] for r in results], 
         'o-', linewidth=2, markersize=8)

for r in results:
    plt.annotate(f"p{int(r['ratio']*100)}", 
                 (r['idle'], r['miss']), 
                 textcoords="offset points", 
                 xytext=(5,5), ha='left')

plt.xlabel('Idle Ratio', fontsize=12)
plt.ylabel('Miss Ratio', fontsize=12)
plt.title('Case 1: Utilization-Reliability Tradeoff Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./case1_tradeoff_curve.pdf', dpi=300)
print("权衡曲线已保存到: ./case1_tradeoff_curve.pdf")
```

---

## API总结表

| 实验 | 获取数据 | 格式化输出 | 绘图 | 统一导出 |
|------|---------|-----------|------|---------|
| Case 1 | `get_motiv_case1_stats()` | `format_motiv_case1_output()` | N/A | `export_motiv_case_results(case=1)` |
| Case 2 | `get_motiv_case2_stats()` | `format_motiv_case2_output()` | N/A | `export_motiv_case_results(case=2)` |
| Case 3 | `get_motiv_case3_stats()` | `format_motiv_case3_output()` | `plot_motiv_case3()` | `export_motiv_case_results(case=3)` |

### 通用函数
- `print_motiv_case_summary(case)`: 快捷打印到控制台
- `set_motiv3_mode('raw'|'binned')`: 设置Case 3数据模式
- `set_path(key, path)`: 设置输出路径

---

## 注意事项

1. **Case 3 的模式选择**:
   - `raw` 模式：适合点数较少（<1000个周期）的情况，保留所有原始数据
   - `binned` 模式：适合大数据量（≥1000个周期），自动分箱降低存储开销

2. **相关系数的解释**:
   - Spearman ρ ∈ [-1, 1]
   - ρ > 0.7: 强正相关（负载↑延迟↑趋势明显）
   - 0.3 < ρ < 0.7: 中等相关
   - ρ < 0.3: 弱相关或无相关

3. **延迟分解的总和**:
   - `exec_ratio + realloc_ratio + wait_ratio`
   - 理想情况：总和 ≈ 1（恰好满足约束）
   - 总和 > 1：表示超时（实际延迟超过端到端约束）
   - 总和 < 1：表示有余量

4. **文件路径**:
   - 使用 `set_path('stat', path)` 设置默认统计输出路径
   - 使用 `set_path('motiv3', path)` 设置Case 3图表路径

