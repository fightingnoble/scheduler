# Motivation Experiments Runner 使用说明

## 概述

`motiv_exp_runner.py` 是一个统一的Python脚本，用于运行三个motivation实验。它负责：

1. ✅ **参数扫描**: 根据实验需求自动扫描参数组合
2. ✅ **流程调用**: 调用`main_approach.py`标准流程
3. ✅ **数据收集**: 收集每次运行的`StatisticsCollector`
4. ✅ **结果汇总**: 生成统计报告（JSON + 图表）
5. ✅ **对象缓存**: 保存collector对象供后续深度分析

## 与文档的对应关系

### main_approach.py 流程检查

**标准流程**（与`doc/test_plan.md`一致）:
```python
1. setup_benchmark() → 生成/加载workload和调度信息（step 1-3）
2. instantiate_processors() → 创建处理器和collector
3. run_simulation() → 运行随机测试（step 4）
4. collector统计 → 通过专用API获取结果
```

**主要组件**:
- ✅ `setup_benchmark`: 实现了step 1-3的调度信息生成
- ✅ `run_simulation`: 实现了step 4的随机测试
- ✅ `stats_collector`: 与三个case的专用API完全匹配
- ✅ `num_hp`: 默认100个超周期，可通过参数调整

### 与Collector API的匹配

| Experiment | Collector API | Runner实现 |
|------------|--------------|-----------|
| Case 1 | `get_motiv_case1_stats()` | ✅ `MotivExp1Runner` |
| Case 1 | `plot_motiv_case1()` | ✅ 自动调用 |
| Case 2 | `get_motiv_case2_stats()` | ✅ `MotivExp2Runner` |
| Case 2 | `plot_motiv_case2()` | ✅ 自动调用（breakdown + utilization） |
| Case 3 | `get_motiv_case3_stats()` | ✅ `MotivExp3Runner` |
| Case 3 | `set_motiv3_mode()` | ✅ 自动设置（raw/binned） |
| Case 3 | `plot_motiv_case3()` | ✅ 自动调用 |

## 快速开始

### Case 1: 纯静态调度 - 利用率问题

```bash
# 基本用法：扫描默认的预留分位数
python scripts/motiv_exp_runner.py --case 1 --output_dir ./results

# 自定义扫描范围
python scripts/motiv_exp_runner.py --case 1 \
    --case1_ratios 0.5,0.6,0.7,0.8,0.9,0.95,0.99 \
    --output_dir ./results/case1 \
    --num_hp 200 \
    --verbose
```

**输出文件**:
```
./results/case1/
├── ratio_0.50/          # 每个配置的详细结果
├── ratio_0.60/
├── ...
├── case1_tradeoff.pdf   # 权衡图（并排双柱状图）
├── case1_summary.json   # JSON摘要
├── collector_ratio_0.50.pkl  # 缓存的collector对象
├── collector_ratio_0.60.pkl
└── ...
```

### Case 2: 纯动态调度 - 可扩展性问题

```bash
# 基本用法：扫描默认的硬件和任务规模
python scripts/motiv_exp_runner.py --case 2 --output_dir ./results

# 自定义扫描范围
python scripts/motiv_exp_runner.py --case 2 \
    --case2_tiles 300,400,500 \
    --case2_chains 1,2,4,8 \
    --case2_loads 0.5,0.75,1.0,1.25 \
    --output_dir ./results/case2 \
    --num_hp 150
```

**输出文件**:
```
./results/case2/
├── tiles_300_chains_1_load_0.5/
├── tiles_300_chains_4_load_1.0/
├── ...
├── case2_breakdown.pdf      # 延迟分解堆叠柱状图
├── case2_utilization.pdf    # 资源利用率堆叠柱状图
├── case2_summary.json
├── collector_300t_1c_0.5l.pkl
└── ...
```

### Case 3: 切换行为的不确定性

```bash
# 基本用法：运行对照实验（基线组 + 实验组）
python scripts/motiv_exp_runner.py --case 3 --output_dir ./results

# 只运行基线组（禁用切换开销）
python scripts/motiv_exp_runner.py --case 3 \
    --case3_baseline \
    --case3_num_periods 500 \
    --output_dir ./results/case3

# 只运行实验组（启用切换开销）
python scripts/motiv_exp_runner.py --case 3 \
    --case3_experiment \
    --case3_mode binned \
    --case3_num_periods 10000 \
    --output_dir ./results/case3

# 完整对照实验
python scripts/motiv_exp_runner.py --case 3 \
    --case3_baseline \
    --case3_experiment \
    --case3_mode binned \
    --case3_num_periods 5000 \
    --output_dir ./results/case3
```

**输出文件**:
```
./results/case3/
├── baseline/              # 基线组（无切换开销）
├── experiment/            # 实验组（有切换开销）
├── case3_baseline.pdf     # 基线组散点图
├── case3_experiment.pdf   # 实验组binned曲线
├── case3_summary.json     # 对比结果（Δρ等）
├── collector_baseline.pkl
└── collector_experiment.pkl
```

## 命令行参数完整列表

### 通用参数

```bash
--case {1,2,3}              # [必需] 实验编号
--output_dir PATH           # 输出目录（默认: ./motiv_exp_results）
--cache_collectors          # 缓存collector对象（默认: True）
--num_hp INT                # 仿真超周期数（默认: 100）
--verbose                   # 详细输出
--dry_run                   # 只打印命令，不执行
--extra_args "..."          # 传递给main_approach.py的额外参数
```

### Case 1 特定参数

```bash
--case1_ratios "0.5,0.6,..." # exec_t_comp_ratioA扫描值
                             # 默认: 0.5,0.6,0.7,0.8,0.9,0.99
```

### Case 2 特定参数

```bash
--case2_tiles "300,500"      # 硬件tile数扫描值（默认: 300,500）
--case2_chains "1,4"         # 任务链数量扫描值（默认: 1,4）
--case2_loads "0.5,1.0"      # 负载倍数扫描值（默认: 0.5,1.0）
```

### Case 3 特定参数

```bash
--case3_mode {raw,binned}    # 数据收集模式（默认: binned）
--case3_baseline             # 运行基线组（禁用切换开销）
--case3_experiment           # 运行实验组（启用切换开销）
--case3_num_periods INT      # 仿真周期数（默认: 1000）
```

**注意**: Case 3如果不指定`--case3_baseline`或`--case3_experiment`，则默认运行两者。

## 高级用法

### 1. 传递额外参数给main_approach.py

```bash
# 例如：添加profiling, 修改benchmark等
python scripts/motiv_exp_runner.py --case 1 \
    --extra_args "--profiling_filename profiling.csv --gen_benchmark"
```

### 2. 干运行（调试模式）

```bash
# 只打印将要执行的命令，不实际运行
python scripts/motiv_exp_runner.py --case 1 --dry_run
```

输出示例:
```
[DRY RUN] Running: python main_approach.py --test_case cyclic --exec_t_comp_ratioA 0.5 ...
[DRY RUN] Running: python main_approach.py --test_case cyclic --exec_t_comp_ratioA 0.6 ...
...
```

### 3. 后续深度分析

```python
#!/usr/bin/env python3
"""从缓存的collector进行深度分析"""
import pickle
from pathlib import Path

# 加载Case 1的所有collectors
case1_dir = Path('./results/case1')
collectors = {}

for pkl_file in case1_dir.glob('collector_ratio_*.pkl'):
    with open(pkl_file, 'rb') as f:
        ratio = pkl_file.stem.split('_')[-1]
        collectors[ratio] = pickle.load(f)

# 深度分析：获取完整的分布数据
for ratio, collector in collectors.items():
    summary = collector.get_summary(num_bins=50, p_list=[0.5, 0.9, 0.95, 0.99, 0.999])
    print(f"\nRatio {ratio}:")
    print(f"  Idle p99: {summary['overall_idle_time']['percentiles']['p99.0']:.4f}")
    print(f"  Miss p99: {summary['overall_missed_load']['percentiles']['p99.0']:.4f}")
    
    # 获取分位数分解
    breakdown = collector.get_latency_breakdown_avg_ratio()
    # ... 更多分析
```

### 4. 批量运行所有实验

```bash
#!/bin/bash
# run_all_motiv_exps.sh

# Case 1: 纯静态
python scripts/motiv_exp_runner.py --case 1 \
    --output_dir ./final_results \
    --num_hp 200 &

# Case 2: 纯动态
python scripts/motiv_exp_runner.py --case 2 \
    --output_dir ./final_results \
    --num_hp 200 &

# Case 3: 对照实验
python scripts/motiv_exp_runner.py --case 3 \
    --output_dir ./final_results \
    --case3_num_periods 10000 &

wait
echo "所有实验完成！"
```

## 与旧脚本的对比

| 特性 | 旧脚本 (bash) | 新脚本 (python) |
|------|--------------|----------------|
| 参数扫描 | 手动循环，难以修改 | 声明式配置，易于调整 |
| 结果收集 | 需要手动解析日志 | 自动收集collector |
| 图表生成 | 需要单独脚本 | 自动调用专用API |
| 缓存管理 | 无 | 自动pickle缓存 |
| 错误处理 | 基本无 | 完善的异常处理 |
| 可读性 | Shell嵌套复杂 | Python清晰易懂 |
| 可扩展性 | 难以扩展 | 面向对象，易扩展 |

## 实现细节

### 流程对接

```python
# Runner调用main_approach.py的标准流程
run_main_approach({
    'test_case': 'cyclic',     # 对应test_plan中的policy
    'exec_t_comp_ratioA': 0.8, # 对应step 1的配置
    ...
})
↓
main_approach.py:
    setup_benchmark()          # step 1-3: 生成调度信息
    instantiate_processors()   # 创建collector
    run_simulation()           # step 4: 随机测试
    collector收集统计
↓
Runner加载collector
    collector.get_motiv_caseN_stats()  # 获取统计
    collector.plot_motiv_caseN()       # 生成图表
```

### Collector加载机制

**注意**: 当前实现假设`main_approach.py`会保存collector到pickle文件。如果没有，需要修改`main_approach.py`：

```python
# 在main_approach.py末尾添加:
if __name__ == "__main__":
    # ... 现有代码 ...
    
    # 保存collector
    import pickle
    collector_path = path_ctx.get_output_dir() / 'stats_collector.pkl'
    with open(collector_path, 'wb') as f:
        pickle.dump(stats_collector, f)
```

或者，修改Runner的`load_collector()`函数以适应实际的保存方式。

## 故障排查

### 问题1: "Could not load collector"

**原因**: `main_approach.py`未保存collector到pickle文件

**解决**: 
1. 修改`main_approach.py`添加保存逻辑（见上方）
2. 或修改Runner的`load_collector()`以适应实际保存方式

### 问题2: 运行时间过长

**原因**: 参数组合过多或`num_hp`设置过大

**解决**:
- 减少扫描参数范围
- 降低`--num_hp`
- 使用`--dry_run`先验证参数组合数量

### 问题3: 图表未生成

**原因**: collector加载失败或结果为空

**解决**:
- 检查`--verbose`输出
- 验证collector是否正确保存和加载
- 检查实验是否正常完成

## 最佳实践

1. **先干运行**: 使用`--dry_run`验证参数配置
2. **小范围测试**: 先用少量参数测试流程
3. **监控资源**: 注意内存和磁盘使用（特别是Case 3的binned模式）
4. **保留日志**: `--verbose`输出重定向到日志文件
5. **定期备份**: 缓存的collector文件很重要，定期备份

## 扩展建议

如果需要添加新的实验或修改现有逻辑：

1. **继承Runner类**:
```python
class CustomMotivExp1Runner(MotivExp1Runner):
    def run(self):
        # 自定义扫描逻辑
        ...
```

2. **添加新的统计指标**:
```python
def _generate_report(self):
    # 除了标准统计，添加自定义分析
    for r in self.results:
        custom_metric = self._compute_custom(r['collector'])
        ...
```

3. **修改参数映射**:
```python
run_args = {
    # 根据实际需求修改参数映射
    'custom_param': value,
    ...
}
```

## 联系与反馈

如有问题或建议，请参考：
- 详细文档: `doc/motiv_case_api_usage.md`
- 快速参考: `doc/motiv_case_api_quick_ref.md`
- 实验计划: `doc/test_plan.md`

