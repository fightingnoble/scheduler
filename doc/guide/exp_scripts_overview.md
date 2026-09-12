# 实验脚本模块概览

本文档概述 `/home/zhangchg/git_repo/scheduler/scripts/` 目录下的实验脚本模块架构。

## 模块概览

```
scripts/
├── exp_common.py         # 共享组件：参数模板、主程序调用、绘图辅助
├── motiv_exp_runner.py   # 动机实验运行器 (Motivation Experiments)
└── abla_exp_runner.py    # 消融实验运行器 (Ablation Experiments)
```

---

## 1. exp_common.py - 共享组件模块

### 1.1 核心导出

| 导出项 | 类型 | 用途 |
|--------|------|------|
| `_PHYSICAL_CORES` | `int` | 物理CPU核心数，用于并行任务控制 |
| `mapping_args` | `Dict` | 基础mapping参数模板 |
| `specific_args` | `Dict` | 基础specific参数模板 |
| `runtime_args` | `Dict` | 基础runtime参数模板 |
| `ParamTemplate` | `class` | 参数模板封装类 |
| `run_main_approach_inproc` | `function` | 进程内调用main_approach |

### 1.2 ParamTemplate 类

**职责**: 封装三类参数(mapping/runtime/specific)，提供克隆、增量更新和合并导出能力。

```python
class ParamTemplate:
    def __init__(self, mapping, runtime, specific)
    def clone(self) -> ParamTemplate
    def with_updates(mapping=None, runtime=None, specific=None) -> ParamTemplate
    def to_run_args(self) -> Dict[str, Any]
```

**核心方法**:
- `clone()`: 深拷贝当前模板
- `with_updates(**kwargs)`: 创建增量更新的新模板（推荐用法）
- `to_run_args()`: 合并三组参数并过滤None值，生成最终运行参数

### 1.3 run_main_approach_inproc 函数

**职责**: 以函数方式调用 `main_approach.main()`，避免子进程与磁盘往返。

**调用流程**:
```
run_main_approach_inproc(args_dict)
    │
    ├── 构造 sys.argv (添加固定参数: --profiling_filename, --gen_benchmark)
    │
    ├── 临时替换 sys.argv
    │
    └── 调用 main_approach.main() → 返回 StatisticsCollector
```

**固定附加参数**:
- `--profiling_filename profiling/profiling_light.csv`
- `--gen_benchmark`

### 1.4 绘图辅助函数

| 函数 | 用途 |
|------|------|
| `group_by_key(data_points, key)` | 按指定key对数据点分组 |
| `compute_group_means(grouped_data, value_keys)` | 计算每组平均值，支持嵌套key |
| `setup_dual_axis_plot(figsize)` | 创建双轴图 (fig, ax1, ax2) |
| `save_and_close_figure(fig, save_path, msg)` | 保存图形并关闭 |
| `add_value_labels(ax, bars, fmt)` | 在柱状图上方添加数值标签 |

---

## 2. motiv_exp_runner.py - 动机实验运行器

### 2.1 模块职责

运行三个Motivation实验，验证现有调度策略的问题：
- **Case 1**: 纯静态调度 - 利用率问题
- **Case 2**: 纯动态调度 - 可扩展性问题
- **Case 3**: 切换行为不确定性

### 2.2 Runner 类

#### MotivExp1Runner

**实验目的**: 展示静态调度中利用率与可靠性的Trade-off

**扫描参数**: `exec_t_comp_ratioA` (默认: `[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]`)

**关键配置**:
```python
mapping = {
    'test_case': 'cyclic',
    'exec_t_comp_ratioA': ratio,  # 扫描变量
    'exec_t_comp_ratioB': -1,
    'num_bins': -1,
}
runtime = {'policy': 'cyc'}
```

**输出指标**:
- `idle_mean_ratio`: 空闲时间比例
- `miss_mean_ratio`: 超时比例
- `miss_mean_count`: 超时计数
- `realloc_mean_ratio`: 重分配比例

---

#### MotivExp2Runner

**实验目的**: 展示动态调度在大规模场景下的可扩展性瓶颈

**扫描参数**:
- `tiles`: 硬件tile数 (默认: `[400, 400, 200, 200]`)
- `loads`: 负载倍数 (默认: `[0.5, 1.0, 0.5, 1.0]`)
- `chains`: 任务链数 (默认: `[1, 4, 9]`)

**约束**: `tiles` 和 `loads` 必须等长且一一对应

**关键配置**:
```python
mapping = {
    'test_case': 'dynamic',
    'num_cores': tiles,
    'aux_scale_factor': chains,
    'load_factor': load_factor,
    'num_bins': 1,
}
runtime = {'policy': 'glb'}
```

**输出图表**:
- `case2_breakdown.pdf`: 延迟分解图
- `case2_utilization.pdf`: 利用率图

---

#### MotivExp3Runner

**实验目的**: 展示切换开销的不确定性导致调度决策不可预测

**扫描参数**:
- `--case3_mode`: 数据模式 (`raw`/`binned`)
- `--case3_baseline`: 运行基线组（禁用切换开销）
- `--case3_experiment`: 运行实验组（启用切换开销）
- `--case3_num_periods`: 仿真周期数 (默认: 1000)

**关键配置**:
```python
# 基线组
specific = {
    'barrier_dis': True,  # 禁用切换开销
    'stat_param': "{'motiv3_mode': 'raw', 'motiv3_en': True}"
}

# 实验组
specific = {
    'barrier_dis': False,  # 启用切换开销
    'stat_param': f"{{'motiv3_mode': '{mode}', 'motiv3_en': True}}"
}
```

**验证指标**:
- `spearman_rho`: Spearman相关系数
- `rmse`: 均方根误差
- 假设验证条件: `(rho_base > 0.85) and (rho_exp < 0.6) and (delta_rho > 0.25)`

---

### 2.3 Worker 函数

| Worker | 对应Case | 返回值 |
|--------|----------|--------|
| `_case1_worker(payload)` | Case 1 | `{ratio, label, idle_mean_ratio, miss_mean_ratio, ...}` |
| `_case2_worker(payload)` | Case 2 | `{tiles, chains, load_factor, label, utilization, latency_breakdown, ...}` |
| `_case3_worker(payload)` | Case 3 | `{exp_name, stats}` |

---

## 3. abla_exp_runner.py - 消融实验运行器

### 3.1 模块职责

运行三个Ablation实验，验证Reserv调度策略各组件的贡献：
- **Case 1**: cyc-S vs cyc - 预留在串行执行下的影响
- **Case 2**: pglb vs glb - 空间隔离的作用
- **Case 3**: reserv vs pglb - 预留在并行下的影响

### 3.2 Runner 类

#### AblaExp1Runner

**实验对比**: cyc-S (软预留) vs cyc (硬隔离)

**扫描参数**:
- `ratioBs`: cyc-S的`exec_t_comp_ratioB` (默认: `[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]`)
- `ratioA`: cyc-S的固定`exec_t_comp_ratioA` (默认: `0.7`)
- `cyc_ratios`: cyc参考点的`exec_t_comp_ratioA` (用于投影线)

**关键配置**:
```python
# cyc 参考点 (硬隔离)
mapping = {
    'test_case': 'cyclic',
    'exec_t_comp_ratioA': ratio,
    'exec_t_comp_ratioB': -1,
    'num_bins': -1,
}
runtime = {'policy': 'cyc'}

# cyc-S (软预留)
mapping = {
    'test_case': 'cyclic',
    'exec_t_comp_ratioA': ratioA,  # 固定
    'exec_t_comp_ratioB': ratioB,  # 扫描
    'num_bins': -1,
}
runtime = {'policy': 'reserv'}
```

**输出图表**:
- `case1_motiv1_style.pdf`: 复用Motiv-Exp-1绘图风格
- `case1_satisfy_projection.pdf`: 延迟满足率投影图

---

#### AblaExp2Runner

**实验对比**: pglb (分区动态) vs glb (全局动态)

**扫描参数**:
- `num_bins`: 空间分区数 (默认: `[1, 2, 4, 8]`)
- `tiles`: 硬件tile数 (默认: `[200, 400]`)
- `chains`: 任务链数 (默认: `[1, 4]`)
- `loads`: 负载倍数 (默认: `[0.5, 1.0]`)

**关键配置**:
```python
# glb (num_bins=1)
mapping = {
    'test_case': 'dynamic',
    'exec_t_comp_ratioA': 0.7,
    'exec_t_comp_ratioB': -1,
    'num_bins': 1,
    'num_cores': tiles,
    'aux_scale_factor': chains,
    'load_factor': load_factor,
}
runtime = {'policy': 'glb'}

# pglb (num_bins>1)
runtime = {'policy': 'pglb'}
```

**输出图表**:
- `case2_switching_overhead.pdf`: 切换开销对比图
- `case2_breakdown.pdf`: 延迟分解图
- `case2_utilization.pdf`: 利用率图

---

#### AblaExp3Runner

**实验对比**: reserv (预留+分区) vs pglb (仅分区)

**扫描参数**:
- `ratioBs`: reserv的`exec_t_comp_ratioB` (默认: `[0.5, 0.6, 0.7, 0.8, 0.9, 0.99]`)
- `bins`: num_bins扫描值 (默认: `[1, 2, 4, 8]`，`-1`表示单分区)
- `tiles/chains/loads`: 负载强度参数
- `fixed_strength`: 可选固定负载强度模式

**关键配置**:
```python
# pglb 基线
mapping = {
    'test_case': 'dynamic',
    'exec_t_comp_ratioA': 0.7,
    'exec_t_comp_ratioB': -1,
    'num_bins': num_bins,
}
runtime = {'policy': 'pglb'}

# reserv
mapping = {
    'exec_t_comp_ratioB': ratioB,  # 扫描
    'num_bins': num_bins,
}
runtime = {'policy': 'reserv'}
```

**输出图表**:
- `case3_switching_bins{N}.pdf`: 各bins配置下的切换指标图
- `case3_breakdown.pdf`: 延迟分解图
- `case3_utilization.pdf`: 利用率图

---

### 3.3 Worker 函数

| Worker | 对应Case | 额外返回字段 |
|--------|----------|--------------|
| `_case1_worker(payload)` | Case 1 | `exp_type` (cyc/cyc-S) |
| `_case2_worker(payload)` | Case 2 | `num_bins`, `realloc_mean_count` |
| `_case3_worker(payload)` | Case 3 | `num_bins`, `exec_t_comp_ratioB`, `realloc_mean_count/ratio` |

---

## 4. 参数传递流程

### 4.1 整体流程图

```
命令行参数 (parse_args)
        │
        ▼
┌───────────────────────────────────────┐
│  构造基础参数模板                       │
│  base_tpl = ParamTemplate(            │
│      mapping = {**mapping_args},      │
│      runtime = {**runtime_args},      │
│      specific = {**specific_args}     │
│  )                                    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Runner 实例化                         │
│  runner = XxxRunner(base_tpl, args)   │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  _run_simulations()                    │
│  ├─ 遍历扫描参数                        │
│  ├─ tpl = base_tpl.with_updates(...)  │
│  ├─ 构造 tasks 列表                    │
│  └─ ProcessPoolExecutor 并行执行       │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Worker 函数                           │
│  ├─ run_main_approach_inproc(args)    │
│  ├─ collector.get_xxx_stats()         │
│  └─ 返回结果字典                        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  _generate_report()                    │
│  ├─ 保存 JSON 摘要                     │
│  ├─ StatisticsCollector.plot_xxx()    │
│  └─ 生成 PDF 图表                      │
└───────────────────────────────────────┘
```

### 4.2 参数分组说明

| 分组 | 含义 | 典型参数 |
|------|------|----------|
| `mapping` | 传递给 `utils.input_parser()` 的业务参数 | `exec_t_comp_ratioA/B`, `num_bins`, `e2e_latency`, `test_case` |
| `runtime` | 运行时控制参数 | `n_p`, `policy` |
| `specific` | 实验特定参数 | `root_dir`, `verbose`, `stat_param`, `barrier_dis` |

### 4.3 缓存机制

- **JSON缓存**: 每个Case运行后保存 `{output_dir}/case{N}/case{N}_summary.json`
- **缓存加载**: 使用 `--use_plot_cache` 跳过仿真，直接从JSON生成图表
- **缓存验证**: 检查缓存数据点数量是否与扫描参数预期一致

---

## 5. 命令行用法

### 5.1 Motivation Experiments

```bash
# Case 1: 静态调度利用率问题
python -m scripts.motiv_exp_runner --case 1 --output_dir ./motiv_results --num_hp 100

# Case 2: 动态调度可扩展性问题
python -m scripts.motiv_exp_runner --case 2 --output_dir ./motiv_results --num_hp 100

# Case 3: 切换行为不确定性
python -m scripts.motiv_exp_runner --case 3 --output_dir ./motiv_results --case3_num_periods 200
```

### 5.2 Ablation Experiments

```bash
# Case 1: cyc-S vs cyc
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100

# Case 2: pglb vs glb
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100

# Case 3: reserv vs pglb
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
```

### 5.3 通用选项

| 选项 | 用途 |
|------|------|
| `--dry_run` | 只打印命令，不执行 |
| `--use_plot_cache` | 从缓存JSON生成图表，跳过仿真 |
| `--verbose` | 详细输出 |
| `--extra_args` | 传递额外参数给main_approach |

---

## 6. 关键代码路径

### 6.1 实验脚本调用链

```
scripts/xxx_exp_runner.py
    │
    ├── exp_common.ParamTemplate.to_run_args()
    │
    └── exp_common.run_main_approach_inproc()
            │
            └── main_approach.main()
                    │
                    ├── approach_setup.setup_benchmark()
                    │       └── sim_main.perform_bin_packing()
                    │
                    └── approach_sim.run_simulation()
                            │
                            └── approach_collector.StatisticsCollector
```

### 6.2 统计收集调用链

```
StatisticsCollector (approach/approach_collector.py)
    │
    ├── get_motiv_case1_stats()     # Case 1 指标
    ├── get_motiv_case2_stats()     # Case 2 指标 (利用率+延迟分解)
    ├── get_motiv_case3_stats()     # Case 3 指标 (相关性+RMSE)
    ├── get_realloc_info()          # 重分配信息
    ├── get_utilization_avg_ratio() # 利用率统计
    │
    └── plot_xxx()                  # 静态绘图方法
```

### 6.3 并行执行机制

```python
# ProcessPoolExecutor 并行执行
max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(_caseN_worker, t) for t in tasks]
    for fut in as_completed(futures):
        res = fut.result()
        self.results.append(res)
```

### 6.4 错误处理

```python
# 捕获子进程异常，避免主进程卡死
try:
    result = future.result()
except ResourceInsufficientError:
    print(f"Resource insufficient for ratio={ratio}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 7. 结果输出

### 7.1 目录结构

```
output_dir/
├── case1/
│   ├── case1_summary.json     # 结果汇总
│   ├── case1_utilization.pdf  # Case 1 图
│   └── cache/
│       └── *.pkl              # 仿真缓存
├── case2/
│   ├── case2_summary.json
│   ├── case2_breakdown.pdf
│   └── case2_utilization.pdf
└── case3/
    ├── case3_summary.json
    └── case3_correlation.pdf
```

### 7.2 JSON 格式

```json
{
    "experiment": "Case 1: Utilization-Reliability Tradeoff",
    "timestamp": "2026-03-11T10:00:00",
    "parameters": {
        "ratios": [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    },
    "results": [
        {
            "ratio": 0.5,
            "idle_mean_ratio": 0.45,
            "miss_mean_ratio": 0.15,
            "realloc_mean_ratio": 0.0,
            "miss_mean_count": 12.5
        },
        ...
    ]
}
```

---

## 8. 实验状态

### 8.1 Motivation 实验

| Case | 状态 | 结果目录 |
|------|------|----------|
| Case 1 | 完成 | `motiv_exp_results/case1/` |
| Case 2 | 完成 | `motiv_exp_results/case2/` |
| Case 3 | 完成 | `motiv_exp_results/case3/` |

### 8.2 消融实验

| Case | 状态 | 问题 |
|------|------|------|
| Exp 1 | 部分完成 | cyc-S 结果可能有问题（p70-p99 相同） |
| Exp 2 | 未开始 | 目录为空 |
| Exp 3 | 未开始 | 无 case3 目录 |

---

## 9. 扩展指南

### 9.1 添加新实验

1. 在 `exp_common.py` 中添加新的基础参数（如需要）
2. 创建新的 Runner 类，继承相同模式：
   - `__init__(base_tpl, args)`: 初始化参数和输出目录
   - `run()`: 主入口
   - `_run_simulations()`: 构造任务列表并并行执行
   - `_generate_report()`: 生成JSON和图表
3. 创建对应的 worker 函数
4. 在 `parse_args()` 中添加命令行参数
5. 在 `main()` 中添加分支选择

### 9.2 复用绘图功能

```python
# 使用 StatisticsCollector 静态方法
StatisticsCollector.plot_motiv_case1(data_points, save_path)
StatisticsCollector.plot_motiv_case2(data_points, plot_type, group_order, save_path)
StatisticsCollector.plot_load_latency_raw(data_groups, save_path)
StatisticsCollector.plot_load_latency_binned(binned_summary, spearman_rho, save_path)
```

---

## 10. 注意事项

1. **环境依赖**: 运行前需激活 `conda activate gurobi`
2. **并行限制**: 并行度受物理核心数 `_PHYSICAL_CORES` 限制
3. **缓存一致性**: 使用 `--use_plot_cache` 时会验证缓存数据点数量
4. **参数覆盖**: `--extra_args` 可覆盖 specific 组参数
5. **Policy映射**:
   - `cyc`: 纯静态循环调度
   - `glb`: 全局动态调度
   - `pglb`: 分区动态调度
   - `reserv`: 预留调度（软预留）
