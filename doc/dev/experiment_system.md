# 实验系统架构

> 本文档说明 Motivation 实验和消融实验的架构设计。

---

## 1. 实验分类

### 1.1 Motivation 实验（3 个 Case）

| Case | 验证问题 | 对照策略 | 扫描参数 |
|------|----------|----------|----------|
| Case 1 | 纯静态调度的利用率问题 | cyc | `exec_t_comp_ratioA ∈ [0.5, 0.99]` |
| Case 2 | 纯动态调度的延迟开销 | glb | `tiles × chains × load_factor` |
| Case 3 | 切换行为的不确定性 | glb | 负载-延迟相关性 |

### 1.2 消融实验（3 个 Case）

| Case | 对比策略 | 验证机制 | 扫描参数 |
|------|----------|----------|----------|
| Exp 1 | cyc-S vs cyc | 预留（串行执行） | `ratioB ∈ [0.5, 0.99]` |
| Exp 2 | pglb vs glb | 空间隔离 | `num_bins ∈ [1, 2, 4, 8]` |
| Exp 3 | reserv vs pglb | 预留+隔离 | `ratioB × num_bins` 网格 |

---

## 2. 代码结构

### 2.1 共享组件

```
scripts/exp_common.py
    ├── ParamTemplate         # 参数模板类
    ├── run_main_approach_inproc()  # 仿真调用包装
    └── process_results()    # 结果处理工具
```

### 2.2 Motivation 实验运行器

```
scripts/motiv_exp_runner.py
    ├── MotivExp1Runner (L98-232)   # Case 1: 利用率问题
    ├── MotivExp2Runner (L235-396)  # Case 2: 延迟开销
    └── MotivExp3Runner (L399-542)  # Case 3: 不确定性
```

### 2.3 消融实验运行器

```
scripts/abla_exp_runner.py
    ├── AblaExp1Runner (L139-323)   # Exp 1: cyc-S vs cyc
    ├── AblaExp2Runner (L326-542)   # Exp 2: pglb vs glb
    └── AblaExp3Runner (L545-814)   # Exp 3: reserv vs pglb
```

---

## 3. 运行命令

### 3.1 Motivation 实验

```bash
# Case 1: 利用率-可靠性权衡
python -m scripts.motiv_exp_runner --case 1 --output_dir ./motiv_results --num_hp 100

# Case 2: 可扩展性分析
python -m scripts.motiv_exp_runner --case 2 --output_dir ./motiv_results --num_hp 100

# Case 3: 负载-延迟相关性
python -m scripts.motiv_exp_runner --case 3 --output_dir ./motiv_results --case3_num_periods 200
```

### 3.2 消融实验

```bash
# Exp 1: 预留机制验证
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100

# Exp 2: 隔离机制验证
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100

# Exp 3: 预留+隔离联合验证
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
```

### 3.3 从缓存重绘

```bash
# 跳过仿真，从缓存重绘
python -m scripts.motiv_exp_runner --case 1 --use_plot_cache
python -m scripts.abla_exp_runner --case 2 --use_plot_cache
```

---

## 4. 参数模板

### 4.1 基础参数

```python
mapping_args = {
    'G_decomp_mode': "full",
    'exec_t_comp_ratioA': 0.99,
    'exec_t_comp_ratioB': -1,
    'e2e_latency': 0.1,
    'aux_scale_factor': 1,
    'test_case': 'bin_pack_new',
    'num_bins': -1,
    'bin_pack_cfg': 'Bp_guided.json',
    'n_p': 3,
}
```

### 4.2 参数更新模式

```python
# 创建基础模板
base_params = ParamTemplate(mapping_args, runtime_args, specific_args)

# 增量更新
case1_params = base_params.with_updates(
    mapping={'exec_t_comp_ratioA': ratio},
    runtime={'num_hp': 100}
)

# 转换为运行参数
args = case1_params.to_run_args()
```

---

## 5. 并行执行

### 5.1 ProcessPoolExecutor

```python
# motiv_exp_runner.py:L140-160
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {}
    for ratio in self.ratios:
        params = self._build_params(ratio)
        future = executor.submit(
            run_main_approach_inproc,
            params.to_run_args()
        )
        futures[future] = ratio

    for future in as_completed(futures):
        try:
            result = future.result()
            self.results.append(result)
        except Exception as e:
            print(f"Worker failed: {e}")
```

### 5.2 错误处理

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

## 6. 结果输出

### 6.1 目录结构

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

### 6.2 JSON 格式

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

## 7. 实验状态

### 7.1 Motivation 实验

| Case | 状态 | 结果目录 |
|------|------|----------|
| Case 1 | ✅ 完成 | `motiv_exp_results/case1/` |
| Case 2 | ✅ 完成 | `motiv_exp_results/case2/` |
| Case 3 | ✅ 完成 | `motiv_exp_results/case3/` |

### 7.2 消融实验

| Case | 状态 | 问题 |
|------|------|------|
| Exp 1 | ⚠️ 部分完成 | cyc-S 结果可能有问题（p70-p99 相同） |
| Exp 2 | ❌ 未开始 | 目录为空 |
| Exp 3 | ❌ 未开始 | 无 case3 目录 |

---

## 8. 相关文档

- [test_plan.md](../spec/test_plan.md) - 详细实验参数
- [result_collection.md](./result_collection.md) - 结果收集与绘图
- [ABLA_EXP_FIX_PLAN.md](../ABLA_EXP_FIX_PLAN.md) - 消融实验修复计划
