# Motivation实验实现检查清单

## 1. main_approach.py 流程检查

### ✅ 与test_plan.md的对应关系

| test_plan描述 | main_approach.py实现 | 状态 |
|--------------|---------------------|------|
| **Step 1-3**: 生成调度信息 | `setup_benchmark(args, time_norm_factor)` | ✅ |
| **Step 4**: 随机测试 | `run_simulation(processors, event_t_rt, G, ...)` | ✅ |
| 支持多种policy | `policy` 参数：'cyc', 'glb', 'pglb', 'reserv' | ✅ |
| 配置超周期数 | `num_hp=100` | ✅ |
| 配置时间单位 | `set_time_unit(1e-6, False)` | ✅ |
| 创建StatisticsCollector | `instantiate_processors()` 返回 | ✅ |
| 设置输出路径 | `stats_collector.set_path()` | ✅ |

### ⚠️ 需要的修改

**当前缺失**: `main_approach.py` **未保存** `stats_collector` 到文件

**建议修改**（在main_approach.py末尾添加）:
```python
# 在main_approach.py第60行之后添加:
import pickle

# 保存collector供后续分析
collector_path = path_ctx.get_output_dir() / 'stats_collector.pkl'
with open(collector_path, 'wb') as f:
    pickle.dump(stats_collector, f)
print(f"StatisticsCollector saved to: {collector_path}")

# 或者，如果希望立即导出统计（可选）
stats_collector.export_summary(verbose=args.verbose)
```

## 2. StatisticsCollector API匹配度

### ✅ Case 1接口完整性

| 需求 | 实现 | 状态 |
|------|------|------|
| 获取idle/miss/count | `get_motiv_case1_stats()` | ✅ |
| 格式化输出 | `format_motiv_case1_output()` | ✅ |
| 绘制权衡图 | `plot_motiv_case1()` | ✅ 并排双柱状图 |
| 统一导出 | `export_motiv_case_results(case=1)` | ✅ |

### ✅ Case 2接口完整性

| 需求 | 实现 | 状态 |
|------|------|------|
| 获取utilization | `get_motiv_case2_stats()['utilization']` | ✅ |
| 获取breakdown | `get_motiv_case2_stats()['latency_breakdown']` | ✅ |
| 绘制breakdown图 | `plot_motiv_case2(plot_type='breakdown')` | ✅ |
| 绘制utilization图 | `plot_motiv_case2(plot_type='utilization')` | ✅ |
| 统一导出 | `export_motiv_case_results(case=2)` | ✅ |

### ✅ Case 3接口完整性

| 需求 | 实现 | 状态 |
|------|------|------|
| 设置数据模式 | `set_motiv3_mode('raw'/'binned')` | ✅ |
| 获取相关系数 | `get_motiv_case3_stats()['spearman_rho']` | ✅ |
| 绘制关系图 | `plot_motiv_case3()` | ✅ 自动选择模式 |
| 统一导出 | `export_motiv_case_results(case=3)` | ✅ |

## 3. motiv_exp_runner.py 实现检查

### ✅ 三个实验类的实现

| Experiment | Runner类 | 功能 | 状态 |
|------------|---------|------|------|
| Case 1 | `MotivExp1Runner` | 扫描exec_t_comp_ratioA | ✅ |
| Case 2 | `MotivExp2Runner` | 扫描tiles/chains/loads | ✅ |
| Case 3 | `MotivExp3Runner` | 对照实验（baseline vs experiment） | ✅ |

### ✅ 核心功能

| 功能 | 实现 | 状态 |
|------|------|------|
| 参数扫描 | 各Runner类的`run()` | ✅ |
| 调用main_approach | `run_main_approach()` | ✅ |
| 加载collector | `load_collector()` | ⚠️ 依赖main_approach保存 |
| 缓存collector | `save_collector()` | ✅ |
| 调用统计API | `collector.get_motiv_caseN_stats()` | ✅ |
| 生成图表 | `collector.plot_motiv_caseN()` | ✅ |
| JSON摘要 | `_generate_report()` | ✅ |

### ⚠️ 依赖关系

```
motiv_exp_runner.py
    ↓ 调用
main_approach.py
    ↓ 返回（需要添加）
stats_collector.pkl
    ↓ 加载
motiv_exp_runner.py
    ↓ 使用
Collector专用API
```

**关键点**: `load_collector()` 需要 `main_approach.py` 保存collector

## 4. 参数映射检查

### Case 1参数映射

| test_plan参数 | main_approach参数 | runner传递 | 状态 |
|--------------|------------------|-----------|------|
| cyc policy | `--test_case cyclic` | ✅ | ✅ |
| exec_t_comp_ratioA | `--exec_t_comp_ratioA` | ✅ | ✅ |
| num_bin = -1 | `--num_bins -1` | ✅ | ✅ |

### Case 2参数映射

| test_plan参数 | main_approach参数 | runner传递 | 状态 |
|--------------|------------------|-----------|------|
| glb policy | `--test_case dynamic` | ✅ | ✅ |
| 硬件tile数 | `--num_tiles` | ✅ | ⚠️ 需验证 |
| 任务链数 | `--num_chains` | ✅ | ⚠️ 需验证 |
| 负载倍数 | `--load_factor` | ✅ | ⚠️ 需验证 |

**注意**: 需要确认`main_approach.py`/`setup_benchmark()`是否支持这些参数。

### Case 3参数映射

| test_plan参数 | main_approach参数 | runner传递 | 状态 |
|--------------|------------------|-----------|------|
| glb policy | `--test_case dynamic` | ✅ | ✅ |
| 启用/禁用开销 | `--realloc_overhead_enabled` | ✅ | ⚠️ 需验证 |
| 仿真周期数 | `--num_hp` | ✅ | ✅ |

## 5. 输出文件检查

### Case 1预期输出

```
./motiv_exp_results/case1/
├── ratio_0.50/
│   ├── statistics.txt
│   └── stats_collector.pkl  ← 需要main_approach.py生成
├── ratio_0.60/
├── ...
├── case1_tradeoff.pdf        ← runner生成
├── case1_summary.json        ← runner生成
└── collector_ratio_*.pkl     ← runner缓存
```

### Case 2预期输出

```
./motiv_exp_results/case2/
├── tiles_*_chains_*_load_*/
├── case2_breakdown.pdf       ← runner生成
├── case2_utilization.pdf     ← runner生成
├── case2_summary.json
└── collector_*.pkl
```

### Case 3预期输出

```
./motiv_exp_results/case3/
├── baseline/
├── experiment/
├── case3_baseline.pdf        ← runner生成
├── case3_experiment.pdf      ← runner生成
├── case3_summary.json        ← 包含Δρ对比
└── collector_*.pkl
```

## 6. 待验证/修改清单

### 🔴 必须修改

1. **main_approach.py**: 添加保存`stats_collector`的代码
   ```python
   # 在run_simulation()之后添加
   import pickle
   collector_path = path_ctx.get_output_dir() / 'stats_collector.pkl'
   with open(collector_path, 'wb') as f:
       pickle.dump(stats_collector, f)
   ```

### 🟡 需要验证

2. **参数支持**: 确认`main_approach.py`/`setup_benchmark()`支持：
   - `--num_tiles`
   - `--num_chains`
   - `--load_factor`
   - `--realloc_overhead_enabled`

3. **path_ctx**: 确认`path_ctx.get_output_dir()`返回正确路径

4. **collector初始化**: 确认collector在`instantiate_processors()`中正确初始化

### 🟢 建议改进

5. **错误处理**: `motiv_exp_runner.py`中的`load_collector()`应该有更好的错误处理

6. **进度显示**: 添加进度条或状态更新（特别是Case 2的多个配置）

7. **并行执行**: Case 2可以并行运行不同配置（可选优化）

## 7. 测试建议

### 单元测试

```bash
# 1. 测试dry_run模式
python scripts/motiv_exp_runner.py --case 1 --dry_run

# 2. 测试单个配置（快速验证）
python scripts/motiv_exp_runner.py --case 1 \
    --case1_ratios 0.8 \
    --num_hp 10 \
    --verbose

# 3. 测试Shell包装
./scripts/run_motiv_exps.sh -d 1  # dry run
./scripts/run_motiv_exps.sh -v -n 10 1  # 小规模测试
```

### 集成测试

```bash
# 完整的小规模测试
./scripts/run_motiv_exps.sh -n 50 all
```

### 检查点

- [ ] main_approach.py 能够正常运行
- [ ] stats_collector 能够被保存和加载
- [ ] Case 1能够完成参数扫描并生成图表
- [ ] Case 2能够完成参数扫描并生成两种图表
- [ ] Case 3能够完成对照实验并计算Δρ
- [ ] JSON摘要正确生成
- [ ] Collector缓存文件能够被重新加载和分析

## 8. 下一步行动

1. ✅ **已完成**:
   - Collector专用API（三个case）
   - 绘图函数（三种图表类型）
   - motiv_exp_runner.py实现
   - Shell包装脚本
   - 完整文档

2. 🔴 **立即行动**:
   - 修改`main_approach.py`添加collector保存逻辑
   - 验证参数支持（num_tiles等）

3. 🟡 **后续优化**:
   - 添加单元测试
   - 优化错误处理
   - 添加进度显示
   - 考虑并行化（可选）

4. 🟢 **文档完善**:
   - 添加实际运行示例截图
   - 补充故障排查FAQ
   - 提供论文图表生成指南

