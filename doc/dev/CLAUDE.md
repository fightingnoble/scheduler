# Claude 开发指南

> 本文档是 Claude Code 开发助手的快速参考索引，帮助快速理解项目结构和定位代码。

---

## 1. 项目概览

**核心问题**：为多核嵌入式系统（如自动驾驶）设计实时任务调度器，评估静态、动态和混合调度算法在执行时间波动下的表现。

**调度策略速查**：

| 策略 | 预留 | 隔离 | 动态时间共享 | 动态空间共享 | num_bins | repack |
|------|:----:|:----:|:-----------:|:-----------:|:--------:|:------:|
| cyc | √ | √ | × | × | -1（最多分区） | × |
| glb | × | × | √ | √ | 1（单分区） | × |
| pglb | × | √ | √ | √ | >1（多分区） | × |
| reserv | √ | √ | √ | √ | >= 2 | √ |
| cyc-S | √ | √ | √ | × | -1 | √（软预留） |

---

## 2. 核心文档索引

### 2.1 规范文档（doc/spec/）

| 文档 | 层级 | 用途 | 何时参考 |
|------|------|------|----------|
| [key_COT.md](../spec/key_COT.md) | 学术表达 | 核心论点、机制解耦、消融动机 | 论文思路变更时 |
| [e2e_sched_sim_flow.md](../spec/e2e_sched_sim_flow.md) | 设计规范 | Step 定义、参数定义、执行路径 | 代码架构变更时 |
| [test_plan.md](../spec/test_plan.md) | 实现细节 | 实验参数、绘图方法、脚本对齐 | 实验/脚本变更时 |
| [readme.md](../spec/readme.md) | 索引 | 规范文档导航、关键约定 | 查找文档时 |

### 2.2 开发文档（doc/dev/）

| 文档 | 用途 |
|------|------|
| [simulation_flow.md](./simulation_flow.md) | 仿真主流程详解 |
| [experiment_system.md](./experiment_system.md) | 实验系统架构 |
| [configuration_system.md](./configuration_system.md) | 配置系统详解 |
| [result_collection.md](./result_collection.md) | 结果收集与绘图 |

### 2.3 变更日志（doc/dev/）

| 文档 | 内容 |
|------|------|
| [change_log_2025.md](./change_log_2025.md) | 2025 年代码变更记录 |
| [change_log_2026.md](../change_log_2026.md) | 2026 年代码变更记录 |

---

## 3. 代码结构速查

### 3.1 入口点

```
main_approach.py          # 主入口：单次仿真
scripts/motiv_exp_runner.py  # Motivation 实验
scripts/abla_exp_runner.py   # 消融实验
```

### 3.2 核心模块

```
approach_setup.py         # setup_benchmark() - 基准测试设置流水线
sim_main.py               # perform_bin_packing() - 装箱算法核心
approach_sim.py           # run_simulation() - 事件驱动仿真运行时
approach_collector.py     # StatisticsCollector - 统计收集
sched/global_sched.py     # coleasing_alloc_cluster(), push_task_into_bins_new()
sched/binpack_config.py   # BinPackConfig - 配置类
```

### 3.3 关键函数定位

| 功能 | 文件:行号 | 说明 |
|------|-----------|------|
| 基准设置入口 | approach_setup.py:90 | `setup_benchmark()` |
| 装箱算法入口 | sim_main.py:397 | `perform_bin_packing()` |
| Phase 1 (Split) | sched/global_sched.py | `coleasing_alloc_cluster()` |
| Phase 2 (Repack) | sched/global_sched.py:42 | `push_task_into_bins_new()` |
| 仿真运行时 | approach_sim.py:43 | `run_simulation()` |
| 统计收集器 | approach_collector.py:40 | `StatisticsCollector` 类 |

---

## 4. 仿真流程速查

```
runner.py
    │
    └── run_main_approach_inproc() [scripts/exp_common.py:99]
            │
            └── main() [main_approach.py]
                    │
                    ├── setup_benchmark() [approach_setup.py:90]
                    │       ├── build_workload_and_criticality()
                    │       ├── create_scheduler_elements_with_config()
                    │       ├── build_simulation_env()
                    │       ├── perform_bin_packing() [sim_main.py:397]
                    │       │       ├── coleasing_alloc_cluster() [Phase 1]
                    │       │       └── push_task_into_bins_new() [Phase 2]
                    │       └── apply_forced_num_cores() [非 repack 时]
                    │
                    └── run_simulation() [approach_sim.py:43]
```

---

## 5. 参数流向速查

| 参数 | 流向 | 作用 |
|------|------|------|
| `exec_t_comp_ratioA` | args → `coleasing_alloc_cluster(quantile=...)` | Phase 1 资源估计用的保守分位数 |
| `exec_t_comp_ratioB` | args → `push_task_into_bins_new(quantile=...)` | Phase 2 时间窗分配用的激进分位数 |
| `num_bins` | args → `coleasing_alloc_cluster(n_partition=...)` | 空间分区数量 |
| `num_cores` | args → `apply_forced_num_cores()` | 资源约束（仅非 repack） |

---

## 6. 实验系统速查

### 6.1 Motivation 实验

| Case | 验证问题 | 扫描参数 |
|------|----------|----------|
| Case 1 | 纯静态调度的利用率问题 | `exec_t_comp_ratioA ∈ [0.5, 0.99]` |
| Case 2 | 纯动态调度的延迟开销问题 | `tiles × chains × load_factor` |
| Case 3 | 切换行为的不确定性 | 负载-延迟相关性 |

### 6.2 消融实验

| Case | 对比 | 验证机制 | 扫描参数 |
|------|------|----------|----------|
| Exp 1 | cyc-S vs cyc | 预留（串行） | `ratioB ∈ [0.5, 0.99]` |
| Exp 2 | pglb vs glb | 隔离 | `num_bins ∈ [1, 2, 4, 8]` |
| Exp 3 | reserv vs pglb | 预留+隔离 | `ratioB × num_bins` 网格 |

### 6.3 运行命令

```bash
# Motivation 实验
python -m scripts.motiv_exp_runner --case 1 --output_dir ./motiv_results --num_hp 100
python -m scripts.motiv_exp_runner --case 2 --output_dir ./motiv_results --num_hp 100
python -m scripts.motiv_exp_runner --case 3 --output_dir ./motiv_results --case3_num_periods 200

# 消融实验
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100

# 从缓存重绘
python -m scripts.motiv_exp_runner --case 1 --use_plot_cache
```

---

## 7. 统计收集速查

### 7.1 核心方法

| 方法 | 用途 | 输出 |
|------|------|------|
| `record_task_finish()` | 任务完成时调用 | 更新任务级分布 |
| `record_e2e_finish()` | 链完成时调用 | 更新链级分布 |
| `forward_hyperperiod()` | 周期边界调用 | 归一化并累积到系统级分布 |
| `get_motiv_case1_stats()` | Case 1 统计 | 利用率指标 |
| `get_motiv_case2_stats()` | Case 2 统计 | 延迟分解 |
| `get_motiv_case3_stats()` | Case 3 统计 | 负载-延迟相关性 |

### 7.2 绘图方法

| 方法 | 位置 | 用途 |
|------|------|------|
| `plot_motiv_case1()` | approach_collector.py:714 | 利用率-可靠性权衡图 |
| `plot_motiv_case2()` | approach_collector.py:951 | 可扩展性分析图 |

---

## 8. 常见问题排查

| 问题 | 可能原因 | 排查位置 |
|------|----------|----------|
| 资源不足异常 | `num_cores` 约束过紧 | `apply_forced_num_cores()` |
| 结果全相同 | repack 未生效或缓存问题 | 检查 `exec_t_comp_ratioB` 值 |
| 参数未传递 | `binpack_cfg` 更新遗漏 | `utils.py:build_path_old()` |
| 统计数据缺失 | `forward_hyperperiod()` 未调用 | 仿真主循环周期边界 |

---

## 9. 开发环境

```bash
# 激活环境
conda activate gurobi

# 验证 Gurobi
python -c "import gurobipy; print('OK')"

# 运行测试
python -m pytest tests/
```

---

## 10. 文档更新规则

1. **代码变更时**：更新 `doc/change_log_2026.md`
2. **架构变更时**：更新 `doc/spec/e2e_sched_sim_flow.md`
3. **实验参数变更时**：更新 `doc/spec/test_plan.md`
4. **新增模块时**：在本文件添加索引
