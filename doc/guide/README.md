# 代码库理解指南

本目录包含代码库探索生成的模块化理解文档，帮助快速理解各部分代码。

## 文档索引

| 模块 | 文档 | 描述 |
|------|------|------|
| 架构总览 | [architecture_overview.md](./architecture_overview.md) | 高层架构、目录结构、调度策略 |
| 实验脚本 | [exp_scripts_overview.md](./exp_scripts_overview.md) | motiv/abla实验运行器 |
| 调度核心 | [sched_core_overview.md](./sched_core_overview.md) | 两阶段装箱算法 |
| 仿真流程 | [simulation_flow_overview.md](./simulation_flow_overview.md) | 端到端流程图 |
| 统计收集 | [collector_overview.md](./collector_overview.md) | StatisticsCollector |
| 绘图函数 | [plotting_overview.md](./plotting_overview.md) | 全仓绘图函数总览（matplotlib 统一后端） |
| 任务模型 | [task_model_overview.md](./task_model_overview.md) | ProcessInt生命周期 |
| 废弃代码 | [deprecated_code.md](./deprecated_code.md) | 废弃模块记录 |

## 快速开始

### 环境设置
```bash
conda activate gurobi
```

### 运行实验
```bash
# 动机实验
python -m scripts.motiv_exp_runner --case 1 --output_dir motiv_exp_results --num_hp 100
python -m scripts.motiv_exp_runner --case 2 --output_dir motiv_exp_results --num_hp 100
python -m scripts.motiv_exp_runner --case 3 --output_dir motiv_exp_results --case3_num_periods 200

# 消融实验
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
```

## 关键入口点

| 入口 | 文件 | 职责 |
|------|------|------|
| 主入口 | `main_approach.py` | 端到端流程编排 |
| 基准设置 | `approach_setup.py` | workload生成和装箱 |
| 仿真运行 | `approach_sim.py` | 事件驱动仿真 |
| 装箱算法 | `sched/global_sched.py` | 两阶段装箱 |

## 调度策略

| 策略 | 预留 | 隔离 | 动态共享 | 配置 |
|------|:----:|:----:|:--------:|------|
| cyc | √ | √ | × | `num_bins=-1`, 无repack |
| glb | × | × | √ | `num_bins=1`, 无repack |
| reserv | √ | √ | √ | `num_bins>=2`, 有repack |
| pglb | × | √ | √ | `num_bins>1`, 无repack |
| cyc-S | √ | √ | √ | `num_bins=-1`, 有repack |

## 文档层次

```
doc/spec/key_COT.md (最高 - 学术逻辑)
    ↓
doc/spec/e2e_sched_sim_flow.md (设计规范)
    ↓
doc/spec/test_plan.md (实现细节)
    ↓
doc/guide/*.md (本目录 - 模块概览)
```

---
*生成时间: 2026-03-11*
