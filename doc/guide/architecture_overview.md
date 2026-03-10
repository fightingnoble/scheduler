# 代码库架构总览

本文档提供scheduler代码库的高层架构视图，帮助快速定位和理解各模块。

## 项目定位

**实时任务调度仿真器** - 用于多核嵌入式系统（如自动驾驶）的调度算法评估。

### 核心问题
- DAG任务图的端到端延迟约束
- 执行时间变异性（随机性，不同分位数）
- 利用率、延迟满足率、调度开销的权衡

## 目录结构

```
scheduler/
├── main_approach.py          # 主入口点
├── approach_setup.py         # 基准设置流程
├── approach_sim.py           # 事件驱动仿真后端
├── approach_collector.py     # 统计收集器
├── sim_main.py               # 仿真主模块（装箱流程）
│
├── sched/                    # 调度核心模块
│   ├── global_sched.py       # 两阶段装箱算法
│   ├── pre_alloc_new.py      # 预分配逻辑
│   ├── slack_estim.py        # Slack估算
│   ├── binpack_config.py     # 装箱配置
│   ├── scheduler_agent.py    # [废弃] 旧版运行时调度器
│   └── monitor_agent.py      # [废弃] 旧版监控代理
│
├── task/                     # 任务模型模块
│   ├── task_agent.py         # ProcessInt任务类
│   ├── task_cfg.py           # 任务配置
│   ├── graph_scaling.py      # 图缩放
│   └── load_cfg/             # 负载配置
│
├── scripts/                  # 实验脚本
│   ├── motiv_exp_runner.py   # 动机实验
│   ├── abla_exp_runner.py    # 消融实验
│   ├── exp_common.py         # 共享组件
│   └── run_*.sh              # 运行脚本
│
├── doc/                      # 文档
│   ├── spec/                 # 规范文档
│   │   ├── key_COT.md        # 学术逻辑（最高优先级）
│   │   ├── e2e_sched_sim_flow.md
│   │   └── test_plan.md
│   └── guide/                # 代码库理解指南 (doc/guide/)
│
├── model/                    # 系统模型
│   ├── barrier_agent.py      # 屏障代理
│   ├── message/              # 消息传递
│   └── streaming_processing/ # 流处理
│
├── allocator_agent.py        # 资源分配代理
├── global_var.py             # 全局变量
├── paths.py                  # 路径解析
└── utils.py                  # 工具函数
```

## 核心流程

### 端到端仿真流程

```
main_approach.py::main()
    │
    └── approach_setup.py::setup_benchmark()
            │
            ├── Step 0-1: build_workload_and_criticality()
            │       └── 生成workload，计算初始时间片分派
            │
            ├── Step 2: perform_bin_packing() [non-repack]
            │       └── coleasing_alloc_cluster() - 空间分区
            │
            ├── Step 3: perform_bin_packing() [repack, 可选]
            │       └── push_task_into_bins_new() - 时间调度
            │
            └── Step 4: approach_sim.py::run_simulation()
                    └── 事件驱动仿真
```

### 调度策略对照

| 策略 | 预留 | 隔离 | 动态时间共享 | 动态空间共享 | 配置 |
|------|:----:|:----:|:-----------:|:-----------:|------|
| cyc | √ | √ | × | × | `num_bins=-1`, 无repack |
| glb | × | × | √ | √ | `num_bins=1`, 无repack |
| reserv | √ | √ | √ | √ | `num_bins>=2`, 有repack |
| pglb | × | √ | √ | √ | `num_bins>1`, 无repack |
| cyc-S | √ | √ | √ | × | `num_bins=-1`, 有repack |

## 关键参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `exec_t_comp_ratioA` | Phase 1时间片分派（保守） | 0.99, 0.7 |
| `exec_t_comp_ratioB` | Phase 2时间片分派（激进） | 0.80, 0.5 |
| `num_bins` | 空间分区数 | -1(自动), 1(无), >=2(有) |
| `num_cores` | 强制核心数（可选） | None或具体值 |
| `policy` | 仿真调度策略 | cyc/glb/pglb/reserv |

## 两阶段装箱算法

### Phase 1: 空间分区 (Split)
- **函数**: `coleasing_alloc_cluster()`
- **参数**: `exec_t_comp_ratioA` (保守分位数，如0.99)
- **输出**: `bin_list` (任务到bin的映射)

### Phase 2: 时间调度 (Repack)
- **函数**: `push_task_into_bins_new()`
- **参数**: `exec_t_comp_ratioB` (激进分位数，如0.80)
- **输出**: 每个bin内的静态时间窗口

## 实验设计

### 动机实验
1. **Case 1**: 静态调度利用率问题 - 扫描预留分位数
2. **Case 2**: 动态调度可扩展性问题 - 扫描负载规模
3. **Case 3**: 切换行为不确定性 - 负载-延迟关系

### 消融实验
1. **Case 1**: cyc-S vs cyc - 预留在串行执行下的影响
2. **Case 2**: pglb vs glb - 隔离的作用
3. **Case 3**: reserv vs pglb - 预留在并行下的影响

## 废弃代码

参见 [deprecated_code.md](./deprecated_code.md)

## 文档层次

| 文档 | 优先级 | 内容 |
|------|--------|------|
| `doc/spec/key_COT.md` | ⭐⭐⭐ | 学术逻辑 |
| `doc/spec/e2e_sched_sim_flow.md` | ⭐⭐ | 设计规范 |
| `doc/spec/test_plan.md` | ⭐ | 实现细节 |
| `doc/guide/*.md` | 参考 | 模块概览 |

---
*生成时间: 2026-03-11*
