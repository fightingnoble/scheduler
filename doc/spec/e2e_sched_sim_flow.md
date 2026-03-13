# 端到端调度仿真流程设计规范

## 1. 概述

本文档定义调度仿真系统的端到端执行流程规范。系统通过参数化配置生成 workload 和调度信息，支持多种调度策略的消融实验。

main_approach.py 旨在实现，端到端的调度和仿真流程：
1. workload 生成：
    - 根据参数，生成physical graph

2. 根据配置参数，生成调度信息
    - **step 0**（generate_workload_and_criticality）：
        - 生成workload，物理graph
    - **step 1**（deduce_cfg2）：
        - 计算时间片初始分派（per-task 的 deadline）
        - 计算资源需求
        - **所有调度方法都需要 step 1**：包括 cyc, glb, pglb, reserv, cyc-S
        - 由 `exec_t_comp_ratioA` 参数控制初始预留分位数
        - 对于 cyc 相关的消融实验和motiv实验需要用exec_t_comp_ratioA 控制资源总量
        - 实现文档：doc/spec/algorithm/chain_slack_assignment_algorithm.md

    - step 2：bin_split （可选）
    - step 3：repack （可选）
    其中 bin_split 和 repack 封装在perform_bin_packing中，根据num_bins,以及是否need_repack决定是否执行step2和step3；
    - num_bins == 1: 
        - 跳过step2
        - 否则执行step2
    - need_repack == True：
        - 执行step3
        - 否则跳过step3
    
3. 进行随机测试（**step4** ）：
    - run_simulation, 包含 "cyc, glb, pglb, reserv" 四种方法
    - 根据调度信息，进行随机测试
    - 收集统计信息
    - 绘图

代码结构重要说明：

现在的代码由两大块组成：
sim_main.py 和 approach_sim.py相关的两个大块。

  - 之前 sim_main.py 同时支持配置生成和仿真，test_case 用于选择模式
  - 后来添加了 approach_sim.py（event-driven 仿真后端），用来替代sim_main.py，流程中的仿真部分。
  - 两者的流程结合在，main_approach.py 中。
  - 因为原有sim_main.py 相关的代码专注于配置生成，所以 test_case 在main_approach.py 中固定为 'bin_pack_new'
  - 配置生成过程中，具体使用哪一种调度行为由 num_bins, exec_t_comp_ratioA, exec_t_comp_ratioB 参数组合决定。
  - 仿真过程中，具体使用哪一种调度行为由 policy 参数控制（'cyc', 'glb', 'pglb', 'reserv'）

## 2. 流程架构

```
setup_benchmark (approach_setup.py)
    │
    ├── [Phase 1] run_benchmark_setup_pipeline(need_repack=False)
    │       │   args.quantile = ratioA
    │       ├── Step 0-1: build_workload_and_criticality
    │       │       └── deduce_cfg2(quantile=ratioA)
    │       ├── Step 4: create_scheduler_elements_with_config
    │       ├── Step 5: build_simulation_env
    │       ├── Step 6: perform_bin_packing (sim_main.py)
    │       │           └── 返回: (bin_list, max_core_num, glb_p_list, hyper_p)
    │       ├── Step 7: apply_forced_num_cores (仅 non-repack)
    │       └── Step 8-10: generate_bin_paths, Bin_list_print, dump_and_check
    │
    └── [Repack, if ratioB != -1 and ratioA != ratioB]
            run_benchmark_setup_pipeline(need_repack=True)
            │   args.quantile = ratioB
            ├── Step 0-1 re-run: build_workload_and_criticality
            │       └── deduce_cfg2(quantile=ratioB) → 重算 per-task deadline
            ├── Step 6: bypassed — bin_list 保留 Phase 1 布局
            │       └── max_core_num = sum(b.num_resources for b in bin_list)
            └── Step 8-10: generate_bin_paths, Bin_list_print, dump_and_check
```

## 3. 参数定义

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `exec_t_comp_ratioA` | Phase 1 时间片初始分派比例（保守分位数） | 0.99, 0.7 |
| `exec_t_comp_ratioB` | Repack 分位数 — 触发 Step 1 重算任务 deadline；bin 布局不变 | 0.50–0.99 (repack时); -1 = 不 repack |
| `args.quantile` | `deduce_cfg2` 使用的活跃分位数 — Phase 1 设为 ratioA，repack 设为 ratioB | 由 ratioA/B 派生 |
| `num_bins` | 分箱数量 | -1 (cyc), 1 (glb), >1 (pglb) |
| `num_cores` | 强制指定的核心数（可选） | None 或具体数值 |
| `binpack_cfg['algorithm']` | 装箱算法 | "guided" 或 "scratch" |

## 4. 执行流程

### 4.1 是否需要 Repack 判断

```python
if args.exec_t_comp_ratioB != -1 and args.exec_t_comp_ratioA != args.exec_t_comp_ratioB:
    need_repack = True
else:
    need_repack = False
```

> **注意**：触发条件是 `ratioA != ratioB`（非 `ratioA > ratioB`）。只要 ratioB 与 ratioA 不同且 ratioB != -1，即触发 repack。

### 4.2 Non-Repack 阶段 (need_repack=False)

**目标**: 执行装箱，应用资源约束，生成初始调度信息

```
run_benchmark_setup_pipeline(need_repack=False)
    │
    ├── perform_bin_packing(..., need_repack=False)
    │       └── 返回 (bin_list, max_core_num, ...)
    │
    ├── 应用资源约束:
    │   if args.num_cores is not None:
    │       num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
    │   else:
    │       num_cores = max_core_num
    │
    └── dump (使用约束后的 num_cores)
```

**说明**:
- `max_core_num`: 装箱算法计算的资源需求
- `apply_forced_num_cores`: 修改 `bin_list` 中每个 bin 的 `num_resources`，使总和等于 `args.num_cores`
- dump 路径使用约束后的 `num_cores`

### 4.3 Repack 阶段 (need_repack=True)

**目标**: 用 ratioB 重算任务 deadline（Step 1），使仿真中任务的 ERT 和 deadline 更激进。bin 空间布局保持 Phase 1 不变。

```
args.quantile = ratioB                          ← 切换 quantile 为激进分位数
run_benchmark_setup_pipeline(need_repack=True)
    │
    ├── Step 0-1 re-run: build_workload_and_criticality(args)
    │       └── deduce_cfg2(quantile=ratioB)    ← 重算 per-task deadline
    │           注：bin num_resources 从已有 bin_list 加载，不受 quantile 影响
    │
    ├── Step 2/3: bypassed — bin packing 被跳过
    │       └── max_core_num = sum(b.num_resources for b in bin_list)
    │           bin_list 保留 Phase 1 的空间布局
    │
    ├── 不应用资源约束 (repack 不改变资源数量)
    │
    └── dump (使用 non-repack 阶段约束后的 num_cores)
```

**说明**:
- `args.quantile = ratioB` 是正确行为：repack 的核心目的是用更激进的分位数重算 `deduce_cfg2`，改变任务的时间片分配（deadline 更早、ERT 更早）
- bin 的 `num_resources` 在 repack 阶段从已有 `bin_list` 读取，不随 quantile 变化 — 因此不会出现资源不匹配
- 装箱算法（`perform_bin_packing`）被完全跳过，bin_list 的 scheduling_table 保留 Phase 1 结果
- TODO: 未来可为 reserv 策略启用 `push_task_into_bins_new`，在固定空间布局下微调 ERT/deadline

## 5. 调度策略配置

### 5.1 策略对照表

| 策略 | 预留 | 隔离 | 动态时间共享 | 动态空间共享 | 配置 |
|------|:----:|:----:|:-----------:|:-----------:|------|
| cyc | √ | √ | × | × | `num_bins=-1` (最多分区), 无 repack |
| glb | × | × | √ | √ | `num_bins=1` (单分区), 无 repack |
| reserv | √ | √ | √ | √ | `num_bins >= 2` (>=1分区), 有 repack（理论扫描可含 `-1`；脚本默认扫描 `num_bins ∈ [1,2,4,8]`） |
| pglb | × | √ | √ | √ | `num_bins>1` (多分区), 无 repack |
| cyc-S | √ | √ | √ | × | `num_bins=-1` (最多分区), 有 repack (软预留) |

### 5.2 执行路径

```
cyc:     step0-1-2 (guided non-repack)
glb:     step0-1 (scratch)
reserv:  step0-1-2 + repack(step0-1 re-run with ratioB, bin packing bypassed)
pglb:    step0-1-2 (guided non-repack, num_bins>1)
cyc-S:   step0-1-2 + repack(step0-1 re-run with ratioB, bin packing bypassed)
```

> **repack 说明**：repack 阶段用 `args.quantile = ratioB` 重跑 Step 0-1（`deduce_cfg2`），重算任务 deadline。装箱算法被跳过，bin_list 保留 Phase 1 空间布局。bin 的 `num_resources` 从已有 bin_list 加载，不随 quantile 变化。

### 5.3 策略间的特例/退化关系

reserv 是统一框架，其余策略均可视为 reserv 在特定参数配置下的特例（参见 paper.tex §Scheduling Space）：

```
reserv (隔离 + 预留 + 动态时间/空间共享)
  │
  ├─ 去掉 repack (exec_t_comp_ratioB = -1)
  │    └─→ pglb (隔离 + 动态共享, 无预留)
  │          │
  │          └─ num_bins = 1 (去掉隔离)
  │               └─→ glb (纯动态调度)
  │
  └─ num_bins = -1 (最多分区, 去掉动态空间共享)
       └─→ cyc-S (最多分区 + 软预留)
             │
             └─ 去掉 repack (exec_t_comp_ratioB = -1)
                  └─→ cyc (纯静态调度)
```

**参数退化**（key_COT.md）：
- 预留参数中所有任务 `t_start = 0` 时，调度时机退化为 as soon as possible → glb 行为
- 预留参数中所有任务 `t_start` 极晚时，调度器退化为串行执行 → cyc 行为

**实验中的退化**（ABLA_EXP_FIX_PLAN.md）：
- 消融实验3中 `num_bins = 1` 时，pglb 退化为 glb，需使用 `policy='glb'`

## 6. 函数职责

### 6.1 perform_bin_packing (sim_main.py)

**职责**: 执行装箱算法

**参数**:
- `args`, `glb_p_list`, `num_cores`, `bin_list`, `hyper_p`
- `event_iter_dict`, `quantumSize`, `num_periods`
- `physical_graph_nx`, `need_repack`
- `scheduler_list`, `monitor_list`, `msg_dispatcher`, `a_data_pipe`, `w_data_pipe`

**返回**: `(bin_list, max_core_num, glb_p_list, hyper_p)`
- `bin_list`: 装箱后的 bin 列表
- `max_core_num`: 装箱计算的资源需求（未应用约束）

**不负责**:
- 资源约束（由外层 `apply_forced_num_cores` 处理）
- dump（由外层处理）

### 6.2 apply_forced_num_cores (sim_main.py)

**职责**: 强制调整资源数量

**行为**:
```python
def apply_forced_num_cores(bin_list, estimated_num_cores, target):
    if target == estimated_num_cores:
        return estimated_num_cores
    vectorized_core_allocation(bin_list, target)  # 修改 bin_list
    return target
```

**说明**: 修改 `bin_list` 中每个 bin 的 `num_resources`，使总和等于 `target`

### 6.3 run_benchmark_setup_pipeline (approach_setup.py)

**职责**: 编排 benchmark 设置流程

**流程**:
1. 生成 workload 并设置 criticality（Step 0-1，使用 `args.quantile`）
2. 创建调度器元素
3. 构建仿真环境参数
4. 执行装箱算法（`perform_bin_packing`）— **仅 non-repack 时执行；repack 时跳过**
5. 应用资源约束（仅 non-repack）
6. 生成 dump 路径
7. 打印和绘制
8. Dump

**repack 时的行为差异**:
- `args.quantile` 已由调用方设为 `ratioB`
- Step 0-1 用 ratioB 重算 `deduce_cfg2` → 任务 deadline 更激进
- `perform_bin_packing` 被跳过 → `bin_list` 保留 Phase 1 布局
- `max_core_num = sum(b.num_resources for b in bin_list)`（从已有 bin_list 读取）
- 不应用 `apply_forced_num_cores`

**返回**: `(hyper_p, bin_list, num_cores)`

## 7. 关键约束

1. **资源约束只在 non-repack 时应用一次**
2. **Repack 用 `args.quantile = ratioB` 重跑 Step 0-1** — 重算任务 deadline/FLOPS；bin `num_resources` 从已有 bin_list 加载，不受 quantile 影响
3. **Repack 跳过装箱算法 (Step 2/3)** — bin_list 保留 Phase 1 空间布局；TODO: 未来为 reserv 启用 `push_task_into_bins_new`
4. **Repack 触发条件: `ratioB != -1 and ratioA != ratioB`** — 只要 ratioB 与 ratioA 不同即触发
5. **Dump 路径使用约束后的 `num_cores`**
6. **`num_cores` 始终等于 `sum(b.num_resources for b in bin_list)`**


## 8. 实现

### motivation 实验
 脚本：
 ```bash
  python -m scripts.motiv_exp_runner --case 1 --output_dir motiv_exp_results --num_hp 100 --use_plot_cache    
  python -m scripts.motiv_exp_runner --case 2 --output_dir motiv_exp_results --num_hp 100 --use_plot_cache    
  python -m scripts.motiv_exp_runner --case 3 --output_dir motiv_exp_results --case3_num_periods 200          
 ```
 这三个实验标准流程是已经验证过的，并确认是功能良好的。复现流程已经固化在脚本scripts/run_motiv_exps.sh 中。实现的步骤记录在 doc/spec/test_plan.md。

### 消融实验：

1. 实验方案
    实验设计主要思想在于将静态调度分解为两种机制：1. 时间维度上的预留机制 2. 空间维度上的隔离机制
    预留机制对应于，时间维度的分配，由时间片调整repack（step3）实现，参数exec_t_comp_ratioB；特别的，默认所有case有一个时间片初始分派（step1），比例为exec_t_comp_ratioA；
    隔离机制对应于，空间维度的分配，在bin_split（step2），参数为num_bins；默认初始状态为一个分区。
    |case | 预留 | 隔离 | 动态 time sharing| 动态 space sharing|
    |-----|------|------|------|------|
    |cyc | √ | √| × | × |
    |glb|× | × | √ | √ |
    |reserv | √ | √ | √ | √ |
    |pglb | × | √ | √ | √ |
    |cyc-S | √ | √ | √ | × |

    有如下机制组合：
    - 所有case 都需执行step0和step1，并确定资源总量，以及初始时间片分派
    - Baseline:
        - cyc: 设置num_bins = -1（最多分区），额外执行step2，获取分组信息 (step0-1-2)
        - glb: 无需额外调度信息 (step0-1)
    - reserv: 同cyc，但是需要额外执行step3，进行repack (step0-1-2 + repack), 具有任意的num_bins
    - 消融实验：
        - cyc-S: 同cyc执行step2，然后使用软预留进行repack (step0-1-2 + repack)
        - pglb: 设置num_bins > 1，额外执行step2，获取分组信息 (step0-1-2)

2. 脚本abla_exp_runner.py 脚本已经实现完成。以下是实现的摘要：

  1. scripts/abla_exp_runner.py - 消融实验运行脚本
  2. scripts/run_abla_exps.sh - 便捷运行脚本

  实现的三个 Runner 类

  AblaExp1Runner (cyc-S vs cyc)

  - 目的：证明引入时间维度的共享（软预留）能平衡利用率和出错概率
  - 设置：
    - cyc: 扫描 exec_t_comp_ratioA ∈ [0.5, 0.7, 0.99]（硬隔离）
    - cyc-S: 固定 exec_t_comp_ratioA = 0.7，扫描 exec_t_comp_ratioB ∈ [0.5, 0.99]（软预留）
  - 绘图：复用 plot_motiv_case1，并添加 cyc 参考点作为水平线投影

  AblaExp2Runner (pglb vs glb)

  - 目的：证明引入 Spatial Partitioning 能平衡利用率和切换开销
  - 设置：
    - glb: num_bins=1（无隔离）
    - pglb: num_bins ∈ [2, 4, 8]（有隔离）
    - 扫描: tiles ∈ [200, 400], chains ∈ [1, 4], load_factor ∈ [0.5, 1.0]
  - 绘图：切换次数和切换开销对比图

  AblaExp3Runner (reserv vs pglb)

  - 目的：证明调整分箱和 repack，能在规模变化时维持较优范围
  - 设置：
    - pglb: 多 bins 无 repack（脚本默认 num_bins ∈ [1, 2, 4, 8]）
    - reserv: 扫描 exec_t_comp_ratioB ∈ [0.5, 0.99] 和 num_bins ∈ [1, 2, 4, 8]
    - 扫描: tiles ∈ [200, 400], chains ∈ [1, 4], load_factor ∈ [0.5, 1.0]
    - 负载强度模式：默认多组扫描，同时支持固定单一负载强度模式
  - 绘图：复用 plot_motiv_case2 绘制延迟分解和利用率图

  使用方法

  # 运行所有消融实验
  bash scripts/run_abla_exps.sh [输出目录] [超周期数]

  # 或单独运行某个实验
  python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
  python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
  python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100

  # 使用缓存重新绘图（不重新运行仿真）
  python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --use_plot_cache

### 8.3 最新进展（2026-03-11）

**消融实验脚本状态**：
- ✅ `scripts/abla_exp_runner.py` 已实现，包含三个 Runner 类
- ✅ `scripts/run_abla_exps.sh` 便捷运行脚本已创建
- ⏳ 待测试验证（建议先用 `--dry_run` 检查参数生成）

**实际参数配置（与设计文档对齐）**：
- Ablation-1: `ratioB ∈ [0.5,0.9]`，cyc参考点 `ratioA ∈ [0.5,0.7,0.99]`
- Ablation-2: `num_bins ∈ [1,2,4,8]`，扫描 tiles/chains/load_factor
- Ablation-3: reserv 扫描 `ratioB ∈ [0.5,0.9]` + `num_bins ∈ [1,2,4,8,-1]`

**进度条**：
```
消融实验框架: ████████████████████ 100% (脚本已实现)
参数定义对齐: ████████████████████ 100% (文档已同步)
测试验证:     ░░░░░░░░░░░░░░░░░░░░   0% (待执行)
```