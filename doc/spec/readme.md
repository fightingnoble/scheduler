# doc/spec/ 规范文档索引

> 本文档是 `CLAUDE.md` 的下一级索引，提供 `doc/spec/` 目录下所有规范文档的导航，以及跨文档的关键约定和澄清事项。

## 1. 核心文档层级

三个核心文档层层细化，**后面向前面对齐**。不一致时以前面的文件为准。

| 文档 | 定位 | 内容 | 修改时机 |
|------|------|------|----------|
| [key_COT.md](./key_COT.md) | 学术表达（最高层次） | 核心论点、机制解耦、消融实验的学术动机 | 论文思路变更时 |
| [e2e_sched_sim_flow.md](./e2e_sched_sim_flow.md) | 设计规范（中间层） | Step 定义、参数定义、执行路径、函数职责、关键约束 | 代码架构变更时 |
| [test_plan.md](./test_plan.md) | 实现相关（最具体） | 详细实验参数、绘图方法、脚本对齐、统计收集方式 | 实验参数/脚本变更时 |

**对齐规则**：
- key_COT 是学术表达，为权威来源
- e2e 补充 key_COT 未涉及的技术细节（Step 定义、函数签名）
- test_plan 补充 e2e 未涉及的实现细节（脚本参数、绘图 API、缓存机制）
- 冲突时一律以上层文档为准

## 2. 全部文档目录

### 2.1 核心规范（实验设计直接相关）

| 文档 | 内容摘要 |
|------|----------|
| [key_COT.md](./key_COT.md) | 论文核心思路：静态/动态调度的局限 → 解耦为预留+隔离 → 三个消融实验 |
| [e2e_sched_sim_flow.md](./e2e_sched_sim_flow.md) | Step 0-1-2-3-4 定义、策略对照表、perform_bin_packing/apply_forced_num_cores 职责、关键约束 |
| [test_plan.md](./test_plan.md) | Motiv 1/2/3 + 消融 1/2/3 + 端到端实验的完整参数、绘图、脚本对齐 |
| [motiv1_describe.md](./motiv1_describe.md) | Motiv1 绘图细节（柱状图双轴设计） |
| [motiv2_describe.md](./motiv2_describe.md) | Motiv2 绘图细节（延迟分解 + 利用率 breakdown） |

### 2.2 算法规范

| 文档 | 内容摘要 |
|------|----------|
| [algorithm/guided_hybrid_allocation_algorithm.md](./algorithm/guided_hybrid_allocation_algorithm.md) | 两阶段分配算法：Split (`coleasing_alloc_cluster`) + Repack (`push_task_into_bins_new`) |
| [algorithm/chain_slack_assignment_algorithm.md](./algorithm/chain_slack_assignment_algorithm.md) | Step 1 实现：链松弛分配，计算 per-task deadline |
| [algorithm/binpack_solver_spec.md](./algorithm/binpack_solver_spec.md) | 装箱启发式求解器规范 |
| [algorithm/task_layout_plotting.md](./algorithm/task_layout_plotting.md) | 任务布局绘图（bin_list_utils）原理：时间-空间甘特图 + 一维装箱位置分配 |

### 2.3 统计收集规范

层次关系：底层算法 → 业务集成 → 扩展指南 → 运行时测量

```
stat/tdigest_system_spec.md        (T-Digest 流式直方图算法)
        │ 被使用
        v
stat/statistics_collection_spec.md (StatisticsCollector 架构、分布存储、仿真集成)
        │ 提供 API
        v
stat/collector.md                  (E2E 延迟分解、关键路径分析、扩展指南)
        │
        v
stat/runtime_overhead_spec.md      (Algorithm 2 运行时开销测量、审稿回复)
```

| 文档 | 内容摘要 |
|------|----------|
| [stat/tdigest_system_spec.md](./stat/tdigest_system_spec.md) | T-Digest 算法原理、TDigestStreamingHistogram 接口、参数配置 |
| [stat/statistics_collection_spec.md](./stat/statistics_collection_spec.md) | StatisticsCollector 架构、分布存储结构、记录/摘要/绘图 API |
| [stat/collector.md](./stat/collector.md) | **权威文档**：E2E 延迟分解（关键路径分析）、扩展指南 |
| [stat/runtime_overhead_spec.md](./stat/runtime_overhead_spec.md) | **新增**：Algorithm 2 运行时开销测量、关键路径判断、审稿回复要点 |

### 2.4 仿真规范

| 文档 | 内容摘要 |
|------|----------|
| [sim/approach_sim_spec.md](./sim/approach_sim_spec.md) | Event-driven 仿真后端规范 |
| [sim/simulation_numerical_design.md](./sim/simulation_numerical_design.md) | 仿真数值精度与公式分离设计 |

### 2.5 配置与模型

| 文档 | 内容摘要 |
|------|----------|
| [cfg/binpack_config_design.md](./cfg/binpack_config_design.md) | BinPackConfig 参数配置类设计 |
| [model/latency_model.md](./model/latency_model.md) | 访存拥塞延迟建模（排队论与概率分布） |

### 2.6 其他

| 文档 | 内容摘要 |
|------|----------|
| [e2e_flow_refactor.md](./e2e_flow_refactor.md) | 历史重构记录（benchmark 生成与测试串联） |
| [paper.tex](./scheduler_paper/main.tex) | 论文 LaTeX 源文件 |

## 3. 实验设计速查

### 3.1 调度策略配置

| 策略 | 预留 | 隔离 | 动态时间共享 | 动态空间共享 | 常规 num_bins | repack |
|------|:----:|:----:|:-----------:|:-----------:|:-------------:|:------:|
| cyc | √ | √ | × | × | -1（最多分区） | × |
| glb | × | × | √ | √ | 1（单分区） | × |
| pglb | × | √ | √ | √ | >1（多分区） | × |
| reserv | √ | √ | √ | √ | >= 2（多分区） | √ |
| cyc-S | √ | √ | √ | × | -1（最多分区） | √（软预留） |

### 3.2 消融实验参数速查

| 实验 | 对比 | 验证机制 | 扫描参数 | 负载强度 | 参考线 |
|------|------|----------|----------|----------|--------|
| Exp 1 | cyc-S vs cyc | 预留（串行） | `ratioB ∈ [0.5, 0.99]` | 固定 | cyc 在 `ratioA=0.5,0.7,0.99` 的结果投影 Y 轴 |
| Exp 2 | pglb vs glb | 隔离 | `num_bins ∈ [1]+range(2,max,2)` | 同 Motiv2 | glb 作为 `num_bins=1` 的基线 |
| Exp 3 | reserv vs pglb | 预留+隔离 | `ratioB ∈ [0.5, 0.99]` + `num_bins ∈ [1,2,4,8,-1]` | 同 Motiv2 | pglb 作为无 repack 基线 |

### 3.3 当前进度

**Motivation 实验**：已验证完成，复现流程固化在 `scripts/run_motiv_exps.sh`

```bash
python -m scripts.motiv_exp_runner --case 1 --output_dir motiv_exp_results --num_hp 100
python -m scripts.motiv_exp_runner --case 2 --output_dir motiv_exp_results --num_hp 100
python -m scripts.motiv_exp_runner --case 3 --output_dir motiv_exp_results --case3_num_periods 200
```

**消融实验**：脚本 `scripts/abla_exp_runner.py` 已实现，运行脚本 `scripts/run_abla_exps.sh`

```bash
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results --num_hp 100
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results --num_hp 100
```

通用选项：`--use_plot_cache` 跳过仿真，仅从缓存重绘

## 4. 关键澄清与约定

> 本节记录跨文档讨论中确认的重要约定，避免后续修改时重复犯错。

### 4.1 glb 需要 Step 1

glb "无需额外调度信息" 指的是**不需要分区、预留、静态表**（不执行 Step 2/3）。但 per-task 的 deadline（Step 1 给出）仍然需要，因此 glb 必须指定 `exec_t_comp_ratioA` 来决定初始时间片分配。**所有调度策略都需要执行 Step 0 和 Step 1**。

### 4.2 reserv 的双机制特性

reserv 同时具备**空间隔离**（`num_bins`）和**时间预留**（`exec_t_comp_ratioB`）两种可调机制：
- **常规配置**：`num_bins >= 2`（多分区）
- **消融实验 3**：扫描 `num_bins ∈ [1, 2, 4, 8, -1]`，覆盖从单分区到最多分区的全部范围
- `-1` 的含义：最多分区/自动搜索（`n_partition=9999`，保证每个任务声明周期不重叠）

### 4.3 消融实验 2/3 的资源控制

消融实验 2 (pglb vs glb) 和消融实验 3 (reserv vs pglb) 的资源控制方式：
- **不使用** `exec_t_comp_ratioA` 进行资源控制
- 通过与 Motiv2 相同的**负载强度**（负载规模/硬件规模）组合控制
- 内部会触发 `force_num_cores` 逻辑来匹配资源需求

### 4.4 消融实验间的参数复用关系

- 消融实验 2 和 3 使用与 Motiv2 **相同的负载规模扫描**
- 消融实验 3 与消融实验 1 扫描**相同的预留系数** (`exec_t_comp_ratioB`)
- 消融实验 3 与消融实验 2 扫描**相同的分箱数量** (`num_bins`)

### 4.5 cyc-S 需要 repack

cyc-S 不是"无 repack 的 cyc"——它是在 cyc 基础上将硬隔离替换为软预留，**需要执行 Step 3 (repack)**。执行路径为 step0-1-2 + step3。

### 4.6 消融实验 1 额外绘图需求

除了复用 Motiv1 的绘图函数外，需要将 cyc（硬隔离）在不同 `exec_t_comp_ratioA` 值（0.5, 0.7, 0.99）下的**延迟满足率投影在 Y 轴上**作为参考线，体现软预留相对于硬预留的可靠性提升。
