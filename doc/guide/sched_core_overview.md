# 调度核心模块概览 (sched/)

本文档概述调度核心模块的设计思想、核心算法流程和关键实现。

## 1. 模块架构

```
sched/
├── global_sched.py        # 核心装箱算法 (两阶段分配)
├── pre_alloc_new.py       # 运行时资源分配
├── slack_estim.py         # Slack 估算 (时间片分配)
├── binpack_config.py      # 装箱配置封装
├── bin_ops.py             # Bin 操作工具函数
├── scheduling_table.py    # 调度表数据结构
├── scheduler_agent.py     # 运行时调度器
├── monitor_agent.py       # 运行时监控
├── sort_function.py       # 任务排序函数
└── packing_solver/        # 优化求解器
    ├── chain_slack_assign.py      # 链式 Slack 分配求解器
    └── gurobi_MP_semi2DClst.py    # 空间分区求解器
```

## 2. 核心算法流程图

### 2.1 两阶段 Guided Hybrid Allocation 算法

```
                    ┌──────────────────────────────────────┐
                    │         perform_bin_packing          │
                    │           (sim_main.py)              │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │     binpack_cfg["algorithm"]?        │
                    └──────────────────┬───────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
        "guided"                 "scratch"                "full"
              │                        │                        │
              ▼                        │                        │
    ┌─────────────────┐                │                        │
    │  Phase 1: Split │                │                        │
    │ coleasing_alloc │                │                        │
    │    _cluster()   │                │                        │
    │                 │                │                        │
    │ quantile: q_A   │                │                        │
    │ (conservative)  │                │                        │
    └────────┬────────┘                │                        │
             │                         │                        │
             ▼                         │                        │
    ┌─────────────────┐                │                        │
    │ need_repack?    │                │                        │
    └────────┬────────┘                │                        │
             │                         │                        │
      ┌──────┴──────┐                  │                        │
      │             │                  │                        │
      ▼             ▼                  ▼                        ▼
    True         False            glb_alloc                 naive_iso
      │             │
      ▼             │
┌─────────────┐     │
│Phase 2:     │     │
│  Repack     │     │
│push_task_   │     │
│into_bins_new│     │
│             │     │
│quantile: q_B│     │
│(aggressive) │     │
└──────┬──────┘     │
       │            │
       └─────┬──────┘
             │
             ▼
    ┌─────────────────┐
    │    bin_list     │
    │   (调度结果)     │
    └─────────────────┘
```

### 2.2 调度策略与执行路径

```
┌─────────┬────────────────────────────────────────────────────────────┐
│ 策略    │ 执行路径                                                    │
├─────────┼────────────────────────────────────────────────────────────┤
│ cyc     │ Step 0-1-2 (guided non-repack, num_bins=-1)                │
│ glb     │ Step 0-1 (scratch, num_bins=1)                             │
│ reserv  │ Step 0-1-2 + repack (guided + guided, num_bins >= 2)      │
│ pglb    │ Step 0-1-2 (guided non-repack, num_bins > 1)              │
│ cyc-S   │ Step 0-1-2 + repack (guided + guided, num_bins=-1)        │
└─────────┴────────────────────────────────────────────────────────────┘
```

## 3. 核心函数职责

### 3.1 global_sched.py

#### `coleasing_alloc_cluster()` - Phase 1: 空间分区

**位置**: `sched/global_sched.py:580`

**职责**:
- 将任务聚类到资源分区 (bins)
- 基于图拓扑和亲和性进行聚类
- 使用**保守分位数** (`exec_t_comp_ratioA`, 如 0.99) 计算资源需求

**输入**:
- `bin_list`: 初始 bin 列表
- `glb_p_list`: 任务实例列表
- `affinity`: 亲和性配置
- `quantile`: 资源估算分位数 (q_A)
- `n_partition`: 目标分区数

**输出**:
- `max_core_num`: 总核心数需求
- `pid2_bin_id`: 任务到 bin 的映射
- `bin_size_list`: 每个 bin 的资源大小

**核心流程**:
```
coleasing_alloc_cluster
    │
    ├── coleasing_alloc_1bin()  # 初始单 bin 调度
    │
    ├── [if n_partition > 1]
    │       │
    │       ├── get_chains()           # 提取链信息
    │       ├── gurobi_split_solver()  # Gurobi 求解分区
    │       └── update_bp_result2_schedtab()  # 更新调度表
    │
    └── return (max_core_num, pid2_bin_id, bin_size_list)
```

#### `push_task_into_bins_new()` - Phase 2: 时间调度

**位置**: `sched/global_sched.py:42`

**职责**:
- 在每个 bin 内分配静态时间窗口
- 使用**激进分位数** (`exec_t_comp_ratioB`, 如 0.80) 计算时间片
- 实现"软预留"机制

**输入**:
- `bin_list`: Phase 1 输出的 bin 列表 (含 pre_defined mapping)
- `glb_p_list`: 任务实例列表
- `quantile`: 时间片估算分位数 (q_B)
- `binpack_cfg`: 装箱配置

**核心流程**:
```
push_task_into_bins_new
    │
    ├── for n_slot in range(sim_slot_num):
    │       │
    │       ├── message_trigger_event_new()   # 触发事件
    │       │
    │       └── push_step_new()
    │               │
    │               ├── check_complete()      # 检查完成
    │               ├── check_miss()          # 检查超时
    │               ├── data_pipe_read()      # 读取数据
    │               ├── chk_release()         # 检查释放
    │               ├── pendingToReady()      # 就绪转换
    │               │
    │               └── [if trigger_cond]
    │                       └── glb_alloc_new2()  # 动态资源分配
    │
    └── return bin_list
```

### 3.2 pre_alloc_new.py

#### `glb_alloc_new2()` - 运行时动态分配

**位置**: `sched/pre_alloc_new.py:26`

**职责**:
- 为就绪任务分配资源
- 处理抢占逻辑
- 实现 Bin 选择策略

**核心流程**:
```
glb_alloc_new2
    │
    ├── 排序就绪队列 (按 deadline + affinity)
    │
    └── while sorted_ready_queue:
            │
            ├── allocate_rsc_4_process_new2()
            │       │
            │       ├── Step 1: 计算资源请求参数
            │       │       └── rsc_req_estm_quantile()
            │       │
            │       ├── Step 2: 选择 Bin
            │       │       └── bin_sel() / pre_defined mapping
            │       │
            │       └── Step 3: 尝试分配
            │               └── check_and_preemt_alloc()
            │                       ├── push_into_bin()
            │                       └── 抢占低优先级任务
            │
            └── 处理被抢占任务
```

#### `allocate_rsc_4_process_new2()` - 单任务分配

**位置**: `sched/pre_alloc_new.py:138`

**参数**:
- `_p`: 待分配的任务实例
- `binpack_cfg`: 包含 `bin_sel_mod`, `total_cores` 等

**Bin 选择模式**:
- `pre_defined`: 使用 Phase 1 的静态映射
- `search`: 动态搜索合适的 bin

### 3.3 slack_estim.py

#### `deduce_cfg2()` - 配置推导入口

**位置**: `sched/slack_estim.py:393`

**职责**:
- 为所有调度策略计算初始时间片分配
- 调用链式 Slack 分配算法

**核心流程**:
```
deduce_cfg2(taskattr_dict, logical_graph_nx, ..., quantile)
    │
    ├── get_chains()           # 分解 DAG 为链
    │       └── decompose_dag_into_chains()
    │
    ├── rsc_slack_estim()      # 资源和 Slack 估算
    │       └── DistributeSlack()
    │               ├── 排序链 (硬约束优先)
    │               └── GurobiRscSlackEstim / HeuriRscSlackEstim
    │
    └── init_topo_time_attr()  # 设置 ERT 和 ddl
```

#### `DistributeSlack()` - Slack 分配算法

**位置**: `sched/slack_estim.py:107`

**职责**:
- 按链优先级分配资源
- 确保端到端延迟约束

**链排序规则**:
1. 硬约束 (deadline) 优先
2. 计算量/延迟预算比值高的优先 (更紧急)

**求解器选择**:
- `algorithm="gurobi"`: 精确求解 (MIQP)
- `algorithm="avg"`: 启发式求解

### 3.4 binpack_config.py

#### `BinPackConfig` - 配置封装类

**位置**: `sched/binpack_config.py:7`

**设计思想**:
- 继承 `dict`，兼容旧代码
- 提供属性访问模式，增强类型安全

**核心属性**:

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `algorithm` | str | "coalescing" | 算法选择 |
| `sort` | str | "EAT" | Bin 排序方式 |
| `mode` | str | "non-block" | 插入模式 |
| `bin_sel_mod` | str | "search" | Bin 选择模式 |
| `quantile` | float | 0.99 | 资源估算分位数 |
| `exec_t_comp_ratioB` | float | -1.0 | Repack 分位数 |
| `reservation_policy` | str | "manual" | 预留策略 |
| `affinity_en` | bool | True | 启用亲和性 |
| `preempt_en` | bool | True | 允许抢占 |
| `var_dist_map` | Dict | {} | 变异分布映射 (运行时注入) |
| `mapping` | Dict | {} | 任务到 Bin 映射 (运行时注入) |

### 3.5 配置流与参数注入点

**配置流总览**:

```
JSON 文件 (cfgs/*.json)
    │
    ▼
BinPackConfig 类 (sched/binpack_config.py)
    │
    ▼
args.binpack_cfg (运行时)
    │
    ▼
binpack_cfg dict (函数参数)
    │
    ▼
调度算法 (sched/global_sched.py)
```

**参数来源**:

| 来源 | 示例参数 | 注入位置 |
|------|----------|----------|
| JSON 文件 | `algorithm`, `affinity_en` | `utils.input_parser()` |
| 运行时 | `quantile`, `var_dist_map` | `sim_main.prepare_binpack_cfg()` |
| 命令行 | `--bin_pack_cfg`, `--num_bins` | `argparse` |

#### 3.5.1 JSON 加载

**位置**: `utils.py:182-196`

```python
binpack_cfg_dict = json.load(open(cfg_path))
binpack_cfg_dict.update(args.bin_pack_para)
args.binpack_cfg = BinPackConfig(binpack_cfg_dict)

# 后续注入 exec_t_comp_ratioB
args.binpack_cfg.update({"exec_t_comp_ratioB": args.exec_t_comp_ratioB})
```

#### 3.5.2 运行时注入

**位置**: `sim_main.py:423-429`

```python
def prepare_binpack_cfg(cfg, quantile, p_list, graph):
    new_cfg_dict = dict(cfg)
    new_cfg_dict['var_dist_map'] = {
        p.task.name: graph.nodes[p.task.name]['var_dist']
        for p in p_list
    }
    new_cfg_dict['quantile'] = quantile
    return BinPackConfig(new_cfg_dict)
```

#### 3.5.3 全局参数注入

**位置**: `sched/global_sched.py` (coleasing_alloc_cluster 入口)

```python
# 注入 total_cores 到 binpack_cfg
binpack_cfg["total_cores"] = total_cores
```

### 3.6 已知配置问题

#### 3.6.1 僵尸参数

**位置**: `sched/global_sched.py:push_step_new()`

```python
def push_step_new(..., percentile: float, ...):
    # percentile 参数在函数签名中，但未在函数体中使用
    # 实际 quantile 通过 binpack_cfg['quantile'] 传递
```

#### 3.6.2 双重传递

部分参数同时通过两种方式传递：
1. 直接函数参数
2. `binpack_cfg` 字典

可能导致不一致。

#### 3.6.3 更新位置耦合

`exec_t_comp_ratioB` 在 `build_path_old()` 中更新，与路径构建耦合。

### 3.7 配置文件示例

#### 3.7.1 Bp_guided.json

```json
{
    "algorithm": "guided",
    "sort": "barycenter",
    "mode": "block",
    "bin_sel_mod": "search",
    "reservation_policy": "static_1_bin",
    "affinity_en": true,
    "affinity_level": 2,
    "preempt_en": false,
    "partial_alloc_en": false,
    "core_size": "induced"
}
```

#### 3.7.2 Bp_scratch.json

```json
{
    "algorithm": "scratch",
    "sort": "barycenter",
    "mode": "block",
    "bin_sel_mod": "search",
    "reservation_policy": "manual",
    "affinity_en": true,
    "preempt_en": true,
    "core_size": "specified"
}
```

## 4. 参数影响分析

### 4.1 双分位数机制

```
exec_t_comp_ratioA (Phase 1)     exec_t_comp_ratioB (Phase 2)
        │                                │
        ▼                                ▼
┌───────────────────┐            ┌───────────────────┐
│   资源分区大小     │            │   时间片分配       │
│   (保守估计)       │            │   (激进估计)       │
│                   │            │                   │
│ 高 q_A → 多资源   │            │ 低 q_B → 紧凑调度  │
│ 低 q_A → 少资源   │            │ 高 q_B → 松散调度  │
└───────────────────┘            └───────────────────┘
        │                                │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   软预留机制            │
        │                        │
        │ • 静态时间窗口隐藏开销  │
        │ • 保守资源提供动态回退  │
        └────────────────────────┘
```

### 4.2 策略参数组合

> `num_bins` 含义: `-1` = 自动搜索最大分区数; `1` = 单分区 (跳过聚类); `>1` = 指定分区数

| 策略 | num_bins | need_repack | ratioA | ratioB |
|------|----------|-------------|--------|--------|
| cyc | -1 | False | 扫描 | - |
| glb | 1 | False | 扫描 | - |
| reserv | >=2 | True | 固定 | 扫描 |
| pglb | >1 | False | 扫描 | - |
| cyc-S | -1 | True | 固定 | 扫描 |

## 5. 与 Spec 文档的对应关系

| Spec 文档 | 对应实现 | 说明 |
|-----------|----------|------|
| `doc/spec/algorithm/guided_hybrid_allocation_algorithm.md` | `global_sched.py` | 两阶段算法设计 |
| `doc/spec/algorithm/chain_slack_assignment_algorithm.md` | `slack_estim.py`, `packing_solver/chain_slack_assign.py` | Slack 分配算法 |
| `doc/spec/e2e_sched_sim_flow.md` | `sim_main.py`, `approach/approach_setup.py` | 端到端流程 |
| `doc/spec/key_COT.md` | 全局 | 学术论点与消融实验设计 |
| `doc/spec/cfg/binpack_config_design.md` | `binpack_config.py` | 配置设计 |

## 6. 关键数据结构

### 6.1 SchedulingTableInt (Bin)

**位置**: `sched/scheduling_table.py`

```python
class SchedulingTableInt:
    """
    调度表 (Bin) 的核心数据结构

    属性:
        num_resources: int      # 资源数量 (核心数)
        scheduling_table: List  # 时间维度的资源分配
        sparse_list: List       # 稀疏配置列表
        sparse_cores: List      # 稀疏核心记录
        sparse_flops: List      # 稀疏 FLOPS 记录
    """
```

### 6.2 ProcessInt (任务实例)

**位置**: `task/task_agent.py`

```python
class ProcessInt:
    """
    任务实例

    关键属性:
        pid: int                # 进程 ID
        task: TaskInt           # 关联的任务模板
        release_time: float     # 释放时间
        deadline: float         # 截止时间
        exp_comp_t: float       # 预期执行时间
        remburst: float         # 剩余计算量
    """
```

## 7. 求解器模块

### 7.1 GurobiRscSlackEstim

**位置**: `sched/packing_solver/chain_slack_assign.py:83`

**用途**: 精确求解链式 Slack 分配 (MIQP)

**特点**:
- 最优解保证
- 支持复杂约束
- 计算开销较高

### 7.2 ClusterGurobiSolverSemi2D

**位置**: `sched/packing_solver/gurobi_MP_semi2DClst.py:7`

**用途**: 空间分区求解

**目标函数**:
- `mux_min_nbin`: 最小化 bin 数量
- `colocate_fix_nbin_min_size`: 固定 bin 数量下最小化资源

## 8. 错误处理

### 8.1 ResourceInsufficientError

**位置**: `sched/pre_alloc_new.py:21`

```python
class ResourceInsufficientError(Exception):
    """当 Bin 资源不足以容纳任务时抛出"""
```

**触发场景**:
- Phase 1 资源估算与 Phase 2 任务需求不匹配
- 静态映射下 bin 资源不足

## 9. 调试与日志

### 9.1 关键打印点

- `push_task_into_bins_new`: 周期切换、WARMUP/DRAIN 状态
- `coleasing_alloc_cluster`: max_core_num, pid2_bin_id, bin_size_list
- `glb_alloc_new2`: 任务分配成功/失败信息
- `check_and_preemt_alloc`: 抢占操作

### 9.2 DEBUG 标志

- `verbose`: 详细输出
- `DEBUG_FG`: 调试模式
- `show_warnings`: 警告显示
