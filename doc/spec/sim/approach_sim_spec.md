# Event-Driven 仿真后端规范

> **状态**: 草稿 (2026-02)
>
> **相关文件**:
> - `approach_sim.py`: 仿真主循环
> - `approach_sched.py`: 调度策略定义
> - `approach_def.py`: 处理器基类和图结构
> - `approach_initiator.py`: 实例化工厂
> - `approach_collector.py`: 统计收集器

## 1. 概述

### 1.1 与 sim_main.py 的关系

| 模块 | 职责 | 调用位置 |
|------|------|----------|
| `sim_main.py` | **配置生成** - Bin packing, 资源约束, Dump | `main_approach.py` 第一阶段 |
| `approach_sim.py` | **仿真执行** - Event-driven 调度模拟 | `main_approach.py` 第二阶段 |

### 1.2 流程结合 (main_approach.py)

```
main_approach.py
    │
    ├── Stage 1: 配置生成 (sim_main.py)
    │   ├── setup_benchmark()
    │   │   ├── build_workload_and_criticality()
    │   │   ├── perform_bin_packing() → bin_list
    │   │   └── dump(bin_list)
    │   └── 输出: bin_list.pkl, G, pid2name
    │
    └── Stage 2: 仿真执行 (approach_sim.py)
        ├── get_partition_info(bin_list) → PartitionConfig
        ├── instantiate_processors(partition_cfg, policy)
        └── run_simulation(processors, event_t, G)
```

## 2. 核心数据结构

### 2.1 PartitionConfig

**定义位置**: `approach_sched.py:243-266`

**作用**: 描述所有分区的调度配置，是仿真输入的核心数据结构

```python
class PartitionConfig:
    def __init__(
        self,
        num_partitions: int,      # 分区数量
        cap_list: List[int],       # 每个分区的资源容量 (tiles)
        base_pwr_list: List[float], # 每个分区的算力 (FLOPS/core)
        G: MyGraph,                # 任务图
        TSmap_list: List = None,   # 每个分区的静态调度表 (cyc专用)
        mapped_node_list: List = None, # 每个分区的任务映射
        swt_lat_list: List = None, # 每个分区的切换延迟
        T_hp: float = None         # 超周期长度
    ):
```

**字段说明**:

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `num_partitions` | int | ✅ | 分区数量 (= len(bin_list)) |
| `cap_list` | List[int] | ✅ | 每个分区的 tile 数量 |
| `base_pwr_list` | List[float] | ✅ | 每个分区的 FLOPS/core |
| `G` | MyGraph | ✅ | 任务图实例 |
| `TSmap_list` | List[List[tuple]] | cyc/cyc-S | 静态调度表 `[(t, {task: size})]` |
| `mapped_node_list` | List[List[str]] | ✅ | 每个分区负责的任务名列表 |
| `swt_lat_list` | List[float] | ✅ | 每个分区的切换开销 |
| `T_hp` | float | cyc | 超周期长度 |

**从 bin_list 构造**:

```python
# approach_initiator.py:24-97
def get_partition_info(bin_list, graph, pid2name):
    # 1. 分区的任务映射
    partition_task_map = []
    for _bin in bin_list:
        task_set = set()
        for rsc_agent in _bin.scheduling_table:
            task_set.update(rsc_agent.rsc_map.keys())
        partition_task_map.append([pid2name[pid] for pid in task_set])

    # 2. 分区大小
    partition_size = [_bin.num_resources for _bin in bin_list]

    # 3. 静态调度表 (从 sparse_list 提取)
    TSMap_list = []
    for _bin in bin_list:
        TSmap = []
        for cfg_slot_s, next_cfg, cfg_slot_num in _bin.sparse_list:
            # 转换为 (time, {task_name: size}) 格式
            TSmap.append((loc_t, cfg_))
        TSMap_list.append(TSmap)

    return num_partitions, partition_size, flops_list, partition_task_map, TSMap_list
```

### 2.2 MyGraph

**定义位置**: `approach_def.py:100-264`

**作用**: 任务图，包含节点属性和状态管理

**关键属性**:
- `logical_graph`: NetworkX DiGraph，节点属性包括 `type`, `exp_comp_t`, `ddl`, `ert`, `offset`
- `n_pred_map`: 未就绪前驱计数
- `ddl_map`, `ert_map`, `offset_map`: 时间缓存

**关键方法**:
- `duplicate_for_hyperperiod(hp_idx, seed, T_hp, var_en)`: 为新超周期复制任务
- `mark_finish(node)`: 标记任务完成，更新后继状态
- `mark_ready(node)`: 标记任务就绪

### 2.3 GlobalEvent_t

**定义位置**: `approach_def.py:267-340`

**作用**: 全局事件队列，管理仿真时间推进

**事件类型**:

| 类型 | 触发时机 | 作用 |
|------|----------|------|
| `"external"` | 传感器触发 / ERT 到达 | 唤醒任务执行 |
| `"table"` | 静态调度表时间点 | cyc 策略的时间驱动 |
| `"finish"` | 任务完成 | 仿真内部生成 |

**核心数据结构**:
```python
class GlobalEvent_t:
    def __init__(self, event_t: List):
        # 当前事件队列 (排序)
        self.event_t = [(float("inf"), "external")]  # 哨兵事件
        # 原始事件模式 (用于动态复制)
        self._original_events = sorted(set(event_t))
```

**关键方法**:

| 方法 | 功能 | 返回值 |
|------|------|--------|
| `get_next_event_time(curr_t)` | 获取下一个事件时间 | `float` |
| `confirm_next_event(curr_t)` | 确认并弹出下一个事件 | `(time, type)` |
| `add_events_for_hyperperiod(hp_idx, T_hp, curr_t, type_list)` | 为新超周期添加事件 | `None` |
| `remove_ood_events(curr_t)` | 移除过期事件 | `None` |

**事件生命周期**:
```
初始化 (第一个超周期)
    ↓
_original_events 存储 [(t, type), ...]
    ↓
进入新超周期 → add_events_for_hyperperiod()
    ↓
事件时间 += hp_idx * T_hp
    ↓
加入 event_t 队列
    ↓
get_next_event_time() → 确定下一时间点
    ↓
confirm_next_event() → 消费事件
```

## 3. 四种调度策略

### 3.1 策略对比表

| 策略 | 配置生成路径 | 触发条件 | 资源分配 | TSmap | ERT过滤 |
|------|-------------|----------|----------|-------|---------|
| **cyc** | step0-1-2 | 无 (时间驱动) | 查表 (force=True) | ✅ 必需 | ❌ |
| **glb** | step0-1 | 动态触发 | 最小需求 (reserv_en=False) | ❌ | ❌ |
| **pglb** | step0-1-2 | 动态触发 | 最小需求 (reserv_en=False) | ❌ | ❌ |
| **reserv** | step0-1-2+repack | 动态触发 | 最小需求 (reserv_en=True) | ❌ | ✅ |

### 3.2 策略绑定逻辑

**位置**: `approach_sched.py:269-324`

```python
def acc_p_factory(policy, cfg, stats_collector):
    for i in range(cfg.num_parts):
        acc_p = Acc_p(...)

        if policy in ["pglb", "glb"]:
            # 动态调度，无预留
            acc_p.alloc_fn = alloc_fn_pglb(reserv_en=False)
            acc_p.task_filter = task_filter(reserv_en=False)
            acc_p.trigger_cond = trigger_cond_dyn()

        elif policy in ["cyc", "cyc-S"]:
            # 静态调度，查表
            acc_p.static_schedule_map = cfg.TSmap_list[i]
            force = True if policy == "cyc" else False
            acc_p.alloc_fn = alloc_fn_cyclic(T_hp=cfg.T_hp, force=force)
            acc_p.trigger_cond = no_trigger()

        elif policy == "reserv":
            # 动态调度，有预留
            acc_p.alloc_fn = alloc_fn_pglb(reserv_en=True)
            acc_p.task_filter = task_filter(reserv_en=True, op_miss_en=True)
            acc_p.trigger_cond = trigger_cond_dyn()
```

### 3.3 各策略详细说明

#### 3.3.1 cyc (纯静态周期调度)

**输入要求**:
- `TSmap_list`: ✅ 必需 (静态调度表)
- `T_hp`: ✅ 必需
- `num_partitions`: 任意 (通常 = 实际分区数 当 num_bins=-1)

**调度逻辑** (`alloc_fn_cyclic` with `force=True`):
1. 根据 `curr_t` 查找 `static_schedule_map` 中的配置
2. 严格按照静态表分配资源
3. 如果资源不足 → **assert 报错**
4. 不触发动态重调度 (`trigger_cond = no_trigger`)

**事件初始化**:
```python
# 从 TSmap 中提取 table 事件
for TSmap in TSmap_list:
    for t, cfg in TSmap:
        event_t.add((t, "table"))
```

#### 3.3.2 cyc-S (软预留静态调度)

**与 cyc 的区别**:
- `force=False`: 资源不足时**动态降级**而非报错
- 允许任务在静态窗口 missed 时继续运行

**调度逻辑** (`alloc_fn_cyclic` with `force=False`):
1. 查找静态调度表
2. 尝试按静态表分配
3. 如果资源不足 → 分配可用资源，任务继续执行

#### 3.3.3 glb (全局动态调度)

**输入要求**:
- `TSmap_list`: ❌ 不需要
- `num_partitions`: = 1 (单分区)

**调度逻辑** (`alloc_fn_pglb` with `reserv_en=False`):
1. 按 slack (ddl - curr_t) 排序任务
2. 计算每个任务的最小资源需求
3. 剩余资源均匀分配给所有任务
4. 触发条件: 有新就绪任务 或 有空闲资源

**触发条件** (`trigger_cond_dyn`):
```python
cond1 = (free_tiles <= 0) and new_ready_has_higher_priority
cond2 = (free_tiles > 0) and (new_ready or starving)
```

#### 3.3.4 pglb (分区全局动态调度)

**与 glb 的区别**:
- `num_partitions > 1` (多分区)
- 每个分区独立调度，但使用相同的 `alloc_fn_pglb`

**特殊处理**:
```python
if policy == "pglb" and cfg.num_parts <= 1:
    warn("pglb policy is not supported for single partition, use glb instead")
    policy = "glb"
```

#### 3.3.5 reserv (预留调度)

**输入要求**:
- `TSmap_list`: ❌ 不需要
- `ERT`: ✅ 需要 (用于预留窗口过滤)
- `num_partitions`: >= 1

**调度逻辑** (`alloc_fn_pglb` with `reserv_en=True`):
1. **ERT 过滤**: 只调度 `ert <= curr_t` 的任务
2. 按 slack 排序
3. 计算最小资源需求
4. **不均匀分配**: 剩余资源保留而非分配

**任务过滤** (`task_filter` with `reserv_en=True`):
```python
filter_cond = [
    lambda node: node != "R",                    # 过滤系统任务
    lambda node: ddl > curr_t or not drop,       # 过滤超时任务
    lambda node: ert <= curr_t if reserv_en else True,  # ERT 过滤
]
```

**事件初始化**:
```python
# 添加 ERT 事件
for node in op_nodes:
    event_t.add((node['ert'], "external"))
```

## 4. 仿真主循环

### 4.1 run_simulation 流程

**位置**: `approach_sim.py:43-168`

```
run_simulation(processors, event_t, G, num_hp, T_hp)
    │
    while curr_hp < num_hp:
        │
        ├── 1. 超周期边界处理
        │   ├── record_miss(timeout_tasks)
        │   ├── forward_hyperperiod()
        │   └── duplicate_for_hyperperiod()
        │
        ├── 2. update_run(pred_t, curr_t)
        │   ├── 更新运行队列
        │   ├── 记录 idle/完成
        │   └── 返回 new_comp
        │
        ├── 3. update_ready(curr_t)
        │   ├── 检查 n_pred_map
        │   └── 返回 new_ready_list
        │
        ├── 4. sched(curr_t, new_comp, new_ready_list)
        │   ├── trigger_cond() → realloc?
        │   ├── alloc_fn() → alloc_map
        │   └── predict_next() → duration
        │
        └── 5. 推进时间
            └── curr_t = min(curr_t + duration, next_event_time)
```

### 4.2 处理器类型

| 类型 | 职责 | 调度策略 |
|------|------|----------|
| `Sen_p` | 传感器处理器 | FCFS |
| `Acc_p` | 加速器处理器 | cyc/glb/pglb/reserv |

### 4.3 事件驱动的详细流程

#### 4.3.1 仿真时间轴

```
时间轴:
-T_hp    0      T_hp    2*T_hp   3*T_hp   ...   num_hp*T_hp
  │      │       │       │        │              │
  │      │       │       │        │              └─ 仿真结束
  │      │       │       │        └─ hp_idx = num_hp-1
  │      │       │       └─ hp_idx = 2
  │      │       └─ hp_idx = 1
  │      └─ hp_idx = 0 (第一个有效超周期)
  └─ hp_idx = -1 (预热阶段)
```

#### 4.3.2 超周期边界处理

**触发条件**: `curr_t >= (curr_hp + 1) * T_hp`

```python
# approach_sim.py:54-79
next_hp_boundary = (curr_hp + 1) * T_hp
if curr_t >= next_hp_boundary and pred_t < next_hp_boundary:
    # 1. 记录上一超周期的超时任务
    if curr_hp >= 0:
        stats_collector.record_miss(iter_timeout(curr_t, curr_hp * T_hp))
        stats_collector.forward_hyperperiod(T_hp)

    # 2. 递增超周期计数
    curr_hp += 1
    tgt_hp = curr_hp + 1

    # 3. 为下一超周期准备
    if tgt_hp < num_hp:
        # a) 复制任务图
        G.duplicate_for_hyperperiod(tgt_hp, seed=0, T_hp=T_hp, var_en=var_en)
        # b) 更新处理器映射
        for proc in processors:
            proc.update_mapped_nodes_for_hyperperiod(tgt_hp)
        # c) 添加事件
        event_t.add_events_for_hyperperiod(tgt_hp, T_hp, curr_t)
    elif tgt_hp < sim_hp:
        # 最后一个超周期只添加 table 事件 (用于 drain)
        event_t.add_events_for_hyperperiod(tgt_hp, T_hp, curr_t, type_list=["table"])
```

#### 4.3.3 事件类型与处理

| 事件类型 | 触发条件 | 处理逻辑 |
|----------|----------|----------|
| `"external"` | `curr_t == event_t.pop(0)[0]` | 传感器触发，任务就绪 |
| `"table"` | `curr_t == event_t.pop(0)[0]` | cyc 策略的静态调度点 |
| `"finish"` | `curr_t == curr_t + duration` | 任务完成，内部生成 |

#### 4.3.4 时间推进机制

```python
# approach_sim.py:145-164

# 1. 计算各处理器的下一个事件时间
duration_dict = {}
for proc in processors:
    duration_dict[proc] = proc.predict_next(curr_t)

# 2. 取最小值 (最快到达的事件)
duration = min(duration_dict.values())

# 3. 与外部事件队列比较
next_timer_event = event_t.get_next_event_time(curr_t)

# 4. 推进到最近的时间点
next_curr_t = curr_t + duration
curr_t = min(next_curr_t, next_timer_event, sim_hp * T_hp)

# 5. 确定事件类型
if curr_t == next_timer_event:
    _, event_type = event_t.confirm_next_event(curr_t)  # external 或 table
else:
    event_type = "finish"
```

#### 4.3.5 处理器状态转换

**Acc_p 状态机**:

```
        ┌─────────────────────────────────────┐
        │                                     │
        ▼                                     │
    ┌───────┐   trigger_cond()=True    ┌───────┐
    │   S   │ ───────────────────────► │   R   │
    │(执行) │                          │(重调度)│
    └───────┘ ◄─────────────────────── └───────┘
              R 任务完成
```

**状态说明**:
- `S` (Steady): 正常执行任务
- `R` (Reallocation): 执行重调度，所有计算任务暂停

**状态转换代码**:
```python
# approach_sched.py:69-77
if cond:  # trigger_cond 返回 True
    realloc = True
    acc_p.sys_state = "R"
    acc_p.running["R"] = cal_load(acc_p.swt_lat, 1)  # 系统任务
else:
    realloc = False
```

### 4.4 事件初始化 (按策略)

#### 4.4.1 通用事件 (所有策略)

```python
# 源节点触发事件
for node in src_nodes:
    t = G.nodes[node]['offset']
    event_t.add((t, "external"))
```

#### 4.4.2 cyc 专用事件

```python
# 静态调度表事件
for TSmap in TSmap_list:
    for t, cfg in TSmap:
        event_t.add((t, "table"))
```

#### 4.4.3 reserv 专用事件

```python
# ERT (Earliest Release Time) 事件
for node in op_nodes:
    t = G.nodes[node]['ert']
    event_t.add((t, "external"))
```

### 4.5 任务生命周期

```
任务状态:
┌─────────┐    n_pred=0    ┌─────────┐   sched()    ┌─────────┐
│ 未就绪  │ ─────────────► │  Ready  │ ───────────► │ Running │
│(n_pred) │               │ (ready) │              │(running)│
└─────────┘               └─────────┘              └────┬────┘
     ▲                                                   │
     │                                                   │
     │         ┌─────────┐    load≤0    ┌─────────┐     │ ddl<t
     │         │  完成   │ ◄──────────── │  超时   │ ◄───┴─────┐
     └─────────┤ (finish)│               │(timeout)│           │
      mark_    └─────────┘               └─────────┘           │
      finish()                                               │
                                                           drop=True
                                                              │
                                                              ▼
                                                        [任务丢弃]
```

### 4.6 图复制机制 (duplicate_for_hyperperiod)

**目的**: 为每个超周期创建独立的任务实例

**流程**:
```python
# approach_def.py:167-264
def duplicate_for_hyperperiod(self, hp_idx, seed, T_hp, var_en):
    # 1. 复制节点
    for node, attr in orig_nodes:
        new_name = f"{node}_{hp_idx}"
        new_attr = attr.copy()

        # 2. 随机化执行时间 (如果 var_en)
        if var_en and node in var_dist_map:
            new_attr['exp_comp_t'] = dist.get_var_fn()(rng)

        # 3. 添加时间偏移
        time_offset = hp_idx * T_hp
        new_attr['offset'] += time_offset
        new_attr['ert'] += time_offset
        new_attr['ddl'] += time_offset

        self.add_node(new_name, **new_attr)

    # 4. 复制边
    for u, v, eattr in orig_edges:
        new_u = f"{u}_{hp_idx}"
        new_v = f"{v}_{hp_idx}"
        self.add_edge(new_u, new_v, **eattr)

    # 5. 更新 n_pred_map
    for node in non_src_nodes:
        dup = f"{node}_{hp_idx}"
        self.n_pred_map[dup] = len(list(self.predecessors(dup)))
```

**任务命名规则**:
- 原始: `TaskA`
- 超周期 0: `TaskA_0`
- 超周期 1: `TaskA_1`
- ...

## 5. 与配置生成的参数映射

### 5.1 num_bins → num_partitions

| num_bins | 含义 | num_partitions |
|----------|------|----------------|
| -1 | 最多分区/自动搜索 (n_partition=9999，每个任务独立分区) (cyc, cyc-S) | 实际分区数 |
| 1 | 单分区 (glb) | 1 |
| >1 | 多分区 (pglb, reserv) | num_bins |

### 5.2 exec_t_comp_ratioB → 策略选择

| exec_t_comp_ratioB | 是否 repack | 策略影响 |
|-------------------|-------------|----------|
| -1 | 否 | 使用 step0-1-2 的 bin_list |
| < ratioA | 是 | 使用 repack 后的 bin_list (含 TSmap) |

### 5.3 完整参数映射表

| 配置参数 | 仿真参数 | 说明 |
|----------|----------|------|
| `num_bins` | `num_partitions` | 分区数量 |
| `bin.num_resources` | `cap_list[i]` | 分区资源容量 |
| `bin.sparse_list` | `TSmap_list[i]` | 静态调度表 |
| 任务 PID → name | `mapped_node_list[i]` | 任务映射 |
| `exec_t_comp_ratioB` | 影响 ERT/DDL | repack 后的时间约束 |

## 6. 统计收集

### 6.1 StatisticsCollector 集成

每个处理器共享同一个 `stats_collector` 实例：

```python
# approach_initiator.py:176-193
def instantiate_processors(partition_cfg, event_t, policy, stat_param):
    stats_collector = StatisticsCollector(**stat_param)
    sen_p0 = Sen_p(..., stats_collector)
    acc_p_list = acc_p_factory(policy, partition_cfg, stats_collector)
```

### 6.2 记录点

| 事件 | 记录位置 | 指标 |
|------|----------|------|
| 任务完成 | `update_run` | `record_task_finish`, `record_compute_progress` |
| 超时 | `iter_timeout` | `record_miss` |
| 重调度 | `sched` | `record_realloc`, `record_realloc_num` |
| 空闲 | `update_run` | `record_idle_capacity` |
| E2E 完成 | `update_ready` (sink) | `record_e2e_finish` |

## 7. 与消融实验的对应关系

| 消融实验 | 策略 | num_bins | ratioB | TSmap | ERT过滤 |
|----------|------|----------|--------|-------|---------|
| cyc | cyc | -1 | -1 | ✅ | ❌ |
| cyc-S | cyc-S (force=False) 或 reserv | -1 | 扫描 | ✅ | 取决于实现 |
| glb | glb | 1 | -1 | ❌ | ❌ |
| pglb | pglb | >1 | -1 | ❌ | ❌ |
| reserv | reserv | >=1 | 扫描 | ❌ | ✅ |

## 8. 任务延迟分布处理

### 8.1 延迟分布类型

**定义位置**: `approach_Eq.py:149-315`

| 类型 | 适用任务 | 组成 |
|------|----------|------|
| `SenVarDist` | src (传感器) | 截断正态分布 |
| `LoadVarDist` | op (计算) | 离散分布 (Poisson 权重) |
| `IOVarDist` | op (访存) | 截断指数分布 |
| `AccVarDist` | op (联合) | LoadVarDist + IOVarDist |

### 8.2 分布初始化

**位置**: `approach_Eq.py:318-352` (初始化) → `approach_def.py:105-138` (重建)

```python
# approach_Eq.py: init_var_dist()
def init_var_dist(args, logical_graph):
    for _node, _type in logical_graph.nodes(data="type"):
        if _type == "src":
            # 传感器抖动：截断正态
            range_max = 1 / logical_graph.nodes[_node]['freq']
            trunc_ratio = args.jitter_sim_para['scale']
            zscore = args.jitter_sim_para['zscore']
            loc, scale, myclip_b, a, b = get_truncnorm_para(range_max, trunc_ratio, zscore)
            var_dist = SenVarDist(a, b, loc, scale, truncate=True)

        elif _type == "op":
            # 计算负载：离散分布
            exp_comp_t = logical_graph.nodes[_node]['flops']
            var_factor_list = logical_graph.nodes[_node]['var_factor']
            probs, values = get_discrete_param(var_factor_list, exp_comp_t, lambda_ld=1.0)
            load_dist = LoadVarDist(values, probs)

            # 访存时间：截断指数
            trunc_ratio_exec = args.exec_var_para['scale']
            lambda_exec = args.exec_var_para['lambda_exp']
            exp_io_t = logical_graph.nodes[_node]['exp_io_t']
            _, loc_e, scale_e, b_e = get_truncexpon_param(exp_io_t, trunc_ratio_exec, lambda_exec)
            exec_dist = IOVarDist(b=b_e, loc=loc_e, scale=scale_e, truncate=True)

            # 联合分布
            var_dist = AccVarDist(load_dist, exec_dist)

        # 存储到节点
        logical_graph.nodes[_node]['dist_info'] = var_dist.to_dict()
        logical_graph.nodes[_node]['var_dist'] = var_dist
```

### 8.3 运行时采样

**位置**: `approach_def.py:167-264`

```python
def duplicate_for_hyperperiod(self, hp_idx: int, seed: int, T_hp: float, var_en: bool):
    """
    为新超周期复制任务，并可选地随机化执行时间
    """
    rng = np.random.default_rng(seed)

    for node, data in self.logical_graph.nodes(data=True):
        new_name = f"{node}_{hp_idx}"
        new_attr = data.copy()

        if node_type in ['src', 'op']:
            if var_en and node in self.var_dist_map:
                dist = self.var_dist_map[node]

                # AccVarDist: 同时采样计算负载与访存时间
                if hasattr(dist, 'load_dist') and hasattr(dist, 'exec_dist'):
                    new_attr['exp_comp_t'] = elim_nume_error(dist.load_dist.get_var_fn()(rng))
                    new_attr['exp_io_t'] = elim_nume_error(dist.exec_dist.get_var_fn()(rng))
                else:
                    # 单分布
                    new_attr['exp_comp_t'] = elim_nume_error(dist.get_var_fn()(rng))
                    new_attr['exp_io_t'] = 0.

        # 添加时间偏移
        time_offset = hp_idx * T_hp
        new_attr['offset'] += time_offset
        new_attr['ert'] += time_offset
        new_attr['ddl'] += time_offset

        self.add_node(new_name, **new_attr)
```

### 8.4 运行时延迟组件是否已知？

**答案**: **取决于 `var_en` 参数**

| 模式 | var_en | 延迟来源 | 延迟是否已知 |
|------|--------|----------|--------------|
| **确定性** | False | 使用任务定义的 `exp_comp_t` | ✅ 已知（固定值） |
| **随机** | True | 从 `var_dist_map` 采样 | ❌ 运行时随机生成 |

**关键区别**:

```python
# 确定性模式 (var_en=False)
exp_comp_t = task_attr['exp_comp_t']  # 使用原始值

# 随机模式 (var_en=True)
exp_comp_t = var_dist.get_var_fn()(rng)  # 从分布采样
```

### 8.5 延迟分布与配置生成的关系

| 阶段 | 使用的分位数 | 用途 |
|------|--------------|------|
| **配置生成** | `exec_t_comp_ratioA` (0.99) | Phase 1 资源估算 |
| **配置生成** | `exec_t_comp_ratioB` (0.80) | Phase 2 时间窗口 |
| **仿真运行** | `var_en` 控制 | 实际执行时间 |

**关键公式** (`approach_Eq.py`):

```python
# 配置生成时：使用分位数
quantile_size = var_dist.quantile(exec_t_comp_ratioA)

# 仿真运行时：随机采样（如果 var_en=True）
actual_size = var_dist.get_var_fn()(rng)
```

### 8.6 调度算法对执行时间的可见性

**关键问题**: 调度算法是知道"真实"执行时间，还是只知道估算值？

**答案**: **调度算法知道"真实"执行时间**

#### 8.6.1 执行时间确定时机

```
任务生命周期:
    │
    ├── 1. 任务图复制 (duplicate_for_hyperperiod)
    │   └── 此时确定 exp_comp_t 和 exp_io_t
    │       - var_en=False: 使用任务定义的固定值
    │       - var_en=True: 从分布随机采样
    │
    ├── 2. 任务就绪 (update_ready)
    │   └── 从任务图读取已确定的值
    │       load = cal_load(G.nodes[node]["exp_comp_t"], base_size)
    │
    └── 3. 调度决策 (alloc_fn_pglb / alloc_fn_cyclic)
        └── 使用 ready[node] 中的负载值进行资源估算
```

#### 8.6.2 关键代码位置

**任务负载初始化** (`approach_def.py:721`):

```python
def update_ready(self, curr_t):
    for node in list(self.mapped_node):
        if n_pred == 0:
            # 从任务图节点读取执行时间（此时已确定）
            load = cal_load(self.G_ptr.nodes[node]["exp_comp_t"],
                           self.G_ptr.nodes[node]["base_size"])
            self.ready[node] = load  # 存入 ready 字典
```

**资源需求估算** (`approach_sched.py:124-126`):

```python
def alloc_fn_pglb(...):
    # 从 ready/running 字典读取负载（调度算法"知道"的值）
    task_load = acc_p.running.get(node, 0) + acc_p.ready.get(node, 0)
    exp_io_t = acc_p.G_ptr.nodes[node]['exp_io_t']  # 访存时间
    req_rsc_size = estimate_resource_requirement(task_load, slack, acc_p.base_pwr, exp_io_t)
```

#### 8.6.3 对调度算法的影响

| 场景 | 调度算法知道的信息 | 实际执行时间 | 是否一致 |
|------|-------------------|--------------|----------|
| **确定性** (var_en=False) | exp_comp_t | exp_comp_t | ✅ 一致 |
| **随机** (var_en=True) | 采样后的值 | 采样后的值 | ✅ 一致 |

**结论**:
- 调度算法**始终知道任务的"真实"执行时间**
- 这个"真实"值在任务进入 ready 队列时已经确定
- 无论 `var_en` 是 True 还是 False，调度算法和执行引擎使用的是**同一个值**

#### 8.6.4 与实际系统的对比

| 特性 | 本仿真器 | 实际系统 |
|------|----------|----------|
| 执行时间 | 任务到达时已知 | 运行时才知道 |
| 资源估算 | 精确计算 | 基于历史/预测 |
| 调度决策 | 最优决策 | 近似决策 |

**仿真的假设**: 仿真器假设调度器能够**精确预测**任务执行时间，这在实际系统中是不成立的。

#### 8.6.5 配置生成 vs 仿真的信息差异

**关键区别**: 配置生成和仿真使用**不同的执行时间值**

| 阶段 | 执行时间来源 | 值类型 |
|------|--------------|--------|
| **配置生成** | `var_dist.quantile(ratioA)` | 分位数（保守估算） |
| **仿真运行** | `var_dist.get_var_fn()(rng)` | 随机采样（实际值） |

**这意味着**:
1. **Phase 1 (资源配置)**: 使用 `ratioA=0.99` 分位数，资源充足
2. **Phase 2 (时间窗口)**: 使用 `ratioB=0.80` 分位数，窗口较紧
3. **仿真运行**: 使用随机采样值，可能与配置时的估算**不同**

**典型场景**:
```python
# 配置时
estimated_size = quantile(0.80)  # 假设 100 tiles

# 仿真时（var_en=True）
actual_size = sample()  # 可能是 80, 90, 110, ...

# 如果 actual_size > estimated_size → 任务可能超时
# 如果 actual_size < estimated_size → 任务提前完成
```

#### 8.6.6 可能的改进方向

1. **引入预测误差**: 调度算法看到的值与实际值有偏差
2. **增量学习**: 调度算法根据历史执行记录更新预测
3. **保守调度**: 调度算法使用分位数估算而非真实值
4. **不确定性建模**: 调度算法同时考虑分布的方差

### 8.6 分布序列化/反序列化

**目的**: 支持从 JSON 文件加载任务图后重建分布对象

```python
# 序列化
dist_info = var_dist.to_dict()
# {'__dist_type__': 'AccVarDist', 'load_dist': {...}, 'exec_dist': {...}}

# 反序列化
var_dist = dist_from_dict(dist_info)
```

### 8.7 延迟分布参数

**传感器 (SenVarDist)**:
| 参数 | 来源 | 说明 |
|------|------|------|
| `loc` | 0 | 均值 |
| `scale` | `range_max * trunc_ratio / zscore` | 标准差 |
| `a, b` | `-zscore, +zscore` | 截断边界 |

**计算负载 (LoadVarDist)**:
| 参数 | 来源 | 说明 |
|------|------|------|
| `values` | `var_factor * flops` | 离散值 |
| `probs` | Poisson PMF | 概率 |

**访存时间 (IOVarDist)**:
| 参数 | 来源 | 说明 |
|------|------|------|
| `loc` | `exp_io_t` | 位置参数 |
| `scale` | `1/lambda_exec` | 尺度参数 |
| `b` | `scope/scale` | 截断边界 |

## 9. 待解决问题

### 9.1 cyc-S 的实现争议

**当前状态**: 代码中 `cyc-S` 与 `cyc` 共用 `alloc_fn_cyclic`，仅通过 `force` 参数区分

**问题**: `force=False` 的行为是否符合消融实验的设计意图？

**可能方案**:
1. 保持现状：cyc-S 使用 `alloc_fn_cyclic(force=False)`
2. 修改为：cyc-S 使用 `reserv` 策略 + `num_partitions=1`

### 9.2 pre_alloc_new.py:192-195 的检查

**代码**:
```python
if bin_list[bin_id].num_resources < req_rsc_size:
    sys.exit(1)
```

**问题**: 这个检查在 `pre_defined` 模式下会阻止 repack 流程

**建议**:
- 改为抛出异常或警告
- 或者移除此检查（因为 repack 不应改变资源分配）
