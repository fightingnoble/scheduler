# 装箱启发式求解器规范

> **状态**: 草稿 (2026-02)
>
> **相关文件**:
> - `sched/global_sched.py`: 主函数和流程控制
> - `sched/pre_alloc_new.py`: 核心分配逻辑
> - `sched/binpack_config.py`: 配置包装器

## 1. 概述

### 1.1 功能定位

`push_task_into_bins_new` 是一个 **2D 装箱启发式求解器**，用于将任务分配到时空资源（时间 × 空间）上。

**输入**:
- 任务列表 (glb_p_list): 按拓扑排序的任务
- 配置 (binpack_cfg): 装箱策略参数

**输出**:
- bin_list: 每个任务的时间-空间分配

### 1.2 与两阶段算法的关系

| 阶段 | 函数 | 作用 |
|------|------|------|
| **Phase 1 (Spatial)** | `coleasing_alloc_cluster` | 空间分区，确定任务到 Bin 的映射 |
| **Phase 2 (Temporal)** | `push_task_into_bins_new` | 时间分配，确定每个任务的时间窗口 |

**关键区别**:
- `push_task_into_bins_new` 是 **Repack 阶段**的核心
- 当 `bin_sel_mod == "pre_defined"` 时，使用 Phase 1 确定的映射

## 2. 核心函数架构

### 2.1 函数调用层次

```
push_task_into_bins_new()                    # 主入口
    │
    ├── get_initlist_and_biniter()           # 初始化 Bin 迭代器
    │
    └── for n_slot in range(sim_slot_num):   # 时间槽循环
            │
            ├── message_trigger_event_new()  # 消息触发
            │
            └── push_step_new()              # 单步推进
                    │
                    ├── check_complete()     # 检查完成的任务
                    ├── check_miss()         # 检查超时的任务
                    ├── data_pipe_read()     # 数据传输
                    ├── chk_release()        # 检查释放
                    ├── pendingToReady()     # 待处理→就绪
                    │
                    └── glb_alloc_new2()     # ★ 核心分配逻辑
                            │
                            ├── get_process_sort()       # 任务排序
                            │
                            └── allocate_rsc_4_process_new2()  # 单任务分配
                                    │
                                    ├── bin_sel()                    # Bin 选择
                                    └── check_and_preemt_alloc()     # 分配+抢占
                                            │
                                            └── push_into_bin()       # 插入 Bin
```

### 2.2 数据结构

#### 2.2.1 SchedulingTableInt (Bin)

```python
class SchedulingTableInt:
    # 核心属性
    num_resources: int          # 资源容量 (tiles)
    scheduling_table: List      # 时间槽调度表

    # 稀疏表示 (repack 后)
    sparse_list: List           # [(slot_s, {pid: size}, slot_num)]
    sparse_cores: List          # [(slot_s, {pid: cores})]
    sparse_flops: List          # [(slot_s, {pid: flops})]
```

#### 2.2.2 ProcessInt (任务实例)

```python
class ProcessInt:
    pid: int                    # 进程 ID
    task: TaskInt               # 任务定义
    release_time: float         # 释放时间
    deadline: float             # 截止时间
    remburst: float             # 剩余计算量
    currentburst: float         # 当前突发
```

## 3. 配置参数 (BinPackConfig)

### 3.1 完整参数列表

| 参数 | 类型 | 默认值 | 作用 |
|------|------|--------|------|
| **策略控制** ||||
| `algorithm` | str | "coalescing" | 算法选择 |
| `bin_sel_mod` | str | "search" | Bin 选择模式 |
| `sort` | str | "EAT" | Bin 排序方式 |
| `mode` | str | "non-block" | 插入模式 |
| `reservation_policy` | str | "manual" | 预留策略 |
| **行为控制** ||||
| `affinity_en` | bool | True | 是否启用亲和性 |
| `affinity_level` | int | 2 | 亲和性层级 |
| `preempt_en` | bool | True | 是否允许抢占 |
| `partial_alloc_en` | bool | False | 是否允许部分分配 |
| `quantum_check_en` | bool | False | 是否启用量子化检查 |
| **运行时注入** ||||
| `quantile` | float | 0.99 | 资源估算分位数 |
| `exec_t_comp_ratioB` | float | -1.0 | repack 使用的分位数 |
| `mapping` | Dict[int, int] | {} | 任务到 Bin 的静态映射 |
| `var_dist_map` | Dict | {} | 任务负载分布 |

### 3.2 bin_sel_mod 模式

| 模式 | 行为 | 使用场景 |
|------|------|----------|
| `"search"` | 动态搜索最佳 Bin | Phase 1 (coleasing_alloc_cluster) |
| `"pre_defined"` | 使用预定义映射 | Phase 2 (repack) |

### 3.3 sort 模式 (Bin 排序)

| 模式 | 排序依据 | 适用场景 |
|------|----------|----------|
| `"EAT"` | 最早可用时间 | 实时调度 |
| `"barycenter"` | 重心位置 | 紧凑装箱 |
| `"bf"` | 最佳适应 (尺寸差最小) | 资源效率 |
| `"reverse_bf"` | 最差适应 | 负载均衡 |

### 3.4 mode 模式 (插入模式)

| 模式 | 行为 |
|------|------|
| `"non-block"` | 允许资源共享，尽量利用空闲 |
| `"block"` | 不允许共享，独占时间槽 |

## 4. 核心算法流程

### 4.1 push_task_into_bins_new 主流程

```python
def push_task_into_bins_new(bin_list, glb_p_list, ...):
    # 1. 初始化
    iter_next_bin_obj, bin_name_list = get_initlist_and_biniter(...)

    # 2. 时间槽循环
    for n_slot in range(sim_slot_num):
        curr_t = n_slot * timestep

        # 2.1 消息触发
        message_trigger_event_new(...)

        # 2.2 推进单步
        push_step_new(sched, ..., n_slot, curr_t, binpack_cfg)
```

### 4.2 push_step_new 单步推进

```python
def push_step_new(...):
    # 1. 检查完成的任务
    check_complete(sched, ...)

    # 2. 检查超时的任务
    check_miss(sched, ...)

    # 3. 数据传输
    data_pipe_read(...)

    # 4. 检查任务释放
    chk_release(...)

    # 5. 待处理→就绪
    pendingToReady(...)

    # 6. ★ 核心分配（当有新任务或任务丢弃时）
    if trigger_condA or trigger_condB:
        glb_alloc_new2(...)
```

### 4.3 glb_alloc_new2 核心分配

```python
def glb_alloc_new2(...):
    # 1. 任务排序（按 deadline + 亲和性）
    process_sort = get_process_sort(...)
    sorted_ready_queue = TaskQueue(ready_queue + running_queue, sort_f=process_sort)

    # 2. 逐任务分配
    while len(sorted_ready_queue):
        _p = sorted_ready_queue.pop(0)

        # 跳过已分配的任务
        if _p not in ready_queue:
            continue

        # 尝试分配
        state = allocate_rsc_4_process_new2(_p, ...)

        # 处理被抢占的任务
        for _p_2b_preempt in preempt_list:
            # 撤回资源，放回就绪队列
            ...

        # 成功分配的任务放入 issue_list
        if state:
            issue_list.put(_p)
```

### 4.4 allocate_rsc_4_process_new2 单任务分配

```python
def allocate_rsc_4_process_new2(_p, ...):
    # Step 1: 计算资源需求
    time_slot_s, time_slot_e = _p.quant_release_deadline(n_slot, timestep)
    req_rsc_size, got_latency, _ = _p.rsc_req_estm_quantile(...)

    # Step 2: 选择 Bin
    if bin_sel_mod == "pre_defined":
        bin_id = pid2bin_id[_p.pid]
        # 检查资源是否足够
        if bin_list[bin_id].num_resources < req_rsc_size:
            sys.exit(1)  # 资源不足，退出
    else:
        # 动态搜索
        affinity_tgt_bin_id_list, affinity_search_bin_id_list = bin_sel(...)

    # Step 3: 尝试分配（允许抢占）
    for bin_id in affinity_tgt_bin_id_list + affinity_search_bin_id_list:
        state, succ_info = check_and_preemt_alloc(_p, bin_list[bin_id], ...)
        if state:
            break

    # Step 4: 如果都失败，创建新 Bin
    if not state:
        bin = next(iter_next_bin_obj)

    # Step 5: 记录分配结果
    if state:
        rsc_recoder[_p.pid] = [alloc_slot_s, alloc_size, allo_slot, bin_id]
```

### 4.5 check_and_preemt_alloc 分配+抢占

```python
def check_and_preemt_alloc(_p, bin, ...):
    # 1. 获取空闲资源
    rsc_avl = bin.idx_free_by_slot(time_slot_s, time_slot_e)

    # 2. 计算可抢占任务
    preemptable_map, preemptable_n = index_occupy_by_id_chunk_ver(...)

    # 3. 计算总可用资源（空闲 + 可抢占）
    tot_avl = size + preemptable_n

    # 4. 尝试插入
    policy, alloc_s, alloc_size, alloc_len = push_into_bin(...)

    # 5. 处理冲突（抢占低优先级任务）
    while np.any(conflict_slot):
        # 找到冲突任务
        conflict_pid = ...

        # 抢占优先级最低的任务
        _pid_2b_preempt = conflict_pid.pop(0)
        preemption_list.append(_p_2b_preempt)

        # 释放资源
        bin.release(_p_2b_preempt, ...)

        # 更新冲突状态
        conflict_slot = rsc_alloc > rsc_avl

    # 6. 最终分配
    bin.allocate(_p.pid, alloc_s, alloc_size, alloc_len)
```

### 4.6 push_into_bin 插入策略

```python
def push_into_bin(_p, bin, ...):
    # 策略 1: block - 连续块插入
    state, chunk_s, chunk_len = bin.block_insert(...)
    if state:
        policy = 'block'

    # 策略 2: asap - 尽快插入
    elif rsc_avl[:expected_slot_num].sum() < expected_req_rsc_size:
        policy = 'asap'
        bin.asap_insert(...)

    # 策略 3: aeap - 尽量均匀
    else:
        policy = 'aeap'
        bin.aeap_insert(...)

    return policy, chunk_s, curr_alloc, chunk_len
```

## 5. 插入策略详解

### 5.1 三种插入策略

| 策略 | 条件 | 行为 |
|------|------|------|
| `block` | 资源充足且连续 | 在连续时间槽中分配完整资源 |
| `asap` | 资源不足 | 尽快分配，接受部分分配 |
| `aeap` | 资源充足但分散 | 在时间窗口内均匀分配 |

### 5.2 策略选择流程

```
push_into_bin()
    │
    ├── block_insert() 成功?
    │   └─ Yes → return 'block'
    │
    ├── mode == "block"?
    │   └─ Yes → return 'N/A'
    │
    ├── 资源不足 (rsc_avl.sum() < required)?
    │   └─ Yes → asap_insert() → return 'asap'
    │
    └── 资源充足
        └─ aeap_insert() → return 'aeap'
```

## 6. 亲和性机制

### 6.1 亲和性来源

1. **历史分配**: `rsc_recoder_his` 记录任务之前在哪个 Bin
2. **预分配**: `task.pre_assigned_resource_flag` 指定任务有专属 Bin
3. **任务名匹配**: 任务名与 Bin 名相同时，绑定到该 Bin

### 6.2 亲和性排序

```python
def get_process_sort(bin_name_list, rsc_recoder_his, tie_break, affinity_en, affinity_level):
    def sort_fn(_p):
        score = 0

        # 1. Deadline (主要排序)
        score += _p.deadline

        # 2. 亲和性 (次要排序)
        if affinity_en:
            if _p.task.name in bin_name_list:
                score -= 1e6  # 高优先级
            elif _p.pid in rsc_recoder_his:
                score -= 1e3  # 中等优先级

        # 3. Tie break
        score += tie_break(_p)[0] * 1e-3

        return score
    return sort_fn
```

### 6.3 Bin 选择优先级

```python
# 1. 目标 Bin（高亲和性）
affinity_tgt_bin_id_list = get_target_bin_id(_p, bin_name_list, rsc_recoder_his)

# 2. 搜索 Bin（低亲和性）
affinity_search_bin_id_list = [n for n in range(len(bin_list))
                               if n not in affinity_tgt_bin_id_list]
```

## 7. 抢占机制

### 7.1 抢占条件

1. **优先级**: 新任务优先级 > 被抢占任务
2. **资源需求**: 新任务需要被抢占任务的资源
3. **量子检查** (可选): 被抢占任务已执行整数倍量子时间

### 7.2 抢占流程

```python
# 1. 找到可抢占任务
preemptable_map = index_occupy_by_id_chunk_ver(...)

# 2. 按优先级排序（降序，低优先级先被抢占）
preemptable_map = sorted(preemptable_map.items(),
                         key=lambda item: process_sort(_p_index_by_pid[item[0]]),
                         reverse=True)

# 3. 逐个抢占直到资源足够
while np.any(conflict_slot):
    _pid_2b_preempt = conflict_pid.pop(0)  # 最低优先级
    preemption_list.append(_p_2b_preempt)
    bin.release(_p_2b_preempt, ...)
```

### 7.3 抢占粒度控制

```python
# quantum_check_en = True 时
cum_exec_quantum = _p_2b_preempt.cumulative_executed_time / quantumSize
reach_preempt_grain = math.isclose(cum_exec_quantum, round(cum_exec_quantum))

# 只有达到抢占粒度才能被抢占
if _p_2b_preempt.currentburst > 0 and not reach_preempt_grain:
    continue  # 跳过，不能抢占
```

## 8. 关键问题与改进建议

### 8.1 当前问题（§8.1.1/8.1.2 均已于 2026-06-16 解决，详见各节标注）

#### 8.1.1 pre_defined 模式的硬退出 ✅ 已解决（2026-06-16）

> **状态**: 已落实建议。`sys.exit(1)` 已改为 `raise ResourceInsufficientError`（`pre_alloc_new.py:206`，异常类定义于 `:21`）。

**位置**（历史）: `pre_alloc_new.py:192-195`

```python
if bin_list[bin_id].num_resources < req_rsc_size:
    sys.exit(1)  # 硬退出
```

**问题**: 当 repack 阶段资源估算与 Phase 1 不一致时，直接退出

**建议**: 改为抛出异常或警告 ✅ 已落实

#### 8.1.2 资源估算依赖外部常量 ✅ 已解决（2026-06-16）

> **状态**: 已落实建议。`tot_cores = 300` 已改为 `tot_cores = binpack_cfg.get("total_cores", 300)`（`pre_alloc_new.py:158`）。

**位置**（历史）: `pre_alloc_new.py:173`

```python
tot_cores = 300  # TODO: Make this a configurable parameter
```

**问题**: 硬编码的最大核心数

**建议**: 从 binpack_cfg 或全局配置读取 ✅ 已落实

### 8.2 功能完整性评估

| 功能 | 状态 | 说明 |
|------|------|------|
| 基本装箱 | ✅ | 支持 block/asap/aeap 策略 |
| 亲和性 | ✅ | 支持历史和预分配 |
| 抢占 | ✅ | 支持优先级抢占 |
| 部分分配 | ⚠️ | 代码存在但默认禁用 |
| 量子检查 | ⚠️ | 代码存在但默认禁用 |
| pre_defined 模式 | ✅ | 硬退出已改为异常（2026-06-16，见 §8.1.1） |

### 8.3 改进建议

1. **移除硬退出**: 将 `sys.exit(1)` 改为异常 ✅ 已落实（ResourceInsufficientError）
2. **配置化 tot_cores**: 添加到 BinPackConfig ✅ 已落实（binpack_cfg["total_cores"]）
3. **增强日志**: 添加更详细的分配/抢占日志
4. **统一参数名**: `quantile` vs `exec_t_comp_ratioB` 的语义澄清

## 9. 使用示例

### 9.1 Phase 2 (repack) 典型调用

```python
# 配置
binpack_cfg = BinPackConfig({
    "bin_sel_mod": "pre_defined",
    "mode": "non-block",
    "mapping": pid2bin_id,  # Phase 1 的输出
    "quantile": 0.80,       # exec_t_comp_ratioB
})

# 调用
bin_list = push_task_into_bins_new(
    bin_list, glb_p_list, affinity, event_iter_dict,
    total_cores, quantum_check_en, quantumSize,
    timestep, hyper_p, quantile,
    scheduler_list, monitor_list, msg_dispatcher,
    a_data_pipe, w_data_pipe,
    n_p=1, binpack_cfg=binpack_cfg,
    warmup=True, drain=True,
)
```

### 9.2 输出数据结构

```python
# bin_list[0].sparse_list 示例
[
    (0, {101: 16, 102: 8}, 100),   # slot 0 开始，101号任务16核，102号任务8核，持续100槽
    (100, {101: 16, 103: 8}, 50),  # slot 100 开始，新配置
    ...
]
```
