# 代码清理记录 - 2026

## 清理日期：2026-02-08

---

## 清理5：资源约束与 Dump 逻辑解耦

### 问题描述

`perform_bin_packing` 函数职责混乱，包含了三个不相关的逻辑：
1. 装箱算法执行
2. 资源约束应用（`apply_forced_num_cores`）
3. Dump 操作（路径生成、序列化）

这导致：
- 职责不清晰，违反单一职责原则
- 资源约束逻辑分散在多个位置
- Repack 和 non-repack 阶段的资源约束判断交织
- `determine_resource_config` 函数包含复杂的条件判断

### 影响文件

1. **sim_main.py**
   - `perform_bin_packing` 函数（398-518行）
   - 移除内部资源约束逻辑（guided non-repack 分支的 471-474 行）
   - 移除内部 dump 逻辑（514-537 行）
   - 修改返回值：从 `(bin_list_save_path, num_cores, ...)` 改为 `(bin_list, max_core_num, ...)`
   - 为 scratch 算法添加 `max_core_num` 计算
   - `determine_resource_config` 函数标记为 deprecated（251-276行）

2. **approach_setup.py**
   - `run_benchmark_setup_pipeline` 函数（11-83行）
   - 接收 `perform_bin_packing` 的新返回值
   - 在外层应用资源约束（60-64行）
   - 在外层执行 dump 逻辑（66-82行）

### 清理前代码

#### sim_main.py - perform_bin_packing (471-474行)
```python
# Guided non-repack 分支：包含资源约束
max_core_num, pid2_bin_id, bin_size_list = coleasing_alloc_cluster(...)
if args.num_cores is not None:
    num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
else:
    num_cores = max_core_num
```

#### sim_main.py - perform_bin_packing (514-537行)
```python
# Dump 逻辑在装箱函数内部
if need_repack:
    extra_suffix = f"_ov_{args.exec_t_comp_ratioB:.2f}_repack(T)"
else:
    extra_suffix = ""
bin_list_save_path, routing_table_save_path = generate_bin_paths(...)
Bin_list_print(bin_list, glb_p_list, sim_step)
if args.plot:
    render_bin_pack_plots(...)
dump_and_check(bin_list_save_path, bin_list)
return bin_list_save_path, num_cores, glb_p_list, hyper_p
```

#### approach_setup.py - run_benchmark_setup_pipeline (48-59行)
```python
# 旧版本：接收 dump 路径
bin_list_save_path, num_cores, glb_p_list, hyper_p = perform_bin_packing(...)
assert bin_list_save_path is not None, "Error: Failed to perform bin packing"
```

### 清理后代码

#### sim_main.py - perform_bin_packing (返回值)
```python
# 只返回装箱结果
return bin_list, max_core_num, glb_p_list, hyper_p
```

#### approach_setup.py - run_benchmark_setup_pipeline (47-84行)
```python
# 6. 执行装箱算法（返回装箱结果，不包含资源约束和 dump）
bin_list, max_core_num, glb_p_list, hyper_p = perform_bin_packing(...)

# 7. 应用资源约束（仅 non-repack 时应用，repack 不改变资源数量）
if not need_repack:
    if args.num_cores is not None:
        num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
    else:
        num_cores = max_core_num

# 8. 生成 dump 路径
if need_repack:
    extra_suffix = f"_ov_{args.exec_t_comp_ratioB:.2f}_repack(T)"
else:
    extra_suffix = ""
bin_list_save_path, routing_table_save_path = generate_bin_paths(...)

# 9. 打印和绘制
Bin_list_print(bin_list, glb_p_list, sim_step)
if args.plot:
    render_bin_pack_plots(...)

# 10. Dump
dump_and_check(bin_list_save_path, bin_list)
```

### 清理原因

1. **单一职责原则**：`perform_bin_packing` 应该只专注于装箱算法
2. **逻辑清晰**：资源约束和 dump 是装箱后的独立操作
3. **减少耦合**：dump 依赖约束后的 `num_cores`，应在同一层级处理
4. **简化条件判断**：repack 相关的逻辑移到外层统一控制
5. **易于测试**：装箱算法可以独立测试，不依赖约束和 dump

### 重构效果

**职责分离**：
```
perform_bin_packing (sim_main.py)
└── 装箱算法
    ├── Scratch: push_task_into_bins_new
    ├── Guided non-repack: coleasing_alloc_cluster
    └── Guided repack: push_task_into_bins_new with pre-defined bins

run_benchmark_setup_pipeline (approach_setup.py)
├── 资源约束（统一位置，仅 non-repack）
└── Dump（路径生成、打印、绘制、序列化）
```

**数据流**：
```
Non-Repack:
  装箱 → max_core_num=20
    ↓
  约束 → num_cores=16
    ↓
  dump(num_cores=16)

Repack:
  装箱 → max_core_num=16（bin_list 已被约束修改）
    ↓
  不应用约束
    ↓
  dump(num_cores=16)
```

### 相关文档

- `doc/spec/resource_constraint_dump_coupling.md` - 耦合分析文档
- `doc/spec/e2e_sched_sim_flow.md` - 更新后的执行流程

---

## 清理日期：2026-02-03

---

## 清理1：删除 `release_temp_rda` 未使用参数

### 问题描述

`release_temp_rda` 是一个配置参数，但代码中完全没有使用它。该参数从 `binpack_cfg` 中读取，但读取后只是设置了注释掉的代码，实际功能已被 `get_rsc_2b_released()` 替代。

### 影响文件

1. **sched/global_sched.py**
   - 第194-195行：删除了对 `release_temp_rda` 的读取和相关注释
   - 第286行：删除了注释掉的代码

2. **sched/pre_alloc_new.py**
   - 第400行：删除了对 `release_temp_rda` 的读取
   - 第491-493行：删除了注释掉的代码块

### 清理前代码

#### sched/global_sched.py (第194-195行)
```python
release_temp_rda = binpack_cfg.get("release_temp_rda", True)
# bp_rls_mode = "future" if release_temp_rda else "none"
```

#### sched/global_sched.py (第286行)
```python
# curr_cfg.updateRunningQueue(timestep, running_queue, mode="verify" if release_temp_rda else "normal")
```

#### sched/pre_alloc_new.py (第400行)
```python
# release_temp_rda = binpack_cfg.get("release_temp_rda", True)
```

#### sched/pre_alloc_new.py (第491-496行)
```python
# resource to be released
if False: #not release_temp_rda:
    # A: the resource occupied from time_slot_s to time_slot_e
    alloc_s_t, alloc_size_t, alloc_len_t = preemptable_map[_pid_2b_preempt]
else:
    # B: the resource occupied from n_slot to future
    bin_id_t, alloc_s_t, alloc_size_t, alloc_len_t = get_rsc_2b_released(rsc_recoder, n_slot, _p_2b_preempt)
```

### 清理后代码

#### sched/global_sched.py (第194-195行)
```python
# 删除了这两行
```

#### sched/global_sched.py (第286行)
```python
curr_cfg.updateRunningQueue(timestep, running_queue)
```

#### sched/pre_alloc_new.py (第400行)
```python
# 删除了这一行
```

#### sched/pre_alloc_new.py (第491-496行)
```python
# resource to be released
# the resource occupied from n_slot to future
bin_id_t, alloc_s_t, alloc_size_t, alloc_len_t = get_rsc_2b_released(rsc_recoder, n_slot, _p_2b_preempt)
```

### 清理原因

1. **参数未实际使用**：`release_temp_rda` 只在被读取后用于注释掉的代码（`if False:`）
2. **功能已被替代**：实际资源释放逻辑使用 `get_rsc_2b_released()` 函数
3. **代码混淆**：未使用的参数和注释代码增加了理解难度
4. **配置冗余**：`release_temp_rda` 在配置文件中出现，但不会影响程序行为

### 相关配置文件

虽然删除了代码中的使用，但配置文件中的此参数不影响运行（因为从未被读取）：
- `cfgs/Bp_guided.json` (第7行)
- `cfgs/Bp_scratch.json` (第7行)

这些配置文件中的 `release_temp_rda` 字段可以后续一并删除。

---

## 清理2：删除 `push_step_new` 的未使用参数 `percentile`

### 问题描述

`push_step_new()` 函数签名中包含 `percentile: float` 参数，但该参数在函数体中完全没有被使用。真正的分位数参数是通过 `binpack_cfg` 字典传递的。这是一个"僵尸参数"，容易误导调用者。

### 影响文件

**sched/global_sched.py**
   - 第162行：从函数签名中删除 `percentile` 参数
   - 第138行：从调用处删除 `quantile` 参数传递

### 清理前代码

#### sched/global_sched.py (第162-165行)
```python
def push_step_new(
        sched: Scheduler, msg_dispatcher: MsgDispatcher,
        a_data_pipe: DataPipe, w_data_pipe: DataPipe,
        n_slot: int, timestep: float, percentile: float,  # ← 未使用参数
        event_range: float, sim_slot_num: int, curr_t: float,

        glb_name_p_dict, res_cfg: Resource_model_int,
        issue_sort_fn: issue_list:TaskQueue,
        iter_next_bin_obj, 

        quantumSize, bin_list:TaskQueue, bin_name_list, 
        binpack_cfg:Dict,
        show_warnings=True, 
        verbose:bool=False, DEBUG_FG:bool=False,
        ):
```

#### sched/global_sched.py (第138行)
```python
push_step_new(
    sched, msg_dispatcher, a_data_pipe, w_data_pipe, 
    n_slot, timestep, quantile,  # ← 传递但未使用
    event_range, sim_slot_num, curr_t, 
    ...
)
```

### 清理后代码

#### sched/global_sched.py (第162-165行)
```python
def push_step_new(
        sched: Scheduler, msg_dispatcher: MsgDispatcher,
        a_data_pipe: DataPipe, w_data_pipe: DataPipe,
        n_slot: int, timestep: float,  # ← 删除 percentile 参数
        event_range: float, sim_slot_num: int, curr_t: float,

        glb_name_p_dict, res_cfg: Resource_model_int,
        issue_sort_fn: issue_list:TaskQueue,
        iter_next_bin_obj, 

        quantumSize, bin_list:TaskQueue, bin_name_list, 
        binpack_cfg:Dict,  # ← 真正的参数传递方式
        show_warnings=True, 
        verbose:bool=False, DEBUG_FG:bool=False,
        ):
```

#### sched/global_sched.py (第138行)
```python
push_step_new(
    sched, msg_dispatcher, a_data_pipe, w_data_pipe, 
    n_slot, timestep,  # ← 删除 quantile 参数传递
    event_range, sim_slot_num, curr_t, 
    ...
)
```

### 清理原因

1. **参数未被使用**：函数体内没有任何地方引用 `percentile` 变量
2. **误导性**：调用者可能误以为通过 `percentile` 参数传递分位数
3. **实际传递方式**：分位数通过 `binpack_cfg['quantile']` 传递（在 `push_task_into_bins_new` 中设置）
4. **代码清晰度**：删除未使用参数后，函数签名更清晰

### 实际参数传递路径（清理后）

```
push_task_into_bins_new(quantile)
    ↓
binpack_cfg['quantile'] = quantile
    ↓
push_step_new(..., binpack_cfg=binpack_cfg)  # ← 通过 binpack_cfg 传递
    ↓
glb_alloc_new2(..., binpack_cfg=binpack_cfg)
    ↓
allocate_rsc_4_process_new2(..., binpack_cfg=binpack_cfg)
    ↓
_p.rsc_req_estm_quantile(..., binpack_cfg=binpack_cfg)
    ↓
q = binpack_cfg.get('quantile')
    ↓
load_q = dist.load_dist.quantile(q)
io_q = dist.exec_dist.quantile(q)
```

---

## 影响范围

### 删除的代码行数

- `sched/global_sched.py`: 4 行
- `sched/pre_alloc_new.py`: 4 行
- **总计**: 8 行

### 修改的函数

1. `push_step_new()` - 删除未使用参数
2. `glb_alloc_new2()` - 清理注释代码

### 无影响区域

- **功能无影响**：清理的都是未使用的参数和注释代码
- **性能无影响**：不影响实际运行逻辑
- **配置兼容**：`exec_t_comp_ratioA/B` 参数传递仍然正常工作

---

## 相关文档

- `doc/parameter_flow_exec_t_comp_ratio.md` - 参数传递路径文档（已更新完整调用链）

---

## 验证建议

清理后应验证：

1. **运行测试**：
   ```bash
   python -m scripts.abla_exp_runner --case 1 --num_hp 10
   python -m scripts.motiv_exp_runner --case 1 --num_hp 10
   ```

2. **检查日志**：确认没有使用 `release_temp_rda` 或 `percentile` 的警告或错误

3. **配置文件**：考虑从 `Bp_guided.json` 和 `Bp_scratch.json` 中删除 `release_temp_rda` 字段

---

## 清理3：移动 `binpack_cfg` 的 `var_dist_map` 和 `quantile` 设置逻辑

### 问题描述

`push_task_into_bins_new()` 内部曾包含一段逻辑，用于从 `job_graph` 提取 `var_dist_map` 并将其与 `quantile` 一起存入 `binpack_cfg`。这段逻辑不依赖于函数内部计算的任何结果，且在 `perform_bin_packing` 中多次调用时存在重复。

### 影响文件

1. **sim_main.py**
   - 在 `perform_bin_packing()` 中添加了 `prepare_binpack_cfg()` 辅助函数。
   - 在调用 `push_task_into_bins_new()` 和 `coleasing_alloc_cluster()` 之前，预先准备好包含 `var_dist_map` 和 `quantile` 的 `binpack_cfg`。

2. **sched/global_sched.py**
   - 从 `push_task_into_bins_new()` 中删除了提取 `var_dist_map` 和设置 `quantile` 的代码块。
   - 从 `push_task_into_bins_new()` 的函数签名中删除了不再使用的 `job_graph` 参数。

### 清理前代码 (sched/global_sched.py)

```python
def push_task_into_bins_new(..., job_graph: DiGraph, ...):
    # ...
    # Provide var_dist_map and quantile to downstream allocator if a job_graph is available
    var_dist_map = {p.task.name: job_graph.nodes[p.task.name]['var_dist'] for p in glb_p_list}
    binpack_cfg = dict(binpack_cfg)  # shallow copy to avoid side effects
    binpack_cfg['var_dist_map'] = var_dist_map
    binpack_cfg['quantile'] = quantile
    # ...
```

### 清理后代码 (sim_main.py)

```python
def perform_bin_packing(...):
    # ...
    # Prepare binpack_cfg with var_dist_map and quantile
    def prepare_binpack_cfg(cfg, quantile, p_list, graph):
        new_cfg = dict(cfg)
        new_cfg['var_dist_map'] = {p.task.name: graph.nodes[p.task.name]['var_dist'] for p in p_list}
        new_cfg['quantile'] = quantile
        return new_cfg

    if args.binpack_cfg["algorithm"] == "scratch":
        binpack_cfg_scratch = prepare_binpack_cfg(args.binpack_cfg, args.exec_t_comp_ratioB, glb_p_list, physical_graph_nx)
        bin_list = push_task_into_bins_new(..., binpack_cfg=binpack_cfg_scratch, ...)
    # ...
```

### 清理原因

1. **职责分离**：`push_task_into_bins_new` 应该专注于装箱逻辑，而不是参数的预处理。
2. **减少依赖**：删除了 `push_task_into_bins_new` 对 `job_graph` 的直接依赖，使其接口更简洁。
3. **逻辑统一**：在调用层统一准备配置，使得参数流向更透明。

---

## 总结

这次清理解决了以下代码质量问题：

1. **删除未使用参数 `release_temp_rda`**：简化了代码，删除了注释掉的代码块。
2. **删除僵尸参数 `percentile`**：使函数签名更清晰，避免误导调用者。
3. **重构配置预处理逻辑**：将 `binpack_cfg` 的准备工作上移至调用层，实现了更好的职责分离。
4. **增强入口参数校验**：在 `perform_bin_packing()` 入口处添加了对 `binpack_cfg` 及其 `algorithm` 键的显式检查。
5. **引入 `BinPackConfig` 包装器**：创建了 `sched/binpack_config.py`，将 `binpack_cfg` 字典包装为类，支持属性访问（如 `cfg.algorithm`）同时保持字典兼容性。

---

## 清理4：删除过期配置参数并替换 `default_binpack_cfg`

### 清理日期：2026-02-03

### 问题描述

代码中存在多个过期配置参数，这些参数：
1. 在 JSON 配置文件中定义
2. 在旧的 `default_binpack_cfg` 字典中定义
3. 但代码中从未实际读取或使用

### 删除的过期参数

| 参数 | 原位置 | 状态 |
|------|--------|------|
| `sort_reverse` | JSON, `default_binpack_cfg` | 代码中从未读取 |
| `release_temp_rda` | JSON, `default_binpack_cfg` | 已在清理1中标记删除 |
| `slack_sharing` | JSON | 代码中不再使用 |

### 影响文件

1. **cfgs/Bp_guided.json**
   - 删除 `sort_reverse`, `release_temp_rda`, `slack_sharing`
   - 保留实际使用的参数

2. **cfgs/Bp_scratch.json**
   - 删除 `sort_reverse`, `release_temp_rda`, `slack_sharing`
   - 保留实际使用的参数

3. **sched/global_sched.py**
   - 将 `default_binpack_cfg` 字典替换为 `BinPackConfig()` 实例
   - 添加 `from sched.binpack_config import BinPackConfig` 导入

4. **utils.py**
   - 删除过期的命令行参数注释 (`--bin_sort`, `--bin_sort_reverse`)

### 清理前代码

#### cfgs/Bp_guided.json
```json
{
    "sort":"barycenter",
    "sort_reverse":true,        // ← 删除
    "quantum_check_en": false,
    "partial_alloc_en": false,
    "mode": "block",
    "release_temp_rda":false,    // ← 删除
    "reservation_policy": "static_1_bin",
    "slack_sharing": false,      // ← 删除
    "algorithm": "guided",
    ...
}
```

#### sched/global_sched.py
```python
default_binpack_cfg = {
    "sort":"EAT",
    "sort_reverse":True,         // ← 过期参数
    "mode": 'non-block',
    "partial_alloc_en":False,
    "quantum_check_en":False,
    "release_temp_rda":True,     // ← 过期参数
    "reservation_policy": "manual",
    "algorithm": "coalescing",   // ← 旧名称，实际使用 "guided"/"scratch"
    "bin_sel_mod": "search",
    "affinity_en": True,
    "affinity_level": 2,
    "mapping": {}
}
```

#### utils.py
```python
# parser.add_argument("--bin_sort", default="EAT", type=str, help="bin sort: EAT, barycenter")
# parser.add_argument("--bin_sort_reverse", default=True, type=bool, help="bin sort reverse")
```

### 清理后代码

#### cfgs/Bp_guided.json
```json
{
    "sort":"barycenter",
    "quantum_check_en": false,
    "partial_alloc_en": false,
    "mode": "block",
    "reservation_policy": "static_1_bin",
    "algorithm": "guided",
    "preempt_en": false,
    "core_size": "induced"
}
```

#### sched/global_sched.py
```python
from sched.binpack_config import BinPackConfig

# 默认配置实例（使用 BinPackConfig 包装器）
# 注意：这些默认值主要作为函数签名的 fallback，实际运行时由 input_parser 从 JSON 文件加载
default_binpack_cfg = BinPackConfig()
```

#### utils.py
```python
# 过期注释已删除
```

### 清理原因

1. **配置一致性**：JSON 文件、代码和文档保持一致
2. **类型安全**：使用 `BinPackConfig()` 提供类型安全的默认值
3. **减少混淆**：删除未使用的参数避免误导
4. **维护性**：单一真实来源（BinPackConfig 的属性定义）

### BinPackConfig 默认值对照

| 参数 | BinPackConfig 默认值 | 来源 |
|------|---------------------|------|
| `algorithm` | "coalescing" | 属性默认 |
| `sort` | "EAT" | 属性默认 |
| `mode` | "non-block" | 属性默认 |
| `bin_sel_mod` | "search" | 属性默认 |
| `reservation_policy` | "manual" | 属性默认 |
| `affinity_en` | True | 属性默认 |
| `affinity_level` | 2 | 属性默认 |
| `preempt_en` | True | 属性默认 |
| `partial_alloc_en` | False | 属性默认 |
| `quantum_check_en` | False | 属性默认 |
| `mapping` | {} | 属性默认 |
| `quantile` | 0.99 | 属性默认（运行时覆盖）|
| `var_dist_map` | {} | 属性默认（运行时覆盖）|
| `core_size` | "specified" | 属性默认 |

---

## 总结

这次清理解决了以下代码质量问题：

1. **删除未使用参数 `release_temp_rda`**：简化了代码，删除了注释掉的代码块。
2. **删除僵尸参数 `percentile`**：使函数签名更清晰，避免误导调用者。
3. **重构配置预处理逻辑**：将 `binpack_cfg` 的准备工作上移至调用层，实现了更好的职责分离。
4. **增强入口参数校验**：在 `perform_bin_packing()` 入口处添加了对 `binpack_cfg` 及其 `algorithm` 键的显式检查。
5. **引入 `BinPackConfig` 包装器**：创建了 `sched/binpack_config.py`，将 `binpack_cfg` 字典包装为类，支持属性访问（如 `cfg.algorithm`）同时保持字典兼容性。
6. **删除过期配置参数**：从 JSON 文件和代码中删除 `sort_reverse`, `release_temp_rda`, `slack_sharing` 等过期参数。
7. **替换默认配置字典**：将 `default_binpack_cfg` 字典替换为 `BinPackConfig()` 实例，提供类型安全的默认值。

清理后的代码：
- 更易读：删除了混淆的注释和未使用参数。
- 更准确：参数传递路径更清晰。
- 更简洁：减少了冗余逻辑和不必要的函数参数。
- 更类型安全：使用 `BinPackConfig` 包装器提供属性访问和 IDE 支持。

