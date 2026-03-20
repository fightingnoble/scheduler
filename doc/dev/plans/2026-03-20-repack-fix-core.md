# Repack Fix-Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 repack 阶段用 ratioB 分位数重算每个任务的执行时长（latency），保持 Phase 1 的核心数不变，从而收紧 ERT/DDL，使尾部出现余量。

**Architecture:** 在 `deduce_cfg2` 增加可选参数 `fix_core_map: Dict[str, int]`。当该参数传入时，跳过 `HeuriRscSlackEstim` 的核心数求解，直接用 fix_core_map 里每个节点的核心数加上 ratioB 分位数的 FLOPS 计算 latency，再调用现有的 `init_topo_time_attr` 推算 ERT/DDL。调用方（`gen_workloads`）在 repack 时从 `bin_list` 中读取各任务的 `num_resources` 构造 `fix_core_map` 并传入。

**Tech Stack:** Python，NetworkX，现有 `sched/slack_estim.py`、`sched/packing_solver/chain_slack_assign.py`、`task/task_cfg.py`

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `sched/slack_estim.py` | Modify L393-424 | `deduce_cfg2` 增加 `fix_core_map` 参数；新增内部函数 `_fixcore_slack_estim` |
| `task/task_cfg.py` | Modify L1011-1080 | `gen_workloads` 增加 `fix_core_map` 参数，透传给 `deduce_cfg2` |
| `approach_setup.py` | Modify L134-137 | repack 调用 `run_benchmark_setup_pipeline` 时，将 `fix_core_map` 从 `bin_list` 提取后传入 |
| `sim_main.py` | Modify L11-88 | `run_benchmark_setup_pipeline` 签名增加 `fix_core_map` 参数，透传给 `build_workload_and_criticality`（`gen_workloads`）|

---

## 数据流

```
setup_benchmark (approach_setup.py)
  Phase 1:
    run_benchmark_setup_pipeline(need_repack=False, fix_core_map=None)
      gen_workloads(args, fix_core_map=None)
        deduce_cfg2(..., fix_core_map=None)   → 正常 HeuriRscSlackEstim 求核心数
      → 返回 bin_list (每个 bin 有 scheduling_table，每个 item 有 num_resources)

  Repack:
    fix_core_map = {pid → cores} 从 bin_list.scheduling_table 提取
    run_benchmark_setup_pipeline(need_repack=True, fix_core_map=fix_core_map)
      gen_workloads(args, fix_core_map=fix_core_map)
        deduce_cfg2(..., fix_core_map=fix_core_map)
          → 跳过 HeuriRscSlackEstim
          → 直接用 fix_core_map[node] × ratioB 分位数 FLOPS 计算 latency
          → 调用 init_topo_time_attr → ERT/DDL 收紧
```

---

## Task 1: 增加 `_fixcore_slack_estim` 并修改 `deduce_cfg2`

**Files:**
- Modify: `sched/slack_estim.py:393-424`

### 实现说明

`fix_core_map` 格式：`Dict[str, int]`，key 为节点名（logical graph 中的 op 节点名），value 为 Phase 1 分配的核心数。

新逻辑：不调用 `rsc_slack_estim`，直接遍历每个节点：
- `AccVarDist`: `latency = load_dist.quantile(ratioB) / cores / FLOPS_PER_CORE + exec_dist.quantile(ratioB)`
- `SenVarDist`: `latency = dist.quantile(ratioB)`, cores=1

然后构造 `rsc_map_w` 并调用 `init_topo_time_attr`（复用现有函数）。

- [ ] **Step 1: 阅读 `deduce_cfg2` 上下文，确认 `FLOPS_PER_CORE` 常量位置**

```bash
grep -n "FLOPS_PER_CORE" sched/slack_estim.py sched/packing_solver/chain_slack_assign.py global_var.py
```

- [ ] **Step 2: 在 `sched/slack_estim.py` 中添加 `_fixcore_slack_estim` 函数**

在 `deduce_cfg2`（L393）**之前**插入：

```python
def _fixcore_slack_estim(
    logical_graph_nx: nx.DiGraph,
    fix_core_map: Dict[str, int],
    quantile: float,
) -> Dict[str, Tuple[int, float, Union[str, None]]]:
    """
    Repack 模式下的 latency 重算：保持 Phase 1 核心数不变，
    用 quantile（ratioB）重算每个节点的 latency。
    返回 rsc_map_w 格式，可直接传给 init_topo_time_attr。
    """
    rsc_map_w: Dict[str, Tuple[int, float, Union[str, None]]] = {}
    for node, data in logical_graph_nx.nodes(data=True):
        if data.get('type') == 'sink':
            continue
        var_dist = data.get('var_dist')
        if var_dist is None:
            continue
        if isinstance(var_dist, SenVarDist):
            latency = float(var_dist.quantile(quantile))
            cores = 1
        else:
            cores = fix_core_map.get(node, 1)
            load_q = float(var_dist.load_dist.quantile(quantile))
            io_q   = float(var_dist.exec_dist.quantile(quantile))
            latency = load_q / max(cores, 1) / FLOPS_PER_CORE + io_q
        rsc_map_w[node] = (cores, elim_nume_error(latency), None)
    return rsc_map_w
```

- [ ] **Step 3: 修改 `deduce_cfg2` 签名，增加 `fix_core_map` 参数**

```python
def deduce_cfg2(taskattr_dict,
               logical_graph_nx, task_graph_srcs, task_graph_sinks,
               quantile, slack_threshold,
               verbose=False, plot=False,
               fix_core_map: Dict[str, int] = None):   # ← 新增
```

- [ ] **Step 4: 在 `deduce_cfg2` 内部，根据 `fix_core_map` 分支**

将 L404-409 的 `get_chains` + `rsc_slack_estim` 替换为：

```python
    if fix_core_map is not None:
        # Repack 模式：固定核心数，只用 quantile（ratioB）重算 latency
        rsc_map_w = _fixcore_slack_estim(logical_graph_nx, fix_core_map, quantile)
    else:
        # 正常 Phase 1 模式：求解最优核心数
        chains_info = get_chains(logical_graph_nx, task_graph_srcs, task_graph_sinks, taskattr_dict,
                                   quantile=quantile, remove_src_sink=False)
        rsc_map_w = rsc_slack_estim(logical_graph_nx, taskattr_dict, chains_info,
                                    slack_threshold,
                                    algorithm="avg",
                                    quantile=quantile)
```

L413 起的 `init_topo_time_attr` 调用不变（两个分支共用）。

- [ ] **Step 5: 手动验证逻辑（无正式测试文件，用 print 检查）**

```bash
conda activate gurobi
python -c "
from sched.slack_estim import deduce_cfg2
print('import OK')
"
```

- [ ] **Step 6: Commit**

```bash
git add sched/slack_estim.py
git commit -m "feat: deduce_cfg2 supports fix_core_map for repack latency recalc"
```

---

## Task 2: `gen_workloads` 透传 `fix_core_map`

**Files:**
- Modify: `task/task_cfg.py:1011-1084`

- [ ] **Step 1: 修改 `gen_workloads` 签名**

```python
def gen_workloads(args, fix_core_map: Dict[str, int] = None):
```

- [ ] **Step 2: 将 `fix_core_map` 透传给 `deduce_cfg2`（L1073）**

```python
    ert, ddl, rsc_map_w = deduce_cfg2(
        taskattr_dict,
        logical_graph_nx, srcs, sinks,
        args.quantile, args.slack_threshold,
        verbose=args.verbose, plot=args.plot,
        fix_core_map=fix_core_map)          # ← 新增
```

- [ ] **Step 3: 验证 import**

```bash
conda activate gurobi
python -c "from task.task_cfg import gen_workloads; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add task/task_cfg.py
git commit -m "feat: gen_workloads accepts fix_core_map for repack"
```

---

## Task 3: `build_workload_and_criticality` 透传 `fix_core_map`

**Files:**
- Modify: `sim_main.py` — `build_workload_and_criticality` 函数

首先找到该函数的实际位置：

```bash
grep -n "def build_workload_and_criticality" sim_main.py
```

- [ ] **Step 1: 修改 `build_workload_and_criticality` 签名**

增加 `fix_core_map=None`，透传给 `gen_workloads`：

```python
def build_workload_and_criticality(args, fix_core_map=None):
    ...
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list = gen_workloads(
        args, fix_core_map=fix_core_map
    )
    ...
```

- [ ] **Step 2: Commit**

```bash
git add sim_main.py
git commit -m "feat: build_workload_and_criticality forwards fix_core_map"
```

---

## Task 4: `run_benchmark_setup_pipeline` 透传 `fix_core_map`

**Files:**
- Modify: `approach_setup.py:11-88`

- [ ] **Step 1: 修改 `run_benchmark_setup_pipeline` 签名**

```python
def run_benchmark_setup_pipeline(
    args, path_ctx, path_params, need_repack, hyper_p,
    bin_list, num_cores,
    fix_core_map=None,          # ← 新增
    ):
```

- [ ] **Step 2: 在 `build_workload_and_criticality` 调用处透传**

```python
            workload = build_workload_and_criticality(args, fix_core_map=fix_core_map)
```

- [ ] **Step 3: Commit**

```bash
git add approach_setup.py
git commit -m "feat: run_benchmark_setup_pipeline forwards fix_core_map"
```

---

## Task 5: `setup_benchmark` 提取 `fix_core_map` 并在 repack 调用时传入

**Files:**
- Modify: `approach_setup.py:109-137`

### 如何从 `bin_list` 提取 `fix_core_map`

Phase 1 的 `bin_list` 里每个 bin 的 `scheduling_table` 记录了 `pid → TimeSlot`，每个 TimeSlot 包含 `req_rsc_size`。

但我们需要的是 **logical graph 节点名 → cores**，而 `pid` 对应 `ProcessInt`，`p.task.name` 就是节点名。

所以提取逻辑为：从 `glb_p_list` 中，对每个 `p`，找它在 Phase 1 的 `req_rsc_size`。Phase 1 `bin_list` 中的 `scheduling_table` 存的是 pid → slot，slot 有 `req_rsc_size`。

等等，要确认 `scheduling_table` 的数据结构：

```bash
grep -n "scheduling_table\|req_rsc_size\|index_occupy_by_id" sched/global_sched.py | head -30
```

**关键**：实际上 `rsc_map_w` 来自 `deduce_cfg2`，它在 `update_taskattr_dict` 中被写入 `taskattr.main_size`，而 `main_size` 就是 Phase 1 的核心数。所以从 Phase 1 run 完之后的 `bin_list` 中的 `scheduling_table` 里取 `req_rsc_size`，或者直接用 `glb_p_list` 里每个 p 的 `task` 属性。

更简单的方式：`bin_list` 里每个 bin 的 `scheduling_table` 里存了 `(pid, req_rsc_size, time_slot_s, time_slot_e)`。

在 `setup_benchmark` 中，Phase 1 返回 `bin_list` 后：

- [ ] **Step 1: 确认 `scheduling_table` 数据结构**

```bash
grep -n "scheduling_table" sched/global_sched.py | head -20
grep -n "index_occupy_by_id\|add_task\|scheduling_table" sched/bin_agent.py 2>/dev/null | head -20
grep -rn "def index_occupy_by_id\|scheduling_table" sched/ | head -20
```

- [ ] **Step 2: 实现 `fix_core_map` 提取**

在 `setup_benchmark` 中，Phase 1 完成后、repack 调用前：

```python
    if need_repack:
        # 从 Phase 1 bin_list 提取每个任务的核心数
        # bin.scheduling_table 的 key 是 pid，value 包含 req_rsc_size
        # ProcessInt 的 task.name 是 logical graph 节点名
        fix_core_map: Dict[str, int] = {}
        # 方法：遍历 bin_list，从 scheduling_table 中读取 pid → req_rsc_size
        # 再通过 glb_p_list 的 pid → task.name 映射转换
        # 注意：glb_p_list 在 run_benchmark_setup_pipeline 内部，这里没有直接访问
        # 解决：改为从 bin.index_occupy_by_id() 获取 pid 集合，
        #      然后通过 bin 的 scheduling_table 读 req_rsc_size
        # 实际数据结构需要 Step 1 确认后填入
        ...

        args.quantile = args.exec_t_comp_ratioB
        hyper_p, bin_list, num_cores = run_benchmark_setup_pipeline(
            args, path_ctx, path_params, True, hyper_p, bin_list, num_cores,
            fix_core_map=fix_core_map,
        )
```

**注意**：`glb_p_list` 和 `bin_list` 之间的连接需要通过 `scheduling_table` 或者另一个方案——直接在 Phase 1 的 `rsc_map_w` 里读取（在 `deduce_cfg2` 返回值 `rsc_map_w` 里，key 是节点名，value 的第 0 元素是 cores）。

**更简洁的方案**：让 `run_benchmark_setup_pipeline` 在 Phase 1 时也返回 `rsc_map_w`（节点名 → cores），这样 `setup_benchmark` 可以直接用它构建 `fix_core_map`。

- [ ] **Step 3: 修改 `run_benchmark_setup_pipeline` 在 Phase 1 时也返回 `rsc_map_w`**

这需要 `gen_workloads` 也返回 `rsc_map_w`。修改 `gen_workloads`：

```python
    return hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w
```

`run_benchmark_setup_pipeline`：

```python
    workload = build_workload_and_criticality(args, fix_core_map=fix_core_map)
    hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w = workload
    ...
    return hyper_p, bin_list, num_cores, rsc_map_w
```

`setup_benchmark`：

```python
    hyper_p, bin_list, num_cores, phase1_rsc_map = run_benchmark_setup_pipeline(
        args, path_ctx, path_params, False, None, bin_list, num_cores
    )
    if need_repack:
        # rsc_map_w[node] = (cores, latency, constr)
        fix_core_map = {node: cores for node, (cores, _, _) in phase1_rsc_map.items()}
        ...
```

**这是最干净的方案**：复用已有的 `rsc_map_w`，无需解析 bin 结构。

- [ ] **Step 4: 更新所有函数的签名和调用链**

  修改顺序（从底层到上层）：
  1. `gen_workloads` → 返回 `(hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list, rsc_map_w)`
  2. `build_workload_and_criticality` → 透传 `rsc_map_w`
  3. `run_benchmark_setup_pipeline` → 返回 `(hyper_p, bin_list, num_cores, rsc_map_w)`
  4. `setup_benchmark` → 在 Phase 1 后用 `rsc_map_w` 构建 `fix_core_map`

- [ ] **Step 5: 端到端冒烟测试**

```bash
conda activate gurobi
python main_approach.py \
  --profiling_filename profiling/profiling_light.csv \
  --gen_benchmark --G_decomp_mode full \
  --exec_t_comp_ratioA 0.7 --exec_t_comp_ratioB 0.5 \
  --e2e_latency 0.1 --aux_scale_factor 1 \
  --test_case cyclic --num_bins -1 \
  --bin_pack_cfg Bp_guided.json --n_p 2 \
  --policy cyc --root_dir test_fixcore/single_test 2>&1 | head -50
```

预期：无 error，能正常运行到仿真阶段。

- [ ] **Step 6: 验证 ERT/DDL 收紧**

在 `deduce_cfg2` 中临时加打印：

```python
    if fix_core_map is not None:
        rsc_map_w = _fixcore_slack_estim(logical_graph_nx, fix_core_map, quantile)
        total_latency = sum(v[1] for v in rsc_map_w.values())
        import sys
        sys.stderr.write(f"[FIXCORE] total latency sum = {total_latency:.4f} (should be < e2e_latency)\n")
```

运行对比：
- Phase 1: `total_latency ≈ e2e_latency`（占满）
- Repack (ratioB=0.5): `total_latency < e2e_latency`（有余量）

- [ ] **Step 7: Commit**

```bash
git add approach_setup.py sim_main.py task/task_cfg.py sched/slack_estim.py
git commit -m "feat: repack fixcore mode - use ratioB latency with Phase1 cores"
```

---

## 回归测试

```bash
# cyc 策略（无 repack）
python main_approach.py \
  --exec_t_comp_ratioA 0.7 --exec_t_comp_ratioB -1 \
  --policy cyc --num_bins -1 ...

# cyc-S（repack with fixcore）
python main_approach.py \
  --exec_t_comp_ratioA 0.7 --exec_t_comp_ratioB 0.5 \
  --policy cyc --num_bins -1 ...

# reserv（repack with fixcore）
python main_approach.py \
  --exec_t_comp_ratioA 0.7 --exec_t_comp_ratioB 0.5 \
  --policy reserv --num_bins 4 ...
```

---

## 关键约束

1. `_fixcore_slack_estim` 不调用 Gurobi，无 ILP 依赖
2. `init_topo_time_attr` 完全复用，不修改
3. Phase 1 路径（`fix_core_map=None`）**零行为变化**
4. `rsc_map_w` 返回给 `update_taskattr_dict`，`req_rsc_size` 为 Phase 1 值，不变
