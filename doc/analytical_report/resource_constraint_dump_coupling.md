# 资源约束与 Dump 逻辑的耦合分析

> **状态**: ✅ 已解决 (2026-02)
>
> **解决方案**: 将资源约束和 dump 逻辑从 `perform_bin_packing` 移到外层 `run_benchmark_setup_pipeline`。
>
> **相关文件**:
> - `sim_main.py`: `perform_bin_packing` 只负责装箱
> - `approach_setup.py`: `run_benchmark_setup_pipeline` 统一处理资源约束和 dump
> - `doc/spec/e2e_sched_sim_flow.md`: 更新后的设计规范

## 1. 耦合位置

### 1.1 资源约束逻辑

**Guided 算法 - Non-Repack 分支** (sim_main.py:471-474):
```python
if args.num_cores is not None:
    num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
else:
    num_cores = max_core_num
```

**Scratch 算法**: 无资源约束逻辑，直接使用输入的 `num_cores`

**Guided 算法 - Repack 分支**: 无资源约束逻辑，使用原始输入的 `num_cores`

### 1.2 Dump 逻辑

**路径生成** (sim_main.py:514-521):
```python
if need_repack:
    extra_suffix = f"_ov_{args.exec_t_comp_ratioB:.2f}_repack(T)"
else:
    extra_suffix = ""
bin_list_save_path, routing_table_save_path = generate_bin_paths(
    path_para_dict, path_ctx, num_cores, "packing save path",
    extra_suffix
)
```

**路径生成函数** (sim_main.py:27-58):
```python
def generate_bin_paths(path_para_dict, path_ctx: PathContext, num_cores, check_hints, extra_suffix=""):
    # 使用 num_cores 生成路径
    path_ctx.num_cores = num_cores
    if extra_suffix:
        path_ctx.file_suffix = f"{path_ctx.file_suffix}{extra_suffix}"
    new_bin_list_save_path = path_ctx.get_bin_list_path()
    new_routing_table_save_path = path_ctx.get_routing_table_path()
    return new_bin_list_save_path, new_routing_table_save_path
```

---

## 2. 耦合关系

### 2.1 数据流依赖

```
装箱算法 (push_task_into_bins_new / coleasing_alloc_cluster)
    ↓
返回 max_core_num (装箱计算的资源需求)
    ↓
资源约束 (apply_forced_num_cores)
    ↓
修改 num_cores (应用约束后的资源数)
    ↓
Dump 路径生成 (generate_bin_paths)
    ↓
使用 num_cores 生成文件路径
```

### 2.2 为什么耦合

**Dump 路径依赖约束后的 `num_cores`**:
- 文件名包含核心数（如 `bin_list_n_cores_16.pkl`）
- 如果使用 `max_core_num=20` 但约束后 `num_cores=16`，路径应该是 `n_cores_16` 而不是 `n_cores_20`
- 因此 **必须在资源约束之后生成路径**

---

## 3. 当前问题

### 3.1 资源约束位置不统一

| 算法分支 | 资源约束位置 | 问题描述 |
|---------|-------------|----------|
| **Scratch** | ❌ 无（外层传入） | 依赖外层正确设置 `num_cores` |
| **Guided Non-Repack** | ✅ 内部应用 (471-474行) | 与 dump 逻辑耦合 |
| **Guided Repack** | ❌ 无（使用输入） | 返回输入 `num_cores`，而非实际 `max_core_num` |

### 3.2 Guided Repack 分支的潜在 Bug

```python
# Line 483: 计算了 max_core_num
max_core_num = sum(b.num_resources for b in bin_list)

# Line 484-494: 调用 push_task_into_bins_new（内部可能修改 bin_list）
bin_list = push_task_into_bins_new(...)

# Line 520: 返回输入参数 num_cores，而非实际 max_core_num
return bin_list_save_path, num_cores, glb_p_list, hyper_p
```

**问题**: 如果 non-repack 阶段应用了资源约束（如 `max_core_num=20` → `num_cores=16`），repack 阶段返回的仍然是 `num_cores=16`，但实际 `bin_list` 可能有 `max_core_num=20` 个核心。

---

## 4. 解耦方案

### 方案 A: 移除内部资源约束，统一到外层

**优点**:
- 职责清晰：`perform_bin_packing` 只负责装箱
- 外层统一处理资源约束和 dump

**缺点**:
- 需要同时移动 dump 逻辑到外层
- `perform_bin_packing` 需要返回 `max_core_num` 而非 `num_cores`

**修改**:
```python
# sim_main.py - perform_bin_packing
def perform_bin_packing(...):
    # ... 装箱逻辑 ...
    # 返回 max_core_num（装箱计算的资源需求），而非应用约束后的 num_cores
    return bin_list, max_core_num, glb_p_list, hyper_p

# approach_setup.py - run_benchmark_setup_pipeline
def run_benchmark_setup_pipeline(...):
    # 执行装箱
    bin_list, max_core_num, glb_p_list, hyper_p = perform_bin_packing(...)

    # 应用资源约束（统一位置）
    if args.num_cores is not None and max_core_num != args.num_cores:
        num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
    else:
        num_cores = max_core_num

    # Dump（使用约束后的 num_cores）
    bin_list_save_path = generate_bin_paths(..., num_cores, ...)
    dump_and_check(bin_list_save_path, bin_list)
    return hyper_p, bin_list, num_cores
```

### 方案 B: 保持现状，但修复 repack bug

**优点**:
- 改动最小
- 不破坏现有结构

**缺点**:
- 资源约束逻辑仍在内部
- 耦合依然存在

**修改**:
```python
# sim_main.py - perform_bin_packing (guided repack 分支)
# Line 483 后添加：
if args.num_cores is not None and max_core_num != args.num_cores:
    num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
else:
    num_cores = max_core_num
```

---

## 5. 推荐方案

### 推荐：方案 A（彻底解耦）

**理由**:
1. **职责清晰**: `perform_bin_packing` 只负责装箱，返回计算的资源需求
2. **统一控制**: 资源约束在外层统一处理
3. **可测试性**: 装箱算法与约束逻辑分离，易于单独测试
4. **符合设计原则**: 单一职责原则（SRP）

**实施步骤**:
1. 修改 `perform_bin_packing` 返回 `max_core_num`
2. 在 `run_benchmark_setup_pipeline` 中应用资源约束
3. 将 dump 逻辑移到 `run_benchmark_setup_pipeline`
4. 修复 repack 分支的 bug

---

## 6. 修改清单

### 修改 1: perform_bin_packing 返回 max_core_num

**文件**: `sim_main.py`

**位置**: 所有 `return` 语句

**修改**:
```python
# 当前
return bin_list_save_path, num_cores, glb_p_list, hyper_p

# 修改后
return bin_list, max_core_num, glb_p_list, hyper_p
```

### 修改 2: 移除内部资源约束

**文件**: `sim_main.py`

**位置**: 471-474 行（guided non-repack 分支）

**删除**:
```python
if args.num_cores is not None:
    num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
else:
    num_cores = max_core_num
```

### 修改 3: 移除内部 dump 逻辑

**文件**: `sim_main.py`

**位置**: 514-533 行

**删除**:
```python
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

### 修改 4: 在外层添加资源约束和 dump

**文件**: `approach_setup.py`

**位置**: `run_benchmark_setup_pipeline` 函数

**添加**:
```python
def run_benchmark_setup_pipeline(...):
    # ... 现有代码 ...

    # 执行装箱
    bin_list, max_core_num, glb_p_list, hyper_p = perform_bin_packing(
        ..., need_repack=need_repack, ...
    )

    # 应用资源约束（统一位置）
    if args.num_cores is not None and max_core_num != args.num_cores:
        num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)
    else:
        num_cores = max_core_num

    # 生成 dump 路径
    if need_repack:
        extra_suffix = f"_ov_{args.exec_t_comp_ratioB:.2f}_repack(T)"
    else:
        extra_suffix = ""
    bin_list_save_path, routing_table_save_path = generate_bin_paths(
        path_para_dict, path_ctx, num_cores, "packing save path",
        extra_suffix
    )

    # Dump
    Bin_list_print(bin_list, glb_p_list, sim_step)
    if args.plot:
        render_bin_pack_plots(args, bin_list, glb_p_list, sim_step, hyper_p, num_periods, plot_path_para, path_ctx)
    dump_and_check(bin_list_save_path, bin_list)

    return hyper_p, bin_list, num_cores
```

---

## 7. 相关文档

- `doc/spec/binpack_config_design.md` - BinPackConfig 设计
- `doc/code_cleanup_2025.md` - 代码清理记录
- `path_migration_guide.md` - PathContext 迁移指南
