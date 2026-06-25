# Bin-packing 函数与文档现状分析

**日期**: 2026-06-16
**类型**: 整理分析（Phase 1 理解，未动代码）
**方法**: AST 调用图可达性闭包（从 `perform_bin_packing` 出发），非文本 grep

## 结论速览

bin_packing 代码确实"写过好几版"——当前存在 **3 代实现**堆叠在 5 个文件里（3431 行），其中约 **40% 是死代码**（旧版本残留 + 已坏死的备选分支）。活路径只走 2 个算法（`guided` / `scratch`），另外 2 个备选分支（`mem_plan` / `full`）中 `full` 已坏死（调用的函数被注释）。

## 一、活路径（perform_bin_packing 默认走 guided/scratch）

```
sim_main.py::perform_bin_packing
  ├─ [guided] coleasing_alloc_cluster  (global_sched.py:580)
  │     ├─ new_bin                     (bin_ops.py)
  │     ├─ get_initlist_and_biniter    (bin_ops.py)
  │     ├─ gurobi_split_solver         (global_sched.py:757)
  │     │     └─ ClusterGurobiSolverSemi2D  (packing_solver/gurobi_MP_semi2DClst.py)
  │     ├─ update_bp_result2_schedtab  (global_sched.py:660)
  │     └─ rename_bins_and_relable_assignments (global_sched.py:796)
  │           └─ build_greedy_obj / build_search_obj  (global_sched.py:826/884)
  └─ [scratch/repack] push_task_into_bins_new  (global_sched.py:42)
        └─ push_step_new               (global_sched.py:154)
              └─ glb_alloc_new2        (pre_alloc_new.py:26)
                    ├─ allocate_rsc_4_process_new2
                    ├─ bin_sel / push_into_bin / check_and_preemt_alloc
                    └─ get_rsc_2b_released / get_target_bin_id (monitor_agent.py)
```

**活的文件**: `sim_main.py`（perform_bin_packing）、`sched/global_sched.py`（部分）、`sched/pre_alloc_new.py`、`sched/bin_ops.py`、`sched/packing_solver/gurobi_MP_semi2DClst.py`、`sched/monitor_agent.py`（2 个 helper）

## 二、死代码（3 类）

### 2.1 旧版本整文件残留

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `sched/bin_ops.old.py` | 775 | **整文件死** | 0 个活文件 import。含旧版 `push_task_into_bins`/`push_step`/`glb_alloc`/`allocate_rsc_4_process`/`preempt_the_conflicts`（无 `_new` 后缀的初代实现） |
| `sched/pre_alloc.py` | 592 | **整文件死** | `glb_alloc_new`/`allocate_rsc_4_process_new`/`bin_select` 只被 bin_ops.old.py 调用。被 `pre_alloc_new.py`（`glb_alloc_new2` 等）取代 |

### 2.2 global_sched.py 内的死函数（约 1/3 文件）

| 函数 | 行 | 状态 | 说明 |
|------|----|------|------|
| `naive_iso` | 281 | 💀死 | 0 调用，隔离度算法旧版 |
| `coleasing_alloc_1bin` | 376 | ✅活 | 误判纠正：被可达闭包覆盖（coleasing_alloc_cluster 路径） |
| `test_mem_planner` | 318 | **条件死** | 仅 `algorithm=="mem_plan"` 分支调用；默认 guided 不走。本身可导入但实际不触发 |
| `single_turn_solver` | (注释) | **已坏死** | `elif algorithm=="full"` 分支调用，但定义在 global_sched.py:947 已被 `#` 注释 → 此分支一旦触发会 ImportError |

> **注意**：可达性闭包把 `coleasing_alloc_1bin`/`build_greedy_obj`/`build_search_obj`/`update_bp_result2_schedtab`/`gurobi_split_solver`/`rename_bins_and_relable_assignments` 判为活——它们被活的 `coleasing_alloc_cluster` 间接调用。第一轮"外部调用者"判定曾把它们误判为死，闭包纠正了。

### 2.3 死的求解器变体

| 文件 | 状态 | 说明 |
|------|------|------|
| `packing_solver/gurobi_semi2Dclst_mapping.py` | 💀死 | `GurobiSemi2DClstMapping` 类 0 活调用 |
| `packing_solver/gurobi_semi2Dclst_mapping2.py` | 💀死 | 同名类 v2，0 活调用（mapping 和 mapping2 是两个失败迭代） |
| `packing_solver/gurobi_MP_semi2DClst.py` | ✅活 | `ClusterGurobiSolverSemi2D` 被 coleasing_alloc_cluster 用 |

> `packing_solver/old/`（5 个：cplex/ortools/gurobi 各 1DBp/semi2DBp）和 `packing_solver/ref/`（5 个）已在 deprecated_code.md 标记，本次确认仍死。

## 三、"乱"的根因

1. **3 代堆叠无清理**：`bin_ops`(v0) → `bin_ops.old`(v0变体) → `global_sched`(`_new` 后缀 v1) → `pre_alloc_new`(`_new2` 后缀 v2)，旧代没删
2. **同名不同版**：`push_task_into_bins` / `push_step` / `glb_alloc` / `allocate_rsc_4_process` 在 bin_ops.old 和 global_sched/pre_alloc_new 各有一份（带/不带 `_new`/`_new2`）
3. **备选分支未清理**：`mem_plan`/`full` 两个 algorithm 分支保留，其中 `full` 的实现已被注释但调用还在（坏死分支）
4. **文档分散**：bin_packing 相关描述散落在 28 个 doc 文件，核心规范在 `doc/spec/algorithm/guided_hybrid_allocation_algorithm.md` + `binpack_solver_spec.md`，但 `doc/dev/` 下有大量历史记录（ablation_dev/repack_debug/change_log）可能描述的是旧版

## 四、建议的整理方向（待用户定，本轮不执行）

按"移动不删除"原则（沿用 B2 策略）：

1. **低风险先做**：移走 `bin_ops.old.py` + `pre_alloc.py`（整文件死，0 活依赖）→ `unused/`
2. **中风险**：移走 `gurobi_semi2Dclst_mapping.py` + `gurobi_semi2Dclst_mapping2.py`（死求解器变体）→ `packing_solver/old/`
3. **需决策**：`single_turn_solver` 坏死分支（perform_bin_packing 里的 `elif algorithm=="full"`）——移走分支还是注释？
4. **需决策**：`test_mem_planner`（`mem_plan` 分支）—— 这个算法还要保留吗？
5. **文档**：`doc/dev/` 下描述旧版的历史文档可归档，但需先确认哪些已被 spec 文档取代

## 五、本报告未覆盖（需进一步分析时再做）

- `packing_solver/chain_slack_assign.py`（slack 分配，Step 1，独立子模块）
- `packing_solver/fit.py`
- `sched/ref_alloc_search.py`、`scripts/test_alloc_lat.py`、`test_core_allocation.py`（独立测试脚本）
- 文档层的详细新旧对照（28 个 doc 文件）
