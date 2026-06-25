# Bin-packing 函数关系图（spec ↔ 代码对照）+ 文档层分析

**日期**: 2026-06-16
**基准 spec**: `doc/spec/algorithm/binpack_solver_spec.md`（2026-02 草稿，质量高，函数层次准确）
**方法**: AST 调用图（`Call` 节点，非 `Import` 节点）+ spec 描述对照

## 一、spec 描述的应然层次 vs 实际代码

`binpack_solver_spec.md §2.1` 给出的函数调用层次**基本准确**。下表标注每个函数的实际位置、活/死、以及 spec 的准确性：

```
push_task_into_bins_new (global_sched.py:42)          ✅活  spec ✓
    ├── get_initlist_and_biniter (bin_ops.py)          ✅活  spec ✓
    ├── message_trigger_event_new (message_handler.py) ✅活  spec ✓（注意有同名旧版 message_handler_old.py，死）
    └── push_step_new (global_sched.py:154)           ✅活  spec ✓
            ├── check_complete (state_trans.py:164)    ✅活* spec ✓（真身在 state_trans；global_sched 经 scheduler_agent 假身 import）
            ├── check_miss (state_trans.py:48)         ✅活* spec ✓（同上）
            ├── data_pipe_read (sched_utils.py:114)    ✅活* spec ✓（同上）
            ├── chk_release (wartermark_strategy.py)   ✅活  spec ✓（注意文件名拼写 wartermark）
            ├── pendingToReady (state_trans.py:444)    ✅活* spec ✓（同上）
            └── glb_alloc_new2 (pre_alloc_new.py:26)   ✅活  spec ✓ ★核心
                    ├── get_process_sort (sort_function.py) ✅活 spec ✓
                    └── allocate_rsc_4_process_new2 (pre_alloc_new.py:138) ✅活 spec ✓
                            ├── bin_sel (pre_alloc_new.py:287)            ✅活 spec ✓
                            ├── get_target_bin_id (monitor_agent.py:23)   ✅活 spec ✓（bin_ops.old.py 有同名死版）
                            └── check_and_preemt_alloc (pre_alloc_new.py) ✅活 spec ✓
                                    ├── index_occupy_by_id_chunk_ver       ✅活 spec ✓
                                    └── push_into_bin (pre_alloc_new.py)   ✅活 spec ✓
                                            └── block/asap/aeap_insert (scheduling_table.py 方法) ✅活 spec ✓
```

**结论**：spec 的函数层次描述**全部准确**，可信赖。带 `*` 的是"经假身 import"——真身在 state_trans/sched_utils，但 global_sched 通过 `from sched.scheduler_agent import ...` 拿到（scheduler_agent 又 `from sched.sched_fn import *` 重导出）。这是 B2 已发现的 `import *` 拓扑问题，不影响功能，但污染了"定义位置"的判断。

## 二、spec 没覆盖的死代码（spec 是对的，但代码里多了垃圾）

### 2.1 整文件死（0 活依赖，可安全移走）

| 文件 | 行数 | 死因 | 替代者 |
|------|------|------|--------|
| `sched/bin_ops.old.py` | 775 | 0 活 import；5 个函数（push_task_into_bins/push_step/glb_alloc/allocate_rsc_4_process/preempt_the_conflicts）只被自身内部调 | global_sched 的 `_new` 版 |
| `sched/pre_alloc.py` | 592 | `glb_alloc_new` 唯一调用者是 bin_ops.old.py；global_sched 有 import 行但从不调用（死 import） | pre_alloc_new.py 的 `_new2` 版 |
| `model/message/message_handler_old.py` | 246 | 0 活 import；12 个函数（含旧版 message_trigger_event_new） | message_handler.py |
| `sched/packing_solver/gurobi_semi2Dclst_mapping.py` | ~640 | `GurobiSemi2DClstMapping` 类 0 活调用；global_sched 的 import 是注释（#1049/#1067） | gurobi_MP_semi2DClst.py |
| `sched/packing_solver/gurobi_semi2Dclst_mapping2.py` | ~? | 同名类 v2，0 活调用 | 同上 |

### 2.2 global_sched.py 内死函数

| 函数 | 行 | 状态 |
|------|----|------|
| `naive_iso` | 281 | 💀死（0 调用，隔离度算法旧版） |
| `test_mem_planner` | 318 | ⏸️条件保留（用户决定留着，`mem_plan` 分支） |
| `single_turn_solver` | 947(注释) | ⏸️坏死分支不管（用户决定） |

### 2.3 死 import 化石（global_sched.py，可清理）

```python
# global_sched.py:23-25 同时 import 了新旧两版，旧的从不调用
from sched.scheduler_agent import Scheduler, check_miss, check_complete  # Scheduler 活(类型注解)，check_* 经假身活
from sched.scheduler_agent import data_pipe_read, pendingToReady          # 经假身活
from sched.pre_alloc import glb_alloc_new                                 # 💀死 import（从不调用）
from sched.pre_alloc_new import glb_alloc_new2                            # ✅活
```

> `pre_alloc` 的 `glb_alloc_new` import 是纯化石，可删（但要确认 global_sched 内无任何 `glb_alloc_new(` 调用——已确认无）。

## 三、文档层分析

### 3.1 文档分层现状（共 ~40 个 doc）

| 层 | 路径 | 数量 | 性质 | 建议 |
|----|------|------|------|------|
| **规范** | `doc/spec/` | 25 | 应然设计（readme 为索引） | **全保留**，活 |
| **开发日志** | `doc/dev/` | 10 | 历史记录/计划 | change_log 系列保留；claude_revise/code_cleanup_2026/ablation_dev/repack_debug 为过程记录，可归档 |
| **模块概览** | `doc/guide/` | 8 | 代码导航 | 保留，但 `deprecated_code.md` 不完整（漏记 pre_alloc/message_handler_old/死求解器）需更新 |
| **散落** | `doc/*.md` | 5 | 零散笔记（sim_flow/sparse_list/setting/ABLA_EXP_FIX_PLAN/assumptions） | 候选归档 |
| **分析报告** | `doc/analytical_report/` | 6 | 参数流分析 | 保留 |

### 3.2 bin_packing 相关文档的准确性

| 文档 | 准确性 | 说明 |
|------|--------|------|
| `spec/algorithm/binpack_solver_spec.md` | ✅准确 | 函数层次完全对应当前活代码 |
| `spec/algorithm/guided_hybrid_allocation_algorithm.md` | 待核 | 两阶段算法（coleasing + repack） |
| `spec/e2e_sched_sim_flow.md` | ✅准确 | perform_bin_packing 流程 |
| `guide/deprecated_code.md` | ⚠️不完整 | 只记了 bin_ops.old.py，漏 pre_alloc.py/message_handler_old.py/死求解器 |
| `guide/sched_core_overview.md` | 待核 | sched 模块概览 |
| `dev/code_cleanup_2026.md` | 历史记录 | 2026-02 的清理过程 |
| `dev/ablation_dev.md` / `repack_debug_instrumentation.md` | 历史记录 | 调试过程，可能描述旧版 |

## 四、整理建议（待用户定，本轮不执行）

### 4.1 代码清理（低风险，move-reference）

1. `bin_ops.old.py` + `pre_alloc.py` → `unused/`（整文件死，0 活依赖）
2. `message_handler_old.py` → `model/message/old/` 或 `unused/`（整文件死）
3. `gurobi_semi2Dclst_mapping.py` + `_mapping2.py` → `packing_solver/old/`（死求解器变体）
4. global_sched.py 删死 import `from sched.pre_alloc import glb_alloc_new`
5. global_sched.py 移走死函数 `naive_iso`（0 调用）

### 4.2 保留（用户已定）

- `test_mem_planner` + `mem_plan` 分支
- `single_turn_solver` 坏死分支（不管）

### 4.3 文档更新

- 更新 `guide/deprecated_code.md`：补全 pre_alloc.py / message_handler_old.py / 死求解器变体的记录
- 候选归档 `doc/dev/` 下的过程性文档（code_cleanup_2026/ablation_dev/repack_debug_instrumentation/claude_revise）→ `doc/dev/archive/`
- 候选归档散落笔记（sim_flow.md/sparse_list.md）

## 五、本报告结论

bin_packing 的"乱"本质是 **3 代实现堆叠 + import 化石未清理**，但**活路径本身是清晰且单一**的（spec 准确描述了它）。死代码边界已用 AST 闭包精确划定，约 2000+ 行死代码可安全移走（占 bin_packing 相关代码 ~50%），全部 0 活依赖。
