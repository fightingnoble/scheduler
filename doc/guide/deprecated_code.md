# 废弃代码记录

本文档记录代码库中发现的废弃/过时代码，帮助开发者避免误用。

## 已知废弃模块

### 1. old/ 目录
路径: `/home/zhangchg/git_repo/scheduler/old/`
- `sweep.py` - 旧参数扫描脚本
- `del_files.py` - 文件删除工具
- `rename_files.py` - 文件重命名工具

### 2. ref/ 目录
路径: `/home/zhangchg/git_repo/scheduler/ref/`
- 包含各种参考实现和实验性代码
- 不应直接使用

### 3. sched/packing_solver/old/
路径: `/home/zhangchg/git_repo/scheduler/sched/packing_solver/old/`
- `ortools_CP_1DBp.py` - 旧版OR-Tools求解器
- `cplex_MP_1DBp.py` - CPLEX求解器（未使用）
- `ortools_MP_1DBp.py` - 旧版混合整数规划
- `gurobi_MP_1DBp.py` - 旧版Gurobi求解器

### 4. sched/packing_solver/ref/
路径: `/home/zhangchg/git_repo/scheduler/sched/packing_solver/ref/`
- 参考实现，包含多种求解器变体

### 5. model/old/
路径: `/home/zhangchg/git_repo/scheduler/model/old/`
- `ctx.py` - 旧版上下文实现

### 6. optimizer/old/
路径: `/home/zhangchg/git_repo/scheduler/optimizer/old/`
- `2try.py`, `1try.py` - 实验性优化代码

### 7. run/old/
路径: `/home/zhangchg/git_repo/scheduler/run/old/`
- `file_path_prepare.py` - 旧版路径准备

### 8. 弃用的调度器实现（部分活，文件级不可删）
> ⚠️ 澄清（2026-06-16 B2 分析）：以下"替代/弃用"仅指**仿真循环**角色。这两个文件的工具函数仍是**活路径**依赖，文件级删除会断 repack 路径：
> - `sched/monitor_agent.py`：`get_target_bin_id` / `get_rsc_2b_released` 被 `sched/pre_alloc_new.py`（repack 路径）使用 → 活
> - `sched/scheduler_agent.py`：`Scheduler` 类被 repack 路径实例化使用（`create_common_scheduler_elements` → `perform_bin_packing`）→ 活
>
> 死的是它们的**仿真 timestep 循环**（已被 `approach/approach_sim.py` 替代）。详见 `scheduler-audit-20260612/cleanup/reports/`。

- `sched/scheduler_agent.py` - 旧版运行时调度器（仿真循环已被 approach/approach_sim.py 替代；Scheduler 类仍活）
- `sched/monitor_agent.py` - 旧版监控代理（仿真循环已弃用；工具函数仍活）

### 9. 弃用的文件
- `unused_fun.py` - 根目录下的未使用函数集合

### 10. bin_packing 死代码清理（2026-06-16，B3 batch）

3 代堆叠的旧实现，已用 move-reference 移走到归档位置（audit worktree，未合并 main）：

| 原位置 | 新位置（归档） | 说明 |
|--------|---------------|------|
| `sched/pre_alloc.py` | `old/pre_alloc.py` | v1 代（glb_alloc_new 等），被 `pre_alloc_new.py`（_new2）取代 |
| `sched/packing_solver/gurobi_semi2Dclst_mapping.py` | `sched/packing_solver/unused/` | 死求解器变体（GurobiSemi2DClstMapping），活的是 `gurobi_MP_semi2DClst.py` |
| `sched/packing_solver/gurobi_semi2Dclst_mapping2.py` | `sched/packing_solver/unused/` | 同名类 v2 |
| `sched/pre_alloc_new.py::bin_select_new` | `sched/pre_alloc_new_old.py` | 死函数（bin_sel 的旧包装，0 调用） |
| `sched/global_sched.py::naive_iso` | `sched/global_sched_old.py` | 死函数（隔离度算法旧版，0 调用） |
| `sched/pre_alloc_new.py` 注释测试块 | `sched/pre_alloc_new_old.py` | 引用 pre_alloc.py 旧函数的死注释 |

> 注：`bin_ops.old.py` 和 `message_handler_old.py` 在主仓库工作树存在，但**不在 test_pipeline 分支**（未跟踪幽灵文件），故未纳入清理。

## 标记约定

代码中常见的废弃标记：
- `_old.py` 后缀
- `depraved_` 前缀
- `ref/` 目录
- `old/` 目录

## 注意事项

1. **不要修改废弃代码** - 除非有明确的迁移计划
2. **不要从废弃模块导入** - 使用当前活跃的实现
3. **参考实现** - `ref/` 目录的代码可用于学习，但不应直接依赖

## 待确认

以下文件/模块需要确认是否废弃：
- [ ] `model/streaming_processing/depraved_ref.py`
- [ ] `task/depraved_ref.py`
- [ ] `sched/bin_ops.old.py` — ⚠️ 主仓库未跟踪幽灵文件，不在 test_pipeline 分支（B3 确认）；不影响 clean main
- [ ] `analyze/` 目录下的多个分析脚本

---
*最后更新: 2026-06-16（B3 bin_packing 清理）*
