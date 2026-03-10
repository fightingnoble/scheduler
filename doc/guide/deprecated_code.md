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

### 8. 弃用的调度器实现
- `sched/scheduler_agent.py` - 旧版运行时调度器（已被approach_sim.py替代）
- `sched/monitor_agent.py` - 旧版监控代理（已弃用）

### 9. 弃用的文件
- `unused_fun.py` - 根目录下的未使用函数集合

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
- [ ] `sched/bin_ops.old.py`
- [ ] `analyze/` 目录下的多个分析脚本

---
*最后更新: 2026-03-10*
