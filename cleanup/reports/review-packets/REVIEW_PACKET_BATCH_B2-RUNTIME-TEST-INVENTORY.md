# B2 runtime/test 清单

日期：2026-06-16  
范围：`B2-RUNTIME-TEST-INVENTORY`  
状态：已完成清单盘点，未执行清理，未修改源码

## 这次做了什么

这批先读 `doc/spec/readme.md`，再顺着规范文档和代码入口确认 runtime 的新旧边界。核心参考包括：

- `doc/spec/key_COT.md`
- `doc/spec/e2e_sched_sim_flow.md`
- `doc/spec/test_plan.md`
- `doc/spec/sim/approach_sim_spec.md`
- `doc/spec/algorithm/guided_hybrid_allocation_algorithm.md`
- `doc/spec/algorithm/chain_slack_assignment_algorithm.md`
- `doc/spec/algorithm/binpack_solver_spec.md`
- `doc/spec/stat/runtime_overhead_spec.md`
- `doc/guide/architecture_overview.md`
- `doc/guide/simulation_flow_overview.md`
- `doc/guide/sched_core_overview.md`
- `doc/guide/deprecated_code.md`
- `CLAUDE.md`

结论很直接：当前 runtime 不是“把老 `sim_main.py` 删掉，换成 `approach_sim.py`”这么简单。现在的主流程是：

```text
main_approach.py
  -> approach_setup.py
  -> sim_main.py::perform_bin_packing()
  -> approach_initiator.py
  -> approach_sim.py::run_simulation()
  -> approach_collector.py
```

`sim_main.py` 已经不再是推荐的运行时仿真入口，但它仍然是配置生成、bin packing、repack/fallback 的后端。`approach_sim.py` 才是现行事件驱动仿真后端。

## 当前 runtime 主链

这些文件在 B2 中继续保持 KEEP：

- `main_approach.py`：当前策略驱动入口。它固定 `test_case = bin_pack_new`，再调用 `setup_benchmark()` 和 `run_simulation()`。
- `approach_setup.py`：把 workload/config 生成、Phase 1、repack、图加载串起来。
- `sim_main.py`：旧的大入口仍保留为配置生成和装箱后端，`perform_bin_packing()` 仍在当前主链上。
- `approach_sim.py`：现行 event-driven runtime 仿真后端。
- `approach_sched.py`：绑定 `cyc/glb/pglb/reserv` 策略。
- `approach_def.py`：定义处理器、事件图和 `Acc_p.sched()`，也是 Algorithm 2 运行时开销测量插入点。
- `approach_collector.py`：统计收集、延迟分解、调度开销摘要。
- `approach_initiator.py`：从 `bin_list` 生成分区信息、事件和处理器。
- `scripts/motiv_exp_runner.py`、`scripts/abla_exp_runner.py`、`scripts/exp_common.py`：当前文档确认的实验入口和共享调用层。

## 旧 runtime 代码的真实状态

`CLAUDE.md` 和 `doc/guide/deprecated_code.md` 都写了：

- `sched/scheduler_agent.py` 已被 `approach_sim.py` 替代。
- `sched/monitor_agent.py` 已被 `approach_collector.py` 替代。

但 B2 不能把它们当成可删文件，因为实际导入关系还在：

- `sim_main.py` 仍导入 `sched.scheduler_agent` 和 `sched.monitor_agent`。
- `sched/global_sched.py` 仍导入它们。
- `allocator_agent.py`、`sched/sched_fn.py`、`sched/state_trans.py` 等也还导入旧组件。

因此这两个文件的状态是 `KEEP_UNTIL_SPLIT`：不要新增依赖，也不要删除。后续如果要清理，需要先把配置生成链和旧运行时对象的导入拆开，再跑 targeted tests。

## 测试入口盘点

已确认的 tracked 测试/测试脚本：

- `test_approach_collector.py`
- `test_closure_fix.py`
- `test_core_allocation.py`
- `test_duplicate.py`
- `test_event_update.py`
- `test_mapping.py`
- `test_repack_diagnostic.py`
- `test_updated_stats.py`
- `scripts/test_alloc_lat.py`
- `optimizer/ops_test.py`
- `simple_test_collector.py`

当前不批准删除任何测试。尤其是之前被 `P1-REMOVE-B19-TESTS` 覆盖的测试，继续保留。

两个测试风险需要先处理：

- `scripts/test_alloc_lat.py` 硬编码 `sys.path.insert(0, '/home/zhangchg/git_repo/scheduler')`。如果直接运行，它会绕过 audit worktree，引用原始仓库。B2 标为 `REVIEW_BLOCKED`，暂不运行。
- `test_event_update.py` 和 `test_mapping.py` 在 Phase 1 unresolved-import 里显示 `approach_plot` 未解析。先确认这个模块名是不是历史拼写或缺失文件，再运行。

## B2 轻量验证

所有命令都在 audit worktree 中运行，使用 WSL Ubuntu-20.04 的 `zsh` 和 `conda activate gurobi`。

通过项：

```bash
wsl -d Ubuntu-20.04 -- true
```

```bash
cd /home/zhangchg/git_repo/scheduler-audit-20260612
conda activate gurobi
PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help
```

```bash
PYTHONDONTWRITEBYTECODE=1 python -m scripts.motiv_exp_runner --help
PYTHONDONTWRITEBYTECODE=1 python -m scripts.abla_exp_runner --help
```

```bash
PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
mods = [
    "main_approach",
    "approach_setup",
    "approach_sim",
    "approach_def",
    "approach_sched",
    "approach_collector",
    "sim_main",
    "sched.global_sched",
    "sched.slack_estim",
    "sched.packing_solver.chain_slack_assign",
    "mapper.mem_planner",
]
for mod in mods:
    __import__(mod)
    print("IMPORT_PASS", mod)
PY
```

没有跑长实验，没有跑全量 pytest，没有生成新的 cleanup 删除候选。

## 产物

- `cleanup/reports/b2-runtime-test-inventory.csv`
- `cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md`
- 本文件：`REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md`

## 建议的下一步

建议继续做 B2 内部的 `B2-RUNTIME-SYMBOL-SPLIT-PREFLIGHT`，仍然不删代码：

1. 枚举 `sched/scheduler_agent.py`、`sched/monitor_agent.py`、`allocator_agent.py` 的导出类、函数和被导入符号。
2. 把符号分成 live、dead、ambiguous 三类。ambiguous 不猜，先问代码作者。
3. 删除前再做 `B2-RUNTIME-SMOKE-MATRIX`：覆盖 `cyc/glb/pglb/reserv` 的最小 help/import/dry-run 路径，并单独跑 collector 相关测试。
4. 暂不跑 `scripts/test_alloc_lat.py`，除非先修正硬编码原始仓库路径。

本批没有新的删除授权 ID。
