# REVIEW_PACKET_BATCH_B3-BINPACK-DEAD-MOVE

Batch: `B3-BINPACK-DEAD-MOVE` (move-reference, not delete)
Strategy: 整文件 → `unused/` / `*/old/`；符号级死代码 → `*_unused.py`；注释测试 → `*_unused.py`。
基础报告: `cleanup/reports/bin_packing_inventory.md` + `cleanup/reports/bin_packing_function_map.md`
方法: AST 调用图可达性闭包（从 perform_bin_packing 出发），非文本 grep。所有"死"判定均经 0 活调用验证。

> **本批次不动**：test_mem_planner（用户决定保留）、single_turn_solver 坏死分支（用户决定不管）、参数语义错位 `index_occupy_by_id_chunk_ver(process_sort←key)`（属改写范畴，记录为已知问题，本轮不改）。

## 1. MUST REVIEW

### 组 A — 整文件 move（5 个，bin_packing 3 代堆叠的旧代）

#### Decision B3-MOVE-001 — `sched/bin_ops.old.py` → `unused/`
```text
Path: sched/bin_ops.old.py (775 lines)
Proposed: move whole file to unused/bin_ops.old.py
Reason: v0 代装箱实现（push_task_into_bins/push_step/glb_alloc/allocate_rsc_4_process/preempt_the_conflicts，无 _new 后缀）。0 活 import，5 个函数仅被自身内部调。
Risk: LOW. 独立无依赖。
Evidence: grep "from sched.bin_ops.old" → 0 活文件；AST 可达性闭包：不可达。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/bin_ops.old.py
Recommended: approve
```

#### Decision B3-MOVE-002 — `sched/pre_alloc.py` → `unused/`（依赖 B3-CLEAN-001）
```text
Path: sched/pre_alloc.py (592 lines)
Proposed: move whole file to unused/pre_alloc.py
Reason: v1 代（glb_alloc_new/allocate_rsc_4_process_new/bin_select）。glb_alloc_new 唯一调用者是 bin_ops.old.py（已死）。
        global_sched.py:25 有 `from sched.pre_alloc import glb_alloc_new` 但从不调用（死 import 化石）。
Risk: LOW. ⚠️ 依赖 B3-CLEAN-001（必须同时/先删 global_sched.py:25，否则 import sched.global_sched 失败）。
Evidence: AST glb_alloc_new() 调用点：仅 bin_ops.old.py；global_sched 有 import 行但 0 Call 节点。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/pre_alloc.py sched/global_sched.py
Recommended: approve（与 B3-CLEAN-001 绑定执行）
```

#### Decision B3-MOVE-003 — `model/message/message_handler_old.py` → `model/message/old/`
```text
Path: model/message/message_handler_old.py (246 lines)
Proposed: move to model/message/old/message_handler_old.py
Reason: 12 个函数（含旧版 message_trigger_event_new）的旧代。0 活 import。活版是 message_handler.py。
Risk: LOW. 独立无依赖。
Evidence: grep "message_handler_old" → 0 活文件引用。
Recovery: git checkout archive/test_pipeline-20260612 -- model/message/message_handler_old.py
Recommended: approve
```

#### Decision B3-MOVE-004 — `sched/packing_solver/gurobi_semi2Dclst_mapping.py` → `packing_solver/old/`
```text
Path: sched/packing_solver/gurobi_semi2Dclst_mapping.py (~640 lines)
Proposed: move to sched/packing_solver/old/gurobi_semi2Dclst_mapping.py
Reason: GurobiSemi2DClstMapping 类 0 活调用。global_sched 的 import 是注释（L1049 #）。
        活的求解器是 gurobi_MP_semi2DClst.py（ClusterGurobiSolverSemi2D）。
Risk: LOW. 独立。
Evidence: AST GurobiSemi2DClstMapping 调用点：0；global_sched.py:1049 `# from ... import GurobiSemi2DClstMapping`（注释）。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/packing_solver/gurobi_semi2Dclst_mapping.py
Recommended: approve
```

#### Decision B3-MOVE-005 — `sched/packing_solver/gurobi_semi2Dclst_mapping2.py` → `packing_solver/old/`
```text
Path: sched/packing_solver/gurobi_semi2Dclst_mapping2.py
Proposed: move to sched/packing_solver/old/gurobi_semi2Dclst_mapping2.py
Reason: 同名类 v2（GurobiSemi2DClstMapping），0 活调用。mapping 和 mapping2 是两个失败迭代。
Risk: LOW. 独立。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/packing_solver/gurobi_semi2Dclst_mapping2.py
Recommended: approve
```

### 组 B — 符号级 move（3 个，活文件内的死代码）

#### Decision B3-MOVE-006 — `pre_alloc_new.py::bin_select_new`(L287-328) → `pre_alloc_new_unused.py`
```text
Path: sched/pre_alloc_new.py L287-328 (42 lines, def bin_select_new)
Proposed: move function to new file sched/pre_alloc_new_unused.py
Reason: 死函数。内部调用图：bin_select_new 只调 bin_sel，0 外部调用。
        是 bin_sel 的旧包装（名字相似、职责重叠），spec §2.1 只提 bin_sel 不提它。
Risk: LOW. 独立（bin_sel 活，由 allocate_rsc_4_process_new2 调，不受影响）。
Evidence: AST bin_select_new() 调用点：0。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/pre_alloc_new.py
Recommended: approve
```

#### Decision B3-MOVE-007 — `global_sched.py::naive_iso`(L281-317) → `global_sched_unused.py`
```text
Path: sched/global_sched.py L281-317 (37 lines, def naive_iso)
Proposed: move function to new file sched/global_sched_unused.py
Reason: 隔离度算法旧版，0 调用（直接+间接均无）。
Risk: LOW. 独立。
Evidence: AST 可达性闭包：不可达；naive_iso() 调用点：0。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
Recommended: approve
```

#### Decision B3-MOVE-008 — `pre_alloc_new.py` 注释测试代码(L627-748) → `pre_alloc_new_unused.py`
```text
Path: sched/pre_alloc_new.py L627-748 (~122 lines, 全是 # 注释的 test code)
Proposed: move commented-out test block to pre_alloc_new_unused.py
Reason: 死注释。引用 check_and_preemt_at_queue/get_preempt_candi/bin_select（全是 pre_alloc.py 旧版函数），
        与当前代码完全脱节。
Risk: LOW. 注释（非可执行代码），移走不影响任何路径。
Evidence: L627 `# test code`；引用的函数在 pre_alloc_new.py 中均不存在（属 pre_alloc.py）。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/pre_alloc_new.py
Recommended: approve（保守：移到 unused；若用户同意也可直接删注释）
```

### 组 C — 配套清理（1 个）

#### Decision B3-CLEAN-001 — 删 `global_sched.py:25` 死 import（配套 B3-MOVE-002）
```text
Path: sched/global_sched.py L25
Proposed: delete line `from sched.pre_alloc import glb_alloc_new`
Reason: 死 import 化石。glb_alloc_new 从未被 global_sched 调用（AST 0 Call 节点）。
        B3-MOVE-002 移走 pre_alloc.py 后这行必须删，否则 import sched.global_sched 会 ImportError。
Risk: LOW. 删 import 属"清理配套"（移动路径的必要配套，非改写）。
Evidence: global_sched.py 内 glb_alloc_new( 调用：0。
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
Recommended: approve（与 B3-MOVE-002 绑定）
```

### 组 D — 文档更新（2 个，doc/spec 为保护路径，需明确批准）

#### Decision B3-DOC-001 — 更新 `guide/deprecated_code.md`（补全记录）
```text
Path: doc/guide/deprecated_code.md
Proposed: 补充 pre_alloc.py / message_handler_old.py / gurobi_semi2Dclst_mapping*.py 的废弃记录
Reason: 当前 deprecated_code.md 只记了 bin_ops.old.py，遗漏了本批次移走的其余死文件。记录不完整会误导。
Risk: LOW. 纯文档追加，不改代码。
Recommended: approve
```

#### Decision B3-DOC-002 — 更新 `spec/algorithm/binpack_solver_spec.md §8.1`（标记已解决）
```text
Path: doc/spec/algorithm/binpack_solver_spec.md §8.1（保护路径 doc/spec/）
Proposed: §8.1.1（sys.exit 硬退出）和 §8.1.2（tot_cores=300 硬编码）标记为"已解决"，附代码现状
Reason: spec §8.1 的两个"当前问题"实际已修复：
  - §8.1.1: sys.exit(1) → 已改为 raise ResourceInsufficientError (pre_alloc_new.py:206)
  - §8.1.2: tot_cores=300 → 已改为 binpack_cfg.get("total_cores", 300) (pre_alloc_new.py:158)
  spec 与代码脱节，需同步。不删内容，只标注"已解决"+代码位置。
Risk: LOW–MEDIUM. doc/spec 是保护路径；但这是"同步文档与现实"，非删除规范。建议标注而非删除原建议。
Recommended: approve（标注"已解决"，保留历史建议文本）
```

## 2. SAFE SUMMARY

- 整文件 move: 5（bin_ops.old.py, pre_alloc.py, message_handler_old.py, gurobi_semi2Dclst_mapping.py, _mapping2.py）
- 符号级 move: 3（bin_select_new, naive_iso, 注释测试块）
- 配套清理: 1（删 global_sched.py:25 死 import）
- 文档更新: 2（deprecated_code.md 补全, spec §8.1 标记已解决）
- 总计清理: ~2400+ 行死代码 + 2 处文档同步
- 依赖关系: B3-MOVE-002 ←→ B3-CLEAN-001（绑定）；其余独立
- Smoke test: NOT YET RUN — 执行后跑回归门
- 本批次不动: test_mem_planner（保留）、single_turn_solver（不管）、参数语义错位（记录，不改写）

## 3. NO NEED TO REVIEW

已知问题（本轮不改写，记录备查）：
- `index_occupy_by_id_chunk_ver(pre_alloc_new.py:578)` 定义参数 `process_sort`，但 `check_and_preemt_alloc`(L453) 调用时传 `key=lambda _p:_p.deadline`。函数内 L593 把它当排序函数调用——能跑（lambda callable），但语义是把"单维 deadline key"当"多维 process_sort"用。属改写范畴，需单独决策。

证据文件：
- `cleanup/reports/bin_packing_inventory.md`
- `cleanup/reports/bin_packing_function_map.md`
- `cleanup/move-ledger.csv`（rows B3-MOVE-001~008 + B3-CLEAN-001）

## 4. Double-check 验证（2026-06-16，对照 readme/e2e/test_plan）

用三份顶层 spec 对 B3 全部决策交叉验证，**零矛盾，全部通过**：

**移走的死符号 — 顶层 spec 零引用 ✓**
`bin_ops.old` / `pre_alloc` / `message_handler_old` / `gurobi_semi2Dclst_mapping*` / `bin_select_new` / `naive_iso` / `glb_alloc_new`：readme.md + e2e_sched_sim_flow.md + test_plan.md 均无引用。

**保留的活路径 — spec 明确描述为活 ✓**
- `coleasing_alloc_cluster`：test_plan:11/25/29/92（Split 实现）；readme:37
- `push_task_into_bins_new`：test_plan:26/29/100（Repack，`bin_sel_mod="pre_defined"`）；readme:37
- `perform_bin_packing`：e2e:24/61/109（Step 6）
- `apply_forced_num_cores`：e2e:63/114/123（Step 7）

**用户决策与 spec 一致 ✓**
`single_turn_solver` / `test_mem_planner` / `mem_plan` / `"full"` 分支：三份顶层 spec 零引用。spec 只认 guided 算法（test_plan:27/288），mem_plan/full 本就不是 spec 设计的正式路径 → 印证"single_turn_solver 不管、test_mem_planner 保留"合理。

**粒度观察**：顶层 spec 粒度是"算法名"（coleasing_alloc_cluster / push_task_into_bins_new），不深入内部实现函数（glb_alloc_new2 等，那些在 binpack_solver_spec.md）。故移走内部死函数不影响顶层 spec。

**结论**：B3 packet 11 决策与顶层 spec 零矛盾，可安全执行。

> 附注（非本批次）：test_plan:11 拼写 `coalesce_alloc_cluster`（少 s），代码实为 `coleasing_alloc_cluster`。spec 笔误，记录备查，不在 B3 处理。

## Response format

```text
approve all
approve B3-MOVE-001 B3-MOVE-003 B3-MOVE-004 B3-MOVE-005   （独立低风险整文件，先做）
approve B3-MOVE-002 B3-CLEAN-001                           （绑定，pre_alloc 配套）
approve B3-MOVE-006 B3-MOVE-007 B3-MOVE-008                （符号级）
approve B3-DOC-001                                          （文档补全）
reject B3-DOC-002                                           （spec 保护路径，暂不动）
pause batch
```

## Regression gate (post-execution, mandatory)

gurobi 环境，audit worktree：
1. import 探针：`sched.global_sched, sched.scheduler_agent, sched.monitor_agent, approach_sim, approach_setup, main_approach` 全 OK
2. `main_approach.py --help` PASS
3. `scripts.motiv_exp_runner --help` PASS
4. `scripts.abla_exp_runner --help` PASS
5. 额外：确认 `bin_ops.old` / `pre_alloc` / `message_handler_old` / `gurobi_semi2Dclst_mapping*` 已不可从原路径导入（符合预期）

任一失败 → `git checkout archive/test_pipeline-20260612 -- <path>` 恢复 + 记录 FILE_ADJUSTMENT_RECORD。
