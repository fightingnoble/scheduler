# Change Log 2026

## [2026-02-14] 配置化硬编码的 tot_cores = 300

### 问题描述
多处代码中硬编码了 `tot_cores = 300`，导致：
- 不同硬件规模下需要手动修改代码
- 无法适应不同的资源上限配置
- 代码可维护性差

### 修改内容

1. **`sched/global_sched.py`**
   - 在 `push_task_into_bins_new` 函数中，将 `total_cores` 参数注入到 `binpack_cfg`
   - 位置：第 73-76 行

2. **`sched/pre_alloc_new.py`**
   - 将 `tot_cores = 300` 改为从 `binpack_cfg` 读取
   - 位置：第 178 行
   - 修改后：`tot_cores = binpack_cfg.get("total_cores", 300)`

3. **`sched/slack_estim.py`**
   - 为 `rsc_slack_estim` 函数添加 `total_cores` 参数（默认值 300）
   - 位置：第 198-204 行
   - 将函数内的硬编码改为使用参数
   - 位置：第 174 行

### 效果
- `total_cores` 现在可以通过 `binpack_cfg` 配置传递
- 保持向后兼容（默认值 300）
- 支持不同硬件规模的实验

---

## [2026-02-14] 修复文档中 `num_bins = -1` 的错误描述

### 问题描述
多处文档错误地将 `num_bins = -1` 描述为"单分区"，实际含义应为"最多分区/自动搜索"。
- 正确含义：`num_bins = -1` 表示 `n_partition=9999`，在初始时间片分派下，任何分区内的任务都具有不重叠的生命区间
- 错误描述：将 `-1` 描述为"单分区"（实际 `num_bins = 1` 才是单分区）

### 修改内容

1. **`doc/spec/test_plan.md`**
   - 第 89 行: Step 2 参数说明，更新 `num_bins = -1` 的描述
   - 第 110 行: cyc 配置，添加"(最多分区)"说明
   - 第 114 行: cyc-S 配置，添加"(最多分区)"说明
   - 第 293-297 行: 消融实验执行路径对照表，统一添加分区类型说明
   - 第 306 行: 消融实验1 固定参数，添加"(最多分区)"说明

2. **`doc/spec/e2e_sched_sim_flow.md`**
   - 第 138 行: cyc 配置，添加"(最多分区)"说明
   - 第 142 行: cyc-S 配置，添加"(最多分区)"说明
   - 第 241 行: cyc 的 num_bins 设置说明，更新为"最多分区"

3. **`doc/spec/readme.md`**
   - 第 40-45 行: 更新 `-1` 的描述，从"单分区"改为"最多分区/自动搜索"

4. **`doc/spec/sim/approach_sim_spec.md`**
   - 第 216 行: cyc 的 num_partitions 说明，从"通常 = 1"改为"通常 = 实际分区数"
   - 第 555 行: num_bins 含义表，更新 `-1` 的描述

### 正确的 num_bins 对照表
| num_bins | 含义 |
|----------|------|
| `-1` | **最多分区/自动搜索** (n_partition=9999，每个任务独立分区) |
| `1` | **单分区** (所有任务共享一个分区，glb 使用) |
| `>1` | **多分区** (pglb, reserv 使用) |

---

## [2026-02-14] 修复 pre_alloc_new.py 硬退出问题

### 问题描述
`pre_alloc_new.py:192-195` 在 `bin_sel_mod == "pre_defined"` 且资源不足时使用 `sys.exit(1)` 硬退出，导致：
- 在并行执行中，worker 进程意外终止
- `fut.result()` 抛出未捕获的异常
- 主进程卡在 `as_completed()` 循环中

### 修改内容

1. **`sched/pre_alloc_new.py`**
   - 新增自定义异常 `ResourceInsufficientError`
   - 将 `sys.exit(1)` 改为 `raise ResourceInsufficientError(...)`
   - 位置：第 19-21 行（异常定义），第 195-200 行（抛出异常）

2. **`scripts/abla_exp_runner.py`**
   - 在 Case 1/2/3 的 `as_completed()` 循环中添加 `try-except`
   - 捕获 worker 进程的异常，打印错误信息后继续处理其他任务
   - 位置：第 215-225 行（Case 1），第 409-419 行（Case 2），第 666-676 行（Case 3）

### 效果
- 单个任务失败不会影响其他任务的执行
- 错误信息会被打印，便于调试
- 主进程能正常完成所有可完成的任务

## [2026-02-13] 消融脚本参数与文档对齐（阶段1-4）

### 修改内容

1. **`scripts/abla_exp_runner.py`**
   - 默认参数调整：
     - `--case2_tiles`: `200,400`
     - `--case3_tiles`: `200,400`
     - `--case3_bins`: `1,2,4,8`（默认不含 `-1`）
   - 保持 `ratioB` 默认扫描包含 `0.99`（Case1/Case3）。
   - Case3 新增“固定负载强度”模式：
     - `--case3_fixed_strength`
     - `--case3_fixed_tiles`
     - `--case3_fixed_chains`
     - `--case3_fixed_load`
   - Case3 worker 与报表增强：
     - 复用 `get_motiv_case2_stats()` 获取 `utilization`/`latency_breakdown`
     - 新增 `case3_breakdown.pdf` 与 `case3_utilization.pdf` 输出
     - 缓存数量校验改为按“强度点数量”计算，兼容固定/多扫两种模式

2. **`doc/spec/test_plan.md`**
   - 更新 reserv 描述：理论可含 `-1`，脚本默认 `num_bins=[1,2,4,8]`。
   - 更新消融2/3默认负载强度组合为 `tiles=[200,400], chains=[1,4], load=[0.5,1.0]`。
   - 明确消融3支持固定单一负载强度模式，默认先多组扫描。

3. **`doc/spec/e2e_sched_sim_flow.md`**
   - 同步 reserv 策略表与实现摘要中的默认扫描口径：
     - 脚本默认 `num_bins=[1,2,4,8]`
     - 默认资源规模 `tiles=[200,400]`
   - 补充 Case3 “默认多扫 + 可固定强度”说明。

## [2026-02-13] 修复量化资源估计返回值未完整赋值

### 修改内容

1. **`task/task_agent.py` (`rsc_req_estm_quantile`, 约 563-594 行)**
   - 修复 `got_latency`/`got_constr` 在部分分支未赋值导致的 `UnboundLocalError`。
   - `SenVarDist` 分支补充：
     - 通过 `find_legal` 应用约束得到 `req_rsc_size`。
     - 使用分位数延迟计算 `got_latency`。
   - `compute_budget <= 0` 分支补充：
     - 通过 `find_legal` 对最大请求资源进行合法化。
   - `AccVarDist` 延迟统一为：
     - `load_q / req_rsc_size / FLOPS_PER_CORE + io_q (+ 数值容差)`，
       与约束驱动求解流程口径保持一致。

### 结果

- 消除 `scripts/abla_exp_runner.py` 执行过程中由 `rsc_req_estm_quantile` 引发的
  `UnboundLocalError: local variable 'got_latency' referenced before assignment`。

## [2026-02-08] 资源控制流程与装箱逻辑重构

### 核心重构：Stage 分离与算法解耦

**问题**：`perform_bin_packing` 内部混合了 Repack 阶段判断、多种装箱算法实现以及资源约束逻辑，导致控制流混乱且难以维护。

**解决方案**：将 Repack 逻辑上移至 `setup_benchmark`，并将装箱算法拆分为独立原子函数。

### 修改内容

1. **`sim_main.py` - 算法拆分**
   - 将 `perform_bin_packing` 重构为分发器（Dispatcher）。
   - 新增原子算法函数：
     - `execute_guided_packing`: 负责 Guided Clustering。
     - `execute_scratch_packing`: 负责 Scratch Bin Packing。
     - `execute_repacking`: 负责 Repack 阶段的时间维度重分配。
   - **简化 `determine_resource_config`**：移除了老旧的 `test_case` 判断逻辑，明确其仅负责初始化资源或应用强制约束的职责。

2. **`approach_setup.py` - 流程控制重构**
   - `setup_benchmark` 显式分离为两个 Stage：
     - **Stage 1 (Initial)**: 执行初始装箱。
     - **Stage 2 (Repack)**: 若满足条件，基于 Stage 1 的 `bin_list` 执行 Repack。
   - `run_benchmark_setup_pipeline` 支持传递现有的 `bin_list` 和 `num_cores`。

### 重构效果

**流程清晰化**：
- Repack 不再是底层函数的内部状态，而是顶层的独立执行阶段。
- 资源约束逻辑（`apply_forced_num_cores`）的调用路径更加明确，避免了重复计算。
- 每个装箱算法都有独立的函数入口，便于后续扩展和维护。

### 文档更新

- **更新**：`doc/spec/e2e_sched_sim_flow.md` - 反映新的 Stage 分离流程。

---

## [2026-02-04] BinPackConfig 重构与配置系统清理

### 1. 核心重构：引入 BinPackConfig 包装器
- **新增文件**：`sched/binpack_config.py`
- **重构内容**：
    - 创建了 `BinPackConfig` 类，继承自 `dict`，实现了字典与属性访问的双模式（Dual-mode access）。
    - 提供了类型安全的属性访问（如 `cfg.algorithm`, `cfg.mode`），支持 IDE 自动补全。
    - 添加了静态方法 `generate_template()`，作为配置参数的权威参考模板。
- **集成**：
    - 修改 `utils.py` 中的 `input_parser`，将加载的配置字典自动转换为 `BinPackConfig` 实例。
    - 修改 `sim_main.py` 中的 `prepare_binpack_cfg`，确保运行时注入的配置同样经过包装。

### 2. 配置清理：删除过期参数
- **清理内容**：彻底删除了代码中不再使用的"僵尸参数"。
- **删除的参数**：
    - `sort_reverse`：排序反转逻辑已废弃。
    - `release_temp_rda`：功能已被 `get_rsc_2b_released()` 替代。
    - `slack_sharing`：旧版松弛量共享开关，新算法不再使用。
    - `percentile`：`push_step_new` 函数签名中的冗余参数。
- **影响范围**：
    - 更新了 `cfgs/Bp_guided.json` 和 `cfgs/Bp_scratch.json`。
    - 修改了 `sched/global_sched.py`，将旧的 `default_binpack_cfg` 字典替换为 `BinPackConfig()` 实例。
    - 清理了 `sched/pre_alloc.py` 中多处函数的默认参数定义。

### 3. 配置显式化与增强
- **JSON 模板更新**：在 `cfgs/` 下的配置文件中显式添加了 `affinity_en`, `affinity_level`, `bin_sel_mod` 等关键参数，使其从"隐式默认"变为"显式可调"。
- **参数路径追踪**：完成了 `exec_t_comp_ratioA/B` 从实验脚本到最底层 `rsc_req_estm_quantile` 的完整传递路径梳理。

### 4. 文档同步
- **新增文档**：`doc/spec/binpack_config_design.md`（详细设计规范）。
- **更新文档**：`doc/code_cleanup_2025.md`（记录清理细节）。
- **更新文档**：`doc/parameter_flow_exec_t_comp_ratio.md`（记录参数流向）。


## 20260311

Claude 开发指南与实验系统文档

1. **新增开发文档**：
   - `doc/dev/CLAUDE.md` - Claude Code 开发助手快速参考索引
   - `doc/dev/simulation_flow.md` - 仿真主流程详解
   - `doc/dev/experiment_system.md` - 实验系统架构
   - `doc/dev/configuration_system.md` - 配置系统详解
   - `doc/dev/result_collection.md` - 结果收集与绘图

2. **文档层次结构**：
   ```
   CLAUDE.md (顶层索引)
       ├── 规范文档 (doc/spec/)
       │   ├── key_COT.md (学术表达)
       │   ├── e2e_sched_sim_flow.md (设计规范)
       │   └── test_plan.md (实现细节)
       │
       └── 开发文档 (doc/dev/)
           ├── simulation_flow.md
           ├── experiment_system.md
           ├── configuration_system.md
           └── result_collection.md
   ```

3. **创建 spec-writer skill**：
   - 位置: `.claude/skills/spec-writer/SKILL.md`
   - 核心原则: 设计思想 > 代码设计 > 具体实现
   - 包含文档结构模板和写作检查清单

4. **代码探索总结**：
   - 消融实验 Case 1 部分完成（cyc-S p70-p99 结果相同，可能有问题）
   - 消融实验 Case 2/3 未开始
   - 仿真流程: runner → setup_benchmark → perform_bin_packing → run_simulation
   - 配置系统: JSON → BinPackConfig → binpack_cfg dict

---

## 20260310

文档整理 - 统计收集系统规范

1. **新增规范文档**：
   - `doc/spec/stat/tdigest_system_spec.md` - T-Digest 流式统计系统规范
   - `doc/spec/stat/statistics_collection_spec.md` - 统计收集与仿真集成规范
   - 旧文档移至 `doc/old/` 保留作为历史参考

2. **文档结构优化**：
   - 采用"设计思想 > 代码设计思路 > 具体实现"的层次结构
   - 设计思想和核心流程前置，实现细节后置
   - 与 `doc/spec/stat/collector.md` 建立引用关系，避免重复

3. **修复内容**：
   - 修正 `TDigestStreamingHistogram` 文件位置：`utils.py` → `ref_tdigest.py`
   - 修正 `statistics_collection_spec.md` 中相对链接路径
   - 删除冗余的 `scripts/collector.md`，保留 `doc/spec/stat/collector.md`
   - 更新 `doc/dev/change_log_2025.md` 中的文件引用

4. **涉及文件**：
   - 新增: `doc/spec/stat/tdigest_system_spec.md`
   - 新增: `doc/spec/stat/statistics_collection_spec.md`
   - 更新: `doc/spec/readme.md` (添加新文档索引)
   - 删除: `scripts/collector.md` (与 `doc/spec/stat/collector.md` 重复)
   - 更新: `doc/dev/change_log_2025.md` (修正 collector.md 引用)
