# Change Log 2026

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
- **清理内容**：彻底删除了代码中不再使用的“僵尸参数”。
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
- **JSON 模板更新**：在 `cfgs/` 下的配置文件中显式添加了 `affinity_en`, `affinity_level`, `bin_sel_mod` 等关键参数，使其从“隐式默认”变为“显式可调”。
- **参数路径追踪**：完成了 `exec_t_comp_ratioA/B` 从实验脚本到最底层 `rsc_req_estm_quantile` 的完整传递路径梳理。

### 4. 文档同步
- **新增文档**：`doc/spec/binpack_config_design.md`（详细设计规范）。
- **更新文档**：`doc/code_cleanup_2025.md`（记录清理细节）。
- **更新文档**：`doc/parameter_flow_exec_t_comp_ratio.md`（记录参数流向）。
