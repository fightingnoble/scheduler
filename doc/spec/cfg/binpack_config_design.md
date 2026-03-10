# BinPackConfig 参数配置设计

## 1. 概述

`BinPackConfig` 是 bin-packing 算法的配置类，负责管理调度过程中的各种参数。该类继承自 `dict`，同时提供类型安全的属性访问。

**设计目标**：
- 解决字典访问缺少类型检查和 IDE 补全的问题
- 保持对旧代码的完全兼容
- 支持渐进式迁移到类型安全的访问方式

---

## 2. 当前设计

### 2.1 类定义

```python
# sched/binpack_config.py

class BinPackConfig(dict):
    """binpack_cfg 包装器，继承自 dict，提供属性访问模式"""

    @property
    def algorithm(self) -> str:
        """算法选择: 'guided', 'scratch', 'mem_plan', 'full'"""
        return self.get("algorithm", "coalescing")

    # ... 其他属性

    @staticmethod
    def generate_template() -> Dict[str, Any]:
        """生成包含所有可用参数及其默认值的模板字典"""
        # ... 实现代码 ...
```

### 2.2 设计优势

| 特性 | 继承 dict 方案 | 包装器方案 (不继承 dict) |
|------|----------------|------------------------|
| `isinstance(cfg, dict)` | ✅ `True` | ❌ `False` |
| 字典方法支持 | ✅ 自动继承 | ❌ 需手动实现 |
| 向后兼容性 | ✅ 完全兼容 | ⚠️ 部分兼容 |
| 属性访问 | ✅ 支持 | ✅ 支持 |
| IDE 类型提示 | ✅ 支持 | ✅ 支持 |

**结论**：继承 `dict` 是最优方案。

---

## 3. 参数定义

### 3.1 参数清单

| 参数名 | 类型 | 访问方式 | 含义 | 所在函数/模块 | 状态 | 来源 |
|:---|:---|:---|:---|:---|:---|:---|
| `algorithm` | `str` | `cfg.algorithm` | 算法选择：'guided', 'scratch', 'mem_plan', 'full' | `perform_bin_packing`, `push_task_into_bins_new` | ✅ 活动 | JSON |
| `sort` | `str` | `cfg.sort` | Bin 排序方式：'EAT', 'barycenter', 'bf' | `bin_sel`, `glb_alloc_new2` | ✅ 活动 | JSON |
| `mode` | `str` | `cfg.mode` | 任务插入模式：'non-block', 'block' | `allocate_rsc_4_process_new2`, `check_and_preemt_alloc` | ✅ 活动 | JSON |
| `bin_sel_mod` | `str` | `cfg.bin_sel_mod` | Bin 选择模式：'search', 'pre_defined' | `push_task_into_bins_new`, `allocate_rsc_4_process_new2` | ✅ 活动 | JSON |
| `reservation_policy` | `str` | `cfg.reservation_policy` | 预留策略：'manual', 'auto', 'static_1_bin' | `push_task_into_bins_new`, `coleasing_alloc_1bin` | ✅ 活动 | JSON |
| `affinity_en` | `bool` | `cfg.affinity_en` | 是否启用亲和性 | `glb_alloc_new2`, `allocate_rsc_4_process_new2` | ✅ 活动 | JSON |
| `affinity_level` | `int` | `cfg.affinity_level` | 亲和性层级 | `glb_alloc_new2`, `allocate_rsc_4_process_new2` | ✅ 活动 | JSON |
| `preempt_en` | `bool` | `cfg.preempt_en` | 是否允许抢占 | `check_and_preemt_alloc` | ✅ 活动 | JSON |
| `partial_alloc_en` | `bool` | `cfg.partial_alloc_en` | 是否允许部分分配 | `check_and_preemt_alloc` | ✅ 活动 | JSON |
| `quantum_check_en` | `bool` | `cfg.quantum_check_en` | 是否启用量子化检查 | `check_and_preemt_alloc` | ✅ 活动 | JSON |
| `mapping` | `Dict[int, int]` | `cfg.mapping` | 任务到 Bin 的静态映射 | `allocate_rsc_4_process_new2`, `bin_select_new` | ✅ 活动 | 运行时 |
| `quantile` | `float` | `cfg.quantile` | 资源估算分位数 | `rsc_req_estm_quantile` | ✅ 活动 | 运行时 |
| `var_dist_map` | `Dict` | `cfg.var_dist_map` | 任务负载分布图 | `rsc_req_estm_quantile` | ✅ 活动 | 运行时 |
| `core_size` | `str` | `cfg.core_size` | 核心大小模式：'specified', 'induced' | `sim_main.py` (环境准备阶段) | ✅ 活动 | JSON |
| `exec_t_comp_ratioB` | `float` | `cfg.exec_t_comp_ratioB` | repack 使用的分位数 | `perform_bin_packing` | ✅ 活动 | 运行时 |
| `sort_reverse` | `bool` | - | 排序反转 | - | ❌ 已移除 | 过期 |
| `release_temp_rda` | `bool` | - | 临时资源释放 | - | ❌ 已移除 | 过期 |
| `slack_sharing` | `bool` | - | 松弛量共享 | - | ❌ 已移除 | 过期 |
| `percentile` | `float` | - | 分位数参数 | `push_step_new` | ❌ 已移除 | 过期 |

### 3.2 参数来源说明

| 来源标记 | 说明 | 示例 | 是否可配置 |
|----------|------|------|-----------|
| **JSON** | 在 `cfgs/*.json` 中定义的静态配置 | `algorithm`, `affinity_en` | ✅ 可通过切换 JSON 文件配置 |
| **运行时** | 代码动态注入，不在 JSON 中 | `var_dist_map`, `quantile`, `mapping` | ❌ 由算法自动生成 |
| **Parser** | 命令行参数（实验参数） | `--bin_pack_cfg`, `--num_bins` | ✅ 可通过命令行配置 |

**注意**：`bin_sel_mod`, `affinity_en`, `affinity_level` 虽然有默认值，但已在 JSON 文件中显式配置，因此来源标记为 **JSON**。

### 3.3 补充说明：已移除参数细节

| 参数名 | 原用途 | 移除原因 | 替代方案 |
|:---|:---|:---|:---|
| `sort_reverse` | 控制 Bin 排序是否反转 | 代码中从未读取该配置，逻辑已硬编码或由 `sort` 参数隐含 | 无需替代 |
| `release_temp_rda` | 允许缩减冗余资源释放 | 功能已被 `get_rsc_2b_released()` 替代，原逻辑已失效 | `get_rsc_2b_released()` |
| `slack_sharing` | 链间松弛量共享开关 | 算法重构后不再使用该布尔开关，由新的分配逻辑统一处理 | 新的 guided 算法逻辑 |
| `percentile` | `push_step_new` 的分位数参数 | 属于“僵尸参数”，函数签名中有但体内未使用，实际通过 `binpack_cfg.quantile` 传递 | `cfg.quantile` |

---

## 4. 关键逻辑补充

### 4.1 统一入口读取 (Entry Logic)

为了保证代码的健壮性，所有子函数在处理 `binpack_cfg` 时应遵循以下模式：
1. **入口处统一读取**：在函数开始处将需要的配置项读入局部变量。
2. **默认值保护**：使用 `.get()` 或属性访问（已内置默认值）。
3. **显式校验**：对于关键参数（如 `mapping` 在 `pre_defined` 模式下），应添加显式校验。

**示例代码**：
```python
def allocate_rsc_4_process_new2(..., binpack_cfg: BinPackConfig):
    # 1. 入口处统一读取
    bin_sel_mod = binpack_cfg.bin_sel_mod
    affinity_en = binpack_cfg.affinity_en
    mapping = binpack_cfg.mapping
    
    # 2. 显式校验
    if bin_sel_mod == "pre_defined" and not mapping:
        raise KeyError("binpack_cfg missing 'mapping' for pre_defined mode")
    
    # ... 后续逻辑使用局部变量 ...
```

### 4.2 职责分离 (Responsibility Separation)

- **`utils.py`**: 负责从磁盘加载 JSON 并初始化 `BinPackConfig` 对象。
- **`sim_main.py`**: 负责在调用分配算法前，根据当前作业图 (`job_graph`) 和分位数 (`quantile`) 动态注入运行时参数。
- **`sched/`**: 业务逻辑层，只读访问配置，不应在深层函数中修改全局配置。

---

## 5. 使用方式 (更新)

### 4.1 访问方式对比

```python
cfg = BinPackConfig({"algorithm": "scratch"})

# 方式 1：字典访问（向后兼容）
algo = cfg["algorithm"]
algo = cfg.get("algorithm", "coalescing")
cfg.update({"mode": "block"})

# 方式 2：属性访问（类型安全，推荐）
algo = cfg.algorithm          # IDE 自动补全
cfg.mode = "block"           # 类型检查
```

### 4.2 类型注解

```python
# 方案 A：使用 BinPackConfig（推荐新代码）
def foo(binpack_cfg: BinPackConfig):
    algo = binpack_cfg.algorithm  # IDE 提示

# 方案 B：使用 Dict（兼容旧代码）
from typing import Dict, Any
def foo(binpack_cfg: Dict[str, Any]):
    algo = binpack_cfg.get("algorithm")

# 方案 C：不指定类型（最灵活）
def foo(binpack_cfg):
    algo = binpack_cfg.get("algorithm")
```

**由于 `BinPackConfig` 继承 `dict`，三种方式都正确**：
```python
isinstance(BinPackConfig(), dict)  # True
```

---

## 5. 调用链

```
perform_bin_packing (sim_main.py)
│
├─── prepare_binpack_cfg()          # 添加 var_dist_map, quantile
│
├─── push_task_into_bins_new        # scratch 算法
│    └─── push_step_new
│         └─── glb_alloc_new2
│              └─── allocate_rsc_4_process_new2
│                   ├─── bin_sel
│                   └─── check_and_preemt_alloc
│
└─── coleasing_alloc_cluster        # guided 算法
     └─── coleasing_alloc_1bin
```

每个函数通过 `binpack_cfg` 参数传递配置。

---

## 6. 长期迁移计划

### 6.1 迁移阶段

| 阶段 | 状态 | 说明 |
|------|------|------|
| **阶段 1** | ✅ 完成 | 创建 `BinPackConfig` 类，继承 `dict` |
| **阶段 2** | ✅ 完成 | 清理过期配置参数（`sort_reverse`, `release_temp_rda`, `slack_sharing`） |
| **阶段 3** | 🔄 进行中 | 新代码使用属性访问 `cfg.algorithm` |
| **阶段 4** | ⏳ 待开始 | 逐步迁移旧代码到属性访问 |
| **阶段 5** | ⏳ 待开始 | 统一类型注解为 `BinPackConfig` |

### 6.2 迁移原则

1. **不破坏现有功能**：所有旧代码保持可用
2. **渐进式迁移**：新代码优先使用新方式
3. **保持一致性**：同一模块内使用统一的访问方式

### 6.3 迁移示例

**旧代码**：
```python
def foo(binpack_cfg: Dict[str, Any]):
    if binpack_cfg["algorithm"] == "scratch":
        mode = binpack_cfg.get("mode", "non-block")
```

**新代码**：
```python
def foo(binpack_cfg: BinPackConfig):
    if binpack_cfg.algorithm == "scratch":
        mode = binpack_cfg.mode  # 有默认值，无需 .get()
```

---

## 7. 类型检查和 IDE 支持

### 7.1 当前支持

- ✅ PyCharm/VSCode 类型提示
- ✅ 属性自动补全
- ✅ 类型检查器（mypy, pyright）

### 7.2 最佳实践

```python
# 推荐写法
def allocate_resources(cfg: BinPackConfig):
    # 使用属性访问
    algo = cfg.algorithm      # 类型: str
    mode = cfg.mode          # 类型: str
    affinity = cfg.affinity_en  # 类型: bool

    # 字典访问仍然可用
    if "custom_key" in cfg:
        value = cfg["custom_key"]
```

---

## 8. 已删除的过期参数

| 参数 | 删除原因 | 删除日期 |
|------|----------|----------|
| `sort_reverse` | 代码中从未读取 | 2026-02-03 |
| `release_temp_rda` | 已被 `get_rsc_2b_released()` 替代 | 2026-02-03 |
| `slack_sharing` | 代码中不再使用 | 2026-02-03 |

详见：`doc/code_cleanup_2025.md` 清理4

---

## 9. 相关文档

- `doc/code_cleanup_2025.md` - 代码清理记录
- `doc/parameter_flow_exec_t_comp_ratio.md` - 参数传递路径
- `sched/binpack_config.py` - 类实现
