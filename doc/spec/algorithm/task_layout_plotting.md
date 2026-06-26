# 任务布局绘图（bin_list_utils）原理

**一句话核心**：把每个 bin 的 `scheduling_table`（时间槽序列 × 资源映射）转换成时间-空间甘特图；核心难点不是"画图"，而是给同一 bin 内共存的多个任务分配**不重叠且稳定**的垂直位置——通过对 bin 空间维度做一维装箱（first-fit + leftmost）维护 `position_dict`。

> **层次**：`doc/spec/algorithm/`（算法原理规范）
> **相关代码**：`sched/bin_list_utils.py`（`get_task_layout_compact` / `get_task_layout_compact1bin` / `Bin_list_print`）
> **上层文档**：`doc/spec/e2e_sched_sim_flow.md`（scheduling_table 何时生成）、`doc/guide/plotting_overview.md`（绘图函数总览）

---

## 1. 设计思想

### 1.1 关键设计决策

| 决策 | 问题 | 解决方案 |
|------|------|----------|
| **按"变化点"绘图，而非逐槽** | scheduling_table 有上千时间槽，逐槽画冗余且图形碎裂 | 连续相同 `rsc_map` 的槽合并为一段，只在 `rsc_map` 变化时切分 → 用 `broken_barh` 画水平条 |
| **维护 `position_dict` 而非每段重算位置** | 同一 bin 内多任务共存，垂直位置（y 轴）必须稳定，否则同任务的位置在相邻段跳动，图无法读 | 全程维护 `position_dict[pid]`，记录每个任务的垂直区间；仅在任务进出/资源量变化时增量更新 |
| **空间分配用 first-fit + leftmost** | bin 的资源（`num_resources` 个 core）是一维空间，多任务要占据不重叠区间 | 贪心一维装箱：新任务优先找第一个能容纳的连续空隙（first-fit），找不到则取最左（leftmost）→ 任务向左聚拢，图紧凑 |
| **先处理 size_minus 再 size_plus** | 资源量变化的任务需要重新分配位置 | 先收缩（释放位置）再扩张（抢占空位），避免扩张时占用了即将被释放的位置 |
| **多 bin 垂直堆叠 + 共享 x 轴** | 不同 bin 是空间隔离的分区，但时间维度需要对齐 | 每个 bin 一个 subplot，垂直堆叠（`subplots(sharex=True)`），x 轴统一为时间 |

### 1.2 核心设计原则

1. **绘图单位是"变化点"不是"时间槽"**：scheduling_table 本质是分段常量，绘图对齐其自然分段。
2. **位置分配是一维装箱问题**：bin 的空间维度（core 数）是被装箱的"容器"，任务的资源需求是"物品"，first-fit 是装箱策略——这正是函数名 **compact**（紧凑）的由来。
3. **位置稳定性优先**：`position_dict` 增量更新，老任务尽量保持原位（`cum_pos` 锚点），只有必要时才迁移。
4. **图形即调试视图**：该图主要用于人工检查装箱/调度结果是否合理，因此信息密度（任务名标注、网格线、bin 标签）优先于美观。

---

## 2. 数据流与流程

### 2.1 整体流程

```
bin_list: List[SchedulingTableInt]    pid2name: {pid: task_name}
    │
    ▼
┌─────────────────────────────────────────────┐
│  对每个 bin (_SchedTab):                      │
│    遍历 scheduling_table[t].rsc_map          │
│      │                                       │
│      ├─ rsc_map 未变 → 继续下槽（合并段）      │
│      │                                       │
│      └─ rsc_map 变化（变化点）:                │
│           ① 画上一段 [s→e] 的 broken_barh     │
│                └ 垂直位置取自 position_dict    │
│           ② 更新 position_dict:               │
│                ├ expired_pid: 释放位置         │
│                ├ size_minus/plus: 重分配       │
│                └ new_pid: first-fit 分配      │
│           ③ 画垂直网格线 axvline(e)           │
└─────────────────────────────────────────────┘
    │
    ▼
多 bin subplot 垂直堆叠（sharex=True）
    │
    ▼
输出 PDF/PNG（save_path，默认 pdf）
```

### 2.2 关键流程说明

- **时间范围**：`plot_start`/`plot_end` 由 `hyper_p`、`n_p`、`warmup`、`drain` 推导；warmup/drain 段会被裁剪到绘图边界（任务跨边界时，起止时间被夹到 `[plot_start, plot_end]`）。
- **颜色**：按 `pid % len(colors)` 从 `XKCD_COLORS` 取色，保证同任务跨段同色。
- **文字标注**：每个任务只在首次出现的位置标注一次（`is_new` 标志），避免重复。
- **`compact1bin` vs `compact`**：前者只画单个 bin（单 subplot），布局逻辑相同；后者遍历整个 `bin_list`。

---

## 3. 核心概念

### 3.1 `rsc_map`（资源映射）

| 项 | 说明 |
|----|------|
| 类型 | `Dict[pid, int]` |
| 含义 | 某时间槽 `t`，bin 内各任务占用的 core 数量 |
| 来源 | `_SchedTab.scheduling_table[t].rsc_map` |
| 性质 | **分段常量**——连续多槽通常相同，是"变化点"检测的基础 |

### 3.2 `position_dict`（垂直位置簿）

这是本算法的核心数据结构，维护"任务 → bin 空间内的垂直区间"映射。

```python
position_dict[pid] = [[start_0, start_1, ...],   # 各段起始 core 索引
                      [size_0,  size_1,  ...],   # 各段 core 数量
                      is_new]                    # 是否尚未标注文字
```

- **为什么是 list 的 list**：任务的资源区间可能不连续（被其他任务挤碎），所以用多段 `[start, size]` 表示。
- **生命周期**：新任务进入时创建，资源量变化时重算，任务退出时 `pop`。
- **is_new 标志**：首次绘制后置 `False`，防止任务名重复标注。

### 3.3 一维空间装箱（位置分配策略）

bin 的 `num_resources`（core 数）被视为一维容器，任务的资源需求是物品。分配规则：

| 情形 | 策略 |
|------|------|
| 新任务（`new_pid`） | **first-fit**：从左扫描可用位置，找第一个能容纳 `p_size` 的连续空隙；找不到则 **leftmost** 取最左 `p_size` 个 |
| 资源量减少（`size_minus`） | 释放旧位置，以原 `cum_pos` 为锚点重新选 `new_size` 个位置（优先向右，不足向左补） |
| 资源量增加（`size_plus`） | 同上，但因先处理 minus，可用位置更充裕 |
| 任务退出（`expired_pid`） | 从 `position_dict` 移除，其位置回归可用池 |

> **顺序约束**：必须先处理 `size_minus` 再 `size_plus`（收缩让位 → 再扩张），否则扩张可能占用尚未释放的位置。

### 3.4 `Bin_list_print`（文本打印）

非图形函数，按时间槽顺序文本打印各 bin 的 `rsc_map`，用于无图形环境下的调试。不涉及位置分配。

---

## 4. 接口速查

### 4.1 API

| 函数 | 位置 | 用途 |
|------|------|------|
| `get_task_layout_compact(bin_list, pid2name, time_step, show, save, save_path, hyper_p, n_p, warmup, drain, plot_legend, plot_start, plot_end, tick_dens, txt_size, *, tool, **kwargs)` | `bin_list_utils.py:23` | 多 bin 紧凑甘特图 |
| `get_task_layout_compact1bin(...)` | `bin_list_utils.py:327` | 单 bin 版（同参） |
| `Bin_list_print(bin_list, glb_p_list, timestep)` | `bin_list_utils.py:639` | 文本打印调度表 |

### 4.2 关键参数

| 参数 | 含义 |
|------|------|
| `time_step` | 单时间槽对应的物理时长（s），用于 x 轴换算 |
| `hyper_p` / `n_p` / `warmup` / `drain` | 推导 `plot_start`/`plot_end` 与 `event_range` |
| `plot_legend` | `True` 时仅生成统一图例（不画任务名） |
| `tool` | 绘图后端，实际仅 `"matplotlib"`（bokeh/plotly 已废弃，见 `bin_list_utils_unused.py`） |

### 4.3 输入数据来源

`bin_list` 由 `perform_bin_packing`（`sim_main.py`）生成，每个 `SchedulingTableInt` 的 `scheduling_table` 是装箱/调度结果。详见 `doc/spec/algorithm/guided_hybrid_allocation_algorithm.md`。

---

## 5. 实现细节（深入）

> 本节仅在需要修改位置分配算法时阅读。

### 5.1 变化点检测与段绘制

```python
for rsc_map_idx in range(bin_temp_size):
    rsc_map = _SchedTab.scheduling_table[rsc_map_idx].rsc_map
    if rsc_map == pre_rsc:
        continue                      # 合并：与前一槽相同，跳过
    # —— 此处为变化点 ——
    s, e = pre_idx*time_step, rsc_map_idx*time_step   # 上一段的起止
    for pid, size in pre_rsc.items():
        for vertical_s, vertical_size in zip(*position_dict[pid][:-1]):
            ax.broken_barh([(s, e-s)], (vertical_s, vertical_size), facecolors=color)
    pre_idx = rsc_map_idx; pre_rsc = rsc_map
```

### 5.2 first-fit 位置搜索

```python
# 为新任务 pid 找连续空位
gap_start, gap_size = None, 0
for pos in aval_pos:
    if gap_start is None: gap_start = pos
    gap_size += 1
    if gap_size == p_size: break      # 找到够大的连续空隙
    if pos + 1 not in aval_pos:       # 不连续，重置
        gap_start, gap_size = None, 0
```

### 5.3 性能特性

- **时间复杂度**：O(T + 变化点数 × 平均共存任务数)。因 scheduling_table 分段常量，变化点数远小于 T，实际很快。
- **瓶颈**：`aval_pos` 的 list 扫描（first-fit）；大 bin（`num_resources` 大）时可用排序结构优化，但当前规模无需。

### 5.4 已归档的相关函数（勿用）

| 函数 | 归档位置 | 说明 |
|------|---------|------|
| `get_task_layout`（无后缀） | `bin_list_utils_old.py` | compact 前身（历史版本） |
| `get_task_layout_sparse` + `add_bar`/`add_text`/`add_v_grid` | `bin_list_utils_unused.py` | 多后端可视化尝试，`backend` 参数仅 matplotlib 分支，未落地 |

---

## 6. 与其他文档的关系

- **数据来源**：`doc/spec/algorithm/guided_hybrid_allocation_algorithm.md`（scheduling_table 怎么生成）
- **绘图总览**：`doc/guide/plotting_overview.md`（全仓绘图函数索引）
- **统计绘图**：`doc/spec/stat/collector.md`（`approach_collector.plot_motiv_*` 是另一类绘图，处理统计摘要而非调度表）

---
*创建: 2026-06-27*
