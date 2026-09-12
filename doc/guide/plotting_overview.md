# 绘图函数总览

> **层次**: `doc/guide/`（模块导航）
> **作用**: 统一梳理全仓绘图函数——画什么图、用哪个函数、在哪里、输入/输出、调用入口
> **相关**: `collector_overview.md`（StatisticsCollector）、`doc/spec/exp_design/`（motiv 绘图 API）、`doc/spec/{motiv1,motiv2,abla1,abla2-3}_describe.md`（各实验绘图细节）

## 1. 概述

- **统一后端**：`matplotlib`。全仓所有活绘图函数都基于 matplotlib（`pyplot`/`axes`）。`bokeh`/`plotly` 已废弃（早期多后端尝试未落地，2026-06-16 B4 已从 `requirement.txt` 移除，相关注释 import 已清理）。
- **统一字体/样式**：`scripts/exp_common.py:252-264` 通过 `plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', ...]` 设置中文字体，`axes.unicode_minus = False` 修正负号。实验脚本统一走此设置。
- **输出格式**：实验图默认 PDF（论文用），部分 PNG。路径由 `path_para` / `save_path` 参数控制。

## 2. 实验结果绘图（`approach/approach_collector.py`，核心）

`StatisticsCollector` 类的 `@staticmethod` 方法，用于 motiv/abla/e2e 实验的结果可视化。输入为统计摘要 `data_points: List[Dict]`，输出 PDF/PNG。

| 函数 | 位置 | 用途 | 典型调用方 |
|------|------|------|-----------|
| `plot_motiv_case1(data_points, save_path, show, ...)` | `:773` | motiv case1：利用率/超时/空闲柱状图（双轴） | motiv_exp_runner / abla_exp_runner (case1) |
| `plot_motiv_case2(data_points, plot_type, ...)` | `:1121` | motiv case2：延迟分解 + 利用率 breakdown | motiv_exp_runner / abla_exp_runner (case2) |
| `plot_motiv_case3(collector_base, ...)` | `:2075` | motiv case3：规模-延迟-利用率 | motiv_exp_runner / abla_exp_runner (case3) |
| `plot_motiv_legend(save_path, show)` | `:1502` | 生成统一图例（跨图复用） | 实验脚本 |
| `plot_load_latency_binned(binned_summary, spearman_rho, ...)` | `:2138` | 负载-延迟分箱散点 + Spearman 相关系数 | motiv case3 |
| `plot_load_latency_raw(data_groups, save_path, show)` | `:2231` | 负载-延迟原始散点 | motiv case3 |

> **API 细节**：见 `doc/spec/exp_design/motiv_case_api_{README,usage}.md` 与 `motiv_case_plotting_examples.md`。
> **注意**：`plot_motiv_case1/2` 被 motiv 与 abla 实验共享，**不要修改其签名/逻辑**（见 CLAUDE.md "Adding New Experiments"）。

## 3. 仿真过程绘图（`sim_main.py`）

装箱/仿真的中间结果可视化，在 `perform_bin_packing` 流程中按 `args` 触发。

| 函数 | 位置 | 用途 |
|------|------|------|
| `render_bin_pack_plots(args, bin_list, glb_p_list, sim_step, hyper_p, num_periods, plot_path_para, path_ctx)` | `:118` | 渲染装箱结果（bin 甘特图） |
| `render_runtime_full_plot(args, actual_sched_record, glb_p_list, sim_step, hyper_p, num_periods, case_pth, plot_path_para, path_ctx)` | `:153` | 渲染运行时全图（调度时间线） |

## 4. 调度表布局与打印（`sched/bin_list_utils.py`）

为 §3 的 render 函数提供布局数据（任务在时间-空间网格中的位置），以及文本打印。

| 函数 | 位置 | 用途 | 状态 |
|------|------|------|------|
| `get_task_layout_compact(bin_list, pid2name, time_step, ...)` | `:23` | 紧凑布局（多 bin），返回绘图数据 | ✅ 活 |
| `get_task_layout_compact1bin(bin_list, pid2name, time_step, ...)` | `:327` | 紧凑布局（单 bin） | ✅ 活 |
| `Bin_list_print(bin_list, glb_p_list, timestep)` | `:639` | 文本打印调度表（调试用） | ✅ 活 |

> **已归档**（2026-06-16 B5）：
> - `get_task_layout`（compact 前身）→ `sched/bin_list_utils_old.py`（历史版本）
> - `get_task_layout_sparse` + `add_bar`/`add_text`/`add_v_grid`（多后端可视化尝试，`backend` 参数仅 matplotlib 分支，未落地）→ `sched/bin_list_utils_unused.py`（未完成独立版本）

## 5. 结构 / 拓扑图

任务图、工作流、计算图的可视化，多用于调试与论文配图。

| 函数 | 位置 | 用途 |
|------|------|------|
| `plot_timeline_graph(logical_graph_nx, path="plot/jobTask_graph_dbg.pdf")` | `sched/slack_estim.py:496` | 任务逻辑图时间线（debug） |
| `plot_workflow_g(node_color_map, edge_color_map, graph_nx, ax)` | `task/task_cfg.py:1297` | 工作流 DAG（按节点/边着色） |
| `draw_computational_graph(model)` | `optimizer/scheduler_base.py:240` | 计算图（optimizer 模块） |

## 6. 实验脚本辅助绘图（`scripts/`）

实验 runner 内部的专用绘图，不对外暴露，仅在对应实验流程中被调用。

| 函数 | 位置 | 用途 |
|------|------|------|
| `_plot_abla_overhead(data_points, x_key, x_values, x_labels, title, save_path, ...)` | `scripts/abla_exp_runner.py:204` | 消融实验：切换开销图（`case{N}_overhead.pdf`） |
| `_plot_abla_tradeoff(data_points, x_key, x_values, x_labels, title, save_path, ...)` | `scripts/abla_exp_runner.py:300` | 消融实验：权衡图（`case{N}_tradeoff.pdf`） |
| `plot_grouped_bar(data, x_key, y_key, ...)` | `scripts/e2e_exp_runner.py:171` | 端到端：分组柱状图 |
| `plot_tradeoff_curve(data, save_path, title)` | `scripts/e2e_exp_runner.py:229` | 端到端：权衡曲线 |

> 消融实验统一配色：`ABLA_COLORS`（`abla_exp_runner.py`，exec=C0, realloc=C1, wait=C2, miss_bar=C3, miss_line=C4, idle=C7），与 motiv 配色对齐。

## 7. 统一字体/样式设置

`scripts/exp_common.py:252-264` 提供 `set_font`（或等价 rcParams 设置）：

```python
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', ...]
plt.rcParams['axes.unicode_minus'] = False
```

实验脚本通过 `from scripts.exp_common import ...` 复用此设置，保证中文标注（坐标轴、图例、标题）在所有图中一致显示。新增绘图函数应沿用此设置，不要单独 `rcParams`。

## 8. 新增绘图函数的约定

1. **后端**：只用 matplotlib，不要引入 bokeh/plotly（已废弃）。
2. **样式**：复用 `exp_common.py` 的字体设置。
3. **实验图**：motiv/abla 共享的图（case1/2）走 `approach_collector.plot_motiv_*`，不要改其签名；实验专用图放对应 runner 内（`_plot_*`）。
4. **输出路径**：通过 `path_para` / `save_path` 参数化，遵循 `paths.py` 的路径约定。
5. **配色**：消融实验用 `ABLA_COLORS`，与 motiv 对齐。

## 9. 相关文档索引

| 文档 | 内容 |
|------|------|
| `doc/spec/exp_design/motiv_case_api_README.md` | motiv 绘图 API 总览 |
| `doc/spec/exp_design/motiv_case_api_usage.md` | motiv 绘图 API 用法 |
| `doc/spec/exp_design/motiv_case_plotting_examples.md` | motiv 绘图示例 |
| `doc/spec/motiv1_describe.md` / `motiv2_describe.md` | motiv case1/2 绘图细节 |
| `doc/spec/abla1_describe.md` / `abla2-3_describe.md` | 消融实验绘图细节 |
| `doc/guide/collector_overview.md` | StatisticsCollector 架构（含 plot 方法） |
| `doc/spec/stat/statistics_collection_spec.md` | 统计收集（绘图数据来源） |

---
*创建: 2026-06-16（B5 之后，梳理全仓绘图函数）*
