# 实验图片绘制上下文

## 任务目标

为毕业论文生成中文版实验图（动机实验 + 消融实验），同时调整英文版图片的字体大小和布局。

## 涉及图片清单

### 动机实验 (motiv) — 3 张图 × 2 语言 = 6 个 PDF

| 图 | 英文路径 | 中文路径 |
|----|---------|---------|
| Case 1 tradeoff | `motiv_exp_results/case1/case1_tradeoff.pdf` | `motiv_exp_results_zh/case1/case1_tradeoff.pdf` |
| Case 2 breakdown | `motiv_exp_results/case2/case2_breakdown.pdf` | `motiv_exp_results_zh/case2/case2_breakdown.pdf` |
| Case 2 utilization | `motiv_exp_results/case2/case2_utilization.pdf` | `motiv_exp_results_zh/case2/case2_utilization.pdf` |

### 消融实验 (abla) — 4 张图 × 2 语言 = 8 个 PDF

| 图 | 英文路径 | 中文路径 |
|----|---------|---------|
| Case 1 motiv1_style | `abla_fixcore_test/case1/case1_motiv1_style.pdf` | `abla_results_zh/case1/case1_motiv1_style.pdf` |
| Case 2 overhead | `abla_fixcore_test/case2/case2_overhead.pdf` | `abla_results_zh/case2/case2_overhead.pdf` |
| Case 2 tradeoff | `abla_fixcore_test/case2/case2_tradeoff.pdf` | `abla_results_zh/case2/case2_tradeoff.pdf` |
| Case 3 tradeoff | `abla_fixcore_test/case3/case3_tradeoff.pdf` | `abla_results_zh/case3/case3_tradeoff.pdf` |

### 端到端实验 — 2 张图（无 matplotlib 脚本，仅有 overlay 中文版）

- `scheduler_paper/figures/pdf/exp_latency_TC.pdf` / `_zh.pdf`
- `scheduler_paper/figures/pdf/exp_throughput0807.pdf` / `_zh.pdf`

## 代码结构

### 绘图函数位置

| 函数 | 文件 | 用途 |
|------|------|------|
| `plot_motiv_case1()` | `approach_collector.py` | 动机 Case 1 + 消融 Case 1 共用；搜索函数名定位 |
| `plot_motiv_case2()` | `approach_collector.py` | 动机 Case 2 breakdown + utilization |
| `_plot_abla_overhead()` | `scripts/abla_exp_runner.py` | 消融 Case 2/3 overhead |
| `_plot_abla_tradeoff()` | `scripts/abla_exp_runner.py` | 消融 Case 2/3 tradeoff |
| `_plot_satisfy_projection()` | `scripts/abla_exp_runner.py` | 消融 Case 1 额外投影图 |

### 参数传递机制

绘图函数主要通过以下可选字典参数传入布局和文字配置：

- **`labels`**: 文字翻译覆盖（键名必须与代码中 `_L`/`_LB`/`_LU` 的键完全一致）
- **`fontsize`**: 字号覆盖（键名: `label`, `tick`, `title`, `legend`, `annot`, `table` 等）
- **`table_cfg`**: 仅 `plot_motiv_case1`，控制表格布局（见下）
- **`style`**: 仅 `_plot_abla_overhead/tradeoff`，控制 ABLA_PLOT_STYLE

### `table_cfg` 参数（`plot_motiv_case1` 专用）

```python
# 当前接口：按每行几组结果自动生成表格；不要再使用旧的 rows 参数。
{
    'groups_per_row': 3,          # 每行显示几组 [Percentile, Idle, Miss]
    'bbox': None,                 # 显式 [left, bottom, width, height]；优先级最高
    'left': -0.12,                # bbox=None 时使用
    'width': 1.24,
    'table_top': -0.62,
    'row_height': 0.14,
    'col_unit': [0.08, 0.10, 0.12],  # 每组 [label_col, idle_col, miss_col]
    'col_widths': None,           # 显式列宽；优先级高于 col_unit
    'row_scale': 1.0,
    'bottom_margin': 0.30,
    'subplots_adjust': None,      # 显式 fig.subplots_adjust 配置
    'font_size': None,            # None=用 _FS['table']
    'cell_pad': None,             # 表格单元格内边距；字体大但放不下时很有用
    'savefig': {'bbox_inches': 'tight'},
}

# 消融 Case 1 英文专用表格配置（四图并列，需要更大的字号和可控画布）
{
    'figsize': (5.0, 3.12),
    'left': -0.21,
    'width': 1.43,
    'table_top': -0.68,
    'row_height': 0.23,
    'col_unit': [0.16, 0.22, 0.22],
    'cell_pad': 0.01,
    'subplots_adjust': {'left': 0.16, 'right': 0.78, 'bottom': 0.52, 'top': 0.82},
    'font_size': MOTIV_FONTSIZE_EN['table'] * ABLA_FONT_SCALE_EN,
}
```

## 字号配置

### 基准字号（中文和非四图横排场景）

```python
# plot_motiv_case1 _FS 默认值:
{'label': 11, 'tick': 10, 'title': 11, 'legend': 8, 'annot': 8, 'annot_bold': 8, 'table': 11, 'cyc_ref': 11}

# plot_motiv_case2 _FS 默认值:
{'label': 11, 'tick': 10, 'title': 11, 'legend': 8, 'annot': 8}

# _plot_abla_overhead/tradeoff 使用 ABLA_PLOT_STYLE:
{'fontsize_label': 11, 'fontsize_tick': 10, 'fontsize_title': 11, 'fontsize_legend': 8, 'fontsize_annot': 8, 'markersize': 4, 'linewidth': 1.5}
```

### 英文版（按图的论文布局放大）

```python
# scripts/exp_common.py
MOTIV_FONTSIZE_EN = {
    'label': 11, 'tick': 10, 'title': 11, 'legend': 8,
    'annot': 8, 'annot_bold': 8, 'table': 11, 'cyc_ref': 11,
}

# scripts/abla_exp_runner.py：消融图英文版四图并列，因此整体放大 30%；中文版不放大。
ABLA_FONT_SCALE_EN = 1.3
ABLA_PLOT_STYLE_EN = {
    'fontsize_label': 14.3, 'fontsize_tick': 13.0, 'fontsize_title': 14.3,
    'fontsize_legend': 10.4, 'fontsize_annot': 10.4, 'markersize': 6, 'linewidth': 1.8,
}
```

## 最重要经验（2026-06-09 复盘）

1. **先确认论文里的排版，再调单张图。** 消融实验英文版是四个小图并列，所以英文消融图需要整体放大 30%；中文版不是这个布局，不能同步放大。动机实验也不是四图并列，不要套用消融英文参数。
2. **中英文参数必须显式分离。** 英文消融使用 `ABLA_PLOT_STYLE_EN`、`ABLA_CASE1_FONTSIZE_EN`、`ABLA_CASE1_TABLE_CFG_EN`；中文消融使用基准 `ABLA_PLOT_STYLE` 和普通 `MOTIV_CASE1_TABLE_CFG`。不要用一个全局变量同时影响中英文。
3. **动机 Case 1 和消融 Case 1 共用 `plot_motiv_case1()`，但表格布局不相同。** 必须通过调用侧传入 `fontsize` 和 `table_cfg`，不要在绘图函数里为某个实验硬编码。现在两者都按每行三组 `[percentile, idle, miss]` 组织，但英文消融 Case 1 需要单独的 `figsize/row_height/col_unit/cell_pad`。
4. **调字体时要同时看画布物理尺寸。** 单纯增大 `figsize` 会让 LaTeX `\includegraphics[width=...]` 重新缩放，最终字号可能反而和预期不一致。英文消融 Case 1 为了容纳表格可略宽，但要控制 PDF 宽度，避免四图并列时被过度缩小。
5. **表格字号变化不明显时，先确认 `table.auto_set_font_size(False)` 和每个 cell 的 `set_text_props(fontsize=...)` 都生效。** 表格放不下时优先调 `col_unit/width/row_height/cell_pad/subplots_adjust`，不要只改 `font_size`。
6. **消融 Case 2/3 的 Low/Mid/High 折线图例已经去掉。** 左上角只保留 latency breakdown 的组件图例；折线颜色仍表示负载组。内部横轴小标签 `1/2/4/8`、`p50/p60/p70` 使用 legend 字号并旋转 30 度。
7. **柱顶数字标签要用算法排布。** `_place_clustered_bar_labels()` 按同一 cluster 内的柱高从低到高排序，用 `max(bar_height + pad, previous_label_y + gap)` 放置，保证最小间隔。注意这里必须是 `max`，不是 `min`；`min` 会把标签压回原位，无法避免重叠。
8. **术语和统计口径要先查代码再改图名。** 动机 Case 1 右轴的 `Miss Rate` 实际来自 `miss_mean_count = mean(missed tasks per hyperperiod) / task_cnt`，是 per-task timeout ratio，不是 E2E deadline miss rate；柱状图里的 `miss_mean_ratio` 是 missed remaining load / `(total_pwr * T_hp)`。
9. **中文标签必须保持中文。** 消融 Case 2/3 中文版中 `Low/Mid/High` 应为 `低负载/中负载/高负载`，`Normalized Latency` 应为 `归一化延迟`。不要把英文轴标签同步到中文图。
10. **每轮改图后必须做视觉检查。** PDF 生成成功不代表图能用；至少 `pdftoppm -png -r 220 ...` 后查看 case1 表格、case2 顶部数字、case3 旋转小标签和 legend 是否重叠。

## 运行命令

在 Windows/Codex 环境中执行时，用 WSL + zsh + gurobi 环境包住命令：

```powershell
wsl.exe -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler && conda activate gurobi && <cmd>'
```

进入 WSL 仓库后可直接执行：

```bash
conda activate gurobi

# 动机实验
python -m scripts.motiv_exp_runner --case 1 --output_dir ./motiv_exp_results --use_plot_cache
python -m scripts.motiv_exp_runner --case 1 --output_dir ./motiv_exp_results_zh --use_plot_cache --lang zh
python -m scripts.motiv_exp_runner --case 2 --output_dir ./motiv_exp_results --use_plot_cache
python -m scripts.motiv_exp_runner --case 2 --output_dir ./motiv_exp_results_zh --use_plot_cache --lang zh

# 消融实验（注意 case1 和 case3 的特殊参数）
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_fixcore_test --use_plot_cache --case1_ratioBs "0.5,0.6,0.7" --case1_cyc_ratios "0.8"
python -m scripts.abla_exp_runner --case 1 --output_dir ./abla_results_zh --use_plot_cache --lang zh --case1_ratioBs "0.5,0.6,0.7" --case1_cyc_ratios "0.8"
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_fixcore_test --use_plot_cache
python -m scripts.abla_exp_runner --case 2 --output_dir ./abla_results_zh --use_plot_cache --lang zh
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_fixcore_test --use_plot_cache --case3_ratioBs "0.5,0.6,0.7"
python -m scripts.abla_exp_runner --case 3 --output_dir ./abla_results_zh --use_plot_cache --lang zh --case3_ratioBs "0.5,0.6,0.7"
```

## 数据源

- **动机实验缓存**: `motiv_exp_results/case{1,2}/case{1,2}_summary.json`
- **消融实验缓存（英文版使用的数据）**: `abla_fixcore_test/case{1,2,3}/case{1,2,3}_summary.json`
- **消融实验缓存（中文版复制的）**: `abla_results_zh/case{1,2,3}/case{1,2,3}_summary.json`（从 `abla_fixcore_test/` 复制）
- **不要用** `abla_results/` 的缓存（旧数据，参数不同）

## 已知的坑

1. **`_FS["abel"]` typo**: 之前 sed 替换把 `"label"` 损坏成了 `"abel"`，已全部修正为 `"label"`
2. **`plot_motiv_case1` 被动机和消融共用**: 通过 `table_cfg` 区分表格布局；不要在函数内部写死某个实验的参数
3. **Case 2 标签键名**: 代码中用 `title_breakdown` / `title_util`，中文标签字典的键必须匹配
4. **中文字体**: WSL2 下通过 `fm.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')` 加载微软雅黑
5. **消融 Case 1 缓存过滤**: 缓存有 4 条数据（3 cyc-S + 1 cyc），需要用 `--case1_cyc_ratios "0.8"` 匹配
6. **消融 Case 3 缓存过滤**: 需要 `--case3_ratioBs "0.5,0.6,0.7"` 过滤到 3 个 ratioB
7. **英文四图并列不等于中文也四图并列**: 30% 字号放大只作用于英文消融图；中文图保持基准字号和布局
8. **`bbox_inches='tight'` 会改变 PDF 物理尺寸**: 调整 `figsize`、表格外扩、文本外扩后都要用 `pdfinfo` 看 Page size，确认 LaTeX 缩放不会抵消字号调整
9. **`latexmk.exe main.tex` 默认可能走 DVI `latex`**: 遇到 PDF 图片 BoundingBox 错误时，用 `pdflatex.exe -interaction=nonstopmode main.tex` 验证论文编译
10. **不要只看退出码**: Matplotlib/PDF 生成成功也可能存在标签重叠、表格裁切或字体缩放问题，必须打开 PNG/PDF 预览

## 论文目录

最终图片复制到: `scheduler_paper/figures/exp_data/TC/motiv/` 和 `scheduler_paper/figures/exp_data/TC/abla/`
中文版文件名加 `_zh` 后缀。
