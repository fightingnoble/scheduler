#!/usr/bin/env python3
"""
实验脚本公共模块

包含 Motivation 和 Ablation 实验共享的组件：
- 基础参数定义
- 参数模板类
- 主程序调用函数
- 绘图辅助函数
"""

import os
import sys
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# 并行与物理核心数
try:
    import psutil
    _PHYSICAL_CORES = psutil.cpu_count(logical=False) or os.cpu_count()
except Exception:
    _PHYSICAL_CORES = os.cpu_count()


# ==================== 基础参数定义 ====================

mapping_args = {
    'G_decomp_mode': "full",
    'exec_t_comp_ratioA': 0.99,
    'exec_t_comp_ratioB': -1,
    'e2e_latency': 0.1,
    'aux_scale_factor': 1,
    'test_case': 'bin_pack_new',
    'num_bins': -1,
    'bin_pack_cfg': 'Bp_guided.json',
    'n_p': 3,
}

specific_args = {
    'root_dir': None,
    'verbose': None,
    'stat_param': None,
}

runtime_args = {
    'n_p': None,
    'policy': None,
}


# ==================== 参数模板类 ====================

class ParamTemplate:
    """参数模板，封装三类参数并提供克隆、增量更新与合并导出能力。"""

    def __init__(self, mapping: Dict[str, Any], runtime: Dict[str, Any], specific: Dict[str, Any]):
        self.mapping = copy.deepcopy(mapping)
        self.runtime = copy.deepcopy(runtime)
        self.specific = copy.deepcopy(specific)

    def clone(self) -> 'ParamTemplate':
        return ParamTemplate(self.mapping, self.runtime, self.specific)

    def with_updates(
        self,
        mapping: Optional[Dict[str, Any]] = None,
        runtime: Optional[Dict[str, Any]] = None,
        specific: Optional[Dict[str, Any]] = None,
    ) -> 'ParamTemplate':
        new_mapping = copy.deepcopy(self.mapping)
        new_runtime = copy.deepcopy(self.runtime)
        new_specific = copy.deepcopy(self.specific)
        if mapping:
            new_mapping.update(mapping)
        if runtime:
            new_runtime.update(runtime)
        if specific:
            new_specific.update(specific)
        return ParamTemplate(new_mapping, new_runtime, new_specific)

    def to_run_args(self) -> Dict[str, Any]:
        # 合并并过滤 None
        merged: Dict[str, Any] = {}
        for group in (self.mapping, self.runtime, self.specific):
            for k, v in group.items():
                if v is not None:
                    merged[k] = v
        return merged


# ==================== 主程序调用函数 ====================

def run_main_approach_inproc(args_dict: Dict[str, Any], dry_run: bool = False):
    """以函数方式调用 main_approach.main()，避免子进程与磁盘往返。

    构造 sys.argv 供 utils.input_parser() 使用，返回 main_approach.main() 的 StatisticsCollector。
    """
    # 固定附加参数
    argv = [
        'main_approach.py',
        '--profiling_filename', 'profiling/profiling_light.csv',
        '--gen_benchmark',
    ]
    for key, value in args_dict.items():
        if value is None:
            raise ValueError(f"Argument {key} is None")
        if isinstance(value, bool):
            if value:
                argv.append(f'--{key}')
        else:
            argv.extend([f'--{key}', str(value)])

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Args: {' '.join(argv[1:])}")
    if dry_run:
        return None

    # 临时替换 sys.argv 调用 main_approach.main()
    import sys as _sys
    from main_approach import main as _main
    old_argv = list(_sys.argv)
    try:
        _sys.argv = argv
        collector = _main()
    finally:
        _sys.argv = old_argv
    return collector


# ==================== 绘图辅助函数 ====================

def group_by_key(data_points: List[Dict], key: str) -> Dict[Any, List[Dict]]:
    """按指定key对数据点进行分组。"""
    groups = defaultdict(list)
    for d in data_points:
        groups[d[key]].append(d)
    return dict(groups)


def compute_group_means(grouped_data: Dict[Any, List[Dict]], 
                        value_keys: List[str]) -> Dict[Any, Dict[str, float]]:
    """计算每个组的平均值。
    
    支持嵌套字典访问，如 'utilization.realloc_mean_ratio'
    """
    result = {}
    for key, items in grouped_data.items():
        result[key] = {}
        for vk in value_keys:
            if '.' in vk:
                parts = vk.split('.')
                vals = []
                for item in items:
                    val = item
                    for p in parts:
                        val = val.get(p, {}) if isinstance(val, dict) else {}
                    if isinstance(val, (int, float)):
                        vals.append(val)
                result[key][vk] = sum(vals) / len(vals) if vals else 0.0
            else:
                result[key][vk] = sum(d.get(vk, 0) for d in items) / len(items)
    return result


def setup_dual_axis_plot(figsize=(5, 3)) -> tuple:
    """创建并返回双轴图 (fig, ax1, ax2)。"""
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    return fig, ax1, ax2


def save_and_close_figure(fig, save_path: str, msg: str = ""):
    """保存图形并关闭，统一处理目录创建。"""
    fig.tight_layout()
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if msg:
        print(msg)
    plt.close(fig)


def add_value_labels(ax, bars, fmt: str = '{:.3f}'):
    """在柱状图上方添加数值标签。"""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               fmt.format(height) if height < 10 else f'{height:.1f}',
               ha='center', va='bottom', fontsize=6)


# ==================== 中文标签支持 ====================

PLOT_MAIN_FONTSIZE = 11

# 动机图统一字号。表格、轴标题、图标题使用同一字号。
MOTIV_FONTSIZE_EN = {
    'label': PLOT_MAIN_FONTSIZE,
    'tick': 10,
    'title': PLOT_MAIN_FONTSIZE,
    'legend': 8,
    'annot': 8,
    'annot_bold': 8,
    'table': PLOT_MAIN_FONTSIZE,
    'cyc_ref': PLOT_MAIN_FONTSIZE,
}

MOTIV_FIG_WIDTH = 4.2

MOTIV_CASE1_TABLE_CFG = {
    'groups_per_row': 3,
    'figsize': (4.6, 2.80),
    'left': -0.20,
    'width': 1.43,
    'table_top': -0.62,
    'row_height': 0.17,
    'col_unit': [0.10, 0.13, 0.25],
    'row_scale': 1.0,
    'subplots_adjust': {'left': 0.16, 'right': 0.78, 'bottom': 0.48, 'top': 0.82},
    'font_size': PLOT_MAIN_FONTSIZE,
    'tight_layout': False,
    'savefig': {'bbox_inches': None},
}

MOTIV_CASE2_LAYOUT_CFG = {
    'figsize_by_type': {
        'breakdown': (MOTIV_FIG_WIDTH, 2.65),
        'utilization': (MOTIV_FIG_WIDTH, 2.35),
    },
    'subplots_adjust_by_type': {
        'breakdown': {'left': 0.16, 'right': 0.78, 'bottom': 0.32, 'top': 0.82},
        'utilization': {'left': 0.16, 'right': 0.78, 'bottom': 0.34, 'top': 0.80},
    },
    'xtick_rotation': 20,
    'xtick_ha': 'right',
    'xtick_rotation_mode': 'anchor',
    'marker_size': 6,
    'line_width': 1.8,
    'legend_ncol': 2,
    'tight_layout': False,
    'savefig': {'bbox_inches': None},
}

def configure_zh_fonts():
    """配置 matplotlib 使用中文字体（微软雅黑）。"""
    import matplotlib.font_manager as fm
    # 直接添加 Windows 字体路径
    _zh_fonts = [
        '/mnt/c/Windows/Fonts/msyh.ttc',      # Microsoft YaHei
        '/mnt/c/Windows/Fonts/msyhbd.ttc',    # Microsoft YaHei Bold
        '/mnt/c/Windows/Fonts/simhei.ttf',     # SimHei
    ]
    for fp in _zh_fonts:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                        'Droid Sans Fallback']
    plt.rcParams['axes.unicode_minus'] = False


# --- plot_motiv_case1 标签 ---
MOTIV_CASE1_LABELS_ZH = {
    'idle': '空闲',
    'miss': '违例',
    'realloc': '重分配',
    'miss_rate': '违例率',
    'xlabel': '预留分位数',
    'ylabel1': '算力占比（对数）',
    'ylabel2': '违例率',
    'title': 'Cyc.：利用率-可靠性权衡',
    'cyc_ref': 'cyc',
}

# --- plot_motiv_case2 (breakdown) 标签 ---
MOTIV_CASE2_BREAKDOWN_LABELS_ZH = {
    'execution': '执行',
    'scheduling': '调度',
    'waiting': '等待',
    'miss_rate': '违例率',
    'ylabel1': r'延迟 / $\mathcal{D}_{e2e}$',
    'ylabel2': '违例率',
    'title_breakdown': 'Tp-driven：延迟分解与规模',
    'xlabel_scale': '规模配置',
    'xlabel_bins': '分区数',
    'chains_suffix': 'chains',
}

# --- plot_motiv_case2 (utilization) 标签 ---
MOTIV_CASE2_UTIL_LABELS_ZH = {
    'realloc': '重分配',
    'effective': '有效',
    'idle': '空闲',
    'ops_miss': '违例算力占比',
    'xlabel_scale': '规模配置',
    'xlabel_bins': '分区数',
    'ylabel1': '瓦片利用率',
    'ylabel2': '违例算力占比',
    'title_util': 'Tp-driven：资源利用率与规模',
    'chains_suffix': 'chains',
}

# --- _plot_abla_overhead 标签 ---
ABLA_OVERHEAD_LABELS_ZH = {
    'ylabel1': '重分配次数（柱）',
    'ylabel2': '重分配比率（线）',
}

# --- _plot_abla_tradeoff 标签 ---
ABLA_TRADEOFF_LABELS_ZH = {
    'exec': '执行',
    'realloc': '重分配',
    'wait': '等待',
    'latency_title': '延迟',
    'ylabel1': '归一化延迟',
    'ylabel2': '违例率',
}

# --- ablation title 映射 ---
ABLA_TITLES_ZH = {
    'Ablation-2: Effect of Spatial Partitioning': '空间分区的效果',
    'Ablation-2: Latency Breakdown vs num_bins': '延迟分解与分区数',
    'Ablation-3: Effect of Reservation (bins=8)': '分区下动态预留效果',
    'Ablation-3: Latency Breakdown vs ratioB (bins=8)': '延迟分解与预留分位数（bins=8）',
}

# --- _plot_satisfy_projection 标签 ---
ABLA_SATISFY_LABELS_ZH = {
    'xlabel': '软预留分位数 (exec_t_comp_ratioB)',
    'ylabel': '延迟满足率',
    'title': '满足率投影 (cyc-S vs cyc)',
    'cyc_S': 'cyc-S',
    'cyc_prefix': 'cyc',
}
