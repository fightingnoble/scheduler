#!/usr/bin/env python3
"""
Ablation Experiments Runner

统一的三个消融实验执行脚本，负责：
1. 参数扫描
2. 调用main_approach.py标准流程
3. 收集StatisticsCollector
4. 生成统计报告和图表
5. 缓存collector对象供后续分析

Usage:
    python scripts/abla_exp_runner.py --case 1 --output_dir ./abla_results
    python scripts/abla_exp_runner.py --case 2 --output_dir ./abla_results
    python scripts/abla_exp_runner.py --case 3 --output_dir ./abla_results
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 从公共模块导入共享组件
from scripts.exp_common import (
    _PHYSICAL_CORES,
    mapping_args,
    specific_args,
    runtime_args,
    ParamTemplate,
    run_main_approach_inproc,
)

from approach_collector import StatisticsCollector


# -------------- Ablation plot settings --------

# Unified color scheme (consistent with motiv experiments in approach_collector.py)
ABLA_COLORS = {
    'exec':      'C0',   # blue — latency: computation
    'effective':  'C0',   # blue — capacity: useful work
    'realloc':   'C1',   # orange — scheduling overhead
    'wait':      'C2',   # green — latency: queuing
    'miss_bar':  'C3',   # red — capacity: overdue load
    'miss_line': 'C4',   # purple — miss rate line (secondary axis)
    'idle':      'C7',   # gray — capacity: unused
}

# 3 representative load configurations (all chains averaged)
ABLA_LOAD_CONFIGS = [
    {'tiles': 400, 'load_factor': 0.5, 'label': 'Low (400T-0.5×)'},
    {'tiles': 400, 'load_factor': 1.0, 'label': 'Mid (400T-1.0×)'},
    {'tiles': 200, 'load_factor': 1.0, 'label': 'High (200T-1.0×)'},
]
ABLA_LOAD_COLORS = ['#1f77b4', '#ff7f0e', '#d62728']  # blue, orange, red


def _filter_and_avg_by_load(data_points, group_key, group_val, load_cfg):
    """Filter data by (group_key==group_val, tiles, load_factor), average over chains."""
    pts = [d for d in data_points
           if d[group_key] == group_val
           and d['tiles'] == load_cfg['tiles']
           and d['load_factor'] == load_cfg['load_factor']]
    if not pts:
        return None
    # Access patterns differ: Case2 nests realloc_mean_ratio in 'utilization', Case3 flattens it
    def _get_realloc_ratio(d):
        if 'realloc_mean_ratio' in d and not isinstance(d['realloc_mean_ratio'], dict):
            return d['realloc_mean_ratio']
        return d['utilization']['realloc_mean_ratio']

    def _get_miss_ratio(d):
        if 'miss_mean_ratio' in d and not isinstance(d.get('miss_mean_ratio'), dict):
            return d['miss_mean_ratio']
        return d['utilization']['miss_mean_ratio']

    n = len(pts)
    return {
        'realloc_mean_count': sum(d['realloc_mean_count'] for d in pts) / n,
        'realloc_mean_ratio': sum(_get_realloc_ratio(d) for d in pts) / n,
        'miss_mean_ratio':    sum(_get_miss_ratio(d) for d in pts) / n,
        'miss_mean_count':    sum(d.get('miss_mean_count', 0) for d in pts) / n,
        'latency_breakdown':  _avg_latency_breakdown(pts),
    }


def _avg_latency_breakdown(pts):
    """Average latency breakdown across data points."""
    bds = [d.get('latency_breakdown', {}).get('overall', {}) for d in pts]
    bds = [b for b in bds if b]
    if not bds:
        return {'exec_ratio': 0, 'realloc_ratio': 0, 'wait_ratio': 0}
    n = len(bds)
    return {
        'exec_ratio':    sum(b.get('exec_ratio', 0) for b in bds) / n,
        'realloc_ratio': sum(b.get('realloc_ratio', 0) for b in bds) / n,
        'wait_ratio':    sum(b.get('wait_ratio', 0) for b in bds) / n,
    }


def _plot_abla_overhead(data_points, x_key, x_values, x_labels, title, save_path,
                        load_configs=None, load_colors=None, pglb_baseline=None):
    """
    Universal ablation overhead plot: clustered bars (realloc_count) + lines (realloc_ratio).
    Clusters = load configs, within-cluster X = x_values (bins or ratioB).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if load_configs is None:
        load_configs = ABLA_LOAD_CONFIGS
    if load_colors is None:
        load_colors = ABLA_LOAD_COLORS

    n_clusters = len(load_configs)
    n_bars = len(x_values)
    bar_width = 0.7 / n_bars
    cluster_width = n_bars * bar_width + 0.3

    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    ax2 = ax1.twinx()

    for ci, (lcfg, color) in enumerate(zip(load_configs, load_colors)):
        cluster_center = ci * cluster_width
        for bi, xv in enumerate(x_values):
            avg = _filter_and_avg_by_load(data_points, x_key, xv, lcfg)
            if avg is None:
                continue
            x_pos = cluster_center + (bi - (n_bars - 1) / 2) * bar_width
            ax1.bar(x_pos, avg['realloc_mean_count'], bar_width * 0.85,
                    color=color, alpha=0.25 + 0.15 * bi, edgecolor=color, linewidth=0.5)
            ax1.text(x_pos, avg['realloc_mean_count'], f"{avg['realloc_mean_count']:.1f}",
                     ha='center', va='bottom', fontsize=5.5, color=color)

        # Line: realloc_ratio
        line_xs, line_ys = [], []
        for bi, xv in enumerate(x_values):
            avg = _filter_and_avg_by_load(data_points, x_key, xv, lcfg)
            if avg is None:
                continue
            x_pos = cluster_center + (bi - (n_bars - 1) / 2) * bar_width
            line_xs.append(x_pos)
            line_ys.append(avg['realloc_mean_ratio'])
        if line_xs:
            ax2.plot(line_xs, line_ys, 'o-', color=color, linewidth=1.5, markersize=4,
                     label=lcfg['label'])

    # pglb baseline horizontal lines (Case 3 only)
    if pglb_baseline is not None:
        for ci, (lcfg, color) in enumerate(zip(load_configs, load_colors)):
            if lcfg['label'] in pglb_baseline:
                bv = pglb_baseline[lcfg['label']]
                ax2.axhline(y=bv, color=color, linestyle=':', alpha=0.5, linewidth=1)

    # X-tick labels at cluster centers
    cluster_centers = [ci * cluster_width for ci in range(n_clusters)]
    ax1.set_xticks(cluster_centers)
    ax1.set_xticklabels([lc['label'] for lc in load_configs], fontsize=8)

    # Inner x labels (bins or ratioB) — add minor ticks
    for ci in range(n_clusters):
        cc = ci * cluster_width
        for bi, xl in enumerate(x_labels):
            x_pos = cc + (bi - (n_bars - 1) / 2) * bar_width
            ax1.text(x_pos, -0.02, xl, ha='center', va='top', fontsize=5.5,
                     transform=ax1.get_xaxis_transform(), color='gray')

    ax1.set_ylabel('Realloc Count (bars)', fontsize=9)
    ax2.set_ylabel('Realloc Ratio (lines)', fontsize=9)
    ax1.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax1.set_title(title, fontsize=10, pad=8)
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Overhead 图已保存: {save_path}")
    plt.close(fig)


def _plot_abla_tradeoff(data_points, x_key, x_values, x_labels, title, save_path,
                        load_configs=None, load_colors=None):
    """
    Universal ablation tradeoff plot: stacked bars (latency breakdown) + line (miss rate).
    Clusters = load configs, within-cluster X = x_values.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if load_configs is None:
        load_configs = ABLA_LOAD_CONFIGS
    if load_colors is None:
        load_colors = ABLA_LOAD_COLORS

    n_clusters = len(load_configs)
    n_bars = len(x_values)
    bar_width = 0.7 / n_bars
    cluster_width = n_bars * bar_width + 0.3

    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    ax2 = ax1.twinx()

    comp_colors = {'exec': ABLA_COLORS['exec'], 'realloc': ABLA_COLORS['realloc'], 'wait': ABLA_COLORS['wait']}

    for ci, (lcfg, color) in enumerate(zip(load_configs, load_colors)):
        cluster_center = ci * cluster_width
        miss_xs, miss_ys = [], []

        for bi, xv in enumerate(x_values):
            avg = _filter_and_avg_by_load(data_points, x_key, xv, lcfg)
            if avg is None:
                continue
            x_pos = cluster_center + (bi - (n_bars - 1) / 2) * bar_width
            bd = avg['latency_breakdown']

            # Stacked bars: exec (bottom) → realloc → wait
            bottom = 0
            for comp, ckey in [('exec', 'exec_ratio'), ('realloc', 'realloc_ratio'), ('wait', 'wait_ratio')]:
                val = bd.get(ckey, 0)
                ax1.bar(x_pos, val, bar_width * 0.85, bottom=bottom,
                        color=comp_colors[comp], alpha=0.3 + 0.12 * ci, edgecolor='gray', linewidth=0.3)
                bottom += val

            miss_xs.append(x_pos)
            miss_ys.append(avg['miss_mean_count'])

        if miss_xs:
            ax2.plot(miss_xs, miss_ys, 's-', color=color, linewidth=1.5, markersize=4,
                     label=lcfg['label'])

    # X-tick labels
    cluster_centers = [ci * cluster_width for ci in range(n_clusters)]
    ax1.set_xticks(cluster_centers)
    ax1.set_xticklabels([lc['label'] for lc in load_configs], fontsize=8)

    for ci in range(n_clusters):
        cc = ci * cluster_width
        for bi, xl in enumerate(x_labels):
            x_pos = cc + (bi - (n_bars - 1) / 2) * bar_width
            ax1.text(x_pos, -0.02, xl, ha='center', va='top', fontsize=5.5,
                     transform=ax1.get_xaxis_transform(), color='gray')

    # Legend for stacked components
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=comp_colors[c], alpha=0.5, label=c.capitalize())
                      for c in ['exec', 'realloc', 'wait']]
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=6.5, framealpha=0.7, title='Latency', title_fontsize=7)

    ax1.set_ylabel('Latency / Constraint', fontsize=9)
    ax2.set_ylabel('Miss Rate', fontsize=9)
    ax1.tick_params(axis='y', labelsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax1.set_title(title, fontsize=10, pad=8)
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Tradeoff 图已保存: {save_path}")
    plt.close(fig)


# -------------- Parallel workers -------------

def _case1_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker for Case 1. Returns a result dict; returns None in dry_run."""
    run_args = payload['run_args']
    ratio = payload['ratio']
    label = payload['label']
    exp_type = payload.get('exp_type', 'cyc')
    dry_run = payload.get('dry_run', False)
    
    collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    if dry_run or collector is None:
        return None
    
    stats = collector.get_motiv_case1_stats()
    return {
        'ratio': ratio,
        'label': label,
        'exp_type': exp_type,
        'idle_mean_ratio': stats['idle_mean_ratio'],
        'miss_mean_ratio': stats['miss_mean_ratio'],
        'miss_mean_count': stats['miss_mean_count'],
        'realloc_mean_ratio': stats['realloc_mean_ratio'],
    }


def _case2_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker for Case 2. Returns a result dict; returns None in dry_run."""
    run_args = payload['run_args']
    num_bins = payload['num_bins']
    tiles = payload['tiles']
    chains = payload['chains']
    load_factor = payload['load_factor']
    label = payload['label']
    exp_type = payload.get('exp_type', 'pglb')
    dry_run = payload.get('dry_run', False)
    
    collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    if dry_run or collector is None:
        return None
    
    stats = collector.get_motiv_case2_stats()
    res = {
        'num_bins': num_bins,
        'tiles': tiles,
        'chains': chains,
        'load_factor': load_factor,
        'label': label,
        'exp_type': exp_type,
    }
    res.update(stats)
    
    # 添加 realloc_count
    realloc_info = collector.get_realloc_info()
    res['realloc_mean_count'] = realloc_info['realloc_mean_count']
    
    return res


def _case3_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker for Case 3. Returns a result dict; returns None in dry_run."""
    run_args = payload['run_args']
    tiles = payload['tiles']
    chains = payload['chains']
    load_factor = payload['load_factor']
    num_bins = payload['num_bins']
    ratioB = payload.get('ratioB', -1)
    label = payload['label']
    exp_type = payload.get('exp_type', 'reserv')
    dry_run = payload.get('dry_run', False)
    
    try:
        collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    except (KeyError, AssertionError) as e:
        print(f"  跳过: {label} — 仿真失败 ({type(e).__name__}: {e})")
        return None
    if dry_run or collector is None:
        return None

    stats = collector.get_motiv_case2_stats()
    util = stats['utilization']
    realloc_info = collector.get_realloc_info()

    res = {
        'tiles': tiles,
        'chains': chains,
        'load_factor': load_factor,
        'num_bins': num_bins,
        'exec_t_comp_ratioB': ratioB,
        'label': label,
        'exp_type': exp_type,
        'idle_mean_ratio': util['idle_mean_ratio'],
        'miss_mean_ratio': util['miss_mean_ratio'],
        'miss_mean_count': stats['miss_mean_count'],
        'realloc_mean_ratio': realloc_info['realloc_mean_ratio'],
        'realloc_mean_count': realloc_info['realloc_mean_count'],
    }
    res.update(stats)
    return res


class AblaExp1Runner:
    """Ablation-1: cyc(S) vs cyc - 预留在串行执行下的影响"""

    def __init__(self, base_tpl: ParamTemplate, args):
        self.base_tpl = base_tpl
        self.args = args
        self.output_dir = Path(args.output_dir) / 'case1'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ratioBs = [float(r) for r in args.case1_ratioBs.split(',')]
        self.cyc_ratios = [float(r) for r in args.case1_cyc_ratios.split(',')]
        self.ratioA = args.case1_ratioA
        self.results = []
    
    def run(self):
        """运行Case 1实验"""
        print("\n" + "="*60)
        print("Ablation-1: cyc(S) vs cyc - 预留在串行执行下的影响")
        print("="*60)
        print(f"cyc(S) 扫描 repack 分位数: {self.ratioBs} (固定 ratioA={self.ratioA})")
        print(f"cyc 参考点: {self.cyc_ratios} (硬隔离，用于投影)")
        print(f"输出目录: {self.output_dir}")
        
        if not self.args.use_plot_cache:
            self._run_simulations()
        
        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        """运行Case 1的所有仿真"""
        tasks: List[Dict[str, Any]] = []
        
        # cyc 参考点（硬隔离，不 repack）
        for ratio in self.cyc_ratios:
            tpl = self.base_tpl.with_updates(
                mapping={
                    'test_case': 'cyclic',
                    'exec_t_comp_ratioA': ratio,
                    'exec_t_comp_ratioB': -1,
                    'num_bins': -1,
                },
                runtime={'policy': 'cyc'},
                specific={'root_dir': str(self.output_dir / f'cyc_p{int(ratio*100)}')}
            )
            tasks.append({
                'run_args': tpl.to_run_args(),
                'ratio': ratio,
                'label': f'cyc p{int(ratio*100)}',
                'exp_type': 'cyc',
                'dry_run': self.args.dry_run,
            })
        
        # cyc(S) 扫描 repack 分位数（软预留）
        for ratioB in self.ratioBs:
            tpl = self.base_tpl.with_updates(
                mapping={
                    'test_case': 'cyclic',
                    'exec_t_comp_ratioA': self.ratioA,
                    'exec_t_comp_ratioB': ratioB,
                    'num_bins': -1,
                },
                runtime={'policy': 'reserv'},
                specific={'root_dir': str(self.output_dir / f'cycS_p{int(self.ratioA*100)}_p{int(ratioB*100)}')}
            )
            tasks.append({
                'run_args': tpl.to_run_args(),
                'ratio': ratioB,
                'label': f'cyc(S) p{int(ratioB*100)}',
                'exp_type': 'cyc-S',
                'dry_run': self.args.dry_run,
            })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行 Case 1 任务数={len(tasks)}, max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_case1_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    # 捕获子进程异常（包括 sys.exit 导致的 SystemExit）
                    print(f"  [ERROR] Worker failed: {type(e).__name__}: {e}")
                    continue
                if res is None:
                    continue
                self.results.append(res)
                print(f"  完成: {res['label']:20s} Idle={res['idle_mean_ratio']:.4e} Miss={res['miss_mean_ratio']:.4e} Count={res['miss_mean_count']:.2f}")

    def _generate_report(self):
        """生成报告和图表"""
        print("\n" + "="*30 + " Case 1: Generating Report " + "="*30)
        json_path = self.output_dir / 'case1_summary.json'
        
        plot_data = None
        
        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                summary = json.load(f)
            results = summary.get('results', [])
            # 检查缓存数据点数量是否与预期扫描参数一致
            expected_count = len(self.ratioBs) + len(self.cyc_ratios)
            if len(results) != expected_count:
                print(f"Error: Cache file is invalid. Expected {expected_count} results, found {len(results)}.")
                return
            # 直接使用缓存数据
            plot_data = results
        else:
            if not self.results:
                print("Warning: No simulation results found to generate a report.")
                return
            
            plot_data = self.results
            
            summary = {
                'experiment': 'Ablation-1: cyc(S) vs cyc',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'ratioBs': self.ratioBs,
                    'cyc_ratios': self.cyc_ratios,
                    'ratioA': self.ratioA,
                },
                'results': self.results
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存到: {json_path}")

        if plot_data:
            cyc_ref = [{'ratio': d['ratio'], 'miss_mean_count': d['miss_mean_count']}
                       for d in plot_data if d.get('exp_type') == 'cyc']

            # 使用 Motiv-Exp-1 的绘图方法
            cycS_points = [{
                'ratio': d['ratio'],
                'label': f"p{int(d['ratio']*100)}",
                'idle_mean_ratio': d['idle_mean_ratio'],
                'miss_mean_ratio': d['miss_mean_ratio'],
                'realloc_mean_ratio': d['realloc_mean_ratio'],
                'miss_mean_count': d['miss_mean_count'],
                'cyc_ref': cyc_ref  # 将参考数据传递给绘图函数
            } for d in plot_data if d.get('exp_type') == 'cyc-S']

            if cycS_points:
                cycS_points = sorted(cycS_points, key=lambda x: x['ratio'])
                for p in cycS_points:
                    p.pop('ratio', None)
                plot_path = self.output_dir / 'case1_motiv1_style.pdf'
                StatisticsCollector.plot_motiv_case1(data_points=cycS_points, save_path=str(plot_path))
            
            # 额外的投影图
            self._plot_satisfy_projection(data_points=plot_data)
        
        print(f"\n✓ Case 1 完成！结果保存在: {self.output_dir}")
    
    def _plot_satisfy_projection(self, data_points: List[Dict]):
        """绘制延迟满足率投影图"""
        import numpy as np
        import matplotlib.pyplot as plt

        cyc_data = [d for d in data_points if d['exp_type'] == 'cyc']
        cycS_data = sorted([d for d in data_points if d['exp_type'] == 'cyc-S'], key=lambda d: d['ratio'])

        fig, ax = plt.subplots(figsize=(7.0, 3.2))

        if cycS_data:
            xs = [d['ratio'] for d in cycS_data]
            ys = [max(0.0, min(1.0, 1.0 - d['miss_mean_count'])) for d in cycS_data]
            ax.plot(xs, ys, 'o-', color=ABLA_COLORS['miss_line'], linewidth=1.5, markersize=4, label='cyc-S')

        for d in cyc_data:
            y = max(0.0, min(1.0, 1.0 - d['miss_mean_count']))
            ax.axhline(y=y, color='gray', linestyle=':', alpha=0.6, linewidth=1)
            ax.text(0.905, y, f"cyc p{int(d['ratio']*100)}", transform=ax.get_yaxis_transform(),
                   ha='left', va='center', fontsize=7, color='gray')

        ax.set_xlabel('Soft Reservation Percentile (exec_t_comp_ratioB)', fontsize=9)
        ax.set_ylabel('Latency Satisfaction Rate', fontsize=9)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.7)
        ax.set_title('Ablation-1: Satisfaction Projection (cyc-S vs cyc)', fontsize=10, pad=8)

        fig.tight_layout()
        save_path = str(self.output_dir / 'case1_satisfy_projection.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Satisfy projection 图已保存: {save_path}")
        plt.close(fig)


class AblaExp2Runner:
    """Ablation-2: pglb vs glb - 隔离的作用"""

    def __init__(self, base_tpl: ParamTemplate, args):
        self.base_tpl = base_tpl
        self.args = args
        self.output_dir = Path(args.output_dir) / 'case2'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_bins_list = [int(b) for b in args.case2_bins.split(',')]
        self.tiles = [int(t) for t in args.case2_tiles.split(',')]
        self.chains = [int(c) for c in args.case2_chains.split(',')]
        self.loads = [float(l) for l in args.case2_loads.split(',')]
        self.results = []
    
    def run(self):
        """运行Case 2实验"""
        print("\n" + "="*60)
        print("Ablation-2: pglb vs glb - 隔离的作用")
        print("="*60)
        print(f"扫描 num_bins: {self.num_bins_list}")
        print(f"扫描 tiles: {self.tiles}, chains: {self.chains}, loads: {self.loads}")
        print(f"输出目录: {self.output_dir}")
        
        if not self.args.use_plot_cache:
            self._run_simulations()
        
        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        """运行Case 2的所有仿真"""
        tasks: List[Dict[str, Any]] = []
        
        for tiles in self.tiles:
            for chains in self.chains:
                for load_factor in self.loads:
                    for num_bins in self.num_bins_list:
                        if num_bins == 1:
                            tpl = self.base_tpl.with_updates(
                                mapping={
                                    'test_case': 'dynamic',
                                    'exec_t_comp_ratioA': 0.7,
                                    'exec_t_comp_ratioB': -1,
                                    'num_bins': 1,
                                    'num_cores': tiles,
                                    'aux_scale_factor': chains,
                                    'load_factor': load_factor,
                                },
                                runtime={'policy': 'glb'},
                                specific={'root_dir': str(self.output_dir / f'glb_tiles{tiles}_chains{chains}_load{load_factor:.1f}')}
                            )
                            exp_type = 'glb'
                        else:
                            tpl = self.base_tpl.with_updates(
                                mapping={
                                    'test_case': 'dynamic',
                                    'exec_t_comp_ratioA': 0.7,
                                    'exec_t_comp_ratioB': -1,
                                    'num_bins': num_bins,
                                    'num_cores': tiles,
                                    'aux_scale_factor': chains,
                                    'load_factor': load_factor,
                                },
                                runtime={'policy': 'pglb'},
                                specific={'root_dir': str(self.output_dir / f'pglb_bins{num_bins}_tiles{tiles}_chains{chains}_load{load_factor:.1f}')}
                            )
                            exp_type = 'pglb'
                        
                        tasks.append({
                            'run_args': tpl.to_run_args(),
                            'num_bins': num_bins,
                            'tiles': tiles,
                            'chains': chains,
                            'load_factor': load_factor,
                            'label': f'{exp_type} bins={num_bins} {tiles}T-{chains}C-{load_factor}x',
                            'exp_type': exp_type,
                            'dry_run': self.args.dry_run,
                        })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行 Case 2 任务数={len(tasks)}, max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_case2_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [ERROR] Worker failed: {type(e).__name__}: {e}")
                    continue
                if res is None:
                    continue
                self.results.append(res)
                util = res['utilization']
                print(f"  完成: {res['label']:35s} Idle={util['idle_mean_ratio']:.4f} Miss={util['miss_mean_ratio']:.4f} Realloc={util['realloc_mean_ratio']:.4f}")

    def _generate_report(self):
        """生成报告和图表"""
        print("\n" + "="*30 + " Case 2: Generating Report " + "="*30)
        json_path = self.output_dir / 'case2_summary.json'

        plot_data = None

        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                summary = json.load(f)
            plot_data = summary.get('results', [])
        else:
            if not self.results:
                print("Warning: No simulation results found to generate a report.")
                return

            plot_data = self.results

            summary = {
                'experiment': 'Ablation-2: pglb vs glb',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'num_bins_list': self.num_bins_list,
                    'tiles': self.tiles,
                    'chains': self.chains,
                    'loads': self.loads,
                },
                'results': self.results
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存到: {json_path}")

        if plot_data:
            x_values = sorted(set(d['num_bins'] for d in plot_data))
            x_labels = [str(b) for b in x_values]

            _plot_abla_overhead(
                data_points=plot_data,
                x_key='num_bins', x_values=x_values, x_labels=x_labels,
                title='Ablation-2: Effect of Spatial Partitioning',
                save_path=str(self.output_dir / 'case2_overhead.pdf'),
            )
            _plot_abla_tradeoff(
                data_points=plot_data,
                x_key='num_bins', x_values=x_values, x_labels=x_labels,
                title='Ablation-2: Latency Breakdown vs num_bins',
                save_path=str(self.output_dir / 'case2_tradeoff.pdf'),
            )

        print(f"\n✓ Case 2 完成！结果保存在: {self.output_dir}")
    
class AblaExp3Runner:
    """Ablation-3: reserv vs pglb - 预留在并行下的影响"""

    def __init__(self, base_tpl: ParamTemplate, args):
        self.base_tpl = base_tpl
        self.args = args
        self.output_dir = Path(args.output_dir) / 'case3'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tiles = [int(t) for t in args.case3_tiles.split(',')]
        self.chains = [int(c) for c in args.case3_chains.split(',')]
        self.loads = [float(l) for l in args.case3_loads.split(',')]
        self.ratioBs = [float(r) for r in args.case3_ratioBs.split(',')]
        self.bins = [int(b) if b != '-1' else -1 for b in args.case3_bins.split(',')]
        self.fixed_strength = args.case3_fixed_strength
        self.fixed_tiles = args.case3_fixed_tiles
        self.fixed_chains = args.case3_fixed_chains
        self.fixed_load = args.case3_fixed_load
        if self.fixed_strength:
            if self.fixed_tiles is None or self.fixed_chains is None or self.fixed_load is None:
                raise ValueError(
                    "启用 --case3_fixed_strength 时，必须同时提供 "
                    "--case3_fixed_tiles/--case3_fixed_chains/--case3_fixed_load"
                )
        self.results = []

    def _strength_points(self):
        if self.fixed_strength:
            return [(int(self.fixed_tiles), int(self.fixed_chains), float(self.fixed_load))]
        return [
            (tiles, chains, load_factor)
            for tiles in self.tiles
            for chains in self.chains
            for load_factor in self.loads
        ]
    
    def run(self):
        """运行Case 3实验"""
        print("\n" + "="*60)
        print("Ablation-3: reserv vs pglb - 预留在并行下的影响")
        print("="*60)
        if self.fixed_strength:
            print(f"固定负载强度: tiles={self.fixed_tiles}, chains={self.fixed_chains}, load={self.fixed_load}")
        else:
            print(f"扫描 tiles: {self.tiles}, chains: {self.chains}, loads: {self.loads}")
        print(f"reserv 扫描 ratioB: {self.ratioBs}, bins: {self.bins}")
        print(f"输出目录: {self.output_dir}")
        
        if not self.args.use_plot_cache:
            self._run_simulations()
        
        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        """运行Case 3的所有仿真"""
        tasks: List[Dict[str, Any]] = []
        pglb_bins = sorted({b for b in self.bins if b >= 1})
        strength_points = self._strength_points()
        
        # pglb 基线
        for tiles, chains, load_factor in strength_points:
            for num_bins in pglb_bins:
                tpl = self.base_tpl.with_updates(
                    mapping={
                        'test_case': 'dynamic',
                        'exec_t_comp_ratioA': 0.7,
                        'exec_t_comp_ratioB': -1,
                        'num_bins': num_bins,
                        'num_cores': tiles,
                        'aux_scale_factor': chains,
                        'load_factor': load_factor,
                    },
                    runtime={'policy': 'glb' if num_bins == 1 else 'pglb'},
                    specific={'root_dir': str(self.output_dir / f'pglb_bins{num_bins}_tiles{tiles}_chains{chains}_load{load_factor:.1f}')}
                )
                tasks.append({
                    'run_args': tpl.to_run_args(),
                    'tiles': tiles,
                    'chains': chains,
                    'load_factor': load_factor,
                    'num_bins': num_bins,
                    'ratioB': -1,
                    'label': f'pglb bins={num_bins} {tiles}T-{chains}C-{load_factor}x',
                    'exp_type': 'pglb',
                    'dry_run': self.args.dry_run,
                })
        
        # reserv 扫描
        for ratioB in self.ratioBs:
            for num_bins in self.bins:
                for tiles, chains, load_factor in strength_points:
                    bins_label = 'single' if num_bins == -1 else str(num_bins)
                    tpl = self.base_tpl.with_updates(
                        mapping={
                            'test_case': 'dynamic',
                            'exec_t_comp_ratioA': 0.7,
                            'exec_t_comp_ratioB': ratioB,
                            'num_bins': num_bins,
                            'num_cores': tiles,
                            'aux_scale_factor': chains,
                            'load_factor': load_factor,
                        },
                        runtime={'policy': 'reserv'},
                        specific={'root_dir': str(self.output_dir / f'reserv_p{int(ratioB*100)}_bins{bins_label}_tiles{tiles}_chains{chains}_load{load_factor:.1f}')}
                    )
                    tasks.append({
                        'run_args': tpl.to_run_args(),
                        'tiles': tiles,
                        'chains': chains,
                        'load_factor': load_factor,
                        'num_bins': num_bins,
                        'ratioB': ratioB,
                        'label': f'reserv p{int(ratioB*100)} bins={bins_label} {tiles}T-{chains}C-{load_factor}x',
                        'exp_type': 'reserv',
                        'dry_run': self.args.dry_run,
                    })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行 Case 3 任务数={len(tasks)}, max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_case3_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [ERROR] Worker failed: {type(e).__name__}: {e}")
                    continue
                if res is None:
                    continue
                self.results.append(res)
                print(f"  完成: {res['label']:45s} ReallocCount={res['realloc_mean_count']:.2f} "
                      f"ReallocRatio={res['realloc_mean_ratio']:.4f} TimeoutRate={res['miss_mean_count']:.4f}")

    def _generate_report(self):
        """生成报告和图表"""
        print("\n" + "="*30 + " Case 3: Generating Report " + "="*30)
        json_path = self.output_dir / 'case3_summary.json'

        plot_data = None
        strength_count = len(self._strength_points())

        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                summary = json.load(f)
            plot_data = summary.get('results', [])
            if not plot_data:
                print("Warning: Cache file has no results.")
                return

        else:
            if not self.results:
                print("Warning: No simulation results found to generate a report.")
                return

            plot_data = self.results

            summary = {
                'experiment': 'Ablation-3: reserv vs pglb',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'tiles': self.tiles,
                    'chains': self.chains,
                    'loads': self.loads,
                    'ratioBs': self.ratioBs,
                    'bins': self.bins,
                    'fixed_strength': self.fixed_strength,
                    'fixed_tiles': self.fixed_tiles,
                    'fixed_chains': self.fixed_chains,
                    'fixed_load': self.fixed_load,
                },
                'results': self.results
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存到: {json_path}")

        if plot_data:
            # Filter reserv data only, fix bins=8 for the main plot
            reserv_data = [d for d in plot_data if d.get('exp_type') == 'reserv']
            # Pick the largest available num_bins as default focus
            available_bins = sorted(set(d['num_bins'] for d in reserv_data))
            focus_bins = max(available_bins) if available_bins else 8
            reserv_focus = [d for d in reserv_data if d['num_bins'] == focus_bins]

            x_values = sorted(set(d['exec_t_comp_ratioB'] for d in reserv_focus))
            x_labels = [f"p{int(x*100)}" for x in x_values]

            # Compute pglb baseline per load config for reference lines
            pglb_data = [d for d in plot_data if d.get('exp_type') == 'pglb'
                         and d['num_bins'] == focus_bins]
            pglb_baseline = {}
            for lcfg in ABLA_LOAD_CONFIGS:
                pts = [d for d in pglb_data
                       if d['tiles'] == lcfg['tiles']
                       and d['load_factor'] == lcfg['load_factor']]
                if pts:
                    pglb_baseline[lcfg['label']] = (
                        sum(d['realloc_mean_ratio'] for d in pts) / len(pts)
                    )

            _plot_abla_overhead(
                data_points=reserv_focus,
                x_key='exec_t_comp_ratioB', x_values=x_values, x_labels=x_labels,
                title=f'Ablation-3: Effect of Reservation (bins={focus_bins})',
                save_path=str(self.output_dir / 'case3_overhead.pdf'),
                pglb_baseline=pglb_baseline,
            )
            _plot_abla_tradeoff(
                data_points=reserv_focus,
                x_key='exec_t_comp_ratioB', x_values=x_values, x_labels=x_labels,
                title=f'Ablation-3: Latency Breakdown vs ratioB (bins={focus_bins})',
                save_path=str(self.output_dir / 'case3_tradeoff.pdf'),
            )

        print(f"\n✓ Case 3 完成！结果保存在: {self.output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Ablation Experiments Runner')
    
    # 实验选择
    parser.add_argument('--case', type=int, required=True, choices=[1, 2, 3],
                       help='实验编号: 1=cyc-S vs cyc, 2=pglb vs glb, 3=reserv vs pglb')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./abla_exp_results',
                       help='输出目录（默认: ./abla_exp_results）')
    
    # Case 1 特定参数
    parser.add_argument('--case1_ratioBs', type=str, default='0.5,0.6,0.7,0.8,0.9,0.99',
                       help='Case 1: cyc(S)的exec_t_comp_ratioB扫描值')
    parser.add_argument('--case1_ratioA', type=float, default=0.7,
                       help='Case 1: cyc(S)的exec_t_comp_ratioA固定值')
    parser.add_argument('--case1_cyc_ratios', type=str, default='0.5,0.6,0.7,0.8,0.9,0.99',
                       help='Case 1: cyc参考点的exec_t_comp_ratioA值')
    
    # Case 2 特定参数
    parser.add_argument('--case2_bins', type=str, default='1,2,4,8',
                       help='Case 2: num_bins扫描值')
    parser.add_argument('--case2_tiles', type=str, default='200,400',
                       help='Case 2: 硬件tile数扫描值')
    parser.add_argument('--case2_chains', type=str, default='1,4',
                       help='Case 2: 任务链数量扫描值')
    parser.add_argument('--case2_loads', type=str, default='0.5,1.0',
                       help='Case 2: 负载倍数扫描值')
    
    # Case 3 特定参数
    parser.add_argument('--case3_tiles', type=str, default='200,400',
                       help='Case 3: 硬件tile数扫描值')
    parser.add_argument('--case3_chains', type=str, default='1,4',
                       help='Case 3: 任务链数量扫描值')
    parser.add_argument('--case3_loads', type=str, default='0.5,1.0',
                       help='Case 3: 负载倍数扫描值')
    parser.add_argument('--case3_ratioBs', type=str, default='0.5,0.6,0.7,0.8,0.9,0.99',
                       help='Case 3: reserv的exec_t_comp_ratioB扫描值')
    parser.add_argument('--case3_bins', type=str, default='1,2,4,8',
                       help='Case 3: reserv/num_bins扫描值（-1表示单分区）')
    parser.add_argument('--case3_fixed_strength', action='store_true',
                       help='Case 3: 使用固定负载强度组合（而非多组扫描）')
    parser.add_argument('--case3_fixed_tiles', type=int, default=None,
                       help='Case 3 固定模式: tiles')
    parser.add_argument('--case3_fixed_chains', type=int, default=None,
                       help='Case 3 固定模式: chains')
    parser.add_argument('--case3_fixed_load', type=float, default=None,
                       help='Case 3 固定模式: load_factor')
    
    # 通用仿真参数
    parser.add_argument('--num_hp', type=int, default=100,
                       help='仿真超周期数（默认: 100）')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('--dry_run', action='store_true',
                       help='只打印命令，不执行')
    parser.add_argument('--use_plot_cache', action='store_true',
                       help='跳过仿真，直接从缓存的JSON结果生成图表')
    
    # 传递给main_approach.py的额外参数
    parser.add_argument('--extra_args', type=str, default='',
                       help='传递给main_approach.py的额外参数')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 构造基础参数模板
    base_mapping = {**mapping_args}
    base_runtime = {**runtime_args, 'n_p': args.num_hp}
    base_specific = {**specific_args, 'verbose': args.verbose}
    
    # 添加额外参数
    if args.extra_args:
        for arg in args.extra_args.split():
            if '=' in arg:
                key, val = arg.split('=', 1)
                base_specific[key.lstrip('-')] = val
    
    base_tpl = ParamTemplate(base_mapping, base_runtime, base_specific)
    
    print("\n" + "="*60)
    print("Ablation Experiments Runner")
    print("="*60)
    print(f"实验: Case {args.case}")
    print(f"输出目录: {args.output_dir}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - 只打印命令，不执行")
    print("="*60)
    
    # 根据case选择运行器
    if args.case == 1:
        runner = AblaExp1Runner(base_tpl, args)
    elif args.case == 2:
        runner = AblaExp2Runner(base_tpl, args)
    elif args.case == 3:
        runner = AblaExp3Runner(base_tpl, args)
    else:
        raise ValueError(f"Invalid case: {args.case}")
    
    # 运行实验
    runner.run()
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()
