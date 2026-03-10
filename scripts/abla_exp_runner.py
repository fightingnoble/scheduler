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
    
    collector = run_main_approach_inproc(run_args, dry_run=dry_run)
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
        'miss_mean_count': util['miss_mean_count'],
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
            # 使用 Motiv-Exp-1 的绘图方法
            cycS_points = [{
                'label': f"p{int(d['ratio']*100)}",
                'idle_mean_ratio': d['idle_mean_ratio'],
                'miss_mean_ratio': d['miss_mean_ratio'],
                'realloc_mean_ratio': d['realloc_mean_ratio'],
                'miss_mean_count': d['miss_mean_count'],
            } for d in plot_data if d.get('exp_type') == 'cyc-S']
            
            if cycS_points:
                cycS_points = sorted(cycS_points, key=lambda x: x['ratio'] if 'ratio' in x else 0)
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
        
        fig, ax = plt.subplots(figsize=(5, 3))
        
        if cycS_data:
            xs = [d['ratio'] for d in cycS_data]
            ys = [max(0.0, min(1.0, 1.0 - d['miss_mean_count'])) for d in cycS_data]
            ax.plot(xs, ys, 'o-', color='C0', linewidth=1.5, markersize=4, label='cyc-S')
        
        for d in cyc_data:
            y = max(0.0, min(1.0, 1.0 - d['miss_mean_count']))
            ax.axhline(y=y, color='gray', linestyle=':', alpha=0.6, linewidth=1)
            ax.text(0.905, y, f"cyc p{int(d['ratio']*100)}", transform=ax.get_yaxis_transform(),
                   ha='left', va='center', fontsize=7, color='gray')
        
        ax.set_xlabel('Soft Reservation Percentile (exec_t_comp_ratioB)', fontsize=9)
        ax.set_ylabel('Latency Satisfaction Rate', fontsize=9)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.7)
        ax.set_title('Ablation-1: Satisfaction Projection (cyc-S vs cyc)', fontsize=9, pad=8)
        
        fig.tight_layout()
        save_path = str(self.output_dir / 'case1_satisfy_projection.pdf')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Case 1 延迟满足率投影图已保存到: {save_path}")
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
        group_order = [
            [(t, l) for t in self.tiles for l in self.loads],
            self.chains
        ]

        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                summary = json.load(f)
            plot_data = summary.get('results', [])
            # 检查缓存数据点数量是否与预期一致
            expected_count = len(self.num_bins_list) * len(self.tiles) * len(self.chains) * len(self.loads)
            if len(plot_data) != expected_count:
                print(f"Error: Cache file is invalid. Expected {expected_count} results, found {len(plot_data)}.")
                return
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
            # 生成消融特有图
            self._plot_switching_overhead(data_points=plot_data)
            
            # 复用 Motiv-Exp-2 的绘图函数
            StatisticsCollector.plot_motiv_case2(
                data_points=plot_data,
                plot_type='breakdown',
                group_order=group_order,
                save_path=str(self.output_dir / 'case2_breakdown.pdf')
            )
            StatisticsCollector.plot_motiv_case2(
                data_points=plot_data,
                plot_type='utilization',
                group_order=group_order,
                save_path=str(self.output_dir / 'case2_utilization.pdf')
            )
        
        print(f"\n✓ Case 2 完成！结果保存在: {self.output_dir}")
    
    def _plot_switching_overhead(self, data_points: List[Dict]):
        """绘制切换次数/开销对比图"""
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        
        by_bins = defaultdict(list)
        for d in data_points:
            by_bins[d['num_bins']].append(d)
        
        bins = sorted(by_bins.keys())
        avg_counts = []
        avg_ratios = []
        labels = []
        
        for b in bins:
            datas = by_bins[b]
            avg_counts.append(sum(d['realloc_mean_count'] for d in datas) / len(datas))
            avg_ratios.append(sum(d['utilization']['realloc_mean_ratio'] for d in datas) / len(datas))
            labels.append(f"{datas[0]['exp_type']} bins={b}")
        
        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax2 = ax1.twinx()
        
        x_pos = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, avg_counts, width, label='Realloc Count', color='C1', alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, avg_ratios, width, label='Realloc Ratio', color='C2', alpha=0.8)
        
        ax1.set_xlabel('Configuration', fontsize=9)
        ax1.set_ylabel('Realloc Count', fontsize=9, color='C1')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
        ax1.tick_params(axis='y', labelcolor='C1', labelsize=8)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        ax2.set_ylabel('Realloc Ratio', fontsize=9, color='C2')
        ax2.tick_params(axis='y', labelcolor='C2', labelsize=8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}' if height < 10 else f'{height:.1f}',
                       ha='center', va='bottom', fontsize=6)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7, framealpha=0.7)
        ax1.set_title('Ablation-2: Effect of Spatial Partitioning on Switching Overhead', fontsize=9, pad=8)
        
        fig.tight_layout()
        save_path = str(self.output_dir / 'case2_switching_overhead.pdf')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"切换开销对比图已保存到: {save_path}")
        plt.close(fig)


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
        group_order = [
            [(t, l) for t in self.tiles for l in self.loads],
            self.chains
        ]
        strength_count = len(self._strength_points())
        
        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                summary = json.load(f)
            plot_data = summary.get('results', [])
            # 检查缓存数据点数量是否与预期一致
            # pglb 基线: len(bins) * len(tiles) * len(chains) * len(loads)
            # reserv 扫描: len(ratioBs) * len(bins) * len(tiles) * len(chains) * len(loads)
            pglb_count = len([b for b in self.bins if b >= 1]) * strength_count
            reserv_count = len(self.ratioBs) * len(self.bins) * strength_count
            expected_count = pglb_count + reserv_count
            if len(plot_data) != expected_count:
                print(f"Error: Cache file is invalid. Expected {expected_count} results, found {len(plot_data)}.")
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
            self._plot_switching_metrics(data_points=plot_data)
            StatisticsCollector.plot_motiv_case2(
                data_points=plot_data,
                plot_type='breakdown',
                group_order=group_order,
                save_path=str(self.output_dir / 'case3_breakdown.pdf')
            )
            StatisticsCollector.plot_motiv_case2(
                data_points=plot_data,
                plot_type='utilization',
                group_order=group_order,
                save_path=str(self.output_dir / 'case3_utilization.pdf')
            )
        
        print(f"\n✓ Case 3 完成！结果保存在: {self.output_dir}")
    
    def _plot_switching_metrics(self, data_points: List[Dict]):
        """绘制切换指标图"""
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        
        by_bins = defaultdict(list)
        for d in data_points:
            if d.get('exp_type') == 'reserv':
                by_bins[d['num_bins']].append(d)
        
        pglb_by_bins = defaultdict(list)
        for d in data_points:
            if d.get('exp_type') == 'pglb':
                pglb_by_bins[d['num_bins']].append(d)
        
        pglb_baseline = {}
        for b, group in pglb_by_bins.items():
            pglb_baseline[b] = {
                'realloc_mean_ratio': sum(g['realloc_mean_ratio'] for g in group) / len(group),
                'realloc_mean_count': sum(g['realloc_mean_count'] for g in group) / len(group),
            }
        
        for num_bins in sorted(by_bins.keys(), key=lambda x: (x == -1, x)):
            pts = by_bins[num_bins]
            by_ratio = defaultdict(list)
            for d in pts:
                by_ratio[d['exec_t_comp_ratioB']].append(d)
            
            xs = sorted(by_ratio.keys())
            avg_ratio = [sum(g['realloc_mean_ratio'] for g in by_ratio[x]) / len(by_ratio[x]) for x in xs]
            avg_count = [sum(g['realloc_mean_count'] for g in by_ratio[x]) / len(by_ratio[x]) for x in xs]
            
            fig, ax1 = plt.subplots(figsize=(5.4, 3.0))
            ax2 = ax1.twinx()
            
            x_pos = np.arange(len(xs))
            ax1.plot(x_pos, avg_count, 'o-', color='C1', linewidth=1.5, markersize=4, label='Realloc Count')
            ax2.plot(x_pos, avg_ratio, 's--', color='C2', linewidth=1.5, markersize=4, label='Realloc Ratio')
            
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([f"p{int(x*100)}" for x in xs], fontsize=8)
            ax1.set_xlabel('exec_t_comp_ratioB', fontsize=9)
            ax1.set_ylabel('Realloc Count', fontsize=9, color='C1')
            ax2.set_ylabel('Realloc Ratio', fontsize=9, color='C2')
            ax1.tick_params(axis='y', labelcolor='C1', labelsize=8)
            ax2.tick_params(axis='y', labelcolor='C2', labelsize=8)
            ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            bins_label = 'single' if num_bins == -1 else str(num_bins)
            ax1.set_title(f'Ablation-3: reserv switching (bins={bins_label})', fontsize=9, pad=8)
            
            baseline = pglb_baseline.get(num_bins)
            if baseline is not None:
                ax1.axhline(y=baseline['realloc_mean_count'], color='gray', linestyle=':', alpha=0.6, linewidth=1)
                ax2.axhline(y=baseline['realloc_mean_ratio'], color='gray', linestyle=':', alpha=0.6, linewidth=1)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7, framealpha=0.7)
            
            fig.tight_layout()
            save_path = str(self.output_dir / f'case3_switching_bins{bins_label}.pdf')
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Case 3 切换指标图已保存到: {save_path}")
            plt.close(fig)


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
