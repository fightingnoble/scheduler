#!/usr/bin/env python3
"""
Motivation Experiments Runner

统一的三个motivation实验执行脚本，负责：
1. 参数扫描
2. 调用main_approach.py标准流程
3. 收集StatisticsCollector
4. 生成统计报告和图表
5. 缓存collector对象供后续分析

Usage:
    python scripts/motiv_exp_runner.py --case 1 --output_dir ./motiv_results
    python scripts/motiv_exp_runner.py --case 2 --output_dir ./motiv_results
    python scripts/motiv_exp_runner.py --case 3 --output_dir ./motiv_results --mode binned
"""

import argparse
import os
import sys
import json
import copy
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
# 并行与物理核心数
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import psutil
    _PHYSICAL_CORES = psutil.cpu_count(logical=False) or os.cpu_count()
except Exception:
    _PHYSICAL_CORES = os.cpu_count()

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from approach_collector import StatisticsCollector

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

# to be specified by the user
specific_args = {
    'root_dir': None,
    'verbose': None,
    'stat_param': None,
}

runtime_args = {
    'n_p': None,
    'policy': None,
}

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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Motivation Experiments Runner')
    
    # 实验选择
    parser.add_argument('--case', type=int, required=True, choices=[1, 2, 3],
                       help='实验编号: 1=静态利用率问题, 2=动态可扩展性问题, 3=切换不确定性')
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='./motiv_exp_results',
                       help='输出目录（默认: ./motiv_exp_results）')
    parser.add_argument('--cache_collectors', action='store_true', default=True,
                       help='缓存StatisticsCollector对象（默认: True）')
    
    # Case 1 特定参数
    parser.add_argument('--case1_ratios', type=str, default='0.5,0.6,0.7,0.8,0.9,0.99',
                       help='Case 1: exec_t_comp_ratioA扫描值（逗号分隔）')
    
    # Case 2 特定参数
    parser.add_argument('--case2_tiles', type=str, default='400,400,200,200',
                       help='Case 2: 硬件tile数扫描值（逗号分隔，与--case2_loads等长且一一对应）')
    parser.add_argument('--case2_loads', type=str, default='0.5,1.0,0.5,1.0',
                       help='Case 2: 负载倍数扫描值（逗号分隔，与--case2_tiles等长且一一对应）')
    parser.add_argument('--case2_chains', type=str, default='1,4,9',
                       help='Case 2: 任务链数量扫描值（逗号分隔）')
    
    # Case 3 特定参数
    parser.add_argument('--case3_mode', type=str, default='raw', choices=['raw', 'binned'],
                       help='Case 3: 数据收集模式（raw或binned）')
    parser.add_argument('--case3_baseline', action='store_true',
                       help='Case 3: 运行基线组（禁用切换开销）')
    parser.add_argument('--case3_experiment', action='store_true',
                       help='Case 3: 运行实验组（启用切换开销）')
    parser.add_argument('--case3_num_periods', type=int, default=1000,
                       help='Case 3: 仿真周期数')
    
    # 通用仿真参数
    parser.add_argument('--num_hp', type=int, default=100,
                       help='仿真超周期数（默认: 100）')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('--dry_run', action='store_true',
                       help='只打印命令，不执行')
                       
    parser.add_argument('--use_plot_cache', action='store_true',
                          help='跳过仿真，直接从缓存的JSON结果生成图表')

    parser.add_argument('--base_ratioA', type=float, default=0.99,
                       help='Case 2/3: 基准 exec_t_comp_ratioA（默认: 0.9）')
    parser.add_argument('--base_ratioB', type=float, default=-1,
                       help='Case 2/3: 基准 exec_t_comp_ratioB（默认: -1）')
        
    # 传递给main_approach.py的额外参数
    parser.add_argument('--extra_args', type=str, default='',
                       help='传递给main_approach.py的额外参数（空格分隔）')
    
    return parser.parse_args()


def run_main_approach_inproc(args_dict: Dict[str, Any], dry_run: bool = False):
    """以函数方式调用 main_approach.main()，避免子进程与磁盘往返。

    构造 sys.argv 供 utils.input_parser() 使用，返回 main_approach.main() 的 StatisticsCollector。
    """
    # 固定附加参数（与sum.md一致）
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


# -------------- Parallel workers --------------

def _case1_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker for Case 1. Returns a result dict; returns None in dry_run."""
    run_args = payload['run_args']
    ratio = payload['ratio']
    label = payload['label']
    dry_run = payload.get('dry_run', False)
    collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    if dry_run or collector is None:
        return None
    stats = collector.get_motiv_case1_stats()
    return {
        'ratio': ratio,
        'label': label,
        'idle_mean_ratio': stats['idle_mean_ratio'],
        'miss_mean_ratio': stats['miss_mean_ratio'],
        'miss_mean_count': stats['miss_mean_count'],
        'realloc_mean_ratio': stats['realloc_mean_ratio'],
    }


def _case2_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker for Case 2. Returns a result dict; returns None in dry_run."""
    run_args = payload['run_args']
    tiles = payload['tiles']
    chains = payload['chains']
    load_factor = payload['load_factor']
    label = payload['label']
    dry_run = payload.get('dry_run', False)
    collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    if dry_run or collector is None:
        return None
    stats = collector.get_motiv_case2_stats()
    res = {
        'tiles': tiles,
        'chains': chains,
        'load_factor': load_factor,
        'label': label,
    }
    res.update(stats)
    return res


def _case3_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker for Case 3. Returns {'exp_name':..., 'stats':...}; returns None in dry_run."""
    run_args = payload['run_args']
    exp_name = payload['exp_name']
    dry_run = payload.get('dry_run', False)
    collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    if dry_run or collector is None:
        return None
    stats = collector.get_motiv_case3_stats(percentile=0.99)
    util = collector.get_utilization_avg_ratio()
    stats['realloc_mean_ratio'] = util['realloc_mean_ratio']

    # Calculate RMSE if raw data is available
    if stats.get('mode') == 'raw' and stats.get('raw_data'):
        stats['rmse'] = StatisticsCollector._calculate_rmse_from_trend(stats['raw_data'])
    else:
        stats['rmse'] = float('nan')

    return {'exp_name': exp_name, 'stats': stats}


class MotivExp1Runner:
    """Motiv-Exp-1: 纯静态调度 - 利用率问题"""

    def __init__(self, base_tpl: ParamTemplate, args):
        self.base_tpl = base_tpl
        self.args = args
        self.output_dir = Path(args.output_dir) / 'case1'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ratios = [float(r) for r in args.case1_ratios.split(',')]
        self.results = []
    
    def run(self):
        """运行Case 1实验：扫描不同预留分位数"""
        print("\n" + "="*60)
        print("Motiv-Exp-1: 纯静态调度 - 利用率问题")
        print("="*60)
        print(f"扫描预留分位数: {self.ratios}")
        print(f"输出目录: {self.output_dir}")
        
        if not self.args.use_plot_cache:
            self._run_simulations()
        
        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        """运行Case 1的所有仿真"""
        # 构造任务列表
        tasks: List[Dict[str, Any]] = []
        for ratio in self.ratios:
            tpl = self.base_tpl.with_updates(
                mapping={
                    'test_case': 'cyclic',
                    'exec_t_comp_ratioA': ratio,
                    'exec_t_comp_ratioB': -1,
                    'num_bins': -1,
                },
                runtime={
                    'policy': 'cyc',
                },
                specific={
                    'root_dir': str(self.output_dir / f'ratio_{ratio:.2f}')
                }
            )
            run_args = tpl.to_run_args()
            tasks.append({
                'run_args': run_args,
                'ratio': ratio,
                'label': f'p{int(ratio*100)}',
                'dry_run': self.args.dry_run,
            })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行 Case 1 任务数={len(tasks)}, max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_case1_worker, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                # 结果结构保持不变
                self.results.append(res)
                print(f"  完成: {res['label']}  Idle={res['idle_mean_ratio']:.4e} Miss={res['miss_mean_ratio']:.4e} Count={res['miss_mean_count']:.2f}")

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
            if len(results) != len(self.ratios):
                print(f"Error: Cache file is invalid. Expected {len(self.ratios)} results, found {len(results)}.")
                return
            
            # 从 'results' 动态重建绘图数据
            plot_data = [{
                'label': f'p{int(r["ratio"]*100)}',
                'idle_mean_ratio': r['idle_mean_ratio'],
                'miss_mean_ratio': r['miss_mean_ratio'],
                'realloc_mean_ratio': r['realloc_mean_ratio'],
                'miss_mean_count': r['miss_mean_count'],
            } for r in results]
        
        else: # 不使用缓存或缓存文件不存在
            if not self.results:
                print("Warning: No simulation results found to generate a report.")
                return

            # 从实时仿真结果准备绘图数据
            plot_data = [{
                'idle_mean_ratio': r['idle_mean_ratio'],
                'miss_mean_ratio': r['miss_mean_ratio'],
                'realloc_mean_ratio': r['realloc_mean_ratio'],
                'miss_mean_count': r['miss_mean_count'],
                'label': r['label']
            } for r in self.results]
            
            # 保存不含 plot_data 的JSON摘要
            summary = {
                'experiment': 'Case 1: Utilization-Reliability Tradeoff',
                'timestamp': datetime.now().isoformat(),
                'parameters': {'ratios': self.ratios},
                'results': [{
                    'ratio': r['ratio'],
                    'idle_mean_ratio': r['idle_mean_ratio'],
                    'miss_mean_ratio': r['miss_mean_ratio'],
                    'realloc_mean_ratio': r['realloc_mean_ratio'],
                    'miss_mean_count': r['miss_mean_count'],
                } for r in self.results]
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存到: {json_path}")

        if plot_data:
            # 使用静态方法绘图
            plot_path = self.output_dir / 'case1_tradeoff.pdf'
            StatisticsCollector.plot_motiv_case1(data_points=plot_data, save_path=str(plot_path))
        
        print(f"\n✓ Case 1 完成！结果保存在: {self.output_dir}")


class MotivExp2Runner:
    """Motiv-Exp-2: 纯动态调度 - 可扩展性问题"""

    def __init__(self, base_tpl: ParamTemplate, args):
        self.base_tpl = base_tpl
        self.args = args
        self.output_dir = Path(args.output_dir) / 'case2'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tiles = [int(t) for t in args.case2_tiles.split(',')]
        self.loads = [float(l) for l in args.case2_loads.split(',')]
        self.chains = [int(c) for c in args.case2_chains.split(',')]
        
        # 验证 tiles 和 loads 列表等长
        if len(self.tiles) != len(self.loads):
            raise ValueError(f"tiles 和 loads 列表长度必须相等：tiles有{len(self.tiles)}个，loads有{len(self.loads)}个")
        
        self.results = []
    
    def run(self):
        """运行Case 2实验：扫描硬件和任务规模"""
        print("\n" + "="*60)
        print("Motiv-Exp-2: 纯动态调度 - 可扩展性问题")
        print("="*60)
        print(f"扫描 (tiles, load) 配置对: {list(zip(self.tiles, self.loads))}")
        print(f"扫描任务链数: {self.chains}")
        print(f"输出目录: {self.output_dir}")
        
        if not self.args.use_plot_cache:
            self._run_simulations()

        if not self.args.dry_run:
            self._generate_report()
            
    def _run_simulations(self):
        """运行Case 2的所有仿真"""
        # 构造任务列表
        tasks: List[Dict[str, Any]] = []
        for tiles, load_factor in zip(self.tiles, self.loads):
            for chains in self.chains:
                tpl = self.base_tpl.with_updates(
                    mapping={
                        'test_case': 'dynamic',
                        'num_cores': tiles,
                        'aux_scale_factor': chains,
                        'load_factor': load_factor,
                        'num_bins': 1,
                    },
                    runtime={
                        'policy': 'glb',
                    },
                    specific={
                        'root_dir': str(self.output_dir / f'tiles_{tiles}_chains_{chains}_load_{load_factor:.1f}')
                    }
                )
                run_args = tpl.to_run_args()
                tasks.append({
                    'run_args': run_args,
                    'tiles': tiles,
                    'chains': chains,
                    'load_factor': load_factor,
                    'label': f'{tiles}T-{chains}C-{load_factor}×',
                    'dry_run': self.args.dry_run,
                })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行 Case 2 任务数={len(tasks)}, max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_case2_worker, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                self.results.append(res)
                util = res['utilization']
                overall = res['latency_breakdown']['overall']
                print(f"  完成: {res['label']}  Util: idle={util['idle_mean_ratio']:.4f}, realloc={util['realloc_mean_ratio']:.4f}, miss={util['miss_mean_ratio']:.4f}")
                print(f"           Breakdown: exec={overall.get('exec_ratio', 0):.4f}, realloc={overall.get('realloc_ratio', 0):.4f}, wait={overall.get('wait_ratio', 0):.4f}")

    def _generate_report(self):
        """生成报告和图表"""
        print("\n" + "="*30 + " Case 2: Generating Report " + "="*30)
        json_path = self.output_dir / 'case2_summary.json'
        
        plot_data = None
        group_order = [list(zip(self.tiles, self.loads)), self.chains]

        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                summary = json.load(f)
            # Case 2的 'results' 结构与绘图数据一致
            plot_data = summary.get('results', [])
            if not plot_data:
                print("Error: Cache file is invalid or contains no results.")
                return
        else:
            if not self.results:
                print("Warning: No simulation results found to generate a report.")
                return
            
            plot_data = self.results
            
            # 保存不含 plot_data 的JSON摘要
            summary = {
                'experiment': 'Case 2: Scalability Bottleneck',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'tiles': self.tiles,
                    'chains': self.chains,
                    'loads': self.loads
                },
                'results': self.results # 仅保存核心结果
            }
            # results中的collector对象无法被JSON序列化，需移除
            for r in summary['results']:
                r.pop('collector', None)
            
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存到: {json_path}")

        if plot_data:
            # 使用静态方法绘图
            plot_path_breakdown = self.output_dir / 'case2_breakdown.pdf'
            StatisticsCollector.plot_motiv_case2(
                data_points=plot_data,
                plot_type='breakdown',
                group_order=group_order,
                save_path=str(plot_path_breakdown)
            )
            
            plot_path_util = self.output_dir / 'case2_utilization.pdf'
            StatisticsCollector.plot_motiv_case2(
                data_points=plot_data,
                plot_type='utilization',
                group_order=group_order,
                save_path=str(plot_path_util)
            )
                
        print(f"\n✓ Case 2 完成！结果保存在: {self.output_dir}")


class MotivExp3Runner:
    """Motiv-Exp-3: 切换行为的不确定性"""

    def __init__(self, base_tpl: ParamTemplate, args):
        self.base_tpl = base_tpl
        self.args = args
        self.output_dir = Path(args.output_dir) / 'case3'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = args.case3_mode
        self.num_periods = args.case3_num_periods
        self.results = {}
    
    def run(self):
        """运行Case 3实验：对照实验（禁用/启用切换开销）"""
        print("\n" + "="*60)
        print("Motiv-Exp-3: 切换行为的不确定性")
        print("="*60)
        print(f"数据模式: {self.mode}")
        print(f"仿真周期数: {self.num_periods}")
        print(f"输出目录: {self.output_dir}")
        
        if not self.args.use_plot_cache:
            self._run_all_experiments()

        if not self.args.dry_run:
            self._generate_report()

    def _run_all_experiments(self):
        """运行Case 3的所有仿真（基线和/或实验组）"""
        # 运行基线组（禁用切换开销）
        exp_list = [] # list of (exp_name, barrier_dis)
        if self.args.case3_baseline or (not self.args.case3_experiment):
            exp_list.append(('baseline',True))
        
        # 运行实验组（启用切换开销）
        if self.args.case3_experiment or (not self.args.case3_baseline):
            exp_list.append(('experiment',False))
        
        # 构造任务
        tasks: List[Dict[str, Any]] = []
        for exp_name, barrier_dis in exp_list:
            exp_mode = 'raw' if exp_name == 'baseline' else self.mode
            tpl = self.base_tpl.with_updates(
                mapping={
                    'test_case': 'dynamic',
                },
                runtime={
                    'policy': 'glb',
                    'n_p': self.num_periods,
                    'num_bins': 1,
                },
                specific={
                    'root_dir': str(self.output_dir / exp_name),
                    'barrier_dis': barrier_dis,
                    'stat_param': f"{{'motiv3_mode': '{exp_mode}', 'motiv3_en': True}}"
                }
            )
            run_args = tpl.to_run_args()
            tasks.append({'run_args': run_args, 'exp_name': exp_name, 'dry_run': self.args.dry_run})
        
        # dry_run: 仅打印
        if self.args.dry_run:
            for t in tasks:
                _ = run_main_approach_inproc(t['run_args'], dry_run=True)
            return
        
        # 并行执行
        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行 Case 3 任务数={len(tasks)}, max_workers={max_workers}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_case3_worker, t) for t in tasks]
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                exp_name = res['exp_name']
                stats = res['stats']
                self.results[exp_name] = stats
                print(f"  {exp_name}: Spearman ρ={stats['spearman_rho']:.4f}, RMSE={stats.get('rmse', float('nan')):.4f}, Realloc={stats['realloc_mean_ratio']:.4f}")

    def _generate_report(self):
        """生成报告和图表"""
        print("\n" + "="*30 + " Case 3: Generating Report " + "="*30)
        json_path = self.output_dir / 'case3_summary.json'
        
        # 统一从 self.results 或缓存加载数据
        if self.args.use_plot_cache:
            assert json_path.exists(), f"Cache file not found at {json_path}"
            print(f"Loading cached results from {json_path}...")
            with open(json_path, 'r') as f:
                self.results = json.load(f).get('results', {})
        
        if not self.results:
            print("Warning: No simulation results found to generate a report.")
            return

        # 保存摘要（如果不是来自缓存）
        if not self.args.use_plot_cache:
            summary_data = {
                'experiment': 'Case 3: Switching Overhead Uncertainty',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'mode': self.mode,
                    'num_periods': self.num_periods
                },
                'results': self.results
            }
            with open(json_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            print(f"摘要已保存到: {json_path}")
        
        # 检查是否有对比数据
        has_baseline = 'baseline' in self.results
        has_experiment = 'experiment' in self.results

        if has_baseline and has_experiment:
            # --- 对比绘图和分析 ---
            rho_base = self.results['baseline']['spearman_rho']
            rho_exp = self.results['experiment']['spearman_rho']
            rmse_base = self.results['baseline'].get('rmse', float('nan'))
            rmse_exp = self.results['experiment'].get('rmse', float('nan'))
            delta_rho = rho_base - rho_exp
            hypo_verified = (rho_base > 0.85) and (rho_exp < 0.6) and (delta_rho > 0.25)

            print("\n--- Comparison Summary ---")
            print(f"Baseline Spearman ρ: {rho_base:.4f}, RMSE: {rmse_base:.4f}")
            print(f"Experiment Spearman ρ: {rho_exp:.4f}, RMSE: {rmse_exp:.4f}")
            print(f"Delta ρ: {delta_rho:.4f}")
            print(f"Hypothesis Verified: {'✅' if hypo_verified else '❌'}")
            
            # Raw模式下合并绘图
            if self.results['baseline'].get('mode') == 'raw':
                data_groups = [
                    {
                        'raw_data': self.results['baseline'].get('raw_data', []),
                        'spearman_rho': rho_base,
                        'rmse': rmse_base,
                        'label': 'w/o overhead',
                        'fit': 'wls',
                        'color': 'C0'
                    },
                    {
                        'raw_data': self.results['experiment'].get('raw_data', []),
                        'spearman_rho': rho_exp,
                        'rmse': rmse_exp,
                        'label': 'w/ overhead',
                        'fit': 'wls',
                        'color': 'C1'
                    }
                ]
                plot_path = self.output_dir / f'case3_comparison_{self.mode}.pdf'
                StatisticsCollector.plot_load_latency_raw(data_groups=data_groups, save_path=str(plot_path))
            else:
                # Binned 模式目前仍分开绘图
                plot_path_base = self.output_dir / 'case3_baseline_binned.pdf'
                StatisticsCollector.plot_load_latency_binned(
                    binned_summary=self.results['baseline'].get('binned_summary', []),
                    spearman_rho=rho_base,
                    percentile=self.args.case3_percentile,
                    fit='wls',
                    save_path=str(plot_path_base)
                )
                
                plot_path_exp = self.output_dir / 'case3_experiment_binned.pdf'
                StatisticsCollector.plot_load_latency_binned(
                    binned_summary=self.results['experiment'].get('binned_summary', []),
                    spearman_rho=rho_exp,
                    percentile=self.args.case3_percentile,
                    iqr_band=(0.25, 0.75),
                    fit=self.args.case3_fit,
                    save_path=str(plot_path_exp)
                )

        else: # 只有一个结果时
            exp_name = list(self.results.keys())[0]
            stats = self.results[exp_name]
            print(f"\n--- Single Result: {exp_name} ---")
            
            if stats.get('mode') == 'raw':
                data_groups = [{
                    'raw_data': stats.get('raw_data', []),
                    'spearman_rho': stats.get('spearman_rho', 0),
                    'rmse': stats.get('rmse', float('nan')),
                    'label': exp_name,
                    'fit': self.args.case3_fit
                }]
                plot_path = self.output_dir / f'case3_{exp_name}_{self.mode}.pdf'
                StatisticsCollector.plot_load_latency_raw(
                    data_groups=data_groups,
                    save_path=str(plot_path),
                )
            else: # binned
                plot_path = self.output_dir / f'case3_{exp_name}_binned.pdf'
                StatisticsCollector.plot_load_latency_binned(
                    binned_summary=stats.get('binned_summary', []),
                    spearman_rho=stats.get('spearman_rho', 0),
                    percentile=self.args.case3_percentile,
                    iqr_band=(0.25, 0.75),
                    fit=self.args.case3_fit,
                    save_path=str(plot_path)
                )
        
        print(f"\n✓ Case 3 完成！结果保存在: {self.output_dir}")


def main():
    """主函数"""
    args = parse_args()
    # 构造基础参数模板
    base_mapping = {**mapping_args}
    base_mapping.update({
        'exec_t_comp_ratioA': args.base_ratioA,
        'exec_t_comp_ratioB': args.base_ratioB,
    })
    base_runtime = {**runtime_args}
    base_runtime.update({
        'n_p': args.num_hp,
    })
    base_specific = {**specific_args}
    base_specific.update({
        'verbose': args.verbose,
    })
    # 添加额外参数（覆盖specific组）
    if args.extra_args:
        for arg in args.extra_args.split():
            if '=' in arg:
                key, val = arg.split('=', 1)
                base_specific[key.lstrip('-')] = val

    base_tpl = ParamTemplate(base_mapping, base_runtime, base_specific)

    print("\n" + "="*60)
    print("Motivation Experiments Runner")
    print("="*60)
    print(f"实验: Case {args.case}")
    print(f"输出目录: {args.output_dir}")
    print(f"缓存collectors: {args.cache_collectors}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE - 只打印命令，不执行")
    print("="*60)
    
    # 根据case选择运行器
    if args.case == 1:
        runner = MotivExp1Runner(base_tpl, args)
    elif args.case == 2:
        runner = MotivExp2Runner(base_tpl, args)
    elif args.case == 3:
        runner = MotivExp3Runner(base_tpl, args)
    else:
        raise ValueError(f"Invalid case: {args.case}")
    
    # 运行实验
    runner.run()
    
    print("\n" + "="*60)
    print("实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()

