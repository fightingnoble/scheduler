#!/usr/bin/env python3
"""
E2E Comparison Experiment Runner

端到端比较实验：横向比较 cyc, glb, reserv 三种策略。
需要先运行 e2e_hyperparam.py 获取最优超参。

子实验优先级:
  - Case 3 (Priority 1): Min Resources — 最小资源需求
  - Case 1 (Priority 2): Max Throughput — 最大吞吐
  - Case 4 (Priority 3): Trade-off Curve — idle vs miss 权衡曲线
  - Case 2 (Priority 4): Min Latency — 最小延迟约束

Usage:
    python -m scripts.e2e_exp_runner --case 3 --output_dir ./e2e_results --num_hp 100
    python -m scripts.e2e_exp_runner --case 1 --output_dir ./e2e_results --num_hp 100
    python -m scripts.e2e_exp_runner --case 4 --output_dir ./e2e_results --num_hp 100
    python -m scripts.e2e_exp_runner --case 2 --output_dir ./e2e_results --num_hp 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.exp_common import (
    _PHYSICAL_CORES,
    mapping_args,
    specific_args,
    runtime_args,
    ParamTemplate,
    run_main_approach_inproc,
)

# ==================== 全局配置 ====================

STRATEGIES = ['cyc', 'glb', 'reserv']
STRATEGY_COLORS = {'cyc': 'C0', 'glb': 'C1', 'reserv': 'C3'}
MISS_THRESHOLD = 0.01  # 1%

# 子实验3 (min resources) 的固定变量 combo 列表
EXP3_COMBOS = [
    {'aux_scale_factor': 1, 'load_factor': 0.5, 'e2e_latency': 0.1, 'label': '1C-0.5x'},
    {'aux_scale_factor': 1, 'load_factor': 1.0, 'e2e_latency': 0.1, 'label': '1C-1.0x'},
    {'aux_scale_factor': 4, 'load_factor': 1.0, 'e2e_latency': 0.1, 'label': '4C-1.0x'},
]

# 子实验1 (max throughput) 的固定变量 combo 列表
EXP1_COMBOS = [
    {'num_cores': 400, 'e2e_latency': 0.1, 'label': '400C-Lat0.1'},
]

# 子实验4 (tradeoff) 的扫描参数
EXP4_NUM_CORES_LIST = [200, 250, 300, 350, 400, 450, 500]

# 子实验2 (min latency) 的固定变量 combo 列表
EXP2_COMBOS = [
    {'num_cores': 400, 'aux_scale_factor': 1, 'load_factor': 1.0, 'label': '400C-1C-1.0x'},
]

# Binary search 配置
BINARY_SEARCH_LO = 200
BINARY_SEARCH_HI = 500
BINARY_SEARCH_STEP = 25


# ==================== Worker 函数 ====================

def _e2e_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """通用仿真 worker。"""
    run_args = payload['run_args']
    dry_run = payload.get('dry_run', False)

    try:
        collector = run_main_approach_inproc(run_args, dry_run=dry_run)
    except Exception as e:
        print(f"  [ERROR] Worker failed: {type(e).__name__}: {e}")
        return None
    if dry_run or collector is None:
        return None

    stats = collector.get_motiv_case1_stats()
    realloc_info = collector.get_realloc_info()

    result = {
        'strategy': payload['strategy'],
        'combo_label': payload.get('combo_label', ''),
        'num_cores': payload.get('num_cores'),
        'load_factor': payload.get('load_factor'),
        'e2e_latency': payload.get('e2e_latency'),
        'miss_mean_ratio': stats['miss_mean_ratio'],
        'miss_mean_count': stats['miss_mean_count'],
        'idle_mean_ratio': stats['idle_mean_ratio'],
        'realloc_mean_ratio': realloc_info['realloc_mean_ratio'],
    }
    return result


# ==================== 辅助函数 ====================

def load_best_params(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """加载超参搜索结果。"""
    best_path = output_dir / 'hyperparam' / 'best_params.json'
    if not best_path.exists():
        raise FileNotFoundError(
            f"超参文件不存在: {best_path}\n请先运行: python -m scripts.e2e_hyperparam --output_dir {output_dir}"
        )
    with open(best_path) as f:
        return json.load(f)


def build_strategy_args(strategy: str, best_params: Dict[str, Any],
                        combo: Dict[str, Any], num_cores: Optional[int] = None,
                        extra_mapping: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """构造单个策略的运行参数。

    注意：test_case 由 main_approach.py 硬编码为 'bin_pack_new'，
    调度行为仅由 policy 参数控制。
    """
    params = best_params[strategy].copy()

    # 策略参数约束检查
    num_bins = params.get('num_bins', -1)
    if strategy == 'reserv' and num_bins < 2:
        raise ValueError(f"reserv strategy requires num_bins >= 2, got {num_bins}")
    if strategy == 'glb' and num_bins != 1:
        raise ValueError(f"glb strategy requires num_bins = 1, got {num_bins}")

    # cyc 强制 ratioB = -1（无 repack）
    ratioB = -1 if strategy == 'cyc' else params.get('ratioB', -1)

    base_mapping = {
        'exec_t_comp_ratioA': params.get('ratioA', 0.7),
        'exec_t_comp_ratioB': ratioB,
        'num_bins': num_bins,
        'e2e_latency': combo.get('e2e_latency', 0.1),
        'aux_scale_factor': combo.get('aux_scale_factor', 1),
        'load_factor': combo.get('load_factor', 1.0),
    }

    if num_cores is not None:
        base_mapping['num_cores'] = num_cores
    elif 'num_cores' in combo:
        base_mapping['num_cores'] = combo['num_cores']

    if extra_mapping:
        base_mapping.update(extra_mapping)

    return {
        'mapping': base_mapping,
        'runtime': {'policy': strategy},
    }


def round_to_step(val: int, step: int) -> int:
    """将值舍入到最近的 step 倍数。"""
    return round(val / step) * step


# ==================== 绘图函数 ====================

def plot_grouped_bar(data: List[Dict[str, Any]], x_key: str, y_key: str,
                     group_key: str, bar_key: str, title: str, save_path: str,
                     ylabel: str, strategies: List[str] = None):
    """绘制分组柱状图。

    Args:
        data: 数据点列表
        x_key: X轴分组键（如 'combo_label'）
        y_key: Y轴值键（如 'min_cores'）
        group_key: 分组键（每个X轴位置对应一组）
        bar_key: 柱子区分键（如 'strategy'）
        title: 图标题
        save_path: 保存路径
        ylabel: Y轴标签
        strategies: 策略列表（决定柱子顺序和颜色）
    """
    if strategies is None:
        strategies = STRATEGIES

    # 提取唯一的 X 组
    x_labels = sorted(set(d[x_key] for d in data))
    n_groups = len(x_labels)
    n_bars = len(strategies)
    bar_width = 0.7 / n_bars

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    for bi, strategy in enumerate(strategies):
        ys = []
        for xl in x_labels:
            matches = [d for d in data if d[x_key] == xl and d.get(bar_key) == strategy]
            if matches:
                ys.append(matches[0][y_key])
            else:
                ys.append(0)
        xs = np.arange(n_groups) + (bi - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(xs, ys, bar_width * 0.9,
                      color=STRATEGY_COLORS[strategy], label=strategy, alpha=0.8)
        # 添加数值标签
        for x, y in zip(xs, ys):
            if y > 0:
                ax.text(x, y, f'{y:.0f}' if isinstance(y, (int, float)) and y == int(y) else f'{y:.2f}',
                        ha='center', va='bottom', fontsize=6)

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.tick_params(axis='y', labelsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图已保存: {save_path}")
    plt.close(fig)


def plot_tradeoff_curve(data: List[Dict[str, Any]], save_path: str, title: str):
    """绘制 idle_ratio (x) vs miss_ratio (y) 权衡曲线。"""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    for strategy in STRATEGIES:
        pts = [d for d in data if d.get('strategy') == strategy]
        if not pts:
            continue
        pts = sorted(pts, key=lambda d: d['idle_mean_ratio'])
        xs = [d['idle_mean_ratio'] for d in pts]
        ys = [d['miss_mean_ratio'] for d in pts]
        ax.plot(xs, ys, 'o-', color=STRATEGY_COLORS[strategy], label=strategy,
                linewidth=1.5, markersize=4)

    ax.set_xlabel('Idle Ratio', fontsize=9)
    ax.set_ylabel('Miss Ratio', fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.tick_params(axis='both', labelsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  图已保存: {save_path}")
    plt.close(fig)


# ==================== Sub-exp 3: Min Resources ====================

class E2EExp3Runner:
    """子实验3 (Priority 1): 最小资源需求 — binary search num_cores"""

    def __init__(self, base_tpl: ParamTemplate, args, best_params: Dict[str, Any]):
        self.base_tpl = base_tpl
        self.args = args
        self.best_params = best_params
        self.output_dir = Path(args.output_dir) / 'exp3_min_resources'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run(self):
        print("\n" + "="*60)
        print("Sub-exp 3: Min Resources (Binary Search)")
        print("="*60)
        print(f"扫描范围: num_cores ∈ [{BINARY_SEARCH_LO}, {BINARY_SEARCH_HI}], step={BINARY_SEARCH_STEP}")
        print(f"阈值: miss_mean_ratio < {MISS_THRESHOLD}")
        print(f"Combos: {len(EXP3_COMBOS)}")
        print(f"输出目录: {self.output_dir}")

        if not self.args.use_plot_cache:
            self._run_simulations()

        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        """对每个 combo × strategy 执行 binary search。"""
        for combo in EXP3_COMBOS:
            print(f"\n--- Combo: {combo['label']} ---")
            for strategy in STRATEGIES:
                min_cores = self._binary_search_min_cores(strategy, combo)
                if min_cores is not None:
                    self.results.append({
                        'strategy': strategy,
                        'combo_label': combo['label'],
                        'min_cores': min_cores,
                        'miss_threshold': MISS_THRESHOLD,
                    })
                    print(f"  {strategy}: min_cores = {min_cores}")

    def _binary_search_min_cores(self, strategy: str, combo: Dict[str, Any]) -> Optional[int]:
        """线性扫描找最小 num_cores 使得 miss_ratio < threshold。

        注意：由于扫描范围仅 [200, 500]，步长 25，总共只有 13 个点，
        线性扫描比 binary search 更简单可靠。
        """
        for cores in range(BINARY_SEARCH_LO, BINARY_SEARCH_HI + 1, BINARY_SEARCH_STEP):
            result = self._run_single(strategy, combo, cores)
            if result is None:
                continue
            if result['miss_mean_ratio'] < MISS_THRESHOLD:
                return cores  # 找到第一个满足条件的即返回
        return None  # 没有找到满足条件的 num_cores

    def _run_single(self, strategy: str, combo: Dict[str, Any], num_cores: int) -> Optional[Dict[str, Any]]:
        """运行单次仿真并返回结果。"""
        args_dict = build_strategy_args(strategy, self.best_params, combo, num_cores=num_cores)
        tpl = self.base_tpl.with_updates(
            mapping=args_dict['mapping'],
            runtime=args_dict['runtime'],
            specific={'root_dir': str(self.output_dir / f'{strategy}_{combo["label"]}_cores{num_cores}')},
        )
        payload = {
            'run_args': tpl.to_run_args(),
            'strategy': strategy,
            'combo_label': combo['label'],
            'num_cores': num_cores,
            'dry_run': self.args.dry_run,
        }
        return _e2e_worker(payload)

    def _generate_report(self):
        print("\n" + "="*30 + " Exp3: Generating Report " + "="*30)
        json_path = self.output_dir / 'exp3_summary.json'

        if self.args.use_plot_cache:
            assert json_path.exists(), f"缓存文件不存在: {json_path}"
            print(f"从缓存加载: {json_path}")
            with open(json_path) as f:
                summary = json.load(f)
            self.results = summary.get('results', [])
        else:
            summary = {
                'experiment': 'E2E Sub-exp 3: Min Resources',
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'combos': EXP3_COMBOS,
                    'binary_search': {'lo': BINARY_SEARCH_LO, 'hi': BINARY_SEARCH_HI, 'step': BINARY_SEARCH_STEP},
                    'miss_threshold': MISS_THRESHOLD,
                },
                'results': self.results,
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存: {json_path}")

        if self.results:
            plot_grouped_bar(
                data=self.results,
                x_key='combo_label',
                y_key='min_cores',
                group_key='combo_label',
                bar_key='strategy',
                title='E2E Exp3: Minimum Cores Required (miss < 1%)',
                save_path=str(self.output_dir / 'exp3_min_cores_bar.pdf'),
                ylabel='Min Cores',
            )

        print(f"\n✓ Exp3 完成！结果保存在: {self.output_dir}")


# ==================== Sub-exp 1: Max Throughput ====================

class E2EExp1Runner:
    """子实验1 (Priority 2): 最大吞吐 — 扫描 load_factor 找最大可行值"""

    def __init__(self, base_tpl: ParamTemplate, args, best_params: Dict[str, Any]):
        self.base_tpl = base_tpl
        self.args = args
        self.best_params = best_params
        self.output_dir = Path(args.output_dir) / 'exp1_max_throughput'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.load_factors = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
        self.results = []

    def run(self):
        print("\n" + "="*60)
        print("Sub-exp 1: Max Throughput (Load Factor Sweep)")
        print("="*60)
        print(f"扫描 load_factor: {self.load_factors}")
        print(f"阈值: miss_mean_ratio < {MISS_THRESHOLD}")
        print(f"输出目录: {self.output_dir}")

        if not self.args.use_plot_cache:
            self._run_simulations()

        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        tasks = []
        for combo in EXP1_COMBOS:
            for strategy in STRATEGIES:
                for lf in self.load_factors:
                    combo_with_lf = {**combo, 'load_factor': lf, 'label': f"{combo['label']}_lf{lf}"}
                    args_dict = build_strategy_args(strategy, self.best_params, combo_with_lf)
                    tpl = self.base_tpl.with_updates(
                        mapping=args_dict['mapping'],
                        runtime=args_dict['runtime'],
                        specific={'root_dir': str(self.output_dir / f'{strategy}_lf{lf}')},
                    )
                    tasks.append({
                        'run_args': tpl.to_run_args(),
                        'strategy': strategy,
                        'combo_label': combo['label'],
                        'load_factor': lf,
                        'num_cores': combo.get('num_cores'),
                        'dry_run': self.args.dry_run,
                    })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行, 任务数={len(tasks)}, max_workers={max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_e2e_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    continue
                if res is not None:
                    self.results.append(res)
                    print(f"  完成: {res['strategy']:8s} lf={res['load_factor']:.1f} "
                          f"miss={res['miss_mean_ratio']:.4e}")

    def _generate_report(self):
        print("\n" + "="*30 + " Exp1: Generating Report " + "="*30)
        json_path = self.output_dir / 'exp1_summary.json'

        if self.args.use_plot_cache:
            assert json_path.exists(), f"缓存文件不存在: {json_path}"
            print(f"从缓存加载: {json_path}")
            with open(json_path) as f:
                summary = json.load(f)
            self.results = summary.get('results', [])
        else:
            # 计算每个 (combo, strategy) 的最大可行 load_factor
            max_throughput = []
            for combo in EXP1_COMBOS:
                for strategy in STRATEGIES:
                    pts = [r for r in self.results
                           if r['combo_label'] == combo['label'] and r['strategy'] == strategy]
                    valid = [p for p in pts if p['miss_mean_ratio'] < MISS_THRESHOLD]
                    if valid:
                        max_lf = max(p['load_factor'] for p in valid)
                        max_throughput.append({
                            'strategy': strategy,
                            'combo_label': combo['label'],
                            'max_load_factor': max_lf,
                        })

            summary = {
                'experiment': 'E2E Sub-exp 1: Max Throughput',
                'timestamp': datetime.now().isoformat(),
                'parameters': {'combos': EXP1_COMBOS, 'load_factors': self.load_factors, 'miss_threshold': MISS_THRESHOLD},
                'results': self.results,
                'max_throughput': max_throughput,
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存: {json_path}")
            self.results = max_throughput

        if self.results:
            plot_grouped_bar(
                data=self.results,
                x_key='combo_label',
                y_key='max_load_factor',
                group_key='combo_label',
                bar_key='strategy',
                title='E2E Exp1: Max Throughput (miss < 1%)',
                save_path=str(self.output_dir / 'exp1_throughput_bar.pdf'),
                ylabel='Max Load Factor',
            )

        print(f"\n✓ Exp1 完成！结果保存在: {self.output_dir}")


# ==================== Sub-exp 4: Trade-off Curve ====================

class E2EExp4Runner:
    """子实验4 (Priority 3): 权衡曲线 — idle_ratio vs miss_ratio"""

    def __init__(self, base_tpl: ParamTemplate, args, best_params: Dict[str, Any]):
        self.base_tpl = base_tpl
        self.args = args
        self.best_params = best_params
        self.output_dir = Path(args.output_dir) / 'exp4_tradeoff'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_cores_list = EXP4_NUM_CORES_LIST
        self.results = []

    def run(self):
        print("\n" + "="*60)
        print("Sub-exp 4: Trade-off Curve (Idle vs Miss)")
        print("="*60)
        print(f"扫描 num_cores: {self.num_cores_list}")
        print(f"输出目录: {self.output_dir}")

        if not self.args.use_plot_cache:
            self._run_simulations()

        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        tasks = []
        combo = {'e2e_latency': 0.1, 'aux_scale_factor': 1, 'load_factor': 1.0}

        for strategy in STRATEGIES:
            for num_cores in self.num_cores_list:
                args_dict = build_strategy_args(strategy, self.best_params, combo, num_cores=num_cores)
                tpl = self.base_tpl.with_updates(
                    mapping=args_dict['mapping'],
                    runtime=args_dict['runtime'],
                    specific={'root_dir': str(self.output_dir / f'{strategy}_cores{num_cores}')},
                )
                tasks.append({
                    'run_args': tpl.to_run_args(),
                    'strategy': strategy,
                    'num_cores': num_cores,
                    'dry_run': self.args.dry_run,
                })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行, 任务数={len(tasks)}, max_workers={max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_e2e_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    continue
                if res is not None:
                    self.results.append(res)
                    print(f"  完成: {res['strategy']:8s} cores={res['num_cores']:3d} "
                          f"idle={res['idle_mean_ratio']:.4f} miss={res['miss_mean_ratio']:.4e}")

    def _generate_report(self):
        print("\n" + "="*30 + " Exp4: Generating Report " + "="*30)
        json_path = self.output_dir / 'exp4_summary.json'

        if self.args.use_plot_cache:
            assert json_path.exists(), f"缓存文件不存在: {json_path}"
            print(f"从缓存加载: {json_path}")
            with open(json_path) as f:
                summary = json.load(f)
            self.results = summary.get('results', [])
        else:
            summary = {
                'experiment': 'E2E Sub-exp 4: Trade-off Curve',
                'timestamp': datetime.now().isoformat(),
                'parameters': {'num_cores_list': self.num_cores_list},
                'results': self.results,
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存: {json_path}")

        if self.results:
            plot_tradeoff_curve(
                data=self.results,
                save_path=str(self.output_dir / 'exp4_idle_vs_miss.pdf'),
                title='E2E Exp4: Idle vs Miss Trade-off',
            )

        print(f"\n✓ Exp4 完成！结果保存在: {self.output_dir}")


# ==================== Sub-exp 2: Min Latency ====================

class E2EExp2Runner:
    """子实验2 (Priority 4): 最小延迟 — 扫描 e2e_latency 找最小可行值"""

    def __init__(self, base_tpl: ParamTemplate, args, best_params: Dict[str, Any]):
        self.base_tpl = base_tpl
        self.args = args
        self.best_params = best_params
        self.output_dir = Path(args.output_dir) / 'exp2_min_latency'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.e2e_latencies = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
        self.results = []

    def run(self):
        print("\n" + "="*60)
        print("Sub-exp 2: Min Latency (E2E Latency Sweep)")
        print("="*60)
        print(f"扫描 e2e_latency: {self.e2e_latencies}")
        print(f"阈值: miss_mean_ratio < {MISS_THRESHOLD}")
        print(f"输出目录: {self.output_dir}")

        if not self.args.use_plot_cache:
            self._run_simulations()

        if not self.args.dry_run:
            self._generate_report()

    def _run_simulations(self):
        tasks = []
        for combo in EXP2_COMBOS:
            for strategy in STRATEGIES:
                for lat in self.e2e_latencies:
                    combo_with_lat = {**combo, 'e2e_latency': lat, 'label': f"{combo['label']}_lat{lat}"}
                    args_dict = build_strategy_args(strategy, self.best_params, combo_with_lat)
                    tpl = self.base_tpl.with_updates(
                        mapping=args_dict['mapping'],
                        runtime=args_dict['runtime'],
                        specific={'root_dir': str(self.output_dir / f'{strategy}_lat{lat}')},
                    )
                    tasks.append({
                        'run_args': tpl.to_run_args(),
                        'strategy': strategy,
                        'combo_label': combo['label'],
                        'e2e_latency': lat,
                        'num_cores': combo.get('num_cores'),
                        'dry_run': self.args.dry_run,
                    })

        max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
        print(f"并行执行, 任务数={len(tasks)}, max_workers={max_workers}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_e2e_worker, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    continue
                if res is not None:
                    self.results.append(res)
                    print(f"  完成: {res['strategy']:8s} lat={res['e2e_latency']:.2f} "
                          f"miss={res['miss_mean_ratio']:.4e}")

    def _generate_report(self):
        print("\n" + "="*30 + " Exp2: Generating Report " + "="*30)
        json_path = self.output_dir / 'exp2_summary.json'

        if self.args.use_plot_cache:
            assert json_path.exists(), f"缓存文件不存在: {json_path}"
            print(f"从缓存加载: {json_path}")
            with open(json_path) as f:
                summary = json.load(f)
            self.results = summary.get('results', [])
        else:
            # 计算每个 (combo, strategy) 的最小可行 e2e_latency
            min_latencies = []
            for combo in EXP2_COMBOS:
                for strategy in STRATEGIES:
                    pts = [r for r in self.results
                           if r['combo_label'] == combo['label'] and r['strategy'] == strategy]
                    valid = [p for p in pts if p['miss_mean_ratio'] < MISS_THRESHOLD]
                    if valid:
                        min_lat = min(p['e2e_latency'] for p in valid)
                        min_latencies.append({
                            'strategy': strategy,
                            'combo_label': combo['label'],
                            'min_e2e_latency': min_lat,
                        })

            summary = {
                'experiment': 'E2E Sub-exp 2: Min Latency',
                'timestamp': datetime.now().isoformat(),
                'parameters': {'combos': EXP2_COMBOS, 'e2e_latencies': self.e2e_latencies, 'miss_threshold': MISS_THRESHOLD},
                'results': self.results,
                'min_latencies': min_latencies,
            }
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"摘要已保存: {json_path}")
            self.results = min_latencies

        if self.results:
            plot_grouped_bar(
                data=self.results,
                x_key='combo_label',
                y_key='min_e2e_latency',
                group_key='combo_label',
                bar_key='strategy',
                title='E2E Exp2: Min E2E Latency (miss < 1%)',
                save_path=str(self.output_dir / 'exp2_min_latency_bar.pdf'),
                ylabel='Min E2E Latency (s)',
            )

        print(f"\n✓ Exp2 完成！结果保存在: {self.output_dir}")


# ==================== CLI ====================

def parse_args():
    parser = argparse.ArgumentParser(description='E2E Comparison Experiment Runner')
    parser.add_argument('--case', type=int, required=True, choices=[1, 2, 3, 4],
                        help='实验编号: 1=max throughput, 2=min latency, 3=min resources, 4=tradeoff')
    parser.add_argument('--output_dir', type=str, default='./e2e_results',
                        help='输出目录（默认: ./e2e_results）')
    parser.add_argument('--num_hp', type=int, default=100,
                        help='仿真超周期数（默认: 100）')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--dry_run', action='store_true', help='只打印命令，不执行')
    parser.add_argument('--use_plot_cache', action='store_true',
                        help='跳过仿真，直接从缓存的JSON结果生成图表')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载超参
    output_dir = Path(args.output_dir)
    best_params = load_best_params(output_dir)

    # 构造基础模板
    base_tpl = ParamTemplate(
        {**mapping_args},
        {**runtime_args, 'n_p': args.num_hp},
        {**specific_args, 'verbose': args.verbose},
    )

    print("\n" + "="*60)
    print(f"E2E Comparison Experiment Runner — Case {args.case}")
    print("="*60)
    print(f"输出目录: {args.output_dir}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE")
    print("="*60)

    # 选择 Runner
    case_to_runner = {
        3: E2EExp3Runner,  # Priority 1
        1: E2EExp1Runner,  # Priority 2
        4: E2EExp4Runner,  # Priority 3
        2: E2EExp2Runner,  # Priority 4
    }

    runner_cls = case_to_runner.get(args.case)
    if runner_cls is None:
        raise ValueError(f"Invalid case: {args.case}")

    runner = runner_cls(base_tpl, args, best_params)
    runner.run()

    print("\n" + "="*60)
    print(f"Case {args.case} 完成！")
    print("="*60)


if __name__ == '__main__':
    main()
