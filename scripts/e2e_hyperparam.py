#!/usr/bin/env python3
"""
E2E Hyperparameter Grid Search

为端到端比较实验搜索每种策略的最优超参数组合。
- cyc:    扫描 ratioA ∈ [0.5, 0.7, 0.9, 0.99]
- glb:    固定 ratioA=0.7, num_bins=1（无超参）
- reserv: 扫描 ratioB ∈ [0.5, 0.7, 0.9, 0.99] × num_bins ∈ [2, 4, 8]

选择标准: 最低 miss_mean_ratio；平局用最低 realloc_mean_ratio。

Usage:
    python -m scripts.e2e_hyperparam --output_dir ./e2e_results --num_hp 100
    python -m scripts.e2e_hyperparam --output_dir ./e2e_results --use_cache
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.exp_common import (
    _PHYSICAL_CORES,
    mapping_args,
    specific_args,
    runtime_args,
    ParamTemplate,
    run_main_approach_inproc,
)

# ==================== 网格搜索参数 ====================

CYC_RATIO_A_GRID = [0.5, 0.7, 0.9, 0.99]
RESERV_RATIO_B_GRID = [0.5, 0.7, 0.9, 0.99]
RESERV_NUM_BINS_GRID = [2, 4, 8]

# 调参用的参考负载
TUNING_LOAD = {
    'num_cores': 400,
    'aux_scale_factor': 1,
    'load_factor': 1.0,
    'e2e_latency': 0.1,
}


# ==================== Worker 函数 ====================

def _hyperparam_worker(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """单次仿真 worker，返回结果 dict 或 None。"""
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

    return {
        'strategy': payload['strategy'],
        'params': payload['params'],
        'label': payload['label'],
        'miss_mean_ratio': stats['miss_mean_ratio'],
        'miss_mean_count': stats['miss_mean_count'],
        'idle_mean_ratio': stats['idle_mean_ratio'],
        'realloc_mean_ratio': realloc_info['realloc_mean_ratio'],
        'realloc_mean_count': realloc_info['realloc_mean_count'],
    }


# ==================== 主逻辑 ====================

def build_tasks(base_tpl: ParamTemplate, args) -> List[Dict[str, Any]]:
    """构造所有网格搜索任务。"""
    output_dir = Path(args.output_dir) / 'hyperparam'
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []

    # --- cyc: 扫描 ratioA ---
    # 注意：test_case 由 main_approach.py 硬编码为 'bin_pack_new'，调度行为仅由 policy 控制
    for ratioA in CYC_RATIO_A_GRID:
        tpl = base_tpl.with_updates(
            mapping={
                'exec_t_comp_ratioA': ratioA,
                'exec_t_comp_ratioB': -1,
                'num_bins': -1,
                **TUNING_LOAD,
            },
            runtime={'policy': 'cyc'},
            specific={'root_dir': str(output_dir / f'cyc_ratioA{ratioA}')},
        )
        tasks.append({
            'run_args': tpl.to_run_args(),
            'strategy': 'cyc',
            'params': {'ratioA': ratioA, 'num_bins': -1, 'policy': 'cyc'},
            'label': f'cyc ratioA={ratioA}',
            'dry_run': args.dry_run,
        })

    # --- glb: 固定参数，只跑一次 ---
    tpl = base_tpl.with_updates(
        mapping={
            'exec_t_comp_ratioA': 0.7,
            'exec_t_comp_ratioB': -1,
            'num_bins': 1,
            **TUNING_LOAD,
        },
        runtime={'policy': 'glb'},
        specific={'root_dir': str(output_dir / 'glb_fixed')},
    )
    tasks.append({
        'run_args': tpl.to_run_args(),
        'strategy': 'glb',
        'params': {'ratioA': 0.7, 'num_bins': 1, 'policy': 'glb'},
        'label': 'glb fixed',
        'dry_run': args.dry_run,
    })

    # --- reserv: 扫描 ratioB × num_bins ---
    for ratioB in RESERV_RATIO_B_GRID:
        for num_bins in RESERV_NUM_BINS_GRID:
            tpl = base_tpl.with_updates(
                mapping={
                    'exec_t_comp_ratioA': 0.7,
                    'exec_t_comp_ratioB': ratioB,
                    'num_bins': num_bins,
                    **TUNING_LOAD,
                },
                runtime={'policy': 'reserv'},
                specific={'root_dir': str(output_dir / f'reserv_rB{ratioB}_bins{num_bins}')},
            )
            tasks.append({
                'run_args': tpl.to_run_args(),
                'strategy': 'reserv',
                'params': {'ratioA': 0.7, 'ratioB': ratioB, 'num_bins': num_bins, 'policy': 'reserv'},
                'label': f'reserv ratioB={ratioB} bins={num_bins}',
                'dry_run': args.dry_run,
            })

    return tasks


def select_best(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """从网格搜索结果中为每种策略选择最优超参。

    选择标准: 最低 miss_mean_ratio；平局用最低 realloc_mean_ratio。
    """
    best = {}
    for strategy in ('cyc', 'glb', 'reserv'):
        candidates = [r for r in results if r['strategy'] == strategy]
        if not candidates:
            continue
        candidates.sort(key=lambda r: (r['miss_mean_ratio'], r['realloc_mean_ratio']))
        winner = candidates[0]
        best[strategy] = winner['params']
        print(f"  {strategy} 最优: {winner['label']} "
              f"(miss={winner['miss_mean_ratio']:.4e}, realloc={winner['realloc_mean_ratio']:.4e})")
    return best


def run_grid_search(args):
    """执行完整的网格搜索流程。"""
    output_dir = Path(args.output_dir) / 'hyperparam'
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / 'best_params.json'
    full_path = output_dir / 'grid_search_full.json'

    # --use_cache: 直接从缓存加载
    if args.use_cache:
        assert best_path.exists(), f"缓存文件不存在: {best_path}"
        print(f"从缓存加载最优超参: {best_path}")
        with open(best_path) as f:
            best = json.load(f)
        for strategy, params in best.items():
            print(f"  {strategy}: {params}")
        return best

    # 构造基础模板
    base_tpl = ParamTemplate(
        {**mapping_args},
        {**runtime_args, 'n_p': args.num_hp},
        {**specific_args, 'verbose': args.verbose},
    )

    tasks = build_tasks(base_tpl, args)
    print(f"\n{'='*60}")
    print(f"E2E Hyperparameter Grid Search")
    print(f"{'='*60}")
    print(f"总任务数: {len(tasks)} (cyc={len(CYC_RATIO_A_GRID)}, glb=1, "
          f"reserv={len(RESERV_RATIO_B_GRID)*len(RESERV_NUM_BINS_GRID)})")
    if args.dry_run:
        print("⚠️  DRY RUN MODE")
    print(f"{'='*60}")

    # 并行执行
    results = []
    max_workers = max(1, min(_PHYSICAL_CORES, len(tasks)))
    print(f"并行执行, max_workers={max_workers}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_hyperparam_worker, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [ERROR] Worker failed: {type(e).__name__}: {e}")
                continue
            if res is None:
                continue
            results.append(res)
            print(f"  完成: {res['label']:35s} miss={res['miss_mean_ratio']:.4e} "
                  f"idle={res['idle_mean_ratio']:.4e} realloc={res['realloc_mean_ratio']:.4e}")

    if args.dry_run:
        print("\nDry run 完成，未执行实际仿真。")
        return {}

    if not results:
        print("WARNING: 没有获得任何仿真结果。")
        return {}

    # 保存完整结果
    full_summary = {
        'experiment': 'E2E Hyperparameter Grid Search',
        'timestamp': datetime.now().isoformat(),
        'tuning_load': TUNING_LOAD,
        'grids': {
            'cyc_ratioA': CYC_RATIO_A_GRID,
            'reserv_ratioB': RESERV_RATIO_B_GRID,
            'reserv_num_bins': RESERV_NUM_BINS_GRID,
        },
        'results': results,
    }
    with open(full_path, 'w') as f:
        json.dump(full_summary, f, indent=2)
    print(f"\n完整结果已保存: {full_path}")

    # 选择最优
    print(f"\n{'='*30} 选择最优超参 {'='*30}")
    best = select_best(results)

    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"最优超参已保存: {best_path}")

    return best


def parse_args():
    parser = argparse.ArgumentParser(description='E2E Hyperparameter Grid Search')
    parser.add_argument('--output_dir', type=str, default='./e2e_results',
                        help='输出目录（默认: ./e2e_results）')
    parser.add_argument('--num_hp', type=int, default=100,
                        help='仿真超周期数（默认: 100）')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--dry_run', action='store_true', help='只打印命令，不执行')
    parser.add_argument('--use_cache', action='store_true',
                        help='跳过仿真，直接从缓存的JSON结果加载')
    return parser.parse_args()


def main():
    args = parse_args()
    best = run_grid_search(args)
    print(f"\n{'='*60}")
    print("超参搜索完成！")
    print(f"{'='*60}")
    return best


if __name__ == '__main__':
    main()
