#!/usr/bin/env python3
"""
Diagnostic script to verify repack execution for cyc-S and reserv strategies.

Usage:
    conda activate gurobi
    python test_repack_diagnostic.py
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
from typing import Dict

def create_test_args(policy: str, ratioA: float, ratioB: float, num_bins: int):
    """Create test arguments for repack testing."""
    args = argparse.Namespace()

    # Basic settings
    args.test_case = "bin_pack_new"
    args.policy = policy
    args.timestepxus = 1000  # 1ms
    args.quantumSize = 10000  # 10ms
    args.n_p = 2  # 2 periods for quick test
    args.warmup_dis = False
    args.seed = 42

    # Resource settings
    args.num_bins = num_bins
    args.num_cores = None  # Auto

    # Repack settings
    args.exec_t_comp_ratioA = ratioA
    args.exec_t_comp_ratioB = ratioB
    args.quantile = ratioA  # Will be updated during repack

    # Workload settings
    args.tiles = 200
    args.chains = 1
    args.load_factor = 0.8
    args.e2e_latency = 0.1

    # Plotting and dump
    args.plot = False
    args.dump = False
    args.verbose = True
    args.DEBUG = False

    # Scheduler settings
    args.barrier_en = True
    args.barrier_dis = False
    args.forbid_miss = False
    args.allow_realloc = True
    args.jitter_sim_en = False
    args.jitter_sim_para = {}
    args.exec_var_en = False
    args.exec_var_para = {"scale": 0.0, "lambda_exp": 0.0}
    args.load_var_sim_para = {}
    args.file_suffix = ""
    args.aux_scale_factor = 1
    args.var_sim_cfg = "var_sim_cfg.json"
    args.wsc_slack_ratio = 1.0
    args.comp_rda_ratio = 0.0
    args.comp_size = 1

    # Binpack config
    args.binpack_cfg = {
        "algorithm": "guided",
        "sort": "barycenter",
        "mode": "block",
        "bin_sel_mod": "search",
        "reservation_policy": "static_1_bin",
        "affinity_en": True,
        "affinity_level": 2,
        "preempt_en": False,
        "partial_alloc_en": False,
        "quantum_check_en": False,
        "core_size": "induced"
    }

    return args


def run_repack_test(policy: str, ratioA: float, ratioB: float, num_bins: int):
    """Run a single repack test and return diagnostic info."""
    from approach.approach_setup import setup_benchmark
    from approach.approach_Eq import set_time_unit

    # Set time unit and normalization factor
    time_unit, time_norm_factor = set_time_unit(1e-6, False)

    print(f"\n{'='*70}")
    print(f"Testing: policy={policy}, ratioA={ratioA}, ratioB={ratioB}, num_bins={num_bins}")
    print(f"{'='*70}")

    args = create_test_args(policy, ratioA, ratioB, num_bins)

    try:
        G, pid2name, bin_list, policy, path_ctx, hyper_p = setup_benchmark(args, time_norm_factor)

        # Count placed tasks
        placed_pids = set()
        for _b in bin_list:
            placed_pids.update(_b.index_occupy_by_id().keys())

        result = {
            "success": True,
            "policy": policy,
            "ratioA": ratioA,
            "ratioB": ratioB,
            "num_bins": num_bins,
            "num_cores": sum(b.num_resources for b in bin_list),
            "placed_tasks": len(placed_pids),
            "total_tasks": len(G.logical_graph.nodes),
            "bins": len(bin_list),
        }

        print(f"\nResult: SUCCESS")
        print(f"  - Placed {len(placed_pids)} tasks in {len(bin_list)} bins")
        print(f"  - Total cores: {result['num_cores']}")

    except Exception as e:
        result = {
            "success": False,
            "policy": policy,
            "ratioA": ratioA,
            "ratioB": ratioB,
            "num_bins": num_bins,
            "error": str(e),
        }
        print(f"\nResult: FAILED - {e}")

    return result


def main():
    """Run diagnostic tests for cyc-S and reserv repack."""
    print("="*70)
    print("REPACK DIAGNOSTIC TEST SUITE")
    print("="*70)

    results = []

    # Test 1: cyc-S (num_bins=-1, reserv policy with repack)
    print("\n" + "="*70)
    print("TEST 1: cyc-S (num_bins=-1, ratioB=0.7)")
    print("="*70)
    result = run_repack_test("reserv", ratioA=0.7, ratioB=0.5, num_bins=-1)
    results.append(("cyc-S", result))

    # Test 2: reserv (num_bins=4)
    print("\n" + "="*70)
    print("TEST 2: reserv (num_bins=4, ratioB=0.5)")
    print("="*70)
    result = run_repack_test("reserv", ratioA=0.7, ratioB=0.5, num_bins=4)
    results.append(("reserv_4bins", result))

    # Test 3: reserv (num_bins=8)
    print("\n" + "="*70)
    print("TEST 3: reserv (num_bins=8, ratioB=0.5)")
    print("="*70)
    result = run_repack_test("reserv", ratioA=0.7, ratioB=0.5, num_bins=8)
    results.append(("reserv_8bins", result))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    success_count = sum(1 for _, r in results if r.get("success", False))
    total_count = len(results)
    print(f"Total tests: {total_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {total_count - success_count}")

    print("\nDetailed results:")
    for name, result in results:
        status = "✓ SUCCESS" if result.get("success", False) else "✗ FAILED"
        if result.get("success", False):
            print(f"  {name}: {status} - {result['placed_tasks']} tasks in {result['bins']} bins, {result['num_cores']} cores")
        else:
            print(f"  {name}: {status} - {result.get('error', 'Unknown error')[:50]}...")

    return results


if __name__ == "__main__":
    main()
