#!/usr/bin/env python3
"""Characterization tests for the bin-packing setup pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))  # worker 脚本模式：保证可发现 approach 包
BASELINE_PATH = REPO_ROOT / "cleanup/reports/b10-binpack-behavior-baseline.json"
SIGNATURE_MARKER = "B10_SIGNATURE_JSON="
SCENARIOS = {
    "split": {
        "policy": "pglb",
        "ratio_b": -1.0,
    },
    "repack": {
        "policy": "reserv",
        "ratio_b": 0.5,
    },
}


def _build_pipeline_args(output_root: Path, scenario: str):
    from global_var import case_name_bp
    from utils import input_parser

    scenario_config = SCENARIOS[scenario]
    old_argv = sys.argv
    sys.argv = [
        "b10-contract-worker",
        "--test_case",
        case_name_bp,
        "--root_dir",
        "b10-artifacts",
        "--gen_benchmark",
        "--var_sim_cfg",
        str(REPO_ROOT / "cfgs/var_sim_cfg.json"),
        "--bin_pack_cfg",
        str(REPO_ROOT / "cfgs/Bp_guided.json"),
        "--profiling_filename",
        str(REPO_ROOT / "profiling/profiling_light.csv"),
        "--G_decomp_mode",
        "full",
        "--num_bins",
        "2",
        "--n_p",
        "2",
        "--seed",
        "42",
        "--exec_t_comp_ratioA",
        "0.7",
        "--exec_t_comp_ratioB",
        str(scenario_config["ratio_b"]),
        "--policy",
        scenario_config["policy"],
    ]
    try:
        args = input_parser()
    finally:
        sys.argv = old_argv
    return args


def _normalize_scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return round(value, 12)
    return value


def _normalize_bin_list(bin_list):
    bins = []
    all_pids = set()
    for scheduling_bin in sorted(bin_list, key=lambda item: str(item.id)):
        occupancy = scheduling_bin.index_occupy_by_id()
        tasks = []
        for pid, (starts, core_counts, lengths) in sorted(occupancy.items()):
            all_pids.add(int(pid))
            intervals = [
                {
                    "start": int(start),
                    "cores": int(cores),
                    "length": int(length),
                }
                for start, cores, length in zip(starts, core_counts, lengths)
            ]
            tasks.append({"pid": int(pid), "intervals": intervals})
        bins.append(
            {
                "id": _normalize_scalar(scheduling_bin.id),
                "num_resources": int(scheduling_bin.num_resources),
                "time_slots": int(scheduling_bin.temp_size),
                "tasks": tasks,
            }
        )
    return {
        "bin_count": len(bins),
        "total_resources": sum(item["num_resources"] for item in bins),
        "pid_count": len(all_pids),
        "pids": sorted(all_pids),
        "bins": bins,
    }


def _normalize_task_timing(graph, pid_to_name, packed_pids):
    task_timing = []
    for pid in packed_pids:
        assert pid in pid_to_name, f"packed PID {pid} is missing from pid_to_name"
        task_name = pid_to_name[pid]
        task_attrs = graph.logical_graph.nodes[task_name]
        assert "ert" in task_attrs, f"packed PID {pid} ({task_name}) has no ert"
        assert "ddl" in task_attrs, f"packed PID {pid} ({task_name}) has no ddl"
        task_timing.append(
            {
                "pid": int(pid),
                "name": task_name,
                "ert": _normalize_scalar(task_attrs["ert"]),
                "ddl": _normalize_scalar(task_attrs["ddl"]),
            }
        )
    return task_timing


def _run_pipeline_worker(scenario: str, output_root: Path):
    from approach.approach_Eq import set_time_unit
    import approach.approach_setup as approach_setup

    output_root.mkdir(parents=True, exist_ok=True)
    os.chdir(output_root)
    args = _build_pipeline_args(output_root, scenario)
    _, time_norm_factor = set_time_unit(1e-6, False)
    packing_calls = []
    original_init = approach_setup.init_sched_components

    def recording_init(*init_args, **init_kwargs):
        pack = original_init(*init_args, **init_kwargs)

        def recording_pack(*pack_args, **pack_kwargs):
            result = pack(*pack_args, **pack_kwargs)
            packing_calls.append(
                {
                    "need_repack": bool(pack_args[4]),
                    "bin_count": len(result[0]),
                    "max_core_num": int(result[1]),
                    "repack_success": bool(result[4]),
                }
            )
            return result

        recording_pack.sim_step = pack.sim_step
        recording_pack.num_periods = pack.num_periods
        return recording_pack

    approach_setup.init_sched_components = recording_init
    try:
        graph, pid_to_name, bin_list, policy, _, hyper_period = (
            approach_setup.setup_benchmark(args, time_norm_factor)
        )
    finally:
        approach_setup.init_sched_components = original_init
    signature = _normalize_bin_list(bin_list)
    signature["task_timing"] = _normalize_task_timing(
        graph, pid_to_name, signature["pids"]
    )
    signature.update(
        {
            "scenario": scenario,
            "policy": policy,
            "hyper_period": _normalize_scalar(hyper_period),
            "logical_task_count": len(graph.logical_graph.nodes),
            "pid_to_name": [
                [int(pid), name] for pid, name in sorted(pid_to_name.items())
            ],
            "packing_calls": packing_calls,
        }
    )
    return signature


def _run_worker_subprocess(scenario: str, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-scenario",
            scenario,
            "--output-root",
            str(output_root),
        ],
        cwd=output_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    marker_lines = [
        line for line in completed.stdout.splitlines() if line.startswith(SIGNATURE_MARKER)
    ]
    assert len(marker_lines) == 1, completed.stdout
    return json.loads(marker_lines[0][len(SIGNATURE_MARKER) :])


def _signature_sha256(signature):
    canonical_json = json.dumps(
        signature, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


@pytest.fixture(scope="module")
def gurobi_preflight():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import gurobipy as gp; "
                "model = gp.Model('b10-contract-preflight'); "
                "model.dispose()"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        pytest.fail(
            "Gurobi environment preflight failed before baseline execution:\n"
            + completed.stdout
            + completed.stderr,
            pytrace=False,
        )


def test_init_sched_components_binds_and_forwards_context(monkeypatch):
    import sim_main

    sentinels = {name: object() for name in (
        "args", "path_ctx", "workload", "num_cores", "initial_bin_list",
        "task_spec", "resource_list", "message_dispatcher", "activation_pipe",
        "weight_pipe", "scheduler_list", "monitor_list", "trace_path", "sim_step",
        "num_periods", "warmup", "quantum_size", "event_range", "event_iter_dict",
        "cfg_params", "scan_group_1", "scan_group_2", "path_params", "bin_format",
        "trace_params", "plot_params", "csv_root", "case_path", "global_processes",
        "call_bin_list", "hyper_period", "physical_graph", "need_repack", "result",
    )}
    path_params = (
        sentinels["cfg_params"],
        sentinels["scan_group_1"],
        sentinels["scan_group_2"],
        sentinels["path_params"],
        sentinels["bin_format"],
        sentinels["trace_params"],
        sentinels["plot_params"],
        sentinels["csv_root"],
        sentinels["case_path"],
    )
    calls = {}

    def fake_create(*args):
        calls["create"] = args
        return (
            sentinels["task_spec"],
            sentinels["resource_list"],
            sentinels["message_dispatcher"],
            sentinels["activation_pipe"],
            sentinels["weight_pipe"],
            sentinels["scheduler_list"],
            sentinels["monitor_list"],
            sentinels["trace_path"],
            sentinels["sim_step"],
        )

    def fake_build(*args):
        calls["build"] = args
        return (
            sentinels["num_periods"],
            sentinels["warmup"],
            sentinels["quantum_size"],
            sentinels["event_range"],
            sentinels["event_iter_dict"],
        )

    def fake_perform(*args):
        calls["perform"] = args
        return sentinels["result"]

    monkeypatch.setattr(sim_main, "create_scheduler_elements_with_config", fake_create)
    monkeypatch.setattr(sim_main, "build_simulation_env", fake_build)
    monkeypatch.setattr(sim_main, "perform_bin_packing", fake_perform)

    pack = sim_main.init_sched_components(
        sentinels["args"],
        path_params,
        sentinels["path_ctx"],
        sentinels["workload"],
        sentinels["num_cores"],
        sentinels["initial_bin_list"],
    )

    assert calls["create"] == (
        sentinels["args"],
        path_params,
        sentinels["path_ctx"],
        sentinels["workload"],
        sentinels["num_cores"],
        sentinels["initial_bin_list"],
    )
    assert calls["build"] == (
        sentinels["args"], sentinels["workload"], sentinels["sim_step"]
    )
    assert "perform" not in calls
    assert pack.sim_step is sentinels["sim_step"]
    assert pack.num_periods is sentinels["num_periods"]

    result = pack(
        sentinels["global_processes"],
        sentinels["call_bin_list"],
        sentinels["hyper_period"],
        sentinels["physical_graph"],
        sentinels["need_repack"],
    )

    expected_args = (
        sentinels["args"],
        sentinels["global_processes"],
        sentinels["num_cores"],
        sentinels["call_bin_list"],
        sentinels["hyper_period"],
        sentinels["sim_step"],
        sentinels["path_params"],
        sentinels["scan_group_1"],
        sentinels["event_iter_dict"],
        sentinels["quantum_size"],
        sentinels["num_periods"],
        sentinels["cfg_params"],
        sentinels["physical_graph"],
        sentinels["need_repack"],
        sentinels["plot_params"],
        sentinels["path_ctx"],
        sentinels["scheduler_list"],
        sentinels["monitor_list"],
        sentinels["message_dispatcher"],
        sentinels["activation_pipe"],
        sentinels["weight_pipe"],
    )
    assert len(expected_args) == 21
    assert calls["perform"] == expected_args
    assert result is sentinels["result"]


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_real_pipeline_matches_stable_baseline(tmp_path, scenario, gurobi_preflight):
    with BASELINE_PATH.open(encoding="utf-8") as baseline_file:
        baseline = json.load(baseline_file)

    first = _run_worker_subprocess(scenario, tmp_path / scenario / "run-1")
    second = _run_worker_subprocess(scenario, tmp_path / scenario / "run-2")

    assert first == second, f"{scenario} is not deterministic with the fixed seed"
    expected = baseline["scenarios"][scenario]
    assert _signature_sha256(first) == expected["sha256"]
    assert {
        key: first[key] for key in expected["summary"]
    } == expected["summary"]
    assert first["packing_calls"] == expected["packing_calls"]
    timing_sample_pids = {item["pid"] for item in expected["timing_samples"]}
    assert [
        item for item in first["task_timing"] if item["pid"] in timing_sample_pids
    ] == expected["timing_samples"]


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    signature = _run_pipeline_worker(args.worker_scenario, args.output_root.resolve())
    print(SIGNATURE_MARKER + json.dumps(signature, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    _main()
