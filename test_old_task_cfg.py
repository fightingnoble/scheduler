"""Characterize the intact historical task-graph builder after archival."""

import ast
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "cleanup/reports/b17-task-cfg-legacy-graph-baseline.json"
REPORT = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
ARCHIVE = ROOT / "task/task_cfg_old.py"
CASES = REPORT["baseline_cases"]
EXPECTED = {item["name"]: item["result"] for item in REPORT["behavior_baseline"]["cases"]}


def graph_state(graph):
    return {
        "nodes": [[str(node), dict(data)] for node, data in graph.nodes(data=True)],
        "edges": [
            [str(start), str(end), dict(data)]
            for start, end, data in graph.edges(data=True)
        ],
    }


def run_isolated(code, tmp_path, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, *args],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_archive_exists():
    assert ARCHIVE.is_file(), "Historical task_cfg archive is missing"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_original_graph_and_exception_behavior(case, tmp_path, monkeypatch):
    archived = importlib.import_module("task.task_cfg_old")
    import matplotlib.pyplot as plt

    csv_path = tmp_path / "profiling.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "name", "Throuput factor (Spat.)", "Thread factor (Spat.)", "Freq.",
        ])
        writer.writerows(case["rows"])
    monkeypatch.chdir(tmp_path)
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                graphs = archived.creat_jobTask_graph(
                    case["graph"], case["gcd"], plot=case.get("plot", False),
                    profiling_filename=str(csv_path),
                )
                result = {"status": "returned", "graphs": [graph_state(g) for g in graphs]}
            except Exception as error:
                result = {
                    "status": "raised", "exception": type(error).__name__,
                    "message": str(error).replace(str(csv_path), "<csv>"),
                }
        result.update(stdout=stdout.getvalue(), stderr=stderr.getvalue())
        expected = {key: value for key, value in EXPECTED[case["name"]].items() if key != "pdf"}
        assert result == expected
        pdf = tmp_path / "jobTask_graph.pdf"
        assert pdf.exists() == EXPECTED[case["name"]]["pdf"]["exists"]
        if pdf.exists():
            assert pdf.read_bytes().startswith(b"%PDF-")
            assert pdf.stat().st_size > 1000
    finally:
        plt.close("all")


@pytest.mark.parametrize("fragment", REPORT["slices"], ids=lambda item: item["name"])
def test_archived_fragment_is_byte_identical(fragment):
    raw = ARCHIVE.read_bytes()
    lines = raw.splitlines(keepends=True)
    if fragment["name"] == "creat_jobTask_graph":
        node = next(
            node for node in ast.parse(raw).body
            if isinstance(node, ast.FunctionDef) and node.name == fragment["name"]
        )
        block = b"".join(lines[node.lineno - 1:node.end_lineno])
    else:
        start = next(
            index for index, line in enumerate(lines)
            if line.startswith(b"# def vis_task_static_timeline(")
        )
        block = b"".join(lines[start:start + fragment["line_count"]])
        assert b"# def vis_task_static_timeline(" not in (ROOT / "task/task_cfg.py").read_bytes()
    assert len(block.splitlines()) == fragment["line_count"]
    assert hashlib.sha256(block).hexdigest() == fragment["sha256"]
    assert raw.count(block) == 1


@pytest.mark.parametrize("archive_first", [False, True])
def test_live_exports_preserved_except_retired_name(tmp_path, archive_first):
    run_isolated(
        "import importlib, inspect, json, sys\n"
        "report = json.load(open(sys.argv[1], encoding='utf-8'))\n"
        "if sys.argv[2] == 'True': importlib.import_module('task.task_cfg_old')\n"
        "live = importlib.import_module('task.task_cfg')\n"
        "archive = importlib.import_module('task.task_cfg_old')\n"
        "expected = report['behavior_baseline']['public_exports'].copy()\n"
        "del expected['creat_jobTask_graph']\n"
        "assert {n for n in vars(live) if not n.startswith('_')} == set(expected)\n"
        "def describe(v):\n"
        "    if inspect.ismodule(v): return {'kind': 'module', 'name': v.__name__}\n"
        "    if inspect.isclass(v) or inspect.isfunction(v):\n"
        "        return {'kind': 'class' if inspect.isclass(v) else 'function', 'module': v.__module__, 'name': v.__qualname__}\n"
        "    if isinstance(v, (str, int, float, bool, type(None))): return {'kind': type(v).__name__, 'value': v}\n"
        "    return {'kind': type(v).__module__ + '.' + type(v).__qualname__}\n"
        "assert {n: describe(getattr(live, n)) for n in expected} == expected\n"
        "assert not hasattr(live, 'creat_jobTask_graph')\n"
        "assert archive.creat_jobTask_graph.__module__ == 'task.task_cfg_old'\n"
        "assert callable(live.creat_physical_graph) and callable(live.vis_task_static_timeline)\n",
        tmp_path, str(REPORT_PATH), str(archive_first),
    )


def test_archive_import_does_not_load_live_task_cfg(tmp_path):
    run_isolated(
        "import sys\n"
        "import task.task_cfg_old\n"
        "assert 'task.task_cfg' not in sys.modules\n",
        tmp_path,
    )


def test_original_signature_annotations_and_defaults_preserved():
    function = importlib.import_module("task.task_cfg_old").creat_jobTask_graph
    assert function.__annotations__ == {
        "task_graph": "Dict[str, List[str]]",
        "plot": "bool",
        "profiling_filename": "str",
    }
    assert function.__defaults__ == (False, "profiling/profiling.csv")
    assert function.__code__.co_varnames[:4] == (
        "task_graph", "f_gcd", "plot", "profiling_filename",
    )
