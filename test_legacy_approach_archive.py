"""Characterize the archived standalone approach simulator without fixing it."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "cleanup/reports/b12-legacy-approach-archive.json"


@pytest.fixture(scope="module")
def baseline():
    return json.loads(REPORT.read_text(encoding="utf-8"))


def run_isolated(args, tmp_path):
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        MPLBACKEND="Agg",
    )
    result = subprocess.run(
        [sys.executable, *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stderr == b""
    assert list(tmp_path.iterdir()) == []
    return result.stdout


def test_archive_layout_and_exact_source(baseline):
    for name in ("appoach_plot6.py", "approach_util33.py"):
        target = ROOT / "old" / name
        assert target.is_file(), f"Archive target is missing: {target}"
        assert not (ROOT / name).exists(), f"Old root entry still exists: {name}"
        content = target.read_bytes()
        if name == "appoach_plot6.py":
            new_import = b"from old.approach_util33 import Acc_p, Sen_p, MyGraph"
            old_import = b"from approach_util33 import Acc_p, Sen_p, MyGraph"
            assert content.count(new_import) == 1
            content = content.replace(new_import, old_import, 1)
        assert hashlib.sha256(content).hexdigest() == baseline["source_sha256"][name]


def test_archived_helper_import_is_silent(tmp_path):
    probe = """
import old.approach_util33 as helper
from example import bm4
assert helper.MyGraph.__module__ == "old.approach_util33"
assert helper.Acc_p.__module__ == "old.approach_util33"
assert helper.Sen_p.__module__ == "old.approach_util33"
graph = helper.MyGraph(
    bm4.task_graph_srcs, bm4.task_graph_ops, bm4.task_graph_sinks,
    bm4.task_attr, bm4.src_attr,
)
assert len(graph.nodes()) == 9
"""
    assert run_isolated(["-c", probe], tmp_path) == b""


@pytest.mark.parametrize("repeat", range(2))
@pytest.mark.parametrize("entry", ("module", "script"))
def test_archived_script_matches_independent_pre_move_output(
    entry, repeat, tmp_path, baseline
):
    args = (
        ["-m", "old.appoach_plot6"]
        if entry == "module"
        else [str(ROOT / "old/appoach_plot6.py")]
    )
    stdout = run_isolated(args, tmp_path)
    assert stdout == baseline["baseline"]["stdout"].encode("utf-8")
    assert hashlib.sha256(stdout).hexdigest() == baseline["baseline"]["stdout_sha256"]


def test_archived_simulation_final_state(tmp_path, baseline):
    probe = """
import json
import runpy
s = runpy.run_module("old.appoach_plot6", run_name="__main__")
print("FINAL_STATE=" + json.dumps({
    "nodes": list(s["G"].nodes()),
    "curr_t": s["curr_t"],
    "pred_t": s["pred_t"],
    "state": s["acc_p0"].sys_state,
    "running": s["acc_p0"].running,
    "res_map": s["acc_p0"].res_map,
}, sort_keys=True))
"""
    stdout = run_isolated(["-c", probe], tmp_path)
    transcript, state = stdout.rsplit(b"FINAL_STATE=", 1)
    assert transcript == baseline["baseline"]["stdout"].encode("utf-8")
    assert json.loads(state) == baseline["baseline"]["final_state"]
    assert hashlib.sha256(stdout).hexdigest() == baseline["baseline"]["stdout_with_final_state_sha256"]
