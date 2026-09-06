"""Characterize the unchanged experiment helpers preserved in B23."""

import base64
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "cleanup/reports/b23-unused-experiment-helpers.json"
REPORT = json.loads(REPORT_PATH.read_text())
TARGETS = REPORT["targets"]
BASELINE = REPORT["behavior_baseline"]["functions"]
INPUTS = {
    "round_to_step": [
        (0, 1), (5, 2), (7, 2), (-5, 2), (5, -2), (1.5, 1),
        (1, 0), ("7", 2), (True, 2), (float("inf"), 1), (float("nan"), 1),
    ],
    "extract_num_cores": [
        ("cores12_seq34",), ("no-digits",), ("",), ("-20",), ("1.5",),
        ("\uff11\uff12",), ("core0007",), (None,), (b"12",), (Path("12"),),
    ],
}
CASES = [
    (target, args, sample)
    for target in TARGETS
    for args, sample in zip(
        INPUTS[target["name"]], BASELINE[target["module"]]["samples"]
    )
]


def test_archive_targets_exist():
    assert all((ROOT / target["archive_path"]).is_file() for target in TARGETS)


@pytest.mark.parametrize("target", TARGETS, ids=lambda target: target["name"])
def test_original_function_bytes_preserved(target):
    original = target["block"].encode()
    assert hashlib.sha256(original).hexdigest() == target["block_sha256"]
    assert (ROOT / target["archive_path"]).read_bytes().count(original) == 1


@pytest.mark.parametrize("target", TARGETS, ids=lambda target: target["name"])
def test_remaining_source_bytes(target):
    actual = (ROOT / target["path"]).read_bytes()
    assert hashlib.sha256(actual).hexdigest() == target["remaining_sha256"]
    assert actual.endswith(b"\n") == target["source_eof_lf"]


@pytest.mark.parametrize(
    "target,args,sample", CASES,
    ids=[target["name"] + "-" + str(i) for i, (target, _, _) in enumerate(CASES)],
)
def test_original_behavior(target, args, sample):
    assert repr(args) == sample["args_repr"]
    function = getattr(importlib.import_module(target["archive_module"]), target["name"])
    for _ in range(2):
        try:
            value = function(*args)
            actual = {"value_repr": repr(value), "type": type(value).__name__}
        except Exception as exc:
            actual = {"exception": type(exc).__name__, "message": str(exc)}
        assert actual == sample["result"]


@pytest.mark.parametrize("target", TARGETS, ids=lambda target: target["name"])
def test_interface_surface_in_fresh_processes(target):
    code = """
import importlib,inspect,json,pathlib,sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
target=next(t for t in report["targets"] if t["name"]==sys.argv[2])
order=[target["module"],target["archive_module"]]
if sys.argv[3]=="archive-first":order.reverse()
for name in order:importlib.import_module(name)
live=importlib.import_module(target["module"])
archive=importlib.import_module(target["archive_module"])
expected=report["behavior_baseline"]["functions"][target["module"]]
public=sorted(k for k in vars(live) if not k.startswith("_"))
assert public==sorted(set(expected["public_names"])-{target["name"]}),public
function=getattr(archive,target["name"])
assert str(inspect.signature(function))==expected["signature"]
assert {k:repr(v) for k,v in function.__annotations__.items()}==expected["annotations"]
assert repr(function.__defaults__)==expected["defaults"]
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    for order in ("live-first", "archive-first"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(REPORT_PATH), target["name"], order],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("target", TARGETS, ids=lambda target: target["name"])
def test_pickle_boundary(target):
    original = BASELINE[target["module"]]["pickle_b64"]
    with pytest.raises(AttributeError, match=target["name"]):
        pickle.loads(base64.b64decode(original))
    function = getattr(importlib.import_module(target["archive_module"]), target["name"])
    assert pickle.loads(pickle.dumps(function, protocol=4)) is function


def test_borrowed_num_exec_identity():
    from analyze.analyze_timing import extract_num_exec as timing
    from analyze.analyze_tp import extract_num_exec as throughput
    from analyze.stat_num_exec import extract_num_exec
    assert timing is throughput is extract_num_exec
