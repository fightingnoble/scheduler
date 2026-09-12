"""Characterization for two unused equation helpers moved intact."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads(
    (ROOT / "cleanup/reports/b16-unused-eq-quantiles-baseline.json").read_text(
        encoding="utf-8"
    )
)
ARCHIVE = "approach.approach_Eq_unused"


def test_approved_archive_exists():
    assert (ROOT / "approach/approach_Eq_unused.py").is_file(), (
        "The approved equation helper archive has not been created"
    )


@pytest.mark.parametrize("case", REPORT["behavior_baseline"]["values"]["cases"])
def test_archived_behavior_matches_pre_move(case):
    module = importlib.import_module(ARCHIVE)
    try:
        value = getattr(module, case["name"])(
            *[float(value) for value in case["float_args"]]
        )
        actual = {"type": type(value).__name__, "repr": repr(value)}
    except Exception as exc:
        actual = {"exception": type(exc).__name__, "message": str(exc)}
    assert actual == case["result"]


@pytest.mark.parametrize("fragment", REPORT["slices"], ids=lambda item: item["name"])
def test_complete_original_function_bytes_are_preserved(fragment):
    original = fragment["text"].encode("utf-8")
    archived = (ROOT / "approach/approach_Eq_unused.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == fragment["sha256"]
    assert archived.count(original) == 1


@pytest.mark.parametrize("package_first", [False, True])
def test_live_aliases_keep_exact_remaining_exports(tmp_path, package_first):
    probe = """
import importlib
import json
import sys
import types
names = ("approach.approach_Eq", "approach_Eq") if sys.argv[1] == "True" else ("approach_Eq", "approach.approach_Eq")
a, b = [importlib.import_module(name) for name in names]
assert a is b
assert "approach.approach_Eq_unused" not in sys.modules
before = json.loads(sys.argv[2])
retired = {"norm_inv_cdf", "exp_quantile"}
remaining = set(before) - retired
assert {name for name in vars(a) if not name.startswith("_")} == remaining
def describe(value):
    if isinstance(value, types.ModuleType):
        return {"type": "module", "module": value.__name__}
    if callable(value):
        return {"type": type(value).__name__, "module": getattr(value, "__module__", None), "qualname": getattr(value, "__qualname__", None)}
    return {"type": type(value).__name__, "value_repr": repr(value)}
for name in remaining:
    assert getattr(a, name) is getattr(b, name), name
    assert describe(getattr(a, name)) == before[name], name
from sched.ref_alloc_search import find_legal
assert a.find_legal is find_legal
assert a.set_time_unit(1e-6, False) == (1e-6, 1)
assert b.time_unit == 1e-6
assert b.set_time_unit(1e-6, True) == (1, 1e-6)
assert a.time_unit == 1
"""
    env = dict(
        os.environ, PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, str(package_first),
         json.dumps(REPORT["behavior_baseline"]["values"]["public_descriptors"])],
        cwd=tmp_path, env=env, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == result.stderr == b""
    assert list(tmp_path.iterdir()) == []


def test_archive_import_does_not_load_live_equations(tmp_path):
    probe = """
import sys
import approach.approach_Eq_unused as archive
assert "approach_Eq" not in sys.modules
assert "approach.approach_Eq" not in sys.modules
assert {n for n in vars(archive) if not n.startswith("_")} == {"math", "norm_inv_cdf", "exp_quantile"}
assert archive.norm_inv_cdf(0.5) == 0.0
assert archive.exp_quantile(0.5, 1.0) > 0
"""
    env = dict(
        os.environ, PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=tmp_path, env=env, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == result.stderr == b""
    assert list(tmp_path.iterdir()) == []
