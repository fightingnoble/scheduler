"""Characterization for unused hardware-cost constants moved intact."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
NAMES = (
    "overhead_pushpull_per_core", "overhead_of_enqueuing_op",
    "overhead_of_dequeuing_op", "clock_period", "SRAM_size_per_core",
    "GLB_BUFFER_SIZE", "MIN_CORE_NUM", "W_perc", "A_perc", "O_perc",
)


@pytest.fixture(scope="module")
def baseline():
    return json.loads(
        (ROOT / "cleanup/reports/b15-unused-global-constants-baseline.json").read_text(
            encoding="utf-8"
        )
    )


def test_approved_archive_exists():
    assert (ROOT / "global_var_unused.py").is_file(), (
        "The approved global_var_unused archive has not been created"
    )


@pytest.mark.parametrize("name", NAMES)
def test_archived_values_and_types_match_pre_move_baseline(baseline, name):
    archive = importlib.import_module("global_var_unused")
    item = next(entry for entry in baseline["slices"] if entry["name"] == name)
    value = getattr(archive, name)
    assert type(value).__name__ == item["value_type"]
    assert repr(value) == item["value_repr"]


def test_original_assignment_bytes_are_preserved(baseline):
    raw = (ROOT / "global_var_unused.py").read_bytes()
    for entry in baseline["slices"]:
        original = entry["text"].encode("utf-8")
        assert hashlib.sha256(original).hexdigest() == entry["sha256"], entry["name"]
        assert raw.count(original) == 1, entry["name"]


def test_fresh_global_and_utils_exports_preserve_remaining_bindings(tmp_path, baseline):
    probe = """
import json
import sys
import types
import global_var
import utils
assert "h5py" not in sys.modules
import model.resource_agent as resource
before = json.loads(sys.argv[1])
retired = set(before["retired"])
for module, key in ((global_var, "global_var_public_names"), (utils, "utils_public_names"), (resource, "resource_public_names")):
    actual = {name for name in vars(module) if not name.startswith("_")}
    assert set(before[key]) - actual == retired
    assert not (retired & actual)
scope = {}
exec("from global_var import *", scope)
def describe(value):
    if isinstance(value, types.ModuleType):
        return {"type": "module", "module": value.__name__}
    if callable(value):
        return {"type": type(value).__name__, "module": value.__module__, "qualname": value.__qualname__}
    return {"type": type(value).__name__, "value_repr": repr(value)}
remaining = set(before["global_var_public_names"]) - retired
for name in remaining:
    assert scope[name] is getattr(global_var, name), name
    assert getattr(utils, name) is getattr(global_var, name), name
    assert getattr(resource, name) is getattr(global_var, name), name
    assert describe(getattr(global_var, name)) == before["global_value_descriptions"][name], name
assert "h5py" not in sys.modules
import global_var_unused
assert set(name for name in vars(global_var_unused) if not name.startswith("_")) == retired
assert "h5py" not in sys.modules
"""
    args = dict(baseline["namespace_baseline"], retired=list(NAMES))
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, json.dumps(args)],
        cwd=tmp_path, env=env, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == result.stderr == b""
    assert list(tmp_path.iterdir()) == []
