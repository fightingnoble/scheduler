"""Characterization for the separated, unused path helpers."""

import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
NAMES = ("get_log_path_str", "_normalize_path", "check_paths_equal")
PATH_CASES = {
    "none": (None, None),
    "none_empty": (None, ""),
    "normalized": ("a/../b", "b"),
    "mismatch": ("old/a", "new/a"),
    "double_slash": ("//server//x", "//server/x"),
    "bytes": (b"a/../b", b"b"),
    "pathlib": (Path("a/../b"), Path("b")),
    "invalid": (123, "a"),
}


@pytest.fixture(scope="module")
def baseline():
    return json.loads(
        (ROOT / "cleanup/reports/b14-unused-path-utils-baseline.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def helpers():
    return importlib.import_module("utils_unused")


def test_archived_path_helpers_are_available(helpers):
    assert all(hasattr(helpers, name) for name in NAMES), (
        "The approved path helpers have not been moved to utils_unused"
    )


@pytest.mark.parametrize("name", NAMES)
def test_archived_slices_are_byte_identical(baseline, name):
    entry = next(item for item in baseline["slices"] if item["symbol"] == name)
    original = entry["text"].encode("utf-8")
    assert hashlib.sha256(original).hexdigest() == entry["sha256"]
    assert (ROOT / "utils_unused.py").read_bytes().count(original) == 1


def test_existing_hdf5_archive_prefix_is_preserved(baseline):
    content = (ROOT / "utils_unused.py").read_bytes()
    prefix = content[:baseline["archive_before_bytes"]]
    assert hashlib.sha256(prefix).hexdigest() == baseline["archive_before_sha256"]


@pytest.mark.parametrize("label", PATH_CASES)
def test_path_comparison_matches_pre_move_behavior(helpers, baseline, label):
    old, new = PATH_CASES[label]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            result = helpers.check_paths_equal(old, new, label)
            value = {"result": result}
        except Exception as exc:
            value = {"error": type(exc).__name__, "message": str(exc)}
    actual = {"label": label, **value, "stdout": output.getvalue()}
    expected = next(item for item in baseline["path_baseline"] if item["label"] == label)
    assert actual == expected


@pytest.mark.parametrize("root_dir", ("", ".", "case/sub", "/tmp/absolute", "../parent"))
def test_log_root_matches_pre_move_behavior(helpers, baseline, root_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = next(
        item["result"] for item in baseline["log_baseline"] if item["root_dir"] == root_dir
    )
    assert helpers.get_log_path_str(SimpleNamespace(root_dir=root_dir)) == expected
    assert list(tmp_path.iterdir()) == []


def test_fresh_live_utils_does_not_expose_unused_paths(tmp_path):
    probe = """
import sys
import utils
assert "h5py" not in sys.modules
for name in ("get_log_path_str", "_normalize_path", "check_paths_equal"):
    assert not hasattr(utils, name), name
for name in ("build_path_old", "get_cfg_n", "get_csv_path_str", "get_case_path_str"):
    assert callable(getattr(utils, name)), name
import utils_unused
assert callable(utils_unused.check_paths_equal)
assert utils_unused.check_parents_path is utils.check_parents_path
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=tmp_path, env=env, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == result.stderr == b""
    assert list(tmp_path.iterdir()) == []
