"""Characterization for the separated, unused HDF5 trace helpers."""

import contextlib
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


@pytest.fixture(scope="module")
def baseline():
    return json.loads(
        (ROOT / "cleanup/reports/b13-utils-hdf5-baseline.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def helpers():
    return importlib.import_module("utils_unused")


def test_archived_cluster_is_byte_identical(baseline):
    path = ROOT / "utils_unused.py"
    assert path.is_file(), "The approved utils_unused.py target is missing"
    original = baseline["removed_cluster"].encode("utf-8")
    assert hashlib.sha256(original).hexdigest() == baseline["removed_cluster_sha256"]
    assert path.read_bytes().count(original) == 1


def test_fresh_utils_import_does_not_load_hdf5(tmp_path):
    probe = """
import sys
import utils
assert "h5py" not in sys.modules
for name in ("CHUNK_SIZE", "get_next_chunk_id", "save_chunk", "load_h5_file"):
    assert not hasattr(utils, name), name
import utils_unused
assert "h5py" in sys.modules
assert utils_unused.check_parents_path is utils.check_parents_path
"""
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == result.stderr == b""
    assert list(tmp_path.iterdir()) == []


def test_real_hdf5_sequence_matches_pre_move_baseline(helpers, baseline, tmp_path):
    import h5py

    path = tmp_path / "nested/trace.h5"
    data = [{"pid": i, "kind": "trace"} for i in range(24)]
    helpers.save_chunk(str(path), data)
    short = {
        "file_exists": path.exists(),
        "retained_count": len(data),
        "parent_exists": path.parent.exists(),
    }
    data.append({"pid": 24, "kind": "trace"})
    helpers.save_chunk(str(path), data)
    first = {
        "remaining_buffer": list(data),
        "records": helpers.load_h5_file(str(path)),
    }
    with h5py.File(path, "r") as handle:
        next_id = helpers.get_next_chunk_id(handle)
    data.extend([{"pid": 25, "kind": "tail"}, {"pid": 26, "kind": "tail"}])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        helpers.save_chunk(str(path), data, True)
    forced = {
        "remaining_buffer": list(data),
        "records": helpers.load_h5_file(str(path)),
        "stdout": output.getvalue().replace(str(path), "<FILE>"),
    }
    with h5py.File(path, "r") as handle:
        keys = sorted(handle.keys())
        compression = {key: handle[key].compression for key in keys}
    with h5py.File(tmp_path / "empty.h5", "w") as handle:
        empty_next = helpers.get_next_chunk_id(handle)
        handle.create_group("unrelated")
        with pytest.raises(IndexError) as exc:
            helpers.get_next_chunk_id(handle)
    actual = {
        "chunk_size": helpers.CHUNK_SIZE,
        "short_buffer": short,
        "first_flush": first,
        "next_id_after_first_flush": next_id,
        "forced_flush": forced,
        "dataset_keys": keys,
        "compression": compression,
        "empty_next_id": empty_next,
        "malformed_key_exception": type(exc.value).__name__,
    }
    expected = dict(baseline["behavior_baseline"])
    expected.pop("source_sha256")
    assert actual == expected


@pytest.mark.parametrize("length", (0, 1, 24))
def test_short_buffer_retains_contents_and_only_creates_parent(
    helpers, tmp_path, length
):
    path = tmp_path / "short/trace.h5"
    data = list(range(length))
    helpers.save_chunk(str(path), data)
    assert data == list(range(length))
    assert path.parent.is_dir()
    assert not path.exists()


def test_malformed_group_key_retains_original_index_error(helpers, tmp_path):
    import h5py

    with h5py.File(tmp_path / "malformed.h5", "w") as handle:
        handle.create_group("unrelated")
        with pytest.raises(IndexError):
            helpers.get_next_chunk_id(handle)


@pytest.mark.parametrize("error", (OSError, KeyError))
def test_existing_supported_key_errors_return_zero(helpers, error):
    class UnavailableKeys:
        def keys(self):
            raise error("unavailable")

    assert helpers.get_next_chunk_id(UnavailableKeys()) == 0
