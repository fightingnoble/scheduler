"""Characterize the exact event-generator slices archived by B18."""

import ast
import base64
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
REPORT = json.loads(
    (ROOT / "cleanup/reports/b18-unused-e2e-event-generators-baseline.json")
    .read_text(encoding="utf-8")
)
LIVE = "model.event_gen.e2e_latency"
ARCHIVE = "model.event_gen.e2e_latency_unused"


def normalize(value):
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "value": value.tolist(),
        }
    if isinstance(value, np.generic):
        return {"type": type(value).__name__, "value": value.item()}
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "float", "value": str(value)}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    return {"type": type(value).__name__, "value": value}


def test_archive_exists():
    assert (ROOT / REPORT["archive_path"]).is_file(), "B18 archive is missing"


def observe_generator(case):
    module = importlib.import_module(ARCHIVE)
    generator = getattr(module, case["function"])(*case["args"], **case["kwargs"])
    observed = []
    try:
        for _ in range(case["limit"]):
            observed.append(normalize(next(generator)))
        ending = {"kind": "bounded"}
    except StopIteration as exc:
        ending = {"kind": "return", "value": normalize(exc.value)}
    except Exception as exc:
        ending = {
            "kind": "exception",
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        generator.close()
    return {"yields": observed, "ending": ending}


@pytest.mark.parametrize("case", REPORT["cases"], ids=lambda case: case["name"])
def test_original_generator_behavior(case):
    for _ in range(2):
        assert observe_generator(case) == case["expected"]


@pytest.mark.parametrize("fragment", REPORT["slices"], ids=lambda item: item["name"])
def test_exact_original_function_bytes(fragment):
    data = (ROOT / REPORT["archive_path"]).read_bytes()
    tree = ast.parse(data)
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == fragment["name"]
    ]
    assert len(definitions) == 1
    node = definitions[0]
    lines = data.splitlines(keepends=True)
    block = b"".join(lines[node.lineno - 1:node.end_lineno])
    assert node.end_lineno - node.lineno + 1 == fragment["lines"]
    assert hashlib.sha256(block).hexdigest() == fragment["sha256"]


@pytest.mark.parametrize("archive_first", [False, True])
def test_full_public_exports_in_fresh_process(archive_first):
    code = """
import importlib, json, pathlib, sys
report = json.loads(pathlib.Path(sys.argv[1]).read_text())
live_name = "model.event_gen.e2e_latency"
archive_name = live_name + "_unused"
order = [archive_name, live_name] if sys.argv[2] == "1" else [live_name, archive_name]
for name in order:
    importlib.import_module(name)
    if name == archive_name:
        assert live_name in sys.modules
live = sys.modules[live_name]
archive = sys.modules[archive_name]
retired = {item["name"] for item in report["slices"]}
actual = {
    name: {
        "type": type(value).__name__,
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", None),
    }
    for name, value in vars(live).items() if not name.startswith("_")
}
expected = {
    name: desc for name, desc in report["public_exports"].items()
    if name not in retired
}
assert actual == expected
assert retired.isdisjoint(vars(live))
assert all(callable(getattr(archive, name)) for name in retired)
assert archive.jitter_gen_biside is live.jitter_gen_biside
assert archive.e2e_var_sim.__globals__["jitter_gen_biside"] is live.jitter_gen_biside
print(json.dumps({"before": len(report["public_exports"]), "after": len(actual)}))
"""
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable, "-B", "-c", code,
            str(ROOT / "cleanup/reports/b18-unused-e2e-event-generators-baseline.json"),
            "1" if archive_first else "0",
        ],
        cwd=ROOT, env=env, text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout) == {"before": 81, "after": 79}


@pytest.mark.parametrize("name", sorted(REPORT["candidate_signatures"]))
def test_original_signature_and_generator_kind(name):
    function = getattr(importlib.import_module(ARCHIVE), name)
    assert str(inspect.signature(function)) == REPORT["candidate_signatures"][name]
    assert inspect.isgeneratorfunction(function)
    assert function.__module__ == ARCHIVE


def test_shared_jitter_helper_is_not_copied():
    live = importlib.import_module(LIVE)
    archive = importlib.import_module(ARCHIVE)
    assert archive.jitter_gen_biside is live.jitter_gen_biside
    tree = ast.parse((ROOT / REPORT["archive_path"]).read_bytes())
    assert {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    } == {item["name"] for item in REPORT["slices"]}


@pytest.mark.parametrize("name", sorted(REPORT["pickle_factory_baseline"]))
def test_old_function_reference_pickle_is_retired(name):
    importlib.import_module(ARCHIVE)
    payload = base64.b64decode(
        REPORT["pickle_factory_baseline"][name]["factory_pickle_base64"]
    )
    with pytest.raises(AttributeError, match=name):
        pickle.loads(payload)


@pytest.mark.parametrize("name", sorted(REPORT["pickle_factory_baseline"]))
def test_new_function_reference_pickle_roundtrip(name):
    function = getattr(importlib.import_module(ARCHIVE), name)
    assert pickle.loads(pickle.dumps(function)) is function
