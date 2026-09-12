"""Import/state/serialization contracts for the approach package migration."""

import base64
import importlib
import json
import os
from pathlib import Path
import pickle
import runpy
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
REPORT = json.loads(
    (ROOT / "cleanup/reports/b11-approach-package-compat.json").read_text(
        encoding="utf-8"
    )
)
MODULES = tuple(path[:-3] for path in REPORT["source_sha256"])


def run_isolated(code, tmp_path, *args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_import_does_not_load_submodules(tmp_path):
    run_isolated(
        "import importlib.util, sys\n"
        "assert importlib.util.find_spec('approach') is not None, "
        "'approach package is missing'\n"
        "import approach\n"
        "assert not any(n.startswith('approach.') for n in sys.modules)\n",
        tmp_path,
    )


@pytest.mark.parametrize("first_module", MODULES)
@pytest.mark.parametrize("package_first", [False, True])
def test_import_orders_share_modules_and_symbols(tmp_path, first_module, package_first):
    run_isolated(
        "import importlib, inspect, json, sys\n"
        "names = json.loads(sys.argv[1])\n"
        "first = sys.argv[2]\n"
        "prefixes = ('approach.', '') if sys.argv[3] == 'True' else ('', 'approach.')\n"
        "for name in [first] + [n for n in names if n != first]:\n"
        "    a, b = [importlib.import_module(p + name) for p in prefixes]\n"
        "    assert a is b, name\n"
        "    for symbol, value in vars(a).items():\n"
        "        if inspect.isclass(value) or inspect.isfunction(value):\n"
        "            assert getattr(b, symbol) is value, (name, symbol)\n",
        tmp_path, json.dumps(MODULES), first_module, str(package_first),
    )


@pytest.mark.parametrize("variable,setter", [
    ("VERBOSE_OUTPUT", "set_verbose_output"),
    ("REALLOC_DISABLED", "set_realloc_disabled"),
    ("MISS_DISABLED", "set_miss_disabled"),
    ("DROP_DISABLED", "set_drop_disabled"),
])
def test_control_setters_and_direct_writes_share_state(monkeypatch, variable, setter):
    root = importlib.import_module("approach_def")
    package = importlib.import_module("approach.approach_def")
    monkeypatch.setattr(root, variable, False)
    getattr(root, setter)(True)
    assert getattr(package, variable) is True
    setattr(package, variable, False)
    assert getattr(root, variable) is False
    getattr(package, setter)(True)
    assert getattr(root, variable) is True


def test_time_unit_setter_and_writes_share_state(monkeypatch):
    root = importlib.import_module("approach_Eq")
    package = importlib.import_module("approach.approach_Eq")
    monkeypatch.setattr(root, "time_unit", 1)
    assert root.set_time_unit(1e-6, False) == (1e-6, 1)
    assert package.time_unit == 1e-6
    package.time_unit = 0.25
    assert root.time_unit == 0.25
    assert package.set_time_unit(1e-6, True) == (1, 1e-6)
    assert root.time_unit == 1


def test_root_monkeypatch_reaches_real_setup_pipeline(monkeypatch, tmp_path):
    root = importlib.import_module("approach_setup")
    package = importlib.import_module("approach.approach_setup")
    args = SimpleNamespace(
        exec_t_comp_ratioA=0.7, exec_t_comp_ratioB=-1, num_cores=None
    )
    path_params = tuple(object() for _ in range(9))
    context = SimpleNamespace(
        graph_fn=str(tmp_path / "graph.json"),
        get_log_path=lambda: str(tmp_path / "pipeline.log"),
    )
    workload = (1.0, {}, object(), [], {})
    calls = []

    class BoundaryReached(Exception):
        pass

    def recording_init(*init_args):
        calls.append(init_args)
        raise BoundaryReached

    # Isolate setup inputs; execute the real setup and pipeline up to the boundary.
    monkeypatch.setattr(root, "preprocess_args", lambda args: None)
    monkeypatch.setattr(root, "build_paths_and_ctx", lambda args: (path_params, context))
    monkeypatch.setattr(root, "build_workload_and_criticality", lambda *a, **kw: workload)
    monkeypatch.setattr(root, "export_json_graph_utils", lambda *a: None)
    monkeypatch.setattr(root, "init_sched_components", recording_init)
    with pytest.raises(BoundaryReached):
        package.setup_benchmark(args, 1)
    assert len(calls) == 1
    assert calls[0][:5] == (args, path_params, context, workload, None)
    assert calls[0][5] == []
    assert package.init_sched_components is recording_init


@pytest.mark.parametrize("fixture", REPORT["legacy_pickles"],
                         ids=lambda item: item["module"] + "." + item["symbol"])
def test_pre_move_pickle_loads_and_new_pickle_roundtrips(fixture):
    payload = base64.b64decode(fixture["base64"])
    expected = getattr(
        importlib.import_module("approach." + fixture["module"]), fixture["symbol"]
    )
    restored = pickle.loads(payload)
    if fixture["kind"] == "reference":
        assert restored is expected
        assert pickle.loads(pickle.dumps(restored, protocol=4)) is expected
    else:
        assert type(restored) is expected
        roundtrip = pickle.loads(pickle.dumps(restored, protocol=4))
        assert type(roundtrip) is expected
        for name, value in fixture["state"].items():
            assert json.loads(json.dumps(getattr(restored, name))) == value
            assert getattr(roundtrip, name) == getattr(restored, name)
    assert expected.__module__ == "approach." + fixture["module"]


@pytest.mark.parametrize("name", ["approach_sim", "approach_collector"])
def test_script_entry_forwards_without_aliasing_main(monkeypatch, name):
    calls = []
    original_main = sys.modules["__main__"]

    def recording_run_module(module, *, run_name, alter_sys):
        calls.append((module, run_name, alter_sys))
        assert Path(sys.modules["__main__"].__file__).resolve() == ROOT / (name + ".py")
        return {}

    monkeypatch.setattr(runpy, "run_module", recording_run_module)
    runpy.run_path(str(ROOT / (name + ".py")), run_name="__main__")
    assert calls == [("approach." + name, "__main__", True)]
    assert sys.modules["__main__"] is original_main
