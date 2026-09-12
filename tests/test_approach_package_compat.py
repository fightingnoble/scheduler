#!/usr/bin/env python3
"""package-only 契约测试（REQ-027/B28 重写）。

B28 依用户指令退役七个根 approach_*.py 兼容壳：
- 旧根模块导入必须失败（退役断言）
- 包导入身份稳定；公共符号可用
- 包路径 pickle 往返 identity；根模块名从 sys.modules 消失
"""
import importlib
import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULES = ["approach_Eq", "approach_def", "approach_sched",
           "approach_initiator", "approach_collector", "approach_sim", "approach_setup"]


def _pkg(mod):
    return importlib.import_module("approach." + mod)


def test_package_import_identity():
    for mod in MODULES:
        assert _pkg(mod) is _pkg(mod)


def test_public_symbols_available():
    from approach.approach_def import Acc_p, Sen_p, MyGraph, GlobalEvent_t
    from approach.approach_sched import PartitionConfig
    from approach.approach_sim import run_simulation
    from approach.approach_setup import setup_benchmark
    from approach.approach_collector import StatisticsCollector
    from approach.approach_Eq import set_time_unit, trasfer_realloc_as_task
    from approach.approach_initiator import instantiate_processors, get_partition_info
    assert callable(setup_benchmark) and callable(run_simulation)


def test_root_shells_retired():
    """七个根模块名在干净子进程中必须不再可导入。"""
    for module in MODULES:
        assert not (REPO_ROOT / f"{module}.py").exists()

    code = (
        "import importlib.util\n"
        f"modules = {MODULES!r}\n"
        "found = [name for name in modules if importlib.util.find_spec(name) is not None]\n"
        "if found:\n"
        "    raise SystemExit('ROOT_STILL_IMPORTABLE:' + ','.join(found))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stdout + r.stderr


def test_package_init_lazy():
    """__init__ 零主动导入——须在干净子进程验证（同进程导入子模块会 setattr 到父包）。"""
    code = "import approach; print([n for n in vars(approach) if not n.startswith('_')])"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(REPO_ROOT))
    assert r.returncode == 0 and r.stdout.strip() == "[]", r.stdout + r.stderr


def test_pickle_roundtrip_package_path():
    from approach.approach_Eq import set_time_unit
    from approach.approach_def import Acc_p
    assert pickle.loads(pickle.dumps(set_time_unit)) is set_time_unit
    assert pickle.loads(pickle.dumps(Acc_p)) is Acc_p


def test_no_root_alias_in_sysmodules_after_pkg_import():
    _pkg("approach_sim")
    for m in MODULES:
        assert m not in sys.modules, "根模块名 %s 不应被注册" % m


def test_entrypoints_help():
    for entry in (["main_approach.py"], ["-m", "scripts.motiv_exp_runner"],
                  ["-m", "scripts.abla_exp_runner"]):
        r = subprocess.run([sys.executable, *entry, "--help"], capture_output=True,
                           text=True, cwd=str(REPO_ROOT))
        assert r.returncode == 0, entry
