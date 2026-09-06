"""Preserve the behavior of the complete cache helpers archived in B20."""

import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import pathlib
import pickle
import re
import subprocess
import sys
import tempfile
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent
REPORT = json.loads((ROOT / "cleanup/reports/b20-unused-sim-cache-helpers-baseline.json").read_text())
NAMES = [s["name"] for s in REPORT["slices"]]

def value(v):
    if isinstance(v,re.Match):
        return {'type':'Match','pattern':v.re.pattern,'groups':list(v.groups()),'span':list(v.span()),'text':v.group(0)}
    if isinstance(v,(list,tuple)):
        return {'type':type(v).__name__,'items':[value(x) for x in v]}
    if isinstance(v,dict):
        return {'type':'dict','items':[[value(k),value(x)] for k,x in v.items()]}
    return {'type':type(v).__name__,'value':v}

def observe(case,module):
    from paths import PathContext
    params=dict(root_dir='b20',case='bin_pack_new',num_bins=2,aux_scale_factor=1,e2e_latency=0.1,
                file_suffix='',i_file_suffix='',force_suffix='',exec_t_comp_ratioA=0.7,
                lateness_mode='all_hard',num_cores=case.get('context_cores',8),exec_t_comp_ratioB=-1,
                seed=42,jitter=False)
    previous=pathlib.Path.cwd()
    with tempfile.TemporaryDirectory(prefix='scheduler-b20-case-') as directory:
        try:
            os.chdir(directory)
            ctx=None;trace={'keep':1};plot={'keep':2};stdout=io.StringIO()
            if case['kind']=='csv':
                path=case.get('path','limits.csv')
                if 'existing' in case:pathlib.Path(path).write_text(case['existing'])
                args=(path,case['cfg'],case['scan'])
                name='ensure_csv'
            else:
                ctx=PathContext(**params)
                if case['kind']=='max':
                    for e2e,aux,count in case['cached']:
                        settings=dict(params,aux_scale_factor=aux,e2e_latency=e2e,num_cores=case['cores'])
                        fixture=PathContext(**settings)
                        pathlib.Path(fixture.get_bin_list_path()).write_bytes(pickle.dumps(list(range(count))))
                    args=(types.SimpleNamespace(e2e_var_sim_para={'event_list':case['events']}),case['cores'],
                          case.get('format','./cache/b20/x{0}_{1}s_rda-70.00%(S)_all_hard/bin_list_{2}.pkl'),ctx)
                    name='check_max_bin_num'
                else:
                    for file in case.get('files',[]):
                        path=pathlib.Path(ctx.cache_root)/file['name']
                        data=pickle.dumps(file['pickle']) if 'pickle' in file else file.get('text','').encode()
                        path.write_bytes(data)
                    if case.get('subdir'):(pathlib.Path(ctx.cache_root)/'nested').mkdir()
                    if case.get('remove_folder'):pathlib.Path(ctx.cache_root).rmdir()
                    path_params={'root_dir':ctx.root_dir,'cfg_n':ctx.cfg_n,'force_suffix':ctx.force_suffix,'i_file_suffix':ctx.i_file_suffix}
                    if case['kind']=='trace':
                        name='get_core_num_from_trace_name';args=(path_params,ctx)
                    else:
                        name='prepare_induced_env_if_needed';args=(path_params,ctx,trace,plot)
            def state():
                return {'ctx':vars(ctx).copy() if ctx is not None else None,'trace':dict(trace),'plot':dict(plot),
                        'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pathlib.Path('.').rglob('*')) if p.is_file()},
                        'csv_text':pathlib.Path(args[0]).read_text() if case['kind']=='csv' and pathlib.Path(args[0]).is_file() else None}
            before=state()
            with contextlib.redirect_stdout(stdout):
                try:result={'kind':'return','value':value(getattr(module,name)(*args))}
                except Exception as exc:result={'kind':'exception','type':type(exc).__name__,'message':str(exc)}
            return {'result':result,'stdout':stdout.getvalue(),'before':before,'after':state()}
        finally:
            os.chdir(previous)

def archive():
    return importlib.import_module("sim_main_unused")


def test_archive_file_exists():
    assert (ROOT / REPORT["archive_path"]).is_file()


@pytest.mark.parametrize("case", REPORT["cases"], ids=lambda case: case["name"])
def test_original_file_io_and_state(case):
    assert observe(case, archive()) == case["expected"]
    assert observe(case, archive()) == case["expected"]


@pytest.mark.parametrize("name", NAMES)
def test_original_block_is_contiguous(name):
    block = next(s["text"] for s in REPORT["slices"] if s["name"] == name).encode()
    assert (ROOT / REPORT["archive_path"]).read_bytes().count(block) == 1


@pytest.mark.parametrize("name", NAMES)
def test_original_signature(name):
    assert str(inspect.signature(getattr(archive(), name))) == REPORT["signatures"][name]


@pytest.mark.parametrize("name", NAMES)
def test_original_function_pickle_path_is_retired(name):
    with pytest.raises(AttributeError, match=name):
        pickle.loads(base64.b64decode(REPORT["pickle_original"][name]))


@pytest.mark.parametrize("name", NAMES)
def test_new_function_pickle_identity(name):
    func = getattr(archive(), name)
    assert pickle.loads(pickle.dumps(func)) is func


@pytest.mark.parametrize("archive_first", [False, True])
def test_complete_public_exports_in_fresh_process(archive_first):
    code = """
import importlib,json,pathlib,sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
order=['sim_main_unused','sim_main'] if sys.argv[2]=='True' else ['sim_main','sim_main_unused']
for name in order:importlib.import_module(name)
module=importlib.import_module('sim_main')
def descriptor(v):
    return {'type':type(v).__name__,'module':getattr(v,'__module__',None),'qualname':getattr(v,'__qualname__',None)}
names={s['name'] for s in report['slices']}
actual={k:descriptor(v) for k,v in vars(module).items() if not k.startswith('_')}
expected={k:v for k,v in report['public_exports'].items() if k not in names}
assert actual==expected,(set(actual)^set(expected))
"""
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code,
         str(ROOT / "cleanup/reports/b20-unused-sim-cache-helpers-baseline.json"),
         str(archive_first)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_and_shared_live_dependencies():
    module = archive()
    live = importlib.import_module("sim_main")
    utils = importlib.import_module("utils")
    assert module.prepare_induced_env_if_needed.__globals__["get_core_num_from_trace_name"] is module.get_core_num_from_trace_name
    assert module.compare_paths is live.compare_paths
    assert module.generate_bin_paths is live.generate_bin_paths
    assert module.load_pickle is utils.load_pickle
    assert module.PathContext is live.PathContext
