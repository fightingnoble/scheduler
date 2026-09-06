"""Characterize B19's archived helpers without changing their historical behavior."""

import base64
import copy
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parent
REPORT = json.loads((ROOT / "cleanup/reports/b19-unused-slack-graph-helpers-baseline.json").read_text())
SPECS = REPORT["sources"]
FUNCTIONS = [
    (path, name)
    for path, spec in SPECS.items()
    for name in spec["names"]
]
MODULES = {
    "score": ("sched.slack_estim_unused", "build_score_dict_ref_flops"),
    "chains": ("sched.slack_estim_unused", "get_chains_info"),
    "sort": ("task.graph_breakdown_unused", "sort_chains_by_ddl_flops"),
}

def normalize(v):
    if isinstance(v,dict):return {'type':'dict','items':[[normalize(k),normalize(x)] for k,x in v.items()]}
    if isinstance(v,(tuple,list)):return {'type':type(v).__name__,'items':[normalize(x) for x in v]}
    return {'type':type(v).__name__,'value':v}
def observe(case,func):
    import networkx as nx
    kind=case['kind'];c=copy.deepcopy(case)
    if kind=='score':
        tasks={k:types.SimpleNamespace(**v) for k,v in c['tasks'].items()}
        args=[tasks,c['nodes'],c['scores']]
        def state():return normalize({'nodes':c['nodes'],'scores':c['scores'],'tasks':{k:vars(v) for k,v in tasks.items()}})
    elif kind=='chains':
        graph=nx.DiGraph();graph.add_edges_from(c['edges'])
        for n,a in c['nodes']:graph.add_node(n,**a)
        args=[graph,c['starts'],c['ends']]
        def state():return normalize({'nodes':list(graph.nodes(data=True)),'edges':list(graph.edges(data=True)),'starts':c['starts'],'ends':c['ends']})
    else:
        args=[c['chains'],c['flops'],c['ddls']]
        def state():return normalize(args)
    before=state()
    try:result={'kind':'return','value':normalize(func(*args))}
    except Exception as e:result={'kind':'exception','type':type(e).__name__,'message':str(e)}
    return {'result':result,'before':before,'after':state()}

def archive_module(spec):
    return importlib.import_module(spec["archive"][:-3].replace("/", "."))


def test_archive_files_exist():
    assert all((ROOT / spec["archive"]).is_file() for spec in SPECS.values())


@pytest.mark.parametrize("case", REPORT["cases"], ids=lambda c: c["name"])
def test_original_behavior(case):
    module, name = MODULES[case["kind"]]
    func = getattr(importlib.import_module(module), name)
    assert observe(case, func) == case["expected"]
    assert observe(case, func) == case["expected"]


@pytest.mark.parametrize("path,name", FUNCTIONS)
def test_original_block_is_contiguous(path, name):
    spec = SPECS[path]
    block = next(s["text"] for s in spec["slices"] if s["name"] == name).encode()
    assert (ROOT / spec["archive"]).read_bytes().count(block) == 1


@pytest.mark.parametrize("path,name", FUNCTIONS)
def test_original_signature(path, name):
    spec = SPECS[path]
    assert str(inspect.signature(getattr(archive_module(spec), name))) == spec["signatures"][name]


@pytest.mark.parametrize("archive_first", [False, True])
def test_surviving_public_exports_in_fresh_process(archive_first):
    probe = """
import importlib,json,pathlib,sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
specs=report['sources']
live=[s['module'] for s in specs.values()]
archives=[s['archive'][:-3].replace('/','.') for s in specs.values()]
order=archives+live if sys.argv[2]=='True' else live+archives
for name in order:importlib.import_module(name)
def descriptor(v):
    return {'type':type(v).__name__,'module':getattr(v,'__module__',None),'qualname':getattr(v,'__qualname__',None)}
for spec in specs.values():
    module=importlib.import_module(spec['module'])
    actual={k:descriptor(v) for k,v in vars(module).items() if not k.startswith('_')}
    expected={k:v for k,v in spec['public_exports'].items() if k not in spec['names']}
    assert actual==expected,(spec['module'],set(actual)^set(expected))
"""
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    run = subprocess.run(
        [sys.executable, "-B", "-c", probe,
         str(ROOT / "cleanup/reports/b19-unused-slack-graph-helpers-baseline.json"),
         str(archive_first)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("path,name", FUNCTIONS)
def test_original_function_pickle_path_is_retired(path, name):
    payload = base64.b64decode(SPECS[path]["pickle_original"][name])
    with pytest.raises(AttributeError, match=name):
        pickle.loads(payload)


@pytest.mark.parametrize("path,name", FUNCTIONS)
def test_new_function_pickle_identity(path, name):
    func = getattr(archive_module(SPECS[path]), name)
    assert pickle.loads(pickle.dumps(func)) is func


def test_archived_chain_helper_uses_live_graph_decomposition():
    archive = importlib.import_module("sched.slack_estim_unused")
    live = importlib.import_module("task.graph_breakdown")
    assert archive.get_chains_info.__globals__["decompose_dag_into_chains"] is live.decompose_dag_into_chains
