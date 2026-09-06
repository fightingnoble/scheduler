"""Characterization tests for the approved B24 mapper helper separation."""
import contextlib,io,json,inspect,pickle,base64
from mapper import mem_planner as m
def blocks(rows):
 return [m.Block(*x) for x in rows]
def state(mapper, contention, graph, sparse, work):
 return {'input_remaining':None if work is None else [x.idx for x in work],
 'contention':contention,'graph':None if graph is None else {str(k):sorted(x.idx for x in vs) for k,vs in sorted(graph.items())},
 'sparse':None if sparse is None else [[x.start,x.duatation,x.tot_size,[b.idx for b in x.block_list]] for x in sparse],
 'mapper':None if mapper is None else {'positions':mapper.to_json(),'free':[[x.offset,x.nextoffset,x.size] for x in mapper.free_interv],'max_offset':mapper.max_offset}}
OLD=[
 ('empty',[],1,False,'first_fit',False),
 ('single',[(2,1,3,1)],1,False,'first_fit',False),
 ('overlap',[(2,0,3,1),(3,0,2,2)],1,False,'first_fit',False),
 ('unsorted',[(3,2,5,2),(2,0,3,1),(1,3,4,3)],1,False,'first_fit',False),
 ('touching',[(2,0,1,1),(3,1,2,2)],1,False,'first_fit',False),
 ('zero_duration',[(2,1,1,1)],1,False,'first_fit',False),
 ('negative_time',[(2,-2,0,1)],1,False,'first_fit',False),
 ('zero_step',[(3,2,3,2),(2,0,1,1)],0,False,'first_fit',False),
 ('none_outputs',[(2,0,3,1),(3,1,2,2)],1,False,'first_fit',True),
 ('mapper_first',[(3,2,5,2),(2,0,3,1),(1,3,4,3)],1,True,'first_fit',False),
 ('mapper_best',[(3,2,5,2),(2,0,3,1),(1,3,4,3)],1,True,'best_fit',False),
 ('mapper_worst',[(3,2,5,2),(2,0,3,1),(1,3,4,3)],1,True,'worst_fit',False),
 ('invalid_blocks',None,1,False,'first_fit',False)]
SCAN=[('empty',[]),('single',[(2,3)]),('touching',[(0,2),(2,3)]),('disjoint',[(0,2),(4,1)]),('overlap',[(0,3),(2,2)]),('reversed',[(5,1),(0,2)]),('invalid',None)]
SORT=[('basic','block',[3,2,7,4],{}),('cyclic','cyclic',[3,2,7,4],{'pid':9}),('negative_pid','cyclic',[2,0,1,5],{'pid':-1}),('conflicts','block',[1,0,2,9],{'n_conflict':4}),('negative_values','block',[-2,-3,0,-4],{'n_conflict':-5}),('none','none',[],{}),('dict','dict',[],{})]
def old_case(fn,c):
 name,rows,step,use_mapper,strategy,no_outputs=c
 work=None if rows is None else blocks(rows)
 cont,graph,sparse=(None,None,None) if no_outputs else ({},{},[])
 mapper=m.MemMap() if use_mapper else None
 stdout=io.StringIO()
 with contextlib.redirect_stdout(stdout):
  try:ret={'return':fn(work,step,cont,graph,sparse,mapper,strategy=strategy)}
  except Exception as e:ret={'error':[type(e).__name__,str(e)]}
 ret.update(state(mapper,cont,graph,sparse,work));ret['stdout']=stdout.getvalue()
 return ret
def scan_case(fn,c):
 _,rows=c
 values=None if rows is None else {i:m.Memalloc(*x) for i,x in enumerate(rows)}
 if rows is None:values={0:None}
 try:return {'return':fn(values)}
 except Exception as e:return {'error':[type(e).__name__,str(e)]}
def sort_case(fn,c):
 _,kind,args,kwargs=c
 obj=m.Block(*args,**kwargs) if kind=='block' else m.CyclicBlock(*args,**kwargs) if kind=='cyclic' else {} if kind=='dict' else None
 try:return {'return':list(fn(obj))}
 except Exception as e:return {'error':[type(e).__name__,str(e)]}
def default_sequence(fn):
 before=fn.__defaults__;identity=[id(x) for x in before[:3]]
 out=[]
 for rows in [[(2,0,2,8)],[(3,1,3,9)]]:
  work=blocks(rows);buf=io.StringIO()
  with contextlib.redirect_stdout(buf):result=fn(work,1)
  out.append({'return':result,'stdout':buf.getvalue(),'state':state(None,*fn.__defaults__[:3],work)})
  out[-1]=json.loads(json.dumps(out[-1]))
 return {'initial_types':[type(x).__name__ for x in before],'same_defaults':[id(x)==y for x,y in zip(fn.__defaults__[:3],identity)],'calls':out}
def active_case(which):
 work=blocks([(3,2,5,2),(2,0,3,1),(1,3,4,3)])
 buf=io.StringIO()
 with contextlib.redirect_stdout(buf):
  if which=='scan_conflict':
   graph={};sparse=[];ret=m.scan_conflict(work,1,graph,sparse)
   return {'return':ret,'blocks':[[x.idx,x.n_conflict] for x in work],'graph':{str(k):sorted(x.idx for x in v) for k,v in sorted(graph.items())},'sparse':[[x.start,x.duatation,x.tot_size,[b.idx for b in x.block_list]] for x in sparse]}
  if which=='seq_mapper':
   mapper=m.MemMap();ret=mapper.seq_mapper(work,1)
   return {'positions':mapper.to_json(),'blocks':[[x.idx,x.n_conflict] for x in work],'legal':m.scan_overlap_2d(ret,work)}
  mapper=m.AllocMap();ret,graph=mapper.prority_mapper(work,1)
  return {'positions':mapper.to_json(),'blocks':[[x.idx,x.n_conflict] for x in work],'legal':m.scan_overlap_2d(ret,work),'graph':{str(k):sorted(x.idx for x in v) for k,v in sorted(graph.items())}}
def signature(fn):
 out=[]
 for name,p in inspect.signature(fn).parameters.items():
  d=p.default
  if d is inspect.Parameter.empty:d={'empty':True}
  elif callable(d):d={'callable_name':d.__name__,'sample':d(m.Block(3,0,1,1))}
  out.append({'name':name,'kind':p.kind.name,'default':d,'annotation':None if p.annotation is inspect.Parameter.empty else p.annotation})
 return {'parameters':out,'annotations':fn.__annotations__}

import ast
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parent
REPORT = json.loads((ROOT / "cleanup/reports/b24-mapper-legacy-helpers.json").read_text())
BASELINE = REPORT["behavior_baseline"]["values"]
assert hashlib.sha256(json.dumps(BASELINE, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == "ba3301af2937a8f3a1c05ab7faf08bff98b13dde194681ac18e7bf4849e7cbdb"
TARGETS = {
    "stat_overlapping_old": ("mapper.mem_planner_old", "4c3beab15f805a9002bd025d665096de56ff50163c7338aa4f33d17cba5d020d"),
    "scan_overlap_1d": ("mapper.mem_planner_unused", "1d39f4c26dffb6a6062c5fb4cb360f79b9bae451627efe021246603b18c68b56"),
    "sort_fn_conflict_s_r": ("mapper.mem_planner_unused", "736bfee8c9959b4c1f20c5121004e7f1b22c6fbcc16ace81338869070225afcd"),
}


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def archived(name):
    return getattr(importlib.import_module(TARGETS[name][0]), name)


def fresh(script):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_archive_targets_exist():
    assert (ROOT / "mapper/mem_planner_old.py").is_file()
    assert (ROOT / "mapper/mem_planner_unused.py").is_file()


@pytest.mark.parametrize("case", OLD, ids=[c[0] for c in OLD])
def test_old_stat_behavior(case):
    for _ in range(2):
        assert canonical(old_case(archived("stat_overlapping_old"), case)) == BASELINE["old"][case[0]]


@pytest.mark.parametrize("case", SCAN, ids=[c[0] for c in SCAN])
def test_scan_behavior(case):
    for _ in range(2):
        assert canonical(scan_case(archived("scan_overlap_1d"), case)) == BASELINE["scan"][case[0]]


@pytest.mark.parametrize("case", SORT, ids=[c[0] for c in SORT])
def test_sort_behavior(case):
    for _ in range(2):
        assert canonical(sort_case(archived("sort_fn_conflict_s_r"), case)) == BASELINE["sort"][case[0]]


def test_mutable_defaults_accumulate_in_fresh_processes():
    script = (
        "from test_mapper_legacy_helpers import default_sequence, canonical, BASELINE\n"
        "from mapper.mem_planner_old import stat_overlapping_old\n"
        "assert canonical(default_sequence(stat_overlapping_old)) == BASELINE['defaults']\n"
    )
    for _ in range(2):
        fresh(script)


@pytest.mark.parametrize("name", ["scan_conflict", "seq_mapper", "priority_mapper"])
def test_retained_mapping_paths(name):
    for _ in range(2):
        assert canonical(active_case(name)) == BASELINE["active"][name]


@pytest.mark.parametrize("name", list(TARGETS))
def test_original_contiguous_block(name):
    target = next(x for x in REPORT["targets"] if x["name"] == name)
    block = target["block"].encode()
    assert hashlib.sha256(block).hexdigest() == TARGETS[name][1]
    archive = (ROOT / target["archive_path"]).read_bytes()
    assert archive.count(block) == 1
    tree = ast.parse(archive)
    expected_names = ["stat_overlapping_old"] if name == "stat_overlapping_old" else ["scan_overlap_1d", "sort_fn_conflict_s_r"]
    assert [x.name for x in tree.body if isinstance(x, ast.FunctionDef)] == expected_names
    assert all(isinstance(x, (ast.Expr, ast.ImportFrom, ast.FunctionDef)) for x in tree.body)
    expected_imports = REPORT["archive_imports"][target["archive_path"]]
    assert [ast.get_source_segment(archive.decode(), x) for x in tree.body if isinstance(x, ast.ImportFrom)] == expected_imports


def test_remaining_source_bytes():
    content = (ROOT / "mapper/mem_planner.py").read_bytes()
    assert hashlib.sha256(content).hexdigest() == "482af52853a9ca50ccfef4d940ac6f07bee92b9e4f87e170000031e258e39d10"
    assert content.endswith(b"\n")


def test_complete_interface_in_both_fresh_import_orders():
    check = """
from typing import get_type_hints, get_args, Optional
from test_mapper_legacy_helpers import BASELINE, TARGETS, signature, canonical
assert sorted(n for n in vars(m) if not n.startswith('_')) == sorted(set(BASELINE['public_names']) - set(TARGETS))
for name, module in [('stat_overlapping_old', old), ('scan_overlap_1d', unused), ('sort_fn_conflict_s_r', unused)]:
    assert canonical(signature(getattr(module, name))) == BASELINE['signatures'][name]
assert old.Block is m.Block
assert old.ContentionGroup is m.ContentionGroup
assert old.MemMap is m.MemMap
assert unused.Block is m.Block
assert unused.CyclicBlock is m.CyclicBlock
hints = get_type_hints(old.stat_overlapping_old)
assert get_args(hints['block_list'])[0] is m.Block
assert get_args(hints['sparse_list'])[0] is m.ContentionGroup
assert get_args(get_args(hints['conflict_graph'])[1])[0] is m.Block
assert hints['mapper'] == Optional[m.MemMap]
assert get_type_hints(unused.sort_fn_conflict_s_r)['x'] is m.Block
"""
    for order in [
        "from mapper import mem_planner as m\nfrom mapper import mem_planner_old as old, mem_planner_unused as unused\n",
        "from mapper import mem_planner_old as old, mem_planner_unused as unused\nfrom mapper import mem_planner as m\n",
    ]:
        fresh(order + check)


@pytest.mark.parametrize("name", list(TARGETS))
def test_pickle_boundary(name):
    with pytest.raises(AttributeError):
        pickle.loads(base64.b64decode(BASELINE["pickle_original"][name]))
    function = archived(name)
    assert pickle.loads(pickle.dumps(function, protocol=4)) is function
