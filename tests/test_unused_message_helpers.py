"""Characterize the unchanged message helpers preserved in B21."""

import base64
import contextlib
import importlib
import inspect
import io
import json
import os
import pathlib
import pickle
import subprocess
import sys
from queue import Queue

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "cleanup/reports/b21-unused-message-helpers-baseline.json"
REPORT = json.loads(REPORT_PATH.read_text())
LIVE = "model.message.msg_dispatcher"
ARCHIVE = "model.message.msg_dispatcher_unused"
NAMES = [s["name"] for s in REPORT["slices"]]

def observe(case, module):
    from queue import Queue
    from model.buffer import Buffer
    from types import SimpleNamespace
    import contextlib,copy,io
    messages=copy.deepcopy(case['messages'])
    stdout=io.StringIO()
    if case['kind']=='filter':
        before=copy.deepcopy(messages)
        with contextlib.redirect_stdout(stdout):
            try:
                result=module.msg_filter(messages,case['keyword'])
                outcome={'kind':'return','value':result}
                identities=[any(item is original for original in messages) for item in result]
            except Exception as exc:
                outcome={'kind':'exception','type':type(exc).__name__,'message':str(exc)}
                identities=None
        return {'result':outcome,'stdout':stdout.getvalue(),'before':before,'after':messages,'identity_membership':identities}
    queue=Queue()
    for message in messages:queue.put(message)
    buffer=Buffer(case.get('capacity',-1))
    preds=case.get('preds',['source'])
    process_dict={i:SimpleNamespace(task=SimpleNamespace(name='sink'+str(i)),pred_data={key:{'valid':False,'time':-1,'other':7} for key in preds}) for i in range(case.get('process_count',1))}
    sources={key:SimpleNamespace(pid=pid,io_time=3,task=SimpleNamespace(freq=case.get('freq',10),period=0.1)) for key,pid in [('source',10),('s',11)]}
    def state():
        return {'queue':list(queue.queue),'pred_data':{str(i):copy.deepcopy(p.pred_data) for i,p in process_dict.items()},
                'capacity':buffer.capacity,'remain_cap':buffer.remain_cap,
                'outputs':{str(pid):[vars(data).copy() for data in items] for pid,items in buffer.buffer_o.items()}}
    before=state()
    with contextlib.redirect_stdout(stdout):
        try:
            outcome={'kind':'return','value':module.msg_read(queue,0.25,sources,process_dict,buffer,case.get('bin_name','B0'),case.get('flag',False))}
        except Exception as exc:
            outcome={'kind':'exception','type':type(exc).__name__,'message':str(exc)}
    return {'result':outcome,'stdout':stdout.getvalue(),'before':before,'after':state()}


def archive():
    return importlib.import_module(ARCHIVE)


def test_archive_file_exists():
    assert (ROOT / REPORT["archive_path"]).is_file()


@pytest.mark.parametrize("case", REPORT["cases"], ids=lambda case: case["name"])
def test_original_queue_buffer_and_state(case):
    assert json.loads(json.dumps(observe(case, archive()))) == case["expected"]
    assert json.loads(json.dumps(observe(case, archive()))) == case["expected"]


@pytest.mark.parametrize("name", NAMES)
def test_original_block_is_contiguous(name):
    block = next(s["text"] for s in REPORT["slices"] if s["name"] == name).encode()
    assert (ROOT / REPORT["archive_path"]).read_bytes().count(block) == 1


@pytest.mark.parametrize("name", NAMES)
def test_original_signature(name):
    assert str(inspect.signature(getattr(archive(), name))) == REPORT["signatures"][name]


@pytest.mark.parametrize("name", NAMES)
def test_old_function_pickle_path_is_retired(name):
    with pytest.raises(AttributeError, match=name):
        pickle.loads(base64.b64decode(REPORT["pickle_original"][name]))


@pytest.mark.parametrize("name", NAMES)
def test_new_function_pickle_identity(name):
    function = getattr(archive(), name)
    assert pickle.loads(pickle.dumps(function)) is function


@pytest.mark.parametrize("archive_first", [False, True])
def test_complete_public_exports_in_fresh_process(archive_first):
    code = """
import importlib,json,pathlib,sys
report=json.loads(pathlib.Path(sys.argv[1]).read_text())
live='model.message.msg_dispatcher'
archive='model.message.msg_dispatcher_unused'
for name in ([archive,live] if sys.argv[2]=='True' else [live,archive]):
    importlib.import_module(name)
module=importlib.import_module(live)
def descriptor(v):
    return {'type':type(v).__name__,'module':getattr(v,'__module__',None),'qualname':getattr(v,'__qualname__',None)}
retired={s['name'] for s in report['slices']}
actual={k:descriptor(v) for k,v in vars(module).items() if not k.startswith('_')}
expected={k:v for k,v in report['public_exports'].items() if k not in retired}
assert actual==expected,(actual,expected)
"""
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(REPORT_PATH), str(archive_first)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_internal_and_data_dependencies():
    module = archive()
    live = importlib.import_module(LIVE)
    assert module.msg_read.__globals__["msg_filter"] is module.msg_filter
    assert module.Data is live.Data


@pytest.mark.parametrize("provided", [False, True])
def test_live_dispatcher_queue_behavior(provided):
    live = importlib.import_module(LIVE)
    queues = [Queue(), Queue()] if provided else None
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        dispatcher = live.MsgDispatcher(2, queues)
        dispatcher.broadcast_message("hello", prefix="#")
        dispatcher.send_message(1, "direct", prefix="!")
    actual = {
        "stdout": stream.getvalue(),
        "queues": [list(queue.queue) for queue in dispatcher.queues],
        "num_processes": dispatcher.num_processes,
        "provided_identity": dispatcher.queues is queues if provided else None,
    }
    assert actual == REPORT["dispatcher_behavior"][str(provided)]
