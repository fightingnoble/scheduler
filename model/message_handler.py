from __future__ import annotations
from scipy.stats import truncnorm
from typing import List, Dict, Union, Generator
from typing import TYPE_CHECKING
from global_var import *
import numpy as np
from model.Context_message import ContextMsg

if TYPE_CHECKING:
    from task.task_agent import ProcessBase
    from model.data_pipe import TriggerPipe
    from model.task_queue_agent import TaskQueue

def message_trigger(sim_triggered_list:List[ProcessBase], jitter_sim_en, jitter_sim_para, 
                    timestep, curr_t, DEBUG_FG):
    """
        @ 0: set next_event_time -> sim_trigger -> jitter_sim -> 
    """
    Bin_trigger_state = False
    # set the the property next_event_time
    if curr_t == 0:
        for _p in sim_triggered_list:
            _p.next_event_time = _p.task.i_offset
    for _p in sim_triggered_list:
        trigger_state = _p.sim_trigger(curr_t, timestep)
        if trigger_state:
            _p.next_event_time += _p.task.period
        Bin_trigger_state |= trigger_state
        # inject jitter to the sensor data arrival time
        if jitter_sim_en:
            # case 1: In 1st peroid, i_offset is originally 0 and the task is triggered at the beginning
            # case 2: In 1st peroid, i_offset is originally > 0 and the jitter is introduced
            # case 3: In the later peroid, jitter is lazily introduced, and jitter size is not larger than 0.5*period
            if trigger_state or curr_t == 0:
                init_trigger_state = jitter_sim(_p, jitter_sim_para, curr_t, trigger_state)
                Bin_trigger_state |= init_trigger_state
        if DEBUG_FG and trigger_state:
            print(f"		{_p.task.name} triggered @ {curr_t:.6f}")
    return Bin_trigger_state


def jitter_sim(_p, jitter_sim_para:Dict, curr_t, trigger_state):
    """
        test case: 
        sensor data arrival time varies by injecting jitter
        inject noise to self.task.period, self.task.i_offset
    """
    # jitter parameters: a, b, loc, scale
    a, b, loc, scale = jitter_sim_para["a"], jitter_sim_para["b"], jitter_sim_para["loc"], jitter_sim_para["scale"]
    # 0.2 # truncnorm.rvs(-0.2, 0.2, size=1, scale=1)[0]
    jitter_gen = lambda: _p.task.exp_comp_t * truncnorm.rvs(a, b, loc=loc, scale=scale, size=1)[0]
    jitter = jitter_gen()
    assert abs(jitter) < 0.5*_p.task.period, "jitter is too large"
    _p.i_offset = _p.task.i_offset + jitter
    
    init_trigger_state = False
    # judge whether the offset is negative, which is a illegal value
    if _p.i_offset < 0:
         _p.i_offset += _p.task.period
         if curr_t == 0 and not trigger_state:
             for key in _p.pred_ctrl.keys():
                 _p.pred_ctrl[key]["valid"] = True
                 _p.pred_ctrl[key]["ingestion_time"] = _p.i_offset
                 _p.pred_ctrl[key]["event_time"] = _p.i_offset
                 init_trigger_state = True
    elif _p.i_offset > _p.task.period:
        _p.i_offset -= _p.task.period
    return init_trigger_state

def message_trigger_event(sim_triggered_list:List[ProcessBase], jitter_sim_en, jitter_sim_para, inactive_list,
                    timestep, curr_t, DEBUG_FG):
    Bin_trigger_state = False
    # set the the property next_event_time
    for _p in sim_triggered_list:
        if curr_t == 0:
            _p.next_event_time = _p.task.i_offset
            _p.next_ingestion_time = _p.next_event_time
            if jitter_sim_en:
                jitter = jitter_sim_event(_p, jitter_sim_para)
                _p.next_ingestion_time += jitter

        assert _p.trigger_mode == "event", "trigger mode is not event"
        if curr_t - _p.next_ingestion_time >= -timestep*numerical_error_tol_rel:
            _p.event_triggers.append([_p.next_ingestion_time, _p.next_event_time])
            _p.next_event_time += _p.task.period
            _p.next_ingestion_time = _p.next_event_time
            if jitter_sim_en:
                    jitter = jitter_sim_event(_p, jitter_sim_para)
                    _p.next_ingestion_time += jitter

        if _p not in inactive_list:
            continue

        trigger_state = _p.sim_trigger(curr_t, timestep)
        if DEBUG_FG and trigger_state:
            print(f"		{_p.task.name} triggered @ {curr_t:.6f}")
    return Bin_trigger_state

def jitter_sim_event(_p, jitter_sim_para:Dict, size=1, seed:Union[None, int, np.random.Generator, np.random.RandomState]=None):
    """
        test case: 
        sensor data arrival time varies by injecting jitter
        inject noise to self.task.period, self.task.i_offset
    """
    # jitter parameters: a, b, loc, scale
    a, b, loc, scale = jitter_sim_para["a"], jitter_sim_para["b"], jitter_sim_para["loc"], jitter_sim_para["scale"]
    # 0.2 # truncnorm.rvs(-0.2, 0.2, size=1, scale=1)[0]
    jitter_gen = lambda: _p.task.exp_comp_t * truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=seed)
    jitter = jitter_gen()
    assert abs(jitter.max()) < 0.5*_p.task.period, "jitter is too large"
    return jitter if size>1 else jitter[0]

def gen_sensor_event(glb_p_list:List[ProcessBase], hyper_p, n_p, warmup, jitter_sim_en, jitter_sim_para, seed):
    event_range = hyper_p * (n_p+warmup)

    # filter the processes with trigger_mode is not "N"
    trigger_list = [p for p in glb_p_list if p.task.trigger_mode!='N']
    task_name = [p.task.name for p in trigger_list]
    event_iter_dict = {name:[] for name in task_name}

    for _p in trigger_list:
        event_iter_dict[_p.task.name] = [
            extract_sensor_event(_p, event_range, jitter_sim_en, jitter_sim_para, seed), 
            extract_sensor_event(_p, event_range)
        ]
    return event_iter_dict

def extract_sensor_event(_p, event_range, jitter_sim_en=False, jitter_sim_para=None, seed=0):
    n_event = int(event_range//_p.task.period)
    if jitter_sim_en:
        jitter = jitter_sim_event(_p, jitter_sim_para, size=n_event, seed=seed)

    for i in range(n_event):
        if jitter_sim_en:
            yield _p.task.i_offset + _p.task.period * i + jitter[i]
        else:
            yield _p.task.i_offset + _p.task.period * i

def extract_sensor_event_endless(_p, jitter_sim_en=False, jitter_sim_para=None, seed=0):
    i = 0
    while True:
        if jitter_sim_en:
            jitter = jitter_sim_event(_p, jitter_sim_para)
            yield _p.task.i_offset + _p.task.period * i + jitter
        else:
            yield _p.task.i_offset + _p.task.period * i
        i += 1
    

def message_trigger_event_new(event_iter_dict:Dict, inactive_list, glb_p_list, 
                              sensor_pipe:TriggerPipe, ddl_stream:TaskQueue, load_var_sim_para:Dict,
                              timestep, curr_t, DEBUG_FG):
    
    name2p = {p.task.name:p for p in glb_p_list}
    for name, (ingestion_time_iter, event_time_iter) in event_iter_dict.items():
        _p = name2p[name]
        # initialize the next event time and ingestion time
        if curr_t == 0 and _p.next_event_time is None:
            _p.next_event_time = next(event_time_iter)
            _p.next_ingestion_time = next(ingestion_time_iter)
        
        # judge whether the event is triggered  
        assert _p.trigger_mode == "event", "trigger mode is not event"
        if curr_t - _p.next_ingestion_time >= -timestep*numerical_error_tol_rel:
            # cahce the trigger evnet
            msg:ContextMsg = ContextMsg.create_sensor_ctx(_p.next_ingestion_time, 
                                                            _p.next_event_time,
                                                            _p.task.period)
            # index the ddl
            if ddl_stream is not None:
                e2e_ddl = index_by_timestamp(ddl_stream, _p.next_event_time)
                msg.update_e2e_var(e2e_ddl)
            if load_var_sim_para is not None:
                load_var_handler(load_var_sim_para, name, _p.next_event_time, msg)
            if sensor_pipe is None:
                # _p.event_triggers.append([_p.next_ingestion_time, _p.next_event_time])
                _p.event_triggers.append(msg)
            else:
                # sensor_pipe.broadcast_message([_p.pid, _p.next_ingestion_time, _p.next_event_time])
                sensor_pipe.broadcast_message([_p.pid, msg])
            if DEBUG_FG:
                print(f"		{_p.task.name} triggered @ {_p.next_ingestion_time:.6f}/{_p.next_event_time:.6f}")

            # fetch the next event time
            try:
                _p.next_event_time = next(event_time_iter)
                _p.next_ingestion_time = next(ingestion_time_iter)
            except StopIteration:
                _p.next_event_time = np.inf
                _p.next_ingestion_time = np.inf

        if _p in inactive_list and sensor_pipe is None:
            _p:ProcessBase
            trigger_state = _p.sim_trigger(curr_t, timestep)

def load_var_handler(load_var_sim_para, name, next_event_time, msg):
    thread_n = name.split('_')[-2]
    troughput_n = name.split('_')[-1]
    task_n = name.replace("_"+thread_n, "").replace("_"+troughput_n, "")
    for var_item, var_param in load_var_sim_para.items():
        for source_name in var_param["src_name"]:
            if source_name == task_n:
                load_var_stream = var_param["stream"]
                load_var = index_by_timestamp(load_var_stream, next_event_time)
                msg.update_load_var(
                                {
                                    var_item: {
                                        "typical": var_param["typical"],
                                        "tgt_name": var_param["tgt_name"],
                                        "size": load_var
                                    }
                                }
                            )

def period_trigger_event(_iter:Generator, curr_t, _stream:TaskQueue): 
    """
        ddl_update_iter: a generator of ddl update events
        curr_t: current time
        ddl_stream: a queue of ddl update events
    """
    try:
        next_update_time = _stream.queue[-1][0] if curr_t > 0 else 0 
        # while ddl_update_time <= curr_t:
        while round(next_update_time, numerical_tol_bit) <= round(curr_t, numerical_tol_bit):
            next_update_time, iter_item = next(_iter)
            _stream.put((next_update_time, iter_item))
    except StopIteration:
        pass

def index_by_timestamp(stream:TaskQueue, tgt_event_time:float):
    """
        ddl_stream: a queue of ddl update events
        tgt_event_time: the time of the event to be indexed
    """
    assert len(stream.queue) > 0, "ddl stream is empty"
    result = None
    # the ddl stream is sorted by ddl_update_time in ascending order
    for stream_event_time, items in stream.queue:
        if tgt_event_time >= stream_event_time:
            result = items
        else:
            break
    assert result is not None, "ddl is not found"
    return result
            