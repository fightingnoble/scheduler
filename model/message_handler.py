from scipy.stats import truncnorm
from typing import List, Dict
from task.task_agent import ProcessBase

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

def message_trigger_event(sim_triggered_list:List[ProcessBase], jitter_sim_en, jitter_sim_para, 
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
        if curr_t - _p.next_ingestion_time >= -timestep*0.99:
            _p.event_triggers.append([_p.next_ingestion_time, _p.next_event_time])
        trigger_state = _p.sim_trigger(curr_t, timestep)
        if trigger_state:
            _p.next_event_time += _p.task.period
            _p.next_ingestion_time = _p.next_event_time
            if jitter_sim_en:
                    jitter = jitter_sim_event(_p, jitter_sim_para)
                    _p.next_ingestion_time += jitter
        if DEBUG_FG and trigger_state:
            print(f"		{_p.task.name} triggered @ {curr_t:.6f}")
    return Bin_trigger_state

def jitter_sim_event(_p, jitter_sim_para:Dict):
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
    return jitter
