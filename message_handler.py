from scipy.stats import truncnorm
from typing import List, Dict
from task_agent import ProcessBase

def message_trigger(sim_triggered_list:List[ProcessBase], jitter_sim_en, jitter_sim_para, 
                    timestep, curr_t, DEBUG_FG):
    Bin_trigger_state = False
    for _p in sim_triggered_list:
        trigger_state = _p.sim_trigger(curr_t, timestep)
        Bin_trigger_state |= trigger_state
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
                 _p.pred_ctrl[key]["trigger_time"] = _p.i_offset
                 init_trigger_state = True
    elif _p.i_offset > _p.task.period:
        _p.i_offset -= _p.task.period
    return init_trigger_state
