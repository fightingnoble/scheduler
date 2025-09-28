from __future__ import annotations
from scipy.stats import truncnorm as _scipy_truncnorm, truncexpon as _scipy_truncexpon, expon as _scipy_expon, norm as _scipy_norm
from typing import List, Dict, Union, Callable
from global_var import *
import numpy as np

def naive_period_event_gen(period, curr_t, timestep):
    curr_t = 0
    while True:
        curr_t += period
        yield curr_t


def e2e_var_sim(e2e_latency, jitter_sim_para:Dict, size=1, 
                period:float=1, event_range:float=float("inf"),
                seed:Union[None, int, np.random.Generator, np.random.RandomState]=None):
    """
        test case: 
            simulate behivior of the system which randomly changes deadline allocations with a given period
    """
    # jitter parameters: a, b, loc, scale
    jitter_gen_inst = jitter_gen_biside(e2e_latency, jitter_sim_para, size, seed)
    event_time = 0
    event_no = 0
    while True:
        if event_time == 0:
            yield event_time, 0
        elif event_time > event_range:
            yield np.inf, 0
            return event_no
        else:
            yield event_time, jitter_gen_inst()
        event_time += period
        event_no += 1

def discrete_event_sim(event_list:List, size=1,
                period:float=1, event_range:float=float("inf"),
                seed:Union[None, int, np.random.Generator, np.random.RandomState]=None, bias=0):
    """
        test case: 
            simulate the dyamic object (passenger, vehicle) arrival with a given period
            return: event_time, num_dyn_obj
    """
    # jitter parameters: a, b, loc, scale
    sel_no_gen = get_intger_gen(len(event_list), size, seed)
    event_time = bias
    event_no = 0
    while True:
        if event_time == 0:
            yield event_time, None
        elif event_time > event_range:
            yield np.inf, None
            return event_no
        else:
            yield event_time, event_list[sel_no_gen()]
        event_time += period
        event_no += 1

def jitter_gen_biside(ref_value, jitter_sim_para:Dict, size=1, 
               seed:Union[None, int, np.random.Generator, np.random.RandomState]=None):
    enforce_wc = jitter_sim_para.get("enforce_wc", False) 
    loc, scale, myclip_b, a, b = get_truncnorm_para(ref_value, jitter_sim_para)
    if enforce_wc:
        return lambda: np.full(size, myclip_b) if size>1 else np.float64(myclip_b)
    generator = np.random.default_rng(seed)
    if size == 1:
        jitter_gen_inst = lambda: _scipy_truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=generator)
    else:
        jitter_gen_inst = lambda: _scipy_truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=generator)
    return jitter_gen_inst

def get_truncnorm_para(range_max, jitter_sim_para, ZScore=3, loc=0):
    """Prepare the parameters for function, truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=generator),
        which generates truncated normal distribution, by truncating the normal distribution 
        centered on loc (default 0), with standard deviation scale (default 1), and truncated at myclip_a, myclip_b. 
        The truncnorm.rvs extra requires a, b which are standard deviations from loc at left and right endpoints,
        where a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale, and can be understood as the ratio between the 
        daviation from the mean and the standard deviation, which exactly corrspond to the definition of ZScore.
        
        This simulator assume the maximum/minimum observed jitter falls within ZScore * scale away from the mean, following 
        a truncated normal distribution. The default ZScore is 3, which means we only consider 3sigma away from the mean. 
    Args:
        range_max (Any): the maximum range of the jitter
        jitter_sim_para (Dict): the jitter simulation parameters, including "scale", 
            which is the truncation ratio w.r.t. the range_max.
        ZScore (int, optional): Defines the confidence interval coverage. Defaults to 3.

    Returns:
        loc, scale, abs of myclip_a and myclip_b, a, b
    """
    # half_len: abs of myclip_a and myclip_b
    half_len = range_max * jitter_sim_para["scale"]
    scale=half_len/ZScore
    a, b = -ZScore, ZScore
    return loc,scale,half_len,a,b

def exp_jitter(ref_value, jitter_sim_para:Dict, size=1, 
               seed:Union[None, int, np.random.Generator, np.random.RandomState]=None, lamda_exp:float=100.):
    enforce_wc = jitter_sim_para.get("enforce_wc", False) 
    scope, loc, scale, b = get_truncexpon_param(ref_value, jitter_sim_para, lamda_exp)
    if enforce_wc:
        return lambda: np.full(size, scope) if size>1 else np.float64(scope)
    generator = np.random.default_rng(seed)
    if size == 1:
        jitter_gen_inst = lambda: _scipy_truncexpon.rvs(b, loc=loc, scale=scale, random_state=generator)
    else:
        jitter_gen_inst = lambda: _scipy_truncexpon.rvs(b, loc=loc, scale=scale, size=size, random_state=generator)
    return jitter_gen_inst

def get_truncexpon_param(ref_value, jitter_sim_para, lamda_exp:float=100.):
    scope = ref_value * jitter_sim_para["scale"]
    loc = 0
    scale=1/lamda_exp
    b = scope/scale
    return scope,loc,scale,b

# lambda maxsize, seq_len, seed: np.random.default_rng(np.random.default_rng(seed)).integers(0, maxsize, size=seq_len)
def get_intger_gen(maxsize, seq_len, seed):
    generator = np.random.default_rng(seed)
    if seq_len == 1:
        integer_gen_inst = lambda: generator.integers(0, maxsize)
    else:
        integer_gen_inst = lambda: generator.integers(0, maxsize, size=seq_len)
    return integer_gen_inst