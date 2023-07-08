from __future__ import annotations
from scipy.stats import truncnorm
from typing import List, Dict, Union
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
    jitter_gen_inst = jitter_gen(e2e_latency, jitter_sim_para, size, seed)
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

def dyn_obj_sim(load_var_sim_para:Dict, size=1,
                period:float=1, event_range:float=float("inf"),
                seed:Union[None, int, np.random.Generator, np.random.RandomState]=None):
    """
        test case: 
            simulate the dyamic object (passenger, vehicle) arrival with a given period
            return: event_time, num_dyn_obj
    """
    # jitter parameters: a, b, loc, scale
    num_dyn_obj_gen_inst = num_dyn_obj_gen(load_var_sim_para["maxsize"], size, seed)
    event_time = 0
    event_no = 0
    while True:
        if event_time == 0:
            yield event_time, 0
        else:
            yield event_time, num_dyn_obj_gen_inst()
        if event_time > event_range:
            yield np.inf, 0
            return event_no
        event_time += period
        event_no += 1


def jitter_gen(ref_value, jitter_sim_para:Dict, size=1, 
               seed:Union[None, int, np.random.Generator, np.random.RandomState]=None):
    scope = ref_value * jitter_sim_para["scale"]
    loc = 0
    scale = scope/3
    myclip_a = -scope
    myclip_b = scope
    a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
    generator = np.random.default_rng(seed)
    if size == 1:
        jitter_gen_inst = lambda: truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=generator)
    else:
        jitter_gen_inst = lambda: truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=generator)
    return jitter_gen_inst


# num_dyn_obj_gen = lambda maxsize, seq_len, seed: np.random.default_rng(np.random.default_rng(seed)).integers(0, maxsize, size=seq_len)
def num_dyn_obj_gen(maxsize, seq_len, seed):
    generator = np.random.default_rng(seed)
    if seq_len == 1:
        num_dyn_obj_gen_inst = lambda: generator.integers(0, maxsize)
    else:
        num_dyn_obj_gen_inst = lambda: generator.integers(0, maxsize, size=seq_len)
    return num_dyn_obj_gen_inst