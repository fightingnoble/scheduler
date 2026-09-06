"""Unused periodic event generators preserved without behavior changes."""

from __future__ import annotations
from typing import Dict, Union

import numpy as np

from model.event_gen.e2e_latency import jitter_gen_biside


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
