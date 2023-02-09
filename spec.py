from typing import Dict
from task_agent import TaskBase
class Spec(object):
    e2e_latency = 0
    
    def __init__(self, e2e_latency, thread_factor:Dict[TaskBase, int]):
        self.e2e_latency = e2e_latency
        self.thread_factor = thread_factor
    
    # detect changes
    def if_change (self, new_spec):
        if not isinstance(new_spec, spec):
            return False
        if self.e2e_latency != new_spec.e2e_latency:
            e2e_delta_flg = True
        else:
            e2e_delta_flg = False
        if self.thread_factor != new_spec.thread_factor:
            thread_factor_list_delta_flg = True
            # list new items
            new_item = set(new_spec.thread_factor.items()) - set(self.thread_factor.items())
            # list removed items
            removed_item = set(self.thread_factor.items()) - set(new_spec.thread_factor.items())
            # list changed items
            changed_item = {k:v for k,v in new_spec.thread_factor.items() if k in self.thread_factor and self.thread_factor[k] != v}

        if_change_flg = e2e_delta_flg or thread_factor_list_delta_flg
        return if_change_flg, [[e2e_delta_flg, new_spec.e2e_latency], [thread_factor_list_delta_flg, new_item, removed_item, changed_item]] 

    def __str__(self):
        _str = "e2e_latency: " + str(self.e2e_latency) + " thread_factor: " + str(self.thread_factor)
        return _str
    