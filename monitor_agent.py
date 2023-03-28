from typing import List, Dict
from scheduling_table import SchedulingTableInt
from scheduling_table import Resource_model_int

class Monitor(object):

    def __init__(self, 
                 _SchedTab: SchedulingTableInt, 
                 hyper_period: int,
                 ) -> None:
        self.SchedTab = _SchedTab
        self.trace_recoder = SchedulingTableInt(_SchedTab.num_resources, 0, _SchedTab.id, _SchedTab.name, hyper_period)

    def get_trace(self) -> SchedulingTableInt:
        return self.trace_recoder
    
    def add_a_record(self, _record: Resource_model_int) -> None:
        self.trace_recoder.append(_record)