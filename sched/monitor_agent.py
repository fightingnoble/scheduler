from typing import List, Dict
from sched.scheduling_table import SchedulingTableInt
from sched.scheduling_table import Resource_model_int

class Monitor(object):

    def __init__(self, 
                 num_resources: int, 
                 hyper_period: int, 
                 id: int = None, name: str = None, 
                 ) -> None:
        self.trace_recoder = SchedulingTableInt(num_resources, 0, id, name, hyper_period)

    def get_trace(self) -> SchedulingTableInt:
        return self.trace_recoder
    
    def add_a_record(self, _record: Resource_model_int) -> None:
        self.trace_recoder.append(_record)

    def add_a_placehold_record(self, ) -> None:
        self.trace_recoder.append(Resource_model_int(self.trace_recoder.num_resources))