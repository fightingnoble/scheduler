from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict

if TYPE_CHECKING:
    from buffer import Data, Buffer
    from task_agent import ProcessInt

# Context is a data structure used to record the 
# processing history (by each process or node) and 
# the transfer history (by each link or channel) of 
# of a message at runtime. 
# Processing info:  
    # 1. start time
    # 2. upstream node
    # 3. downstream node
    # 4. ready time
    # 5. finish time
    # 6. processing time
    # 7. id
    # 8. allocated resources

# Transfer info:
    # 1. start time
    # 2. end time

# When passing a message from one node to another, a new context is created. 
# implementation by using a dictionary

class ContextMsg(object):
    def __init__(self, ctx_type='process') -> None:
        self.msg_context = {}
        # trigger event
        self.msg_context["trigger"] = {}
        # upstreaming data source
        self.msg_context["src"] = {}
        # weight data
        self.msg_context["weight"] = {}
        # information of the processing node
        self.msg_context["process_info"] = {}
        # generated data type
        self.msg_context["data_type"] = ctx_type
        # information of the generated data
        self.msg_context["data_info"] = {}
        # transfer details
        self.msg_context["transfer_info"] = {}
        
    def cache_upstreaming(self, process:ProcessInt, glb_n_task_dict:Dict, buffer:Buffer) -> None:
        self.msg_context["src"].update(process.get_upstream_ctx(glb_n_task_dict, buffer))
    
    def cache_trigger(self, process:ProcessInt) -> None:
        self.msg_context["trigger"].update(process.get_trigger_ctx())

    def cache_weight(self, process:ProcessInt, buffer:Buffer) -> None:
        tgt_buffer = buffer.buffer_mux("weight")
        data=tgt_buffer[process.pid][0]
        self.msg_context["weight"].update(data.serialize())

    def cache_processing(self, process:ProcessInt) -> None:
        self.msg_context["process_info"]["start_time"] = process.start_time
        self.msg_context["process_info"]["downstream_node"] = process.get_downstream_ctx()
        self.msg_context["process_info"]["ready_time"] = process.ready_time
        self.msg_context["process_info"]["end_time"] = process.end_time
        self.msg_context["process_info"]["processing_time"] = process.cumulative_executed_time
        # self.msg_context["process_info"]["allocated_resources"] = process.allocated_resources
        
    def cache_msg_transfer(self, time:int) -> None:
        self.msg_context["transfer_info"]["start_time"] = time
    
    def update_receive_time(self, time:int) -> None:
        self.msg_context["transfer_info"]["end_time"] = time

    def cache_data_info(self, data:Data):
        self.msg_context["data_info"] = data.serialize()
    
    def get_downstream_node(self) -> List:
        return self.msg_context["process_info"]["downstream_node"]
    
    def get_type(self) -> str:
        return self.msg_context["data_type"]
    
    def get_trigger(self) -> Dict:
        return self.msg_context["trigger"]
    
    def get_src(self) -> Dict:
        return self.msg_context["src"]
    
    def get_name(self) -> str:
        if self.msg_context["data_type"] == "process":
            return self.msg_context["process_info"]["name"]
    
    def get_end_time(self) -> float:
        return self.msg_context["process_info"]["end_time"]

    def serialize(self) -> Dict:
        return self.msg_context
    
    @staticmethod
    def create_p_ctx(process:ProcessInt) -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context["process_info"]["pid"] = process.pid
        ctx.msg_context["process_info"]["name"] = process.task.name
        return ctx

    @staticmethod
    def create_weight_ctx() -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context["data_type"] = "weight"
        return ctx

    @staticmethod
    def create_sensor_ctx(time) -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context["data_type"] = "sensor"
        ctx.msg_context["trigger"]["trigger_time"] = time
        return ctx
    
    @staticmethod
    def parser_ctx(trace:Dict) -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context = trace
        return ctx
    
    @classmethod
    def find_sensor(cls, serialized_ctx:Dict) -> List:
        dict_o = {}
        _root = cls.parser_ctx(serialized_ctx)
        trigger_dict  = _root.get_trigger()
        for key, ctx in trigger_dict.items():
            if ctx["data_type"] == "sensor":
                dict_o[key] = ctx["trigger"]["trigger_time"]
        _src = _root.get_src()
        for key, ctx in _src.items():
            dict_t, end_t = cls.find_sensor(ctx)
            dict_o.update(dict_t)
        return dict_o, _root.get_end_time()


if __name__ == "__main__":
    from trace_example import trace_example
    import numpy as np

    dict_o, end_time = ContextMsg.find_sensor(trace_example)
    print(dict_o)
    print("end time: ", end_time)

    # load the trace list from the file
    import pickle
    import pandas as pd
    for file_name in ['trace/dynamic_e2e_trace_256var_0.2.pkl',  ]: 
        with open(file_name, "rb") as f:
            trace_list = pickle.load(f)
        n_violation = 0
        row_list = ['sensor', 'time', 'T_e2e']
        for trace in trace_list:
            dict_o, end_time = ContextMsg.find_sensor(trace)
            # print("name: ", trace["process_info"]["name"])
            # print(dict_o)
            # print(f"end time: {end_time:.6f}\n")
            value_array = np.array(list(dict_o.values()))
            e2e_latency = end_time - value_array
            # index the item > 0.1
            index = np.where(e2e_latency > 0.1)
            if len(index[0]):
                name_array = np.array(list(dict_o.keys()))
                df = pd.DataFrame({'sensor': name_array, 'time': value_array, 'T_e2e': e2e_latency}, )
                print(df)
                print(f"{trace['process_info']['name']} end time: {end_time:.6f}\n")
                n_violation += len(index[0])
        print(f"total violation: {n_violation}\n")
        