from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict, Callable
import networkx as nx
from functools import reduce
import re
import numpy as np
import argparse
# load the trace list from the file
import pickle
import pandas as pd
import copy
import os 


if TYPE_CHECKING:
    from model.buffer import Data, Buffer
    from task.task_agent import ProcessInt

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
        # time stamp of the trigger event
        self.msg_context["time_stamp"] = float("inf")
        self.msg_context["stream_domain"] = {}
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
        self.msg_context["e2e_var"] = 0
        self.msg_context["load_var"] = {}
        self.cached_list = ["e2e_var", "load_var"]
   
    def setattr(self, attr_name, attr_value, cached=False):
        self.msg_context[attr_name] = attr_value
        if cached:
            self.cached_list.append(attr_name)
    
    def getattr(self, attr_name):
        return self.msg_context[attr_name]

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

    def get_transfer_delay(self) -> float:
        return self.msg_context["transfer_info"]["end_time"] - self.msg_context["transfer_info"]["start_time"]
    
    def get_e2e_var(self) -> float:
        return self.msg_context["e2e_var"]

    def get_timestamp(self) -> float:
        if self.get_type() == "sensor":
            return self.msg_context["trigger"]["event_time"]
        else:
            return self.msg_context["time_stamp"]
    
    def get_watermark(self) -> float:
        return self.msg_context["watermark"]

    def get_load_var(self) -> Dict:
        return self.msg_context["load_var"]
    
    def cache_upstreaming(self, matched_pair:List[Data]) -> None:
        for key in matched_pair:
            data:Data = matched_pair[key]
            self.msg_context["src"].update({key:data.ctx.serialize()})
        if self.msg_context["trigger"]:
            return
        # max([matched_pair[key].ctx.get_timestamp() for key in matched_pair])
        idx = max([key for key in matched_pair], key=lambda x: matched_pair[x].ctx.get_timestamp())
        event_time = matched_pair[idx].ctx.get_timestamp()
        self.msg_context["stream_domain"] = "multi_stream" if len(matched_pair)>1 else "single_stream"
        self.msg_context["time_stamp"] = event_time
        if self.cached_list:
            for attr_n in self.cached_list:
                self.msg_context[attr_n] = matched_pair[idx].ctx.getattr(attr_n)
    
    def cache_trigger(self, process:ProcessInt, pred_ctrl:Dict[int, Dict]=None) -> None:
        trigger_dict = process.get_trigger_ctx(pred_ctrl)
        assert len(trigger_dict) <= 1
        self.msg_context["trigger"].update(trigger_dict)
        if len(trigger_dict) == 1:
            # set the time stamp of the trigger event
            trigger = self.parser_ctx(list(trigger_dict.values())[0])
            self.msg_context["stream_domain"] = trigger.get_type()
            self.msg_context["time_stamp"] = trigger.get_timestamp()
            if self.cached_list:
                for attr_n in self.cached_list:
                    self.msg_context[attr_n] = trigger.getattr(attr_n)

        else:
            raise ValueError("Multiple trigger events are not supported yet.")
        
    def cache_weight(self, process:ProcessInt, buffer:Buffer) -> None:
        tgt_buffer = buffer.buffer_mux("weight")
        data=tgt_buffer[process.pid][0]
        self.msg_context["weight"].update(data.serialize())

    def cache_msg_transfer(self, time:int) -> None:
        self.msg_context["transfer_info"]["start_time"] = time
    
    def cache_data_info(self, data:Data):
        self.msg_context["data_info"] = data.serialize()
    
    def cache_processing(self, process: ProcessInt) -> None:
        precess_info = {
            "start_time": process.start_time,
            "downstream_node": process.get_downstream_ctx(),
            "ready_time": process.ready_time,
            "end_time": process.end_time,
            "processing_time": process.cumulative_executed_time
        }
        self.msg_context["process_info"].update(precess_info)

    def update_receive_time(self, time:int) -> None:
        self.msg_context["transfer_info"]["end_time"] = time

    def update_e2e_var(self, time:int) -> None:
        self.msg_context["e2e_var"] = time
    
    def update_load_var(self, info:Dict) -> None:
        self.msg_context["load_var"].update(info)

    def get_node_attr(self) -> Dict:
        if self.get_type() == "process":
            return self.msg_context["process_info"]
        elif self.get_type() == "sensor":
            return self.msg_context["trigger"]
        elif self.get_type() == "weight":
            return self.msg_context["weight"]

    def serialize(self) -> Dict:
        return self.msg_context
    
    @staticmethod
    def create_p_ctx(process:ProcessInt) -> ContextMsg:
        ctx = ContextMsg()
        msg_context = {
            "process_info": {
                "pid": process.pid,
                "name": process.task.name,
                "period": process.task.period
            }
        }
        ctx.msg_context.update(msg_context)
        return ctx

    @staticmethod
    def create_weight_ctx() -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context.update({"data_type": "weight"})
        return ctx

    @staticmethod
    def create_sensor_ctx(ingestion_time, event_time=None, period=None) -> ContextMsg:
        ctx = ContextMsg()
        msg_context = {
            "data_type": "sensor",
            "trigger": {
                "ingestion_time": ingestion_time,
                "event_time": event_time,
                "period": period
            }
        }
        ctx.msg_context.update(msg_context)
        return ctx
    
    @staticmethod
    def parser_ctx(trace:Dict) -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context = trace
        return ctx
    
    @classmethod
    def find_sensor(cls, serialized_ctx:Dict, hist_seri_ctx:Dict=None, nx_graph=None) -> List:
        if nx_graph is None:
            nx_graph = nx.DiGraph()
        
        dict_o = {}
        _root = cls.parser_ctx(serialized_ctx)
        his_root = cls.parser_ctx(hist_seri_ctx)
        trigger_dict  = _root.get_trigger()
        for key, ctx in trigger_dict.items():
            if ctx["data_type"] == "sensor":
                dict_o[key] = ctx["trigger"]
        _src = _root.get_src()
        his_src = his_root.get_src() if hist_seri_ctx is not None else {k:None for k in _src}
        
        # add self, trigger node to the graph
        nx_graph.add_node(_root.get_name(), **_root.get_node_attr())
        for key, ctx in trigger_dict.items():
            nx_graph.add_node(key, **ctx["trigger"])
            nx_graph.add_edge(key, _root.get_name(), weight=0) 
        
        for ctx, his_ctx in zip(_src.values(), his_src.values()):
            if his_ctx != ctx:
                # add the edge to the src node
                nx_graph.add_edge(ctx["process_info"]["name"], _root.get_name(), 
                                  weight=cls.parser_ctx(ctx).get_transfer_delay(),
                                  **ctx["transfer_info"], )
                dict_t, end_t, nx_graph = cls.find_sensor(ctx, his_ctx, nx_graph)
                dict_o.update(dict_t)
        return dict_o, _root.get_end_time(), nx_graph


if __name__ == "__main__":
    from model.trace_example import trace_example
    dict_o, end_time, nx_graph = ContextMsg.find_sensor(trace_example)
    print(dict_o)
    print("end time: ", end_time)

