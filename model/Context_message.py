from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List, Dict
import networkx as nx

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

    def cache_upstreaming(self, matched_pair:List[Data]) -> None:
        for key in matched_pair:
            data:Data = matched_pair[key]
            self.msg_context["src"].update({key:data.ctx.serialize()})


    # def cache_upstreaming(self, process:ProcessInt, glb_n_task_dict:Dict, buffer:Buffer) -> None:
    #     # update the upstreaming data source
    #     self.msg_context["src"].update(process.get_upstream_ctx(glb_n_task_dict, buffer))
    #     # # check how many streaming domains are involved
    #     # if not len(self.msg_context["trigger"]):
    #     #     if len(self.msg_context["src"]) == 1: 
    #     #         self.msg_context["stream_domain"] = list(self.msg_context["src"].values())[0]["stream_domain"]
    #     #     elif len(self.msg_context["src"]) > 1:
    #     #         self.msg_context["stream_domain"] = "multi-stream"
    
    def cache_trigger(self, process:ProcessInt, pred_ctrl:Dict[int, Dict]=None) -> None:
        trigger_dict = process.get_trigger_ctx(pred_ctrl)
        assert len(trigger_dict) <= 1
        self.msg_context["trigger"].update(trigger_dict)
        if len(trigger_dict) == 1:
            # set the time stamp of the trigger event
            trigger = list(trigger_dict.values())[0]
            self.msg_context["time_stamp"] = trigger["trigger"]['event_time']
            self.msg_context["stream_domain"] = trigger["data_type"]
        else:
            raise ValueError("Multiple trigger events are not supported yet.")
        
    def get_timestamp(self) -> float:
        return self.msg_context["time_stamp"]
    
    def get_watermark(self) -> float:
        return self.msg_context["watermark"]

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

    def get_transfer_delay(self) -> float:
        return self.msg_context["transfer_info"]["end_time"] - self.msg_context["transfer_info"]["start_time"]
    
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
        ctx.msg_context["process_info"]["pid"] = process.pid
        ctx.msg_context["process_info"]["name"] = process.task.name
        ctx.msg_context["process_info"]["period"] = process.task.period
        return ctx

    @staticmethod
    def create_weight_ctx() -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context["data_type"] = "weight"
        return ctx

    @staticmethod
    def create_sensor_ctx(ingestion_time, event_time=None, period=None) -> ContextMsg:
        ctx = ContextMsg()
        ctx.msg_context["data_type"] = "sensor"
        ctx.msg_context["trigger"]["ingestion_time"] = ingestion_time
        ctx.msg_context["trigger"]["event_time"] = event_time
        ctx.msg_context["trigger"]["period"] = period
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

    import numpy as np
    import argparse
    # load the trace list from the file
    import pickle
    import pandas as pd
    import copy
    import os 
    import torch
    import plotly.graph_objects as go
    import plotly.io as pio   
    pio.kaleido.scope.mathjax = None
    
    parser = argparse.ArgumentParser(description="profiling")
    parser.add_argument("--core_list", type=str, default="300,", help="core list")
    parser.add_argument("--profiling_filename", type=str, default="profiling.csv", help="profiling filename")
    parser.add_argument("--file_suffix", default="", type=str, help="file suffix")
    parser.add_argument("--e2e_latency", type=float, default=0.09, help="e2e latency")
    parser.add_argument("--test_case", type=str, default="dynamic", help="task name")

    args = parser.parse_args()
    e2e_latency = args.e2e_latency
    if args.test_case == "glb_dynamic":
        fn = "dyn_glb_e2e_trace"
    elif args.test_case == "dynamic":
        fn = "dynamic_e2e_trace"

    core_list = [int(i) for i in args.core_list.split(",")]
    if args.profiling_filename == "profiling.csv":
        cfg_n = "heavy"
    else:
        cfg_n = args.profiling_filename.split(".")[-2].split("_")[-1]
    
    # load the criticality 
    from task.task_cfg import load_taskattrib
    glb_n_task_dict, f_gcd = load_taskattrib(args.profiling_filename, verbose=False) 
    timing_flag_dict = {}
    for task_name in glb_n_task_dict:
        timing_flag_dict[task_name] = glb_n_task_dict[task_name].timing_flag

    result_dict = {}
    for num_cores in core_list:
        trace_path = f"trace/{cfg_n}/{fn}_{num_cores}{args.file_suffix}.pkl"
        with open(trace_path, "rb") as f:
            trace_list = pickle.load(f)
        print("="*20, trace_path, "="*20)
        n_violation = 0
        row_list = ['sensor', 'time', 'T_e2e']
        sink_dict = {}
        for trace in trace_list:
            if trace["process_info"]["name"] in sink_dict:
                sink_dict[trace["process_info"]["name"]].append(trace)
            else:
                sink_dict[trace["process_info"]["name"]] = [trace]
        
        e2e_latency_list = [[], []]
        for sink_key in sink_dict:
            hist_seri_ctx = None
            for trace in sorted(sink_dict[sink_key], key=lambda x: x["process_info"]["end_time"]): 
                dict_o, end_time, nx_graph = ContextMsg.find_sensor(trace, hist_seri_ctx)
                # print("name: ", trace["process_info"]["name"])
                # print(dict_o)
                # print(f"end time: {end_time:.6f}\n")
                matched_pair = np.array([trigger["event_time"] for trigger in dict_o.values()])
                event_time = max(matched_pair)
                active_path = matched_pair >= event_time
                e2e_latency = end_time - matched_pair[active_path] 
                task_name = "_".join(trace["process_info"]["name"].split("_")[0:-2])
                if timing_flag_dict[task_name] == "realtime":
                    e2e_latency_list[0].extend(e2e_latency.tolist())
                else:
                    e2e_latency_list[1].extend(e2e_latency.tolist())

                # index the item > e2e_latency
                index = np.where(e2e_latency > e2e_latency)
                if len(index[0]):
                    name_array = np.array(list(dict_o.keys()))
                    df = pd.DataFrame({'sensor': name_array, 'time': matched_pair, 'T_e2e': e2e_latency}, )
                    print(df)
                    print(f"{trace['process_info']['name']} end time: {end_time:.6f}\n")

                    dest = trace["process_info"]["name"]
                    for src in name_array[index]:
                        # 找到节点1到节点3之间的最短路径
                        shortest_path = nx.algorithms.shortest_paths.weighted.dijkstra_path(nx_graph, source=src, target=dest, weight='weight')

                        # 打印每个节点和边的属性，以及边的权重
                        print(f'Node: {shortest_path[0]}, attr: {nx_graph.nodes[shortest_path[0]]}') 
                        for i in range(len(shortest_path) - 1):
                            source = shortest_path[i]
                            target = shortest_path[i + 1]
                            edge_data = nx_graph.get_edge_data(source, target)
                            print(f'Edge: {source} -> {target}, Weight: {edge_data["weight"]: .6f}')
                            print(f'Node: {target}, Start Time: {nx_graph.nodes[target]["start_time"]: .6f}, End Time: {nx_graph.nodes[target]["end_time"]: .6f}')

                        print(f'Shortest Path Length: {nx.algorithms.shortest_paths.weighted.dijkstra_path_length(nx_graph, source=src, target=dest, weight="weight")}')

                    n_violation += len(index[0])
                hist_seri_ctx = copy.deepcopy(trace)
        print(f"total violation: {n_violation}\n")

        # cache the result and the label
        result_dict[num_cores] = e2e_latency_list

        # aplly histogram analysis
        rt_e2e_latency_list = np.array(e2e_latency_list[0])
        ddl_e2e_latency_list = np.array(e2e_latency_list[1])
        print(f"mean: {np.mean(rt_e2e_latency_list):.6f}, std: {np.std(rt_e2e_latency_list):.6f}, max: {np.max(rt_e2e_latency_list):.6f}, min: {np.min(rt_e2e_latency_list):.6f}")
        print(f"mean: {np.mean(ddl_e2e_latency_list):.6f}, std: {np.std(ddl_e2e_latency_list):.6f}, max: {np.max(ddl_e2e_latency_list):.6f}, min: {np.min(ddl_e2e_latency_list):.6f}")
        # calculate the percentile
        rt_percentile = np.percentile(rt_e2e_latency_list, [90, 95, 99, 99.9, 99.99])
        ddl_percentile = np.percentile(ddl_e2e_latency_list, [90, 95, 99, 99.9, 99.99])
        print(f"rt_percentile: {rt_percentile}")
        print(f"ddl_percentile: {ddl_percentile}")

    # plot histogram
    fig = go.Figure()
    bins_num = 20
    for rt_e2e_latency_list, ddl_e2e_latency_list in result_dict.values():
        rt_e2e_latency_list = torch.tensor(rt_e2e_latency_list)
        v_max = rt_e2e_latency_list.max()
        v_min = rt_e2e_latency_list.min()
        bins = [(2*k+1)*(v_max-v_min)/2/bins_num+v_min for k in range(bins_num+1)]
        counts = torch.histc(rt_e2e_latency_list, bins=bins_num, max=v_max, min=v_min)
        fig.add_trace(go.Bar(
                x=bins,
                y=counts.numpy(),
                name=f"{num_cores}_rt", # name used in legend and hover labels
                # marker_color='#EB89B5',
                opacity=0.75
            ))
        ddl_e2e_latency_list = torch.tensor(ddl_e2e_latency_list)
        v_max = ddl_e2e_latency_list.max()
        v_min = ddl_e2e_latency_list.min()
        bins = [(2*k+1)*(v_max-v_min)/2/bins_num+v_min for k in range(bins_num+1)]
        counts = torch.histc(ddl_e2e_latency_list, bins=bins_num, max=v_max, min=v_min)
        fig.add_trace(go.Bar(
                x=bins,
                y=counts.numpy(),
                name=f"{num_cores}_ddl", # name used in legend and hover labels
                # marker_color='#EB89B5',
                opacity=0.75
            ))

    # set axis as log scale
    # fig.update_yaxes(type="log")
    # set axis as linear scale
    fig.update_yaxes(type="linear")
    fig.update_layout(
        title_text=f"{cfg_n}/{fn}_{num_cores}{args.file_suffix}.pkl"+'Sampled Results', # title of plot
        xaxis_title_text='Value', # xaxis label
        yaxis_title_text='Count', # yaxis label
        bargap=0.2, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )
    save_path = f"plot/trace_hist/{cfg_n}/{fn}{args.file_suffix}.pdf"
    dir_path = os.path.dirname(save_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig.write_image(save_path)
    print(f"save to {save_path}")
