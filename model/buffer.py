"""
This file contains the Buffer class. 
This class models the buffer with the following attributes:
    - buffer: a dict, key is the task id, value is the data.
"""
from __future__ import annotations
import collections
import typing
import copy
from queue import Queue
from model.Context_message import ContextMsg
if typing.TYPE_CHECKING:
    from task.task_agent import ProcessBase

class Data(object):
    def __init__(self, pid:int, size:int, data_id:typing.Tuple, data_type:str, 
                 io_time, processed_done_time=0, life_time=float('inf'), period:float=0,
                 event_time=0) -> None:
        self.pid = pid
        self.size = size
        self.data_id = data_id
        self.data_type = data_type
        self.valid = False
        self.waitTime = 0
        self.io_time = io_time
        self.processed_done_time = processed_done_time
        self.event_time = event_time
        self.life_time = life_time
        self.ctx:ContextMsg = None
        self.ref_pid = []
        self.period = period
    
    def serialize(self):
        return {"pid": self.pid, "data_id": self.data_id, 
                "size": self.size, "data_type": self.data_type, "io_time": self.io_time, 
                "event_time": self.processed_done_time, 
                "waitTime": self.waitTime, 
                "life_time": self.life_time}
    
    def cache_data_info(self):
        self.ctx.cache_data_info(self)

    def update_receive_time(self, time:int) -> None:
        self.ctx.update_receive_time(time)
    
    def cache_msg_transfer(self, time:int) -> None:
        self.ctx.cache_msg_transfer(time)
    
    def track_downstream(self) -> typing.List:
        return self.ctx.get_downstream_node()

class Buffer(object):
    def __init__(self, capacity:int=-1, sort_fn:typing.Callable=None) -> None:
        self.capacity = capacity
        self.buffer_w:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.buffer_i:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.buffer_o:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.remain_cap = capacity
        # event_time, data_id,
        self.sort_fn = sort_fn if sort_fn else lambda x: (x.pid, x.data_id)
        self.evict_timestamp = 0.

    
    def buffer_mux(self, data_type):
        if data_type == "weight":
            tgt_buffer = self.buffer_w
        elif data_type == "input":
            tgt_buffer = self.buffer_i
        elif data_type == "output":
            tgt_buffer = self.buffer_o
        return tgt_buffer

    def put(self, data:Data, verbose:bool=False):
        if self.capacity != -1 and self.remain_cap <= data.size:
            return False
        if verbose:
            print("put data: ", data.data_id)
        tgt_buffer = self.buffer_mux(data.data_type)

        if data.pid not in tgt_buffer:
            tgt_buffer[data.pid] = []
        tgt_buffer[data.pid].append(data)
        tgt_buffer[data.pid].sort(key=self.sort_fn)
        self.remain_cap -= data.size
        return True

    def pop(self, pid:int, data_type, verbose:bool=False):
        tgt_buffer = self.buffer_mux(data_type)

        if pid not in tgt_buffer:
            return False, None
        if verbose: 
            print("pop data: ", tgt_buffer[pid][0].data_id)
        data = tgt_buffer[pid].pop(0)
        self.remain_cap += data.size
        tgt_buffer[pid].sort(key=self.sort_fn)
        if len(tgt_buffer[pid]) == 0:
            del tgt_buffer[pid]
        return data
    
    def get(self, pid:int, data_type, verbose:bool=False):
        tgt_buffer = self.buffer_mux(data_type)
        if verbose: 
            print("get data: ", tgt_buffer[pid][0].data_id)
        return tgt_buffer[pid][0]
    
    def pop_timeout(self, data_type, curr_t, verbose:bool=False):
        LifeTimeModel.pop_timeout(self, data_type, curr_t, verbose)

    def recyle_no_ref(self, data_type, verbose:bool=False):
        LifeTimeModel.pop_timeout(self, data_type, verbose=verbose)

    def retain_most_recent(self, data_type, curr_t:float=None, verbose:bool=False):
        LifeTimeModel.retain_most_recent(self, data_type, curr_t, verbose=verbose)

# life time model for data
class LifeTimeModel(object):
    @staticmethod
    def _pop_timeout(buffer, data_type, curr_t, metric_fn, ref_clk, at_least_one=False, verbose:bool=False): 
        tgt_buffer = buffer.buffer_mux(data_type)
        pop_status = False
        for pid in list(tgt_buffer.keys()):
            # group the data by data_id, 
            # then sort by processed_done_time, descending
            # pick the last one as the most recent one
            keyed_by_data_id = {}

            # sort by metric_fn, then data_id(simplified sort_fn with same pid), ascending
            if at_least_one:
                tgt_buffer[pid].sort(key=lambda x: (x.data_id, metric_fn(x)), reverse=True)

                while tgt_buffer[pid]:
                    data = tgt_buffer[pid][0]
                    # preserved if it is the only one or it is valid at clock time
                    if data.data_id not in keyed_by_data_id:
                        keyed_by_data_id[data.data_id] = [data]
                    elif metric_fn(data) - ref_clk>= 0:
                        keyed_by_data_id[data.data_id].append(data)
                    elif verbose: 
                        print("pop data: ", data.data_id, data.pid, data.processed_done_time, data.life_time, curr_t)
                        buffer.remain_cap += data.size
                        pop_status = True
                    tgt_buffer[pid].pop(0)
                # flatten the dict and remove the data_id key
                tgt_buffer[pid] = [data for data_list in keyed_by_data_id.values() for data in data_list]
            else:
                tgt_buffer[pid].sort(key=lambda x: (metric_fn(x), x.data_id))
                while tgt_buffer[pid]:
                    data = tgt_buffer[pid][0]
                    if metric_fn(data) - ref_clk>= 0:
                        break
                    if verbose: 
                        print("pop data: ", data.data_id, data.pid, data.processed_done_time, data.life_time, curr_t)
                    tgt_buffer[pid].pop(0)
                    buffer.remain_cap += data.size
                    pop_status = True
            tgt_buffer[pid].sort(key=buffer.sort_fn)
            if len(tgt_buffer[pid]) == 0:
                del tgt_buffer[pid]
        return pop_status

    @ classmethod
    def pop_timeout(cls, buffer, data_type, curr_t, verbose:bool=False, mode="rel", at_least_one=False):
        if mode == "rel":
            # TODO: check again
            # metric_fn = lambda x: x.event_time + x.life_time
            metric_fn = lambda x: x.processed_done_time + x.life_time
            ref_clk = curr_t
            # the evicted data is the one with the event_time + life_time - curr_t < 0
            return cls._pop_timeout(buffer, data_type, curr_t, metric_fn, ref_clk, at_least_one, verbose)
        elif mode == "abs":
            metric_fn = lambda x: x.ctx.get_timestamp()
            ref_clk = buffer.evict_timestamp
            return cls._pop_timeout(buffer, data_type, curr_t, metric_fn, ref_clk, at_least_one, verbose)
        else:
            raise NotImplementedError

    @staticmethod
    def recyle_no_ref(buffer, data_type, curr_t:float=None, verbose:bool=False):
        sort_fn = lambda x: (x.data_id, x.event_time)
        tgt_buffer = buffer.buffer_mux(data_type)
        for pid in list(tgt_buffer.keys()):
            for data in tgt_buffer[pid]:
                if len(data.ref_pid) == 0:
                    if verbose: 
                        print("pop data: ", data.data_id)
                    tgt_buffer[pid].remove(data)
                    buffer.remain_cap += data.size
            tgt_buffer[pid].sort(key=sort_fn)
            if len(tgt_buffer[pid]) == 0:
                del tgt_buffer[pid]
    
    @staticmethod
    def retain_most_recent(buffer, data_type, curr_t:float=None, verbose:bool=False):
        sort_fn = lambda x: (x.data_id, x.processed_done_time)
        tgt_buffer = buffer.buffer_mux(data_type)
        for pid in list(tgt_buffer.keys()):
            # group the data by data_id, 
            # then sort by processed_done_time, descending
            # pick the last one as the most recent one
            keyed_by_data_id = {}
            tgt_buffer[pid].sort(key=sort_fn, reverse=True)
            while tgt_buffer[pid]:
                data = tgt_buffer[pid][0]
                if data.data_id not in keyed_by_data_id:
                    keyed_by_data_id[data.data_id] = data
                tgt_buffer[pid].pop(0)
            tgt_buffer[pid] = list(keyed_by_data_id.values())
            tgt_buffer[pid].sort(key=buffer.sort_fn)
            if len(tgt_buffer[pid]) == 0:
                del tgt_buffer[pid] 

class EventCache(object):
    def __init__(self, capacity: int=-1, type: str="data"):
        self.capacity = capacity
        self.buffer:typing.Dict[int, typing.Dict[int, typing.Dict]] = {}
        self.remain_cap = capacity
        self.sort_fn = lambda x: x.ctx.get_timestamp()
        self.type = type
    
    def new_process(self, _p:ProcessBase):
        if self.type == "data":
            self.buffer[_p.pid] = copy.deepcopy(_p.pred_data)
        elif self.type == "ctrl":
            self.buffer[_p.pid] = copy.deepcopy(_p.pred_ctrl)
    
    def get_pred_data(self, pid):
        return self.buffer[pid] 

    def put(self, pid, data: Data):
        self.buffer[pid][data.data_id]["event_queue"].put(data)
        if self.capacity > 0:
            self.remain_cap -= data.size
    
    def __getitem__(self, pid):
        return self.buffer[pid]

    # def new_event_queue(self, pid):
    #     self.buffer[pid] = TaskQueue(sort_f=lambda x: x.ctx.get_timestamp(), descending=False)
    
    # def put(self, data: Data):
    #     if data.pid not in self.buffer:
    #         self.new_event_queue(data.pid)
    #     self.buffer[data.pid].put(data)
    #     if self.capacity > 0:
    #         self.remain_cap -= data.size
    
    # def pop(self, pid, curr_t:float=None, verbose:bool=False):
    #     if pid not in self.buffer:
    #         return None
    #     data = self.buffer[pid].pop()
    #     if len(self.buffer[pid]) == 0:
    #         del self.buffer[pid]
    #     if data is not None and self.capacity > 0:
    #         self.remain_cap += data.size
    #     return data

class TriggerCache(EventCache):
    def __init__(self, capacity: int = -1, type: str = "ctrl"):
        super().__init__(capacity, type)
        self.sensor_cache = {}
    
    def new_process(self, _p:ProcessBase):
        super().new_process(_p)
        self.sensor_cache[_p.pid] = []
    

# class SensorCache():
#     def __init__(self, capacity: int=-1, type: str="data"):

if __name__ == "__main__":
    buffer_in = Buffer(100)
    in_data1 = Data(1, 2, (1, 1), "in")
    in_data2 = Data(1, 2, (1, 2), "in")
    in_data3 = Data(1, 2, (1, 3), "in")
    in_data4 = Data(2, 2, (2, 1), "in")
    in_data5 = Data(2, 2, (2, 2), "in")
    in_data6 = Data(2, 2, (2, 3), "in")

    buffer_in.put(in_data1)
    buffer_in.put(in_data2)
    buffer_in.put(in_data3)
    buffer_in.put(in_data4)
    buffer_in.put(in_data5)
    buffer_in.put(in_data6)
    
    for pid in buffer_in.buffer_w:
        print([q.data_id for q in buffer_in.buffer_w[pid]])
    print(buffer_in.remain_cap)

    buffer_in.pop(1)
    buffer_in.pop(1)
    buffer_in.pop(1)
    buffer_in.pop(2)
    buffer_in.pop(2)
    buffer_in.pop(2)
    for pid in buffer_in.buffer_w:
        print([q.data_id for q in buffer_in.buffer_w[pid]])
    print(buffer_in.remain_cap)




        

