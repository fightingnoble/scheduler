"""
This file contains the Buffer class. 
This class models the buffer with the following attributes:
    - buffer: a dict, key is the task id, value is the data.
"""
import collections
import typing
from queue import Queue

class Data(object):
    def __init__(self, pid:int, size:int, data_id:typing.Tuple, data_type:str, io_time, event_time=0, life_time=-1) -> None:
        self.pid = pid
        self.size = size
        self.data_id = data_id
        self.data_type = data_type
        self.valid = False
        self.waitTime = 0
        self.io_time = io_time
        self.event_time = event_time
        self.life_time = life_time

    def check_valid(self, curr_time):
        if self.event_time + self.io_time <= curr_time:
            self.valid = True
            return True
        return False
    
        

class Buffer(object):
    def __init__(self, capacity:int=-1, sort_fn:typing.Callable=None) -> None:
        self.capacity = capacity
        self.buffer_w:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.buffer_i:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.buffer_o:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.remain_cap = capacity
        self.sort_fn = sort_fn if sort_fn else lambda x: x.data_id

    
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
        tgt_buffer = self.buffer_mux(data_type)
        pop_status = False
        for pid in list(tgt_buffer.keys()):
            # sort by data.event_time + data.life_time - curr_t, then sort_fn, ascending
            # the evicted data is the one with the event_time + life_time - curr_t < 0
            tgt_buffer[pid].sort(key=lambda x: (x.event_time + x.life_time - curr_t, self.sort_fn(x)))
            while tgt_buffer[pid]:
                data = tgt_buffer[pid][0]
                if data.event_time + data.life_time - curr_t >= 0:
                    break
                if verbose: 
                    print("pop data: ", data.data_id, data.pid, data.event_time, data.life_time, curr_t)
                tgt_buffer[pid].pop(0)
                self.remain_cap += data.size
                pop_status = True
            tgt_buffer[pid].sort(key=self.sort_fn)
            if len(tgt_buffer[pid]) == 0:
                del tgt_buffer[pid]


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




        

