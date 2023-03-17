"""
This file contains the Buffer class. 
This class models the buffer with the following attributes:
    - buffer: a dict, key is the task id, value is the data.
"""
import collections
import typing
from queue import Queue

class Data(object):
    def __init__(self, pid:int, size:int, data_id:typing.Tuple, data_type:str, io_time) -> None:
        self.pid = pid
        self.size = size
        self.data_id = data_id
        self.data_type = data_type
        self.valid = False
        self.waitTime = 0
        self.io_time = io_time
        self.event_time = 0
        self.life_time = -1

    def check_valid(self, curr_time):
        if self.event_time + self.io_time <= curr_time:
            self.valid = True
            return True
        return False
        

class Buffer(object):
    def __init__(self, capacity:int=-1, sort_fn:typing.Callable=None) -> None:
        self.capacity = capacity
        self.buffer:typing.OrderedDict[int, typing.List[Data]] = collections.OrderedDict()
        self.remain_cap = capacity
        self.sort_fn = sort_fn if sort_fn else lambda x: x.data_id

    
    def put(self, data:Data, verbose:bool=False):
        if self.capacity != -1 and self.remain_cap <= data.size:
            return False
        if verbose:
            print("put data: ", data.data_id)
        if data.pid not in self.buffer:
            self.buffer[data.pid] = []
        self.buffer[data.pid].append(data)
        self.buffer[data.pid].sort(key=self.sort_fn)
        self.remain_cap -= data.size
        return True

    def pop(self, pid:int, verbose:bool=False):
        if pid not in self.buffer:
            return False, None
        if verbose: 
            print("pop data: ", self.buffer[pid][0].data_id)
        data = self.buffer[pid].pop(0)
        self.remain_cap += data.size
        self.buffer[pid].sort(key=self.sort_fn)
        if len(self.buffer[pid]) == 0:
            del self.buffer[pid]
        return data
    
    def get(self, pid:int, verbose:bool=False):
        if verbose: 
            print("get data: ", self.buffer[pid][0].data_id)
        return self.buffer[pid][0]


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
    
    for pid in buffer_in.buffer:
        print([q.data_id for q in buffer_in.buffer[pid]])
    print(buffer_in.remain_cap)

    buffer_in.pop(1)
    buffer_in.pop(1)
    buffer_in.pop(1)
    buffer_in.pop(2)
    buffer_in.pop(2)
    buffer_in.pop(2)
    for pid in buffer_in.buffer:
        print([q.data_id for q in buffer_in.buffer[pid]])
    print(buffer_in.remain_cap)




        

