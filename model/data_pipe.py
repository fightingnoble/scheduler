# this file models a data pipe, 
# which is used to model the data transfer between multiple nodes.
# the data pipe has the following attributes: 
#     - capacity: the capacity of the pipe, in terms of the number of data items.
#     - buffer: a queue, which stores the data items.
#     - remain_cap: the remaining capacity of the pipe.
#     - sort_fn: a function used to sort the data items in the buffer.
#     - data_type: the type of the data items, e.g., weight, input, output.

# the data pipe has the following methods:
#     - put: put a data item into the pipe, send the data to the target node.
#     - get: get a data item from the pipe, receive the data from the source node.
#     - get_all: get all the data items from the pipe.
#     - get_all_by_type: get all the data items of a certain type from the pipe.
#     - get_all_by_id: get all the data items of a certain id from the pipe.
#     - update_wait_time: update the wait time of the data items in the pipe.
#     - sim_delay: simulate the delay of the data transfer, pop the data finished transferring.

import math
from model.task_queue_agent import TaskQueue
from model.buffer import Data, Buffer
from typing import List, Dict, Callable

class DataPipe:
    def __init__(self, data_type, num_reciever:int, 
                 capacity:int=-1, queues_list:List[TaskQueue]=None,
                 sort_fn:Callable=None
                 ) -> None:
        # ["unicast", data, dest] sorted by max(io_time-waitTime, 0)
        self.buffer = TaskQueue(descending=False, sort_f=lambda x: max(x[1].io_time-x[1].waitTime, 0))
        self.remain_cap = capacity
        self.sort_fn = None 
        self.data_type = data_type
        self.sort_fn = sort_fn if sort_fn else lambda x: x.data_id
        if queues_list is None:
            self.queues = [TaskQueue(sort_f=self.sort_fn) for _ in range(num_reciever)]
        else:
            self.queues = queues_list
    
    def put(self, data:Data, mode:bool="broadcast", dest:List[int]=None,):
        if mode == "broadcast":
            self.buffer.put(["broadcast", data, []])
        elif mode == "unicast":            
            self.buffer.put(["unicast", data, dest])
        else:
            raise ValueError("Unknown mode")
        
        if self.remain_cap != -1:
            self.remain_cap -= data.size
    
    def get(self, queue_id:int=0):
        return self.queues[queue_id].get()
    
    def update_wait_time(self, timestep):
        """
        Update the waiting time used in waiting queue
        """
        for mode, data, dest in self.buffer.queue: 
            data.waitTime += timestep

    def data_tranfer_sim(self, curr_t, DEBUG=False):
        # waitingQueue[i]->waitTime != 0 && waitingQueue[i]->waitTime % waitingQueue[i]->io == 0
        l_ready = []
        # for _p in wait_queue:
        #     trans_io_tile = _p.waitTime / _p.io_time
        while self.buffer.queue:
            data:Data
            mode, data, dest = self.buffer.queue[0]
            # trans_io_tile = data.waitTime / data.io_time
            # # trans_io_tile_r = round(trans_io_tile)
            # # trans_comp = np.allclose(trans_io_tile, trans_io_tile_r, atol=1e-2)
            # trans_io_tile_r = int(trans_io_tile)
            # trans_comp = trans_io_tile_r > 1
            # # TODO: _p.waitTime > 0 
            # if trans_comp and data.waitTime > 0: 
            if math.isclose(data.waitTime, data.io_time, rel_tol=1e-2) or data.waitTime > data.io_time:
                self.remain_cap += data.size
                data.valid = True
                data.update_receive_time(curr_t)
                self.buffer.get()
                if mode == "broadcast":
                    self.broadcast_message(data, prefix="  ", DEBUG=DEBUG)
                elif mode == "unicast":
                    for reciever_id in dest:
                        self.one_to_one_message(data, reciever_id, prefix="  ", DEBUG=DEBUG)
            else:
                break

    def broadcast_message(self, data:Data, prefix="", DEBUG=False):
        if DEBUG:
            print(prefix+f"Broadcasting message: {data.pid:d}:{data.data_id}({data.data_type}/{data.size}M)")
        for buffer in self.queues:
            buffer.put(data)
    
    def one_to_one_message(self, data:Data, reciever_id:int, prefix="", DEBUG=False):
        if DEBUG:
            print(prefix+f"Sending message: {data.pid:d}:{data.data_id}({data.data_type}/{data.size}M) to {reciever_id}")
        self.queues[reciever_id].put(data)

# data.dest = [dest_table[tgt_pid] for tgt_pid in data.ctx.msg_context["process_info"]["downstream_node"]] 