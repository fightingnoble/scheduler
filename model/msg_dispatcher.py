# from multiprocessing import Process, Queue
from queue import Queue
from model.buffer import Data

class MsgDispatcher:
    def __init__(self, num_queues, queue_list=None):
        self.num_processes = num_queues
        if queue_list is None:
            self.queues = [Queue() for _ in range(num_queues)]
        else:
            self.queues = queue_list

    def broadcast_message(self, message, prefix=""):
        print(prefix+f"Broadcasting message: {message}")
        for queue in self.queues:
            queue.put(message)
    
    def send_message(self, queue_id, message, prefix=""):
        print(prefix+f"Sending message to queue {queue_id}: {message}")
        self.queues[queue_id].put(message)

#     def start_processes(self, target, args=()):
#         processes = []
#         for i in range(self.num_processes):
#             process = Process(target=target, args=args+(i,self.queues[i]))
#             process.start()
#             processes.append(process)
#         return processes

def msg_read(msg_queue, curr_t, glb_name_p_dict, process_dict, buffer, bin_name, bin_event_flg):
    msg_list = []
    # read out all message and clear the message pipe
    while not msg_queue.empty():
        msg_list.append(msg_queue.get())
    if msg_list:
        for _p in process_dict.values():
            for key, attr in _p.pred_data.items():
                # if msg_pipe.filter(key):
                if msg_filter(msg_list, key):
                    attr["valid"] = True
                    attr["time"] = curr_t
                    # TODO: fix the event time as the actual time
                    if bin_name and not bin_event_flg:
                        bin_event_flg = True
                        print(f"({bin_name})")
                    print(f"		{_p.task.name} received event {key:s} @ {curr_t:.6f}")
                    buffer.put(Data(glb_name_p_dict[key].pid, 1, (0,), "output", glb_name_p_dict[key].io_time, curr_t, 1/glb_name_p_dict[key].task.freq))
    return bin_event_flg

def msg_filter(msg_list:list, keyword:str):
    return [msg for msg in msg_list if keyword in msg]

# def test_msg_dispatcher():
#     num_processes = 3
#     msg_dispatcher = MsgDispatcher(num_processes)

#     def target_func(process_id, queue):
#         while True:
#             message = queue.get()
#             if message == "STOP":
#                 break
#             print(f"Process {process_id} received message: {message}")

#     processes = msg_dispatcher.start_processes(target_func)
#     msg_dispatcher.broadcast_message("Hello, world!")
#     msg_dispatcher.broadcast_message("Goodbye, world!")
#     for queue in msg_dispatcher.queues:
#         queue.put("STOP")
#     for process in processes:
#         process.join()

# if __name__ == '__main__':
#     test_msg_dispatcher()
    