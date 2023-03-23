# from multiprocessing import Process, Queue
from queue import Queue

class MsgDispatcher:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        self.queues = [Queue() for _ in range(num_processes)]

    def broadcast_message(self, message, prefix=""):
        print(prefix+f"Broadcasting message: {message}")
        for queue in self.queues:
            queue.put(message)

#     def start_processes(self, target, args=()):
#         processes = []
#         for i in range(self.num_processes):
#             process = Process(target=target, args=args+(i,self.queues[i]))
#             process.start()
#             processes.append(process)
#         return processes

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
    