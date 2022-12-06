from queue import Queue, PriorityQueue
# from algorithms.tree.red_black_tree.red_black_tree import RBTree
# from algorithms.heap.binary_heap import BinaryHeap

from typing import Dict, List, Tuple, Union, Any, OrderedDict


class RunableQueue(Queue):
    '''Variant of Queue that retrieves open entries in priority order (lowest first).

    Entries are typically tuples of the form:  (priority number, data).
    '''
    def __init__(self, init_list:List=[], maxsize: int = -1) -> None:
        super().__init__(maxsize)
        self.queue:List = init_list
        self.queue.sort(reverse=True)

    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        heap = self.queue
        heap.append(item)
        heap.sort(reverse=True)

    def _get(self):
        return self.queue.pop(0)




if __name__ == "__main__": 
    hp = RunableQueue([11, 2, 14, 1, 7, 15, 5, 8, 4])
    print(hp.queue)
    hp.put(3)
    print(hp.queue)
    print(hp.get())
    print(hp.queue)

    