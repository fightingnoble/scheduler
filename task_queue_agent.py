from queue import Queue, PriorityQueue
# from algorithms.tree.red_black_tree.red_black_tree import RBTree
# from algorithms.heap.binary_heap import BinaryHeap

from typing import Dict, List, Tuple, Union, Any, OrderedDict


class TaskQueue(Queue):
    '''Variant of Queue that retrieves open entries in priority order (lowest first).

    Entries are typically tuples of the form:  (priority number, data).
    '''
    def __init__(self, init_list:List=[], maxsize: int = -1, descending=True, sort_f=None) -> None:
        super().__init__(maxsize)
        self.queue:List = []
        if init_list: 
            self.queue.extend(init_list)
        self.reverse = descending
        self.sort_f = sort_f
        self._sort()

    def _sort(self):
        if self.sort_f:
            self.queue.sort(key=self.sort_f, reverse=self.reverse)
        else:
            self.queue.sort(reverse=self.reverse)
        
    def _qsize(self):
        return len(self.queue)

    def _put(self, item):
        self.queue.append(item)
        self._sort()

    def _get(self):
        return self.queue.pop(0)

    def remove(self, item):
        self.queue.remove(item)
        self._sort()
    
    def __len__(self):
        return len(self.queue)
    
    def __iter__(self):
        return iter(self.queue)

    def __getitem__(self, index):
        return self.queue[index]
    
    def __setitem__(self, index, value):
        self.queue[index] = value
        self._sort()
    
    def __delitem__(self, index):
        del self.queue[index]
        self._sort()

    def clear(self):
        self.queue.clear()

        




if __name__ == "__main__": 
    hp = TaskQueue([11, 2, 14, 1, 7, 15, 5, 8, 4])
    print(hp.queue)
    hp.put(3)
    print(hp.queue)
    print(hp.get())
    print(hp.queue)

    