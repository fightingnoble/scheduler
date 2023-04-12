from collections import OrderedDict
from copy import deepcopy

class LRUCache:

    def __init__(self, capacity: int=-1):
        self.dict=OrderedDict()
        self.remain=capacity
        self.bk = deepcopy(self.dict)
    
    def get(self, key: int) -> int:
        if key not in self.dict:
            return -1
        else:
            v=self.dict.pop(key)
            self.dict[key]=v
            return v
        

    def put(self, key: int, value: int=None) -> None:
        self.bk = deepcopy(self.dict)
        if key in self.dict:
            self.dict.pop(key)
        else:
            if self.remain>0:
                self.remain-=1
            elif self.remain==0:
                # Pairs are returned in FIFO order if false.
                # first item is popped
                self.dict.popitem(last=False)
        self.dict[key]=value

    def withdraw(self):
        delta_size = len(self.dict) - len(self.bk)
        self.dict = deepcopy(self.bk)
        self.remain += delta_size

    def get_lru(self):
        """
        return the least recently used item
        """
        k = list(self.dict.keys())[0]
        return k
    
    def get_mru(self):
        """
        return the most recently used item
        """
        k = list(self.dict.keys())[-1]
        return k
    
# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.get(1))       # 返回  1
    cache.put(3, 3)    # 该操作会使得密钥 2 作废
    print(cache.get(2))       # 返回 -1 (未找到)
    cache.put(4, 4)    # 该操作会使得密钥 1 作废
    print(cache.get(1))       # 返回 -1 (未找到)
    print(cache.get(3))       # 返回  3
    print(cache.get(4))       # 返回  4