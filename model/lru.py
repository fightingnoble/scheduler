from collections import OrderedDict
from copy import deepcopy

class LRUCache:

    def __init__(self, capacity: int=-1):
        self.dict=OrderedDict()
        self.dict_neg=OrderedDict()
        self.remain=capacity
        self.bk = deepcopy(self.dict)
    
    def is_empty(self):
        return len(self.dict)==0
    
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
                # earliest item is popped
                self.dict.popitem(last=False)
        self.dict[key]=value
        if key in self.dict_neg:
            self.dict_neg.pop(key)

    def withdraw(self):
        delta_size = len(self.dict) - len(self.bk)
        self.dict = deepcopy(self.bk)
        self.remain += delta_size

    def get_lrp(self):
        """
        return the latest popped item
        """
        if len(self.dict_neg)==0:
            return None
        k = list(self.dict_neg.keys())[-1]
        return k
    
    def put_neg(self, key: int, value: int=None) -> None:
        if key in self.dict:
            if key != self.get_mru():
                # push back the item
                self.dict.move_to_end(key, last=False)
            else:
                self.dict_neg[key]=self.dict.pop(key)
                self.remain+=1
        else:
            self.dict_neg[key]=value

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