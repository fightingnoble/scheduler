from __future__ import annotations
from typing import Dict, List, Tuple, Union, Any, OrderedDict
# from collections import OrderedDict
import pprint
import numpy as np
import yaml
# from hw_rsc import FLOPS_PER_CORE
# from scheduler_global_cfg import *

class RscMapInt(OrderedDict[int, Tuple[int, ...]]): 
    title_line = "\tTaskID\t->\tRscSize\n"
    # TaskID -> RscSize
    def __str__(self) -> str:
        # return str(dict(self))
        _str = self.title_line
        for k, v in self.items():
            _str += f"\t{k}\t->\t{v}\n"
        return _str

    def print_simple(self, pid2name:Dict[int,str]=None) -> str:
        _str = ""
        for k, v in self.items():
            if pid2name: 
                _str += f"\t{pid2name[k]}\t->\t{v}\n"
            else:
                _str += f"\t{k}\t->\t{v}\n"
        return _str

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, RscMapInt):
            return dict(self) == dict(__o)
        
        
class Resource_model_int(object): 
    # rsc record the usage of the resource
    def __init__(self, size:int, id:int=None, name:str=None):
        self.id = id
        self.task_name = name
        # record the id and the num of the allocated cores
        self.rsc_map:OrderedDict[int, Tuple[int, int]] = RscMapInt()
        self.available_rsc = size
        self.size = size
        self.slot_e = None
        self.slot_s = None
        self.slot_num = None
    
    def update(self, other:RscMapInt):
        # update the rsc map
        self.rsc_map.clear()
        for k, v in other.items():
            self.rsc_map[k] = v
        # update the available rsc
        self.available_rsc = self.size
        for k, v in self.rsc_map.items():
            self.available_rsc -= v
        
    
    def get_available_rsc(self):
        return self.available_rsc
    
    def allocate(self, task_id:int, num:int, verbose:bool=False): 
        self.rsc_map[task_id] = num
        self.available_rsc -= num
        if verbose:
            _str = f"allocate {num} rsc to task {task_id} "
            if self.id: 
                _str += "on node {}\n".format(self.id)
            else:
                _str += "\n"
            print(_str)

    def release(self, task_id:int, num:int=0, verbose:bool=False):
        # release all the rsc allocated by the task
        if not num: 
            _str = f"release {self.rsc_map[task_id]}(all) rsc from task {task_id}"
            self.available_rsc += self.rsc_map[task_id]
            self.rsc_map.pop(task_id)
        # release the num of rsc
        else:
            _str = f"release {num} rsc from task {task_id}" 
            self.available_rsc += num
            self.rsc_map[task_id] -= num
            if self.rsc_map[task_id] == 0: 
                self.rsc_map.pop(task_id)
        if verbose:
            if self.id: 
                _str += "on node {}\n".format(self.id) 
            else: 
                _str += "\n"
            print(_str)
    
    def clear(self):
        self.rsc_map.clear()
        self.available_rsc = self.size

    def __str__(self) -> str: 
        return f"available_rsc: {self.available_rsc},\nrsc_map:\n {self.rsc_map}" 

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Resource_model_int):
            return self.available_rsc == __o.available_rsc and self.rsc_map == __o.rsc_map
    
    def get_plot_col(self): 
        col = []
        for k, v in self.rsc_map.items():
            for i in range(v): 
                col.append(k)
        col += [0] * (self.size - len(col))
        return col


class DDL_reservation(object):
    def __init__(self, main_size, RDA_size,):
        self.main_size = main_size
        self.RDA_size = RDA_size
        self.main_rsc = Resource_model_int(main_size)
        self.RDA_rsc = Resource_model_int(RDA_size)
        self.rsc_map:OrderedDict[int, Tuple[int, int]] = RscMapInt()
        self.available_rsc = main_size + RDA_size

    def get_available_rsc(self):
        return self.available_rsc

    def allocate(self, task_id:int=0, main_num:int=0, RDA_num:int=0, verbose:bool=False):
        self.rsc_map[task_id] = (main_num, RDA_num)
        if verbose: 
            print("main_rsc: ")
        self.main_rsc.allocate(task_id, main_num, verbose)
        if verbose: 
            print("main_rsc: ")
        self.RDA_rsc.allocate(task_id, RDA_num, verbose)
        self.available_rsc = (self.main_rsc.get_available_rsc() + self.RDA_rsc.get_available_rsc())
    
    def release(self, task_id:int=0, main_num:int=0, RDA_num:int=0, verbose:bool=False): 
        if verbose: 
            print("main_rsc: ")
        self.main_rsc.release(task_id, main_num, verbose)
        if verbose: 
            print("main_rsc: ")
        self.RDA_rsc.release(task_id, RDA_num, verbose)
        if task_id in self.main_rsc.rsc_map or task_id in self.RDA_rsc.rsc_map:
            if self.main_rsc.rsc_map[task_id] or self.RDA_rsc.rsc_map[task_id]: 
                self.rsc_map[task_id] = (self.main_rsc.rsc_map[task_id], self.RDA_rsc.rsc_map[task_id])
        else:
            self.rsc_map.pop(task_id)
        self.available_rsc = (self.main_rsc.get_available_rsc() + self.RDA_rsc.get_available_rsc())
    
    def __str__(self) -> str:
        _str = "main_size: {}, RDA_size: {}, available_rsc: {}, \n".format(
            self.main_size, self.RDA_size, self.available_rsc, 
        )
        _str += "rsc_map: \n{}".format(str(self.rsc_map))
        # _str += "main_rsc: {}, \n".format(str(self.main_rsc))
        # _str += "RDA_rsc: {}, \n".format(str(self.RDA_rsc))
        _str += "main_available_rsc: {}, \n".format(self.main_rsc.get_available_rsc())
        _str += "RDA_available_rsc: {}, \n".format(self.RDA_rsc.get_available_rsc())
        return _str

class RT_reservation(DDL_reservation): 
    def __init__(self, main_size, RDA_size):
        assert RDA_size == 0
        super().__init__(main_size, RDA_size)

    def allocate(self, task_id: int, main_num: int, RDA_num: int, verbose: bool = False):
        assert RDA_num == 0
        return super().allocate(task_id, main_num, RDA_num, verbose)

    def release(self, task_id: int = 0, main_num: int = 0, RDA_num: int = 0, verbose: bool = False):
        assert RDA_num == 0
        return super().release(task_id, main_num, RDA_num, verbose)
    
class dummy_reservation(DDL_reservation): 
    def __init__(self, main_size, RDA_size):
        assert RDA_size == 0 and main_size == 0
        super().__init__(main_size, RDA_size)
    
    def allocate(self, task_id:int, main_num:int, RDA_num:int=0):
        assert RDA_num == 0 and main_num == 0
        return super().allocate(task_id, main_num, RDA_num)
    
    def release(self, task_id:int=0, main_num:int=0, RDA_num:int=0):
        assert RDA_num == 0 and main_num == 0
        return super().release(task_id, main_num, RDA_num)


if __name__ == "__main__": 
    verbose  = True
    rsc = DDL_reservation(10, 10)
    rsc.allocate(1, 5, 5, verbose)
    print(rsc)
    rsc.release(1, verbose=verbose)
    print(rsc)