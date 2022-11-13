from typing import Dict
import numpy as np
FLOPS_PER_CORE = 0.5
from noc import NoC_model, ideal_model, aggregate_model

# class Spatial(object):
#     def __init__(
#         self, 
#         main_memory_size=1e12,
#         core_shape=(16, 16),
#         pe_shape=(16, 16),
#         mac_shape=(16,),
#         gb_buffer_size=(65536,),
#         local_buffer_size=(589824,),
#         reg_size=(1,),
#         Noc_model:str="ideal",
#         Noc_params:Dict={"ideal_bw": 128*1e9}
#         ) -> None:
#         self.main_memory = np.zeros_like(main_memory_size)
#         self.core = np.zeros_like(core_shape)
#         self.pe = np.zeros_like(core_shape + pe_shape)
#         self.mac = np.zeros_like(core_shape + pe_shape + mac_shape)
#         self.gb_buffer = np.zeros_like(gb_buffer_size)
#         self.reg = np.zeros_like(reg_size)
#         self.keys = ["core", "pe", "mac", "main_memory", "buffer", "reg", "keys"]
#         self.noc = ideal_model(**Noc_params) if Noc_model=="ideal" else aggregate_model(**Noc_params)
    
#     def allocate(self, reso_list:Dict):
#         for k, v in reso_list.items():
#             assert k in self.keys
#             self.__getattribute__[k][v] = 1
    
#     def release(self, reso_list:Dict):
#         for k, v in reso_list.items():
#             assert k in self.keys
#             self.__getattribute__[k][v] = 0

#     def __getitem__(self, key):
#         if key in self.keys:
#             return self.__getattribute__[key]
#         else:
#             raise "Unedpected key:{}".format(key)            


class Storage(object):
    def __init__(self, size:int=1e12, bandwidth:int=128*1e9) -> None:
        self.size = size

class Compute(object):
    pass

class LMAC(Compute):
    def __init__(self, size, **kwargs) -> None:
        pass

class reg_storage(Storage):
    def __init__(self, data_width, **kwargs) -> None:
        pass

class smartbuffer_SRAM(Storage):
    def __init__(self, data_width, **kwargs) -> None:
        pass

class smartbuffer_RF(Storage):
    def __init__(self, data_width, **kwargs) -> None:
        pass

class PE(Compute):
    def __init__(self, Mac_size, **kwargs) -> None:
        pass

class Core(Compute):
    def __init__(self, pe_shape, **kwargs) -> None:
        pass

class Spatial(object):
    def __init__(self,
        core_shape=(16, 16),
        noc_model:str="ideal",
        noc_params:Dict={"ideal_bw": 128*1e9},
        **kwargs
        ) -> None:
        self.core = np.zeros_like(core_shape)
        self.noc = ideal_model(**noc_params) if noc_model=="ideal" else aggregate_model(**Noc_params)
