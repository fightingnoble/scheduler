
from ctypes import Union
from typing import Collection, Dict, List
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d

class NoC_model(object):
    def __init__(self) -> None:
        self.state = None
        self.last_timestamp = 0
        pass

    def update_overtime(self, time_elapsed): 
        # update status
        pass

    def update_inject(self, packet_size, src, dest, current_time):
        # update status
        pass

    def query_bw(self, packet_size, src, dest, current_time):
        assert current_time>=self.last_timestamp
        time_elapsed = current_time - self.last_timestamp
        self.update_overtime(time_elapsed)
        self.last_timestamp = current_time
        return self.get_equiv_bw(packet_size, src, dest,)

    def get_equiv_bw(self, packet_size, src, dest, ):
        "read out the equivelent bandwidth from current state"
        pass

class ideal_model(NoC_model):
    def __init__(self, ideal_bw) -> None:
        super().__init__()
        self.ideal_bw = ideal_bw
    
    def get_equiv_bw(self, packet_size=None, src=None, dest=None):
        return self.ideal_bw

    def get_comm_latency(self, s_id, d_id, packet_size):
        basic_latency = 0
        interference = 0
        return basic_latency + interference

class aggregate_model(NoC_model):
    def __init__(self, transfer_curve:List[Union(np.ndarray, List), Union(np.ndarray, List)]) -> None:
        super().__init__()
        x, y = transfer_curve
        tck = interpolate.splrep(x, y, s=0)
        self.state = {
            "remained_packets_size": 0, 
            "transfer_curve": tck
        }

    def update_inject(self, packet_size, src, dest, current_time):
        self.state["remained_packets_size"] += packet_size
    
    def get_equiv_bw(self, packet_size, src, dest):
        return interpolate.splev(
            self.state["remained_packets_size"], 
            self.state["transfer_curve"], der=0
        )


