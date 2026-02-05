from __future__ import annotations
import typing
if typing.TYPE_CHECKING:
    from task.task_agent import ProcessInt
    from model.task_queue_agent import TaskQueue 
    from model.resource_agent import Resource_model_int
    from sched.scheduling_table import SchedulingTableInt

from global_var import *
from typing import Dict, List, Tuple, Union, Any, OrderedDict
from model.event_gen.e2e_latency import exp_jitter
from model.performance import slack_comp
from functools import reduce

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
    def __init__(self, size:int, id:int=None, name:str=None, exec_var_en=False, exec_var_para=None, seed=0):
        self.id = id
        self.task_name = name
        # record the id and the num of the allocated cores
        self.rsc_map:OrderedDict[int, Tuple[int, int]] = RscMapInt()
        self.available_rsc = size
        self.size = size
        self.slot_e = None
        self.slot_s = None
        self.slot_num = None
        self.flops_dict= dict()
        self.exec_var_en = exec_var_en
        self.exec_var_para = exec_var_para
        if exec_var_en:
            self.var_gen = exp_jitter(1, exec_var_para, size=1, seed=seed) 
            # self.get_real_ops = lambda exp_ops: (1-self.var_gen()) * exp_ops
            self.get_real_ops = lambda exp_ops: slack_comp(exp_ops, 0, self.var_gen())
        self.event_list = []
        
    def add_rsc_num(self, num:int):
        self.size += num
        self.available_rsc += num
    
    def update(self, other:RscMapInt):
        # update the rsc map
        self.rsc_map.clear()
        for k, v in other.items():
            if v:
                self.rsc_map[k] = v
        # update the available rsc
        self.available_rsc = self.size
        for k, v in self.rsc_map.items():
            self.available_rsc -= v
        
    def is_empty(self):
        return not bool(self.rsc_map)
    
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

    def updateRunningQueue(res_cfg:Resource_model_int, timestep, running_queue:TaskQueue, update_budget=False, bin_id=None, 
                           mode="verify", skiped_tasks:List[ProcessInt]=[]):
        """
        verify: True:
                    running queue is assumed to have same task id as the res_cfg.rsc_map, we use res_cfg.rsc_map to update 
                        the progress of the task in running queue.
                False:
                    **finished** tasks is removed from the running queue but keep the allocated resources unchanged, 
                        to ensure the guardband.
        """
        _p_dict = {p.pid:p for p in running_queue} 
        assert mode in ["verify", "normal"]
        for pid in res_cfg.rsc_map.keys():
            if mode!="verify" and pid not in _p_dict.keys():
                continue
            _p:ProcessInt = _p_dict[pid]
            if _p in skiped_tasks:
                continue
            ops = res_cfg.rsc_map[_p.pid]*timestep*FLOPS_PER_CORE
            # ops can not exceed the (totcpu-totburst) and (_p.rem_flop_budget[bin_id])
            # total _p.rem_flop_budget may exceed the totcpu, 
            # if rem_flop_budget is overly estimated or new budget is complemented before current task finished
            # totcpu-totburst also may exceed the _p.rem_flop_budget
            if res_cfg.exec_var_en:
                ops = elim_error(res_cfg.get_real_ops(ops), numerical_tol_bit, numerical_error_tol_abs, 'up') 
            if update_budget:
                ops = min(ops, _p.rem_flop_budget[bin_id], _p.totcpu-_p.totburst)
            else:
                ops = min(ops, _p.totcpu-_p.totburst)
            _p.currentburst = elim_nume_error(_p.currentburst + ops)
            _p.burst = elim_nume_error(_p.burst + ops)
            _p.totburst = elim_nume_error(_p.totburst + ops)
            _p.remburst = elim_nume_error(_p.remburst - ops)
            _p.cumulative_executed_time = elim_nume_error(_p.cumulative_executed_time + timestep)
            if update_budget:
                assert bin_id is not None
                _p.rem_flop_budget[bin_id] = elim_nume_error(_p.rem_flop_budget[bin_id]-ops)
                pass

    def action_at_start_cfg(curr_cfg:Resource_model_int, budget_recoder, process_dict, bin_id):
        """update the budget at the start of a configuration chunk
            load the budget at the beginning of each cfg chunk rather than the end, 
            which is important as the cfg may be not consecutive, 
            loading at the end may lead to launch some kernels too early. 
            ******************************************************
            Mechanism of budget and progress recoder
            1. rem_flop_budget
            Record the expected operators to be executed in the next few moments
            
            2. budget_recoder: 
            record the upper bound of resource consumption (spatial and temporal)
            The previous budget is covered, when the new chunk is entered.
            Explanation: 
            If the load of privious chunk is uncompleted,
            the previous timeout budget is useless, 
            because the comming computation should be allocated with resources as soon as ponssible
            ******************************************************
            
        Args:
            curr_cfg (Resource_model_int): 
            budget_recoder (_type_): _description_
            rsc_recoder_his (_type_): _description_
            process_dict (_type_): _description_
            bin_id (_type_): _description_

        """
        cfg_slot_s, next_cfg, cfg_slot_num = curr_cfg.slot_s, curr_cfg.rsc_map, curr_cfg.slot_num
        cfg_flops_dict = curr_cfg.flops_dict 
        # replenish the budget
        for pid in next_cfg.keys():
            _p = process_dict[pid]
                        
            if bin_id not in _p.rem_flop_budget:
                _p.rem_flop_budget[bin_id] = 0
                            
            flops_tbd = cfg_flops_dict[pid]
            rem_flop_budget=_p.rem_flop_budget[bin_id]
            if rem_flop_budget> numerical_error_tol_abs or flops_tbd>numerical_error_tol_abs: 
                _p.rem_flop_budget[bin_id] += flops_tbd # * _p.var_scale_factor
                budget_recoder[pid] = [cfg_slot_s, next_cfg[pid], cfg_slot_num, True]

                
    def action_at_end_cfg(curr_cfg:Resource_model_int, timestep, _SchedTab:SchedulingTableInt, tab_temp_size, tab_pointer, hyper_p_n):
        """update the configuration at the end of a configuration chunk

        Args:
            curr_cfg (Resource_model_int): _description_
            timestep (_type_): _description_
            _SchedTab (SchedulingTable): 
            tab_temp_size (_type_): _description_
            tab_pointer (_type_): _description_
            hyper_p_n (_type_): _description_

        Returns:
            _type_: _description_
        """
        # get the next configuration
        cfg_slot_s, next_cfg, cfg_slot_num = _SchedTab.next_item()
        
        # update the rsc_map
        if _SchedTab.alloc_mod == 'compress':
            curr_cfg.update(_SchedTab.sparse_cores[_SchedTab.sparse_idx][1])
        else:
            curr_cfg.update(next_cfg)
        
        # update the deadline
        if cfg_slot_s < tab_pointer: 
            curr_cfg.slot_s = (hyper_p_n + 1) * tab_temp_size + cfg_slot_s
        else:
            curr_cfg.slot_s = hyper_p_n * tab_temp_size + cfg_slot_s
        curr_cfg.slot_e = curr_cfg.slot_s + cfg_slot_num - 1 
        curr_cfg.slot_num = cfg_slot_num
        
        # update the budget
        curr_cfg.flops_dict.clear()
        curr_cfg.flops_dict.update(_SchedTab.sparse_flops[_SchedTab.sparse_idx][1])

        # if _SchedTab.alloc_mod in ['overtime', 'compress']:
        #     curr_cfg.flops_dict.update(_SchedTab.sparse_flops[_SchedTab.sparse_idx][1])
        # else:
        #     curr_cfg.flops_dict.update({pid: n_cores * cfg_slot_num * timestep * FLOPS_PER_CORE for pid, n_cores in next_cfg.items()})

        # 2. clear the old events
        curr_cfg.event_list.clear()
        # 3. process new events
        # list((head_event_time, list((slot_idx, Dict[pid, event_set]))))
        curr_cfg.event_list.extend(map(lambda x:(x[0], reduce(lambda x,y: x+y, map(lambda x:list(x), x[1].values()))), _SchedTab.sparse_event[_SchedTab.sparse_idx][1]))
        # if _SchedTab.sparse_event[_SchedTab.sparse_idx]:
        #     event_set = reduce(lambda x,y: x+y, map(lambda x:list(x), _SchedTab.sparse_event[_SchedTab.sparse_idx][1].values()))
        #     curr_cfg.event_list.extend(list(event_set))
        
        return next_cfg



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