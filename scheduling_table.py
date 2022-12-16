import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Iterable
from collections import OrderedDict
from resource_agent import Resource_model_int
from functools import reduce
from task_agent import TaskInt

class SchedulingTableInt(object): 
    """
    =============== 1. scheduling table ===============
    scheduling table is a 2D array, each row is a resource, each column is a time slot
    the value of each cell is the task id that occupies the resource in the time slot
    the size of the scheduling table is determined by the number of resources and the number of time slots
    the number of time slots is determined by the hyper-period of the tasks
    the number of resources is determined by the number of resources that are available
    """
    def __init__(self, num_resources: int, num_time_slots:int, id: int = None, name: str = None):
        # self.scheduling_table = np.zeros((num_resources, num_time_slots), dtype=int)
        # self.scheduling_table = np.full((num_time_slots), Resource_model_int(num_resources, id, name), dtype=Resource_model_int)
        self.scheduling_table = np.array([Resource_model_int(num_resources, id, name) for _ in range(num_time_slots)], dtype=Resource_model_int)
        

    def index_occupy_by_id(self, time_slot_s, time_slot_e):
        """
        get the task id that occupies the resource
        """
        # rsc_array = self.scheduling_table[:, time_slot_s:time_slot_e]
        # rsc_occp = []
        # for id in np.unique(rsc_array[rsc_array != 0]):
        #     rsc_occp.append((id, np.where(rsc_array == id)[0]))
        # return np.where(rsc_array == 0)[0], rsc_occp
        
        # get task_id set
        task_id_set = set()
        rsc_agents_arr = self.scheduling_table[time_slot_s:time_slot_e]
        for rsc_agent in rsc_agents_arr:
            task_id_set.update(rsc_agent.rsc_map.keys())
        Scheduling_table_index_by_task_id = OrderedDict()
        for task_id in task_id_set: 
            Scheduling_table_index_by_task_id[task_id] = []
        for rsc_agent in rsc_agents_arr:
            for task_id in task_id_set:
                Scheduling_table_index_by_task_id[task_id].append(rsc_agent.rsc_map.get(task_id, 0))
        return Scheduling_table_index_by_task_id

    def idx_free_by_slot(self, time_slot_s, time_slot_e):
        """
        get the available resources in the time slot
        """
        # culculate available resources at each time slot
        rsc_maps_arr = self.scheduling_table[time_slot_s:time_slot_e]
        rsc_avl = []
        for rsc_map in rsc_maps_arr:
            rsc_avl.append(rsc_map.get_available_rsc())
        return rsc_avl
    
    def insert_task(self, task:TaskInt, req_rsc_size, time_slot_s, time_slot_e, expected_slot_num, 
                    verbose=False)->Tuple[bool, Union[int,List[int]], Union[int,List[int]], Union[int,List[int]]]: 
        """
        play a insert-based scheduling: 
        1. search available tensor cores at each slot
        2. insert the task into the scheduling table at a proper interval (here we adapt First-Fit)
            return success or not, the start time slot, the allocated resources, the allocated time slots
        3. release the resources given the list of allocated resources and the list of allocated time slots
        """ 
        rsc_avl = self.idx_free_by_slot(time_slot_s, time_slot_e)
        rsc_avl = np.array(rsc_avl)
        # check if the task can be scheduled with expected resources
        # the task can be scheduled at any time slot
        if np.all(rsc_avl > req_rsc_size):
            # allocate resources 
            for rsc_map in self.scheduling_table[time_slot_s:time_slot_s+expected_slot_num]:
                rsc_map.allocate(task.id, req_rsc_size, verbose)
            return True, time_slot_s, req_rsc_size, expected_slot_num
        else: 
            # divide the rsc_avl into intervals
            boader = (rsc_avl[0:-1] != rsc_avl[1:]).nonzero()[0] + 1
            s = [0] + boader.tolist() 
            e = boader.tolist() + [len(rsc_avl)] 
            # check if the task can be executed on the current interval
            for i in range(len(s)):
                if (e[i] - s[i]) >= expected_slot_num and np.all(rsc_avl[s[i]:e[i]] > req_rsc_size):
                    # allocate resources 
                    for rsc_map in self.scheduling_table[s[i]:s[i]+expected_slot_num]:
                        rsc_map.allocate(task.id, req_rsc_size, verbose)
                    return True, s[i], req_rsc_size, expected_slot_num

            # redistribute the resources to the intervals
            # based on the priciple of as soon as possible
            
            # current allocation (C)
            curr_alloc = np.zeros(len(s), dtype=int)
            curr_slot = np.zeros(len(s), dtype=int)

            # avalable (A)
            rsc_avl_tmp = np.zeros(len(s)) 

            # required (R)
            expected_req_rsc_size = req_rsc_size * expected_slot_num
            cum_rsc_alloc = 0
            cum_slot_length = 0

            if rsc_avl[:expected_slot_num].sum() < expected_req_rsc_size: 
                # as soon as possible
                for i in range(len(s)): 
                    if rsc_avl[s[i]] > 0:
                        cond2 = rsc_avl[:e[i]].sum() >= expected_req_rsc_size
                        # rsc size 
                        curr_alloc[i] = rsc_avl[s[i]]
                        # slot length
                        if cond2:
                            curr_slot[i] = np.ceil((expected_req_rsc_size - rsc_avl[:e[i-1]].sum())/rsc_avl[s[i]]).astype(int)
                            break
                        else:
                            curr_slot[i] = int(e[i] - s[i])
            else: 
                # there is enough rsc in 
                # [0, expected_slot_num] + time_slot_s
                # try to distribute the rsc_lack to the intervals as evenly as possible
                
                # calculate the lacked resources in the interval: 
                for i in range(len(s)): 
                    if rsc_avl[s[i]] > 0:
                        # rsc size 
                        if rsc_avl[s[i]] < req_rsc_size:
                            curr_alloc[i] = rsc_avl[s[i]]
                            rsc_avl_tmp[i] = 0
                        else:
                            curr_alloc[i] = req_rsc_size
                            rsc_avl_tmp[i] = rsc_avl[s[i]] - req_rsc_size
                        # slot length
                        
                        cond1 = (e[i] - s[i]) >= (expected_slot_num - cum_slot_length)
                        if cond1:
                            curr_slot[i] = int(expected_slot_num - cum_slot_length)
                        else: 
                            curr_slot[i] = int(e[i] - s[i])

                        cum_rsc_alloc += curr_alloc[i] * curr_slot[i]
                        cum_slot_length += curr_slot[i]
                        # stop if the available resources are enough 
                        # and the slot length is enough
                        if cond1: 
                            break
                
                rsc_lack = expected_req_rsc_size - cum_rsc_alloc

                # try to distribute the rsc_lack to the intervals as evenly as possible
                while cum_rsc_alloc < expected_req_rsc_size: 
                    size_t = rsc_avl_tmp[rsc_avl_tmp>0].min()
                    avl_slot_idx = np.where(rsc_avl_tmp >0)[0]
                    slot_size_sum = np.sum(curr_slot[avl_slot_idx])
                    if slot_size_sum * size_t >= rsc_lack:
                        size_t = np.ceil(rsc_lack / slot_size_sum).astype(int)
                        rsc_lack = 0
                    else: 
                        rsc_lack -= slot_size_sum * size_t
                    curr_alloc[avl_slot_idx] += size_t
                    rsc_avl_tmp[avl_slot_idx] -= size_t
                    cum_rsc_alloc += size_t * slot_size_sum

            # allocate resources
            idx = curr_alloc.nonzero()[0]
            for i in range(len(idx)):
                for rsc_map in self.scheduling_table[s[idx[i]]:s[idx[i]]+int(curr_slot[idx[i]])]:
                    rsc_map.allocate(task.id, curr_alloc[idx[i]], verbose)
            
        return True, np.array(s)[idx].tolist(), curr_alloc[idx].tolist(), curr_slot[idx].tolist()

    def release(self, task: TaskInt, time_slot_s:Union[int,List[int]], curr_alloc:Union[int,List[int]], curr_slot:Union[int,List[int]], verbose: bool = False):
        if isinstance(curr_alloc, int) and isinstance(curr_slot, int) and isinstance(time_slot_s, int):
            for rsc_map in self.scheduling_table[time_slot_s:time_slot_s+curr_slot]:
                rsc_map.release(task.id, curr_alloc, verbose)
        else:
            assert len(curr_alloc) == len(curr_slot) == len(time_slot_s)
            for i in range(len(curr_alloc)):
                for rsc_map in self.scheduling_table[time_slot_s[i]:time_slot_s[i]+curr_slot[i]]:
                    rsc_map.release(task.id, curr_alloc[i], verbose)


    def print_scheduling_table(self):
        empty = []
        title_line = False
        for i in range(len(self.scheduling_table)):
            if len(self.scheduling_table[i].rsc_map):
                if title_line:
                    _str = f"slot:{i}\n{self.scheduling_table[i].rsc_map.print_simple()}"
                else:
                    _str = f"slot:{i}\n{str(self.scheduling_table[i].rsc_map)}"
                print(_str)
                title_line = True
            else:
                empty.append(i)
        if len(empty)==len(self.scheduling_table):
            print("empty scheduling table")
        elif len(empty):
            print(f"empty slots: {empty}")

if __name__ == "__main__": 
    # create a task set
    # first branch: have free cores and free slots at beginning
    # t1 [15, 19] 25 rsc and 2 slot
    # t2 [3, 12] 24 rsc and 7 slot
    # second branch: select a interval with enough free cores and free slots
    # t3 [0, 20] 10 rsc and 4 slot
    # thrird branch: evently distribute the resources in the expected interval
    # t4 [0, 7] 10 rsc and 4 slot
    # last branch: As soon as possible
    # t5 [0, 16] 10 rsc and 8 slot


    t1 = TaskInt(task_name="task1", task_id=1, task_flag="moveable", timing_flag="deadline",
                ERT=15, ddl=19, period=30, exp_comp_t=2, 
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t2 = TaskInt(task_name="task2", task_id=2, task_flag="moveable", timing_flag="deadline",
                ERT=3, ddl=12, period=30, exp_comp_t=7,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t3 = TaskInt(task_name="task3", task_id=3, task_flag="moveable", timing_flag="deadline", 
                ERT=0, ddl=20, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t4 = TaskInt(task_name="task4", task_id=4, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=7, period=30, exp_comp_t=4,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)
    t5 = TaskInt(task_name="task5", task_id=5, task_flag="moveable", timing_flag="deadline",
                ERT=0, ddl=16, period=30, exp_comp_t=8,
                i_offset=0, jitter_max=0,
                flops=100, pre_assigned_resource_flag=True, main_size=100, RDA_size=20)

    task_list:List[TaskInt] = [None for i in range(10)]
    alloc_info = [None for i in range(10)]
    require_rsc_size = [0 for i in range(10)]
    task_list[0:5] = [t1, t2, t3, t4, t5]
    require_rsc_size[0:5] = [25, 24, 10, 10, 10]

    # create a scheduling table
    scheduling_table = SchedulingTableInt(30, 20)
    # allocate resources

    # alloc_info[0] = scheduling_table.insert_task(t1, 25, t1.ERT, t1.ddl, t1.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[1] = scheduling_table.insert_task(t2, 24, t2.ERT, t2.ddl, t2.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[2] = scheduling_table.insert_task(t3, 10, t3.ERT, t3.ddl, t3.exp_comp_t, verbose=True)
    # # scheduling_table.print_scheduling_table()
    # alloc_info[3] = scheduling_table.insert_task(t4, 10, t4.ERT, t4.ddl, t4.exp_comp_t, verbose=True)
    # alloc_info[4] = scheduling_table.print_scheduling_table()
    # alloc_info[5] = scheduling_table.insert_task(t5, 10, t5.ERT, t5.ddl, t5.exp_comp_t, verbose=True)
    # # print the scheduling table
    # scheduling_table.print_scheduling_table()

    for i in range(5):
        print(f"task {i} allocation\n")
        alloc_info[i] = scheduling_table.insert_task(task_list[i], require_rsc_size[i], 
                                                     task_list[i].ERT, task_list[i].ddl, 
                                                     task_list[i].exp_comp_t, verbose=True)
        scheduling_table.print_scheduling_table()
    
    # release resources
    for i in range(5):
        if alloc_info[i] is not None:
            print(f"task {i} release\n")
            print([task_list[i].id, *alloc_info[i]])
            scheduling_table.release(task_list[i], *alloc_info[i][1:], verbose=True)
    print("after release")
    scheduling_table.print_scheduling_table()