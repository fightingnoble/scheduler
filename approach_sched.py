from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from approach_collector import StatisticsCollector

import networkx as nx
# from example.bm4 import swt_lat
from collections import OrderedDict
from utils import core_distr, elim_nume_error
import functools
import types
from itertools import chain
from warnings import warn
from approach_Eq import (
    sim_comp_time, 
    estimate_resource_requirement, 
    calculate_slack_time,
    cal_load,
    time_gtq,
    time_ltq,
    time_gt,
)
from approach_def import Acc_p, print_if_verbose


def trigger_cond_dyn(acc_p, new_comp, new_ready_list, curr_t):
    """_summary_
    The reallocation is triggered if and only if some tasks need more resources, and we can also find free tiles.
    Need more:
        - A. new ready tasks arrives
        - B. not allocated tasks, allocated by starving tasks
    Free tiles:
        - 1. running tasks are preempted
        - 2. finished tasks at the beginning of this round
        - 3. free tiles in the last round
            |   | 1 | 2 | 3 | 
            | A | o | √ | √ |
            | B | x | √ | - |
            x: must not trigger, √: must trigger, -: impossible case, o: on condition
    
    System state should not affect the judgement of reallocation. 
    Main logic: 
        - 1&A: Under the assumption that all tiles are busy, reallocation is involed if and only if preemption is triggered: 
            if the priority of the new ready tasks is smaller than the running tasks, trigger is need. 
        - 2+3) there are idle tiles, A) new ready tasks B) not allocated or starving tasks. 
        As 3B is impossible we can treat it as true, we have (2 or 3) & (A or B).
    Method: 
        we use running as a mask to detect the reallcation progress, and res_map to detect the free tiles.
    """
    # cond1 = new_comp
    # cond2 = len(self.running) == 0 and len(new_ready_list) > 0
    # cond3 = len(self.running) > 0 and len(new_ready_list) > 0 \
    #     and min(new_ready_list, key=lambda x:self.G_ptr.nodes[x]['ddl']) < \
    #         max(self.running, key=lambda x:self.G_ptr.nodes[x]['ddl'])
    # cond = cond1 or cond2 or cond3

    # filter the new ready tasks by the filter_cond
    new_ready_list = list(acc_p.task_filter(curr_t, new_ready_list))
    # get free tiles
    free_tiles = acc_p.cap - sum(acc_p.res_map.values())
    # case 1: no free tiles
    cond1 = (free_tiles <= 0) and len(new_ready_list) > 0 \
        and min(new_ready_list, key=lambda x:acc_p.G_ptr.ddl_map[x]) < \
            max(acc_p.running, key=lambda x:acc_p.G_ptr.ddl_map[x])
    # case 2: free tiles
    cond2 = (free_tiles > 0) and (len(new_ready_list) > 0 or acc_p.starving) 
    
    cond = cond1 or cond2      
    if cond:
        realloc = True
        acc_p.sys_state = "R"
        acc_p.running["R"] = cal_load(acc_p.swt_lat, 1) 
    else:
        realloc = False
        # self.sys_state = "S"
        # the system exits reallocation progress until the "R" is finished
    return realloc

def task_filter(acc_p, curr_t, iter_tasks, reserv_en=False, op_miss_en=False):
    """
    Filter tasks based on the following conditions:
    filter R task
    filter timeout task if drop
        if op_miss_en, only the sink will be dropped for timeout;
        otherwise, all the tasks will be dropped for timeout.
    filter ert < curr_t task if reserve
    """
    filter_cond = [
        lambda node: node != "R",
        lambda node: (time_gt(acc_p.G_ptr.ddl_map[node], curr_t) or (op_miss_en and node in acc_p.G_ptr.ops)) or not acc_p.drop,
        lambda node: time_ltq(acc_p.G_ptr.ert_map[node], curr_t) if reserv_en else True,
    ]
    for node in iter_tasks:
        if all(cond(node) for cond in filter_cond):
            yield node

def alloc_fn_pglb(acc_p, curr_t, realloc=True, 
                  # static parameters, which will be removed by lambda or functools.partial
                  reserv_en=False):
    """
    Allocation function for reservation-aware scheduler.
    Only allocates minimum required resources, respects EST.
    """

    # calculate slack 
    realloc_slack = 0 if not realloc else sim_comp_time(acc_p.swt_lat, 1, 1)
    iter_tasks = chain(acc_p.running, acc_p.ready)

    acc_p.slack_map = {node: calculate_slack_time(acc_p.G_ptr.ddl_map[node], curr_t, realloc_slack) for node in acc_p.task_filter(curr_t, iter_tasks)}
    score = acc_p.slack_map.copy()
    alloc_map_curr = {}
    acc_p.starving = False
    # calculate min_rsc requirement
    score_dict = OrderedDict(); constr_dict = OrderedDict()
    curr_aval_rsc = acc_p.cap
    for node in sorted(score, key=score.get):
        slack = score[node]
        if slack <= 0:
            # if node != "R":
            #     print(f"\t[{acc_p.id}] {node} is timeout at {curr_t}")
            req_rsc_size = curr_aval_rsc
        else:
            assert not (node in acc_p.ready and node in acc_p.running) 
            task_load = acc_p.running.get(node, 0) + acc_p.ready.get(node, 0)
            exp_io_t = acc_p.G_ptr.nodes[node]['exp_io_t']
            req_rsc_size = estimate_resource_requirement(task_load, slack, acc_p.base_pwr, exp_io_t=exp_io_t)
            
            if req_rsc_size > curr_aval_rsc:
                print_if_verbose(f"\t[{acc_p.id}] {node} is hungry at {curr_t}: lack {req_rsc_size - curr_aval_rsc} tiles") 
                req_rsc_size = curr_aval_rsc
                curr_aval_rsc = 0
                alloc_map_curr[node] = req_rsc_size
                acc_p.starving = True and (not realloc)
                break
                
        curr_aval_rsc -= req_rsc_size
        alloc_map_curr[node] = req_rsc_size
        constr_dict[node] = "N/A"
        score_dict[node] = 1/score[node] if score[node] >0 else float("inf")
        score.pop(node)
            
    # allocate free resource
    # if there are still resources left, 
    # it means no late process is waiting for resources
    if curr_aval_rsc > 0 and len(alloc_map_curr) and not reserv_en:
        print_if_verbose(f"\t[{acc_p.id}] Minimum resource requirement at {curr_t}: {alloc_map_curr}")
        assert sum([score == float('inf') and constr_dict[pid] != "upb" for pid, score in score_dict.items()]) == 0
        # also, there is no process waiting for resources in the ready queue
        assert len(score) == 0
        core_distr(alloc_map_curr, score_dict, curr_aval_rsc)
    
    if reserv_en and curr_aval_rsc > 0 and len(alloc_map_curr):
        print_if_verbose(f"\t[{acc_p.id}] Reservation (en): {curr_aval_rsc} tiles reserved.") 
    return alloc_map_curr

# Cyclic specific alloc_fn and trigger_cond
def no_trigger(acc_p, new_comp, new_ready_list, curr_t=None, static_schedule_map=None):
    """
    Trigger condition for cyclic scheduler: 
        New ready tasks are in current map, or curr_t > next time in static_schedule_map.
    """    
    return False

def alloc_fn_cyclic(acc_p:Acc_p, curr_t:float, realloc:bool=True, T_hp:float=None, force:bool=False):
    """
    Allocation function for cyclic scheduler.
    Allocates resources based on a pre-defined static schedule map.
    :param acc_p: The Acc_p instance.
    :param curr_t: Current time.
    :param static_schedule_map: A nested dictionary {time: {task_id: tile_count}}.
                                 Expected to be passed via the main sched method.
    :param force: Boolean, if True, asserts capacity is sufficient for static allocation.
    """
    alloc_map_curr = OrderedDict()

    if acc_p.static_schedule_map is None:
        assert False, "Error: 'static_schedule_map' must be provided for 'cyclic' policy's alloc_fn."
        

    # 这里考虑到curr_t 可能不属于static_schedule_map的key，要怎么处理，可以保证数学上有较好的抽象
    # 保存一个状态变量，用来记录当前选中的静态调度表中的slot
    # 在数学上，是一个查表操作，循环的行为可以忽略，认为有一个足够长的表，足以覆盖所有的执行时间。
    # find the largest slot index that is less than curr_t
    assert acc_p.curr_slot_index < len(acc_p.static_schedule_map), \
        f"Cyclic scheduler: curr_slot_index {acc_p.curr_slot_index} is out of range {len(acc_p.static_schedule_map)}"
    
    # Special case: the length of static_schedule_map is 1, 
    # where the next slot is always the same as the current slot.
    if len(acc_p.static_schedule_map) == 1:
        # t = (curr_t > t)? t+T_hp: t
        t, cfg = acc_p.static_schedule_map[0]
        if curr_t >= elim_nume_error(t + T_hp):
            acc_p.update_prev_slot_schedule(0, T_hp)
    else: 
        curr_idx = acc_p.curr_slot_index
        nxt_idx = (curr_idx + 1)%len(acc_p.static_schedule_map)
        nxt2_idx = (curr_idx + 2)%len(acc_p.static_schedule_map)
        t, cfg = acc_p.static_schedule_map[nxt_idx]
        if time_gtq(curr_t, t):
            acc_p.update_prev_slot_schedule(curr_idx, T_hp)
            acc_p.curr_slot_index = nxt_idx
            # ensure not slot is skipped: next slot's start time is larger than curr_t
            t_next = acc_p.static_schedule_map[nxt2_idx][0]
            assert time_ltq(curr_t, t_next), \
                f"Cyclic scheduler: next slot's start time {t_next} is less than curr_t {curr_t}"
        else:
            t, cfg = acc_p.static_schedule_map[curr_idx]

    if force:
        if cfg:
            assert acc_p.cap >= sum(cfg.values()), \
                f"Cyclic scheduler (force=True): Capacity Error: {acc_p.cap} is less than required {sum(cfg.values())} at {curr_t}"
            temp_aval_rsc = acc_p.cap
            for node, requested_rsc in cfg.items():
                # ensure the expected task is ready or running
                if node in acc_p.running or node in acc_p.ready:
                    alloc_rsc = cfg[node]
                    alloc_map_curr[node] = alloc_rsc
                    temp_aval_rsc -= alloc_rsc
                    assert temp_aval_rsc >= 0
        else:
            print_if_verbose(f"\t[{acc_p.id}] Cyc-Sched  (force=True): No static allocation defined for time {curr_t}. Allocating nothing.")
    else:
        if cfg:
            print_if_verbose(f"\t[{acc_p.id}] Cyc-Sched: Applying static allocation for time {curr_t}: {cfg}")
            temp_aval_rsc = acc_p.cap
            for node, requested_rsc in cfg.items():
                if temp_aval_rsc <= 0:
                    break
                # ensure the expected task is ready or running
                if node in acc_p.running or node in acc_p.ready:
                    alloc_rsc = min(requested_rsc, temp_aval_rsc)
                    alloc_map_curr[node] = alloc_rsc
                    temp_aval_rsc -= alloc_rsc
                else:
                    print_if_verbose(f"\t[{acc_p.id}] Cyc-Sched: Task {node} not found in running/ready queues, skipping static allocation.")
        else:
            print_if_verbose(f"\t[{acc_p.id}] Cyc-Sched: No static allocation defined for time {curr_t}. Allocating nothing.")    
    return alloc_map_curr


# factory function
class PartitionConfig:
    """
    用于描述所有分区的调度配置。
    Attributes:
        num_partitions: 分区数量
        cap_list: 每个分区的cap
        base_pwr_list: 每个分区的base_pwr
        G_list: 每个分区的MyGraph
        TSmap: 每个分区的static_schedule_map（可选）
        mapped_node_list: 每个分区的节点映射（可选）
        swt_lat_list: 每个分区的切换延迟（可选）
    """
    def __init__(
        self, num_partitions, cap_list, base_pwr_list, 
        G, TSmap_list=None, mapped_node_list=None, swt_lat_list=None, T_hp=None
        ):
        self.num_parts = num_partitions
        self.cap_list = cap_list
        self.base_pwr_list = base_pwr_list
        self.G = G
        self.TSmap_list = TSmap_list or [None]*num_partitions
        self.mapped_node_list = mapped_node_list or [None]*num_partitions
        self.swt_lat_list = swt_lat_list or [None]*num_partitions
        self.T_hp = T_hp


def acc_p_factory(
    policy: str,
    cfg: PartitionConfig,
    stats_collector:StatisticsCollector=None,
    **kwargs
):
    """
    Define scheduler: 
    To define a scheduler, we need first to follow the basic interface of BaseProcessor.
    Then, for specific scheduler, we should define the internal state and static parameters/tables need by the scheduling. 
    policy: pglb, glb, cyc, reserv
    cfg: PartitionConfig对象，包含所有分区的参数
    kwargs: 其他策略相关参数
    返回: acc_p实例列表
    """
    acc_p_list = []
    assert cfg.G is not None, "G_list is not None"
    for i in range(cfg.num_parts):
        # check the validity of the partition config
        assert cfg.cap_list[i] > 0, "cap_list is not None"
        assert cfg.base_pwr_list[i] > 0, "base_pwr_list is not None"
        assert cfg.mapped_node_list[i] is not None, "mapped_node_list is not None"
        
        # create acc_p instance
        acc_p = Acc_p(f"acc_p{i}", cfg.cap_list[i], cfg.base_pwr_list[i], cfg.G, cfg.mapped_node_list[i], stats_collector)

        # policy binding
        if policy in ["pglb", "glb"]:
            if policy == "pglb" and cfg.num_parts <= 1:
                warn("pglb policy is not supported for single partition, use glb instead")
                policy = "glb"
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_pglb, reserv_en=False), acc_p)
            acc_p.task_filter = types.MethodType(functools.partial(task_filter, reserv_en=False), acc_p)
            acc_p.trigger_cond = types.MethodType(trigger_cond_dyn, acc_p)
        elif policy in ["cyc", "cyc-S"]:
            assert cfg.TSmap_list[i] is not None, "TSmap_list is not None"
            # 存储原始静态调度表用于动态更新
            acc_p.static_schedule_map = cfg.TSmap_list[i]
            force = True if policy == "cyc" else False
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_cyclic, T_hp=cfg.T_hp, force=force), acc_p)
            acc_p.trigger_cond = types.MethodType(functools.partial(no_trigger), acc_p)

        elif policy == "reserv":
            acc_p.alloc_fn = types.MethodType(functools.partial(alloc_fn_pglb, reserv_en=True), acc_p)
            acc_p.task_filter = types.MethodType(functools.partial(task_filter, reserv_en=True, op_miss_en=True), acc_p)
            acc_p.trigger_cond = types.MethodType(trigger_cond_dyn, acc_p)
        else:
            raise ValueError(f"Unknown strategy: {policy}")
        acc_p.policy = policy
        # 使用分区配置中的swt_lat，如果没有提供则使用默认计算
        assert cfg.swt_lat_list[i] is not None, f"swt_lat is not None for partition {i}"
        acc_p.swt_lat = cfg.swt_lat_list[i]
        acc_p_list.append(acc_p)
    
    # 返回处理器列表和统计收集器
    return acc_p_list



# latency model: 
# 总执行时间分解为三个关键组成部分来建模延迟（执行时间）：计算、访存/传输和调度。

# 首先，资源分配过程是不aware 访存延迟和调度开销这些运行时数据的。调度器能看到的只有分区内带宽和tile两种资源。
# 任务的计算延迟：
# （NN 任务）等价算数操作数量*非阻塞执行时间占比/设备算力FLOPS 
# （其他程序）指令数量*CPI/CPU频率
# 为了同一抽象，这里统一为，
# 计算延迟=架构无关的操作数量/使用算力/(1-通信时间占比)

# 任务初始定义：基准指令数
# 设备定义：单位计算单元算力，计算单元容量
# 运行时参数：通信延迟占比

