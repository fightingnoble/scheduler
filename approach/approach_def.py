from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from .approach_collector import StatisticsCollector

import networkx as nx
from itertools import chain
from queue import Queue
# from example.bm4 import swt_lat
from collections import OrderedDict
from typing import Callable, Set, List
from utils import elim_nume_error

from warnings import warn
import numpy as np
import time
from .approach_Eq import (
    sim_comp_time, 
    estimate_resource_requirement, 
    find_legal,
    normalize_time_to_unit,
    get_task_load_and_base_size,
    calculate_slack_time,
    update_task_progress,
    cal_load,
    time_eq,
    time_gt,
    time_gtq,
    time_lt,
    time_ltq,
    time_add,
    time_sub,
    # 导入新的工厂函数和分布类
    dist_from_dict, SenVarDist, IOVarDist, LoadVarDist, AccVarDist, Variation
)
import re


# 在文件顶部添加全局控制
VERBOSE_OUTPUT = False
REALLOC_DISABLED = False  # 禁止切换开销
MISS_DISABLED = False     # 禁止miss检测
DROP_DISABLED = False     # 禁止drop

def set_verbose_output(verbose: bool):
    """全局设置是否输出详细信息"""
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = verbose

def set_realloc_disabled(realloc_disabled: bool):
    global REALLOC_DISABLED
    REALLOC_DISABLED = realloc_disabled

def set_miss_disabled(miss_disabled: bool):
    global MISS_DISABLED
    MISS_DISABLED = miss_disabled

def set_drop_disabled(drop_disabled: bool):
    global DROP_DISABLED
    DROP_DISABLED = drop_disabled

def get_drop_disabled():
    global DROP_DISABLED
    return DROP_DISABLED

def get_miss_disabled():
    global MISS_DISABLED
    return MISS_DISABLED

def get_realloc_disabled():
    global REALLOC_DISABLED
    return REALLOC_DISABLED


def print_if_verbose(*args, **kwargs):
    """全局条件打印函数"""
    if VERBOSE_OUTPUT:
        print(*args, **kwargs)


def build_logical_graph(srcs, ops, sinks, task_attr, src_attr, sink_attr):
    dag = nx.DiGraph()
    for src_n in srcs:
        dag.add_node(src_n, type="src", **src_attr[src_n])
        for op_n in srcs[src_n]:
            dag.add_node(op_n)
            dag.add_edge(src_n, op_n, type="control")
    for op_n in ops:
        dag.add_node(op_n, type="op", **task_attr[op_n])
        for sink_n in ops[op_n]:
            dag.add_node(sink_n)
            dag.add_edge(op_n, sink_n, type="data")
    for sink_n in sinks:
        if sink_attr is None:
            dag.add_node(sink_n, type="sink")
        else:
            dag.add_node(sink_n, type="sink", **sink_attr[sink_n])
    
    return dag

class MyGraph(nx.DiGraph):
    def __init__(self, srcs, ops, sinks, task_attr, src_attr, sink_attr=None):
        super(MyGraph, self).__init__()
        self.logical_graph = build_logical_graph(srcs, ops, sinks, task_attr, src_attr, sink_attr)
        
        self.var_dist_map = {}
        self._rebuild_distributions()

        # self.n_pred_map trackes the non-ready, intermediate tasks in the graph
        # The value of n_pred_map is updated when tasks in graph are finished
        # A item is removed from the queue when it is moved to the ready queue
        # srcs are initially excluded from n_pred_map
        self.n_pred_map = {}
        
        self.srcs = []
        self.ops = []
        self.sinks = []
        
        self.offset_map = {}
        self.ddl_map = {"R": -float('inf')}
        self.ert_map = {"R": -float('inf')}

    def _rebuild_distributions(self):
        """
        遍历图节点，使用工厂函数从 dist_info 属性重建分布对象。
        """
        for node, data in self.logical_graph.nodes(data=True):
            var_dist = None
            if data['type'] == "sink":
                continue
            # 先从 dist_info 重建（若存在且尚未有 var_dist）
            if 'dist_info' in data and 'var_dist' not in data:
                var_dist = dist_from_dict(data['dist_info'])
                self.logical_graph.nodes[node]['var_dist'] = var_dist
            elif 'var_dist' in data:
                var_dist: Variation = data['var_dist']
            
            if var_dist:
                self.var_dist_map[node] = var_dist

    def mark_finish(self, node):
        for succ in self.successors(node):
            self.n_pred_map[succ] -= 1
        
        # 从缓存中移除该节点
        if node in self.ddl_map:
            self.ddl_map.pop(node)
        if node in self.ert_map:
            self.ert_map.pop(node)
        if node in self.offset_map:
            self.offset_map.pop(node)
        
        # 从分类列表中移除
        if node in self.ops:
            self.ops.remove(node)
        elif node in self.srcs:
            self.srcs.remove(node)
        else:
            self.sinks.remove(node)
        
        # 移除边和节点
        self.remove_edges_from([(node, succ) for succ in self.successors(node)])
        self.remove_node(node)
        
    def mark_ready(self, node):
        self.n_pred_map.pop(node)
    
    def duplicate_for_hyperperiod(self, hp_idx: int, seed: int, T_hp: float = 0.1, var_en: bool = False):
        """
        按给定的超周期索引 hp_idx 和随机种子 seed，复制当前图中的节点与边：
        - 新节点名称为 "原名_" + hp_idx
        - 除 sink 节点外（src 与 op），为其生成带随机扰动的 exp_comp_t（可控随机，受 seed 影响）
        - 同步更新状态缓存：n_pred_map, srcs, ops, sinks, ddl_map, ert_map
        - offset/ert/ddl 叠加超周期偏移：新值 = 原值 + hp_idx * T_hp

        参数：
            hp_idx: 超周期索引
            seed: 随机种子
            T_hp: 超周期长度，用于计算时间偏移
            jitter_src_std_ratio: src节点执行时间随机化的标准差比例
            jitter_op_std_ratio: op节点执行时间随机化的标准差比例
            clamp_min: 随机化因子的最小限制
            clamp_max: 随机化因子的最大限制

        说明：
        - 随机化策略参考 scheduler_base 的参数化思路：对 src 使用截断近似的正态扰动（以原 exp_comp_t 为均值），
          对 op 采用同样形式的扰动。未显式提供 var_factor/freq/comp_ratio 时，使用固定比例的标准差。
        - offset/ert/ddl 叠加超周期偏移。
        """
        rng = np.random.RandomState(seed + int(hp_idx))

        # 缓存当前节点与边，避免遍历时结构变化
        orig_nodes = list(self.logical_graph.nodes(data=True))
        # 包含边属性
        orig_edges = list(self.logical_graph.edges(data=True))

        # 先创建所有新节点
        for node,attr in orig_nodes:
            new_name = f"{str(node)}_{hp_idx}"
            new_attr = attr.copy()
            node_type = new_attr['type']

            # 对 src/op 做执行时间随机化；sink 保持不变
            if node_type in ['src', 'op']:
                if var_en and node in self.var_dist_map:
                    dist = self.var_dist_map[node]
                    # AccVarDist: 同时采样计算负载与访存时间
                    if hasattr(dist, 'load_dist') and hasattr(dist, 'exec_dist'):
                        new_attr['exp_comp_t'] = elim_nume_error(dist.load_dist.get_var_fn()(rng))
                        new_attr['exp_io_t'] = elim_nume_error(dist.exec_dist.get_var_fn()(rng))
                    else:
                        # 单分布：将采样结果作为计算负载，访存置 0（若已存在则保留）
                        new_attr['exp_comp_t'] = elim_nume_error(dist.get_var_fn()(rng))
                        new_attr['exp_io_t'] = 0.
                else:
                    new_attr['exp_comp_t'] = elim_nume_error(new_attr['exp_comp_t']) # if node_type == "op" else 0
                    new_attr['exp_io_t'] = elim_nume_error(new_attr['exp_io_t']) if node_type == "op" else 0

            # 添加超周期偏移到时间相关属性
            time_offset = hp_idx * T_hp
            if 'offset' in new_attr:
                new_attr['offset'] = elim_nume_error(new_attr['offset'] + time_offset)
            if 'ert' in new_attr:
                new_attr['ert'] = elim_nume_error(new_attr['ert'] + time_offset)
            if 'ddl' in new_attr:
                new_attr['ddl'] = elim_nume_error(new_attr['ddl'] + time_offset)

            self.add_node(new_name, **new_attr)

            # 维护分类列表
            if node_type == 'src':
                self.srcs.append(new_name)
            elif node_type == 'op':
                self.ops.append(new_name)
            elif node_type == 'sink':
                self.sinks.append(new_name)

            # ddl/ert 缓存复制（R 特殊键保持原状，不新增）
            if node_type in ['op', 'sink']:
                if 'ddl' in new_attr:
                    self.ddl_map[new_name] = new_attr['ddl']
                if 'ert' in new_attr:
                    self.ert_map[new_name] = new_attr['ert']
            
            # 更新 offset_map 缓存
            if node_type == 'src':
                if 'offset' in new_attr:
                    self.offset_map[new_name] = new_attr['offset']

        # 再复制所有边（保持原边属性）
        for u, v, eattr in orig_edges:
            new_u = f"{u}_{hp_idx}"
            new_v = f"{v}_{hp_idx}"
            if new_u in self.nodes and new_v in self.nodes:
                self.add_edge(new_u, new_v, **eattr)

        # 更新 n_pred_map：对于非 src 节点，设置其未就绪前驱计数
        for node, new_attr in orig_nodes:
            node_type = new_attr['type']
            if node_type == 'src':
                continue
            dup = f"{str(node)}_{hp_idx}"
            if dup in self.nodes:
                # 以复制后的入度作为初始前驱数
                self.n_pred_map[dup] = len(list(self.predecessors(dup)))


class GlobalEvent_t:
    def __init__(self, event_t:List) -> List:
        """
        Renew the event_t, which is a list of (event_time, event_type)
        assert event_type in ["external"]
        We enforce an event at inf is inserted 
        """
        # self.event_t = []
        # self.event_t.extend(event_t)
        # self.event_t.append((float("inf"), "external"))
        # self.event_t = list(set(self.event_t))
        # self.event_t.sort()
        self.event_t = [(float("inf"), "external")]

        # 存储原始事件模式，用于动态更新
        self._original_events = sorted(set(event_t))

    def add_events_for_hyperperiod(self, hp_idx: int, T_hp: float, curr_t:float, type_list:List=None):
        """
        为新的超周期添加事件
        
        参数：
            hp_idx: 超周期索引
            T_hp: 超周期长度
            src_nodes: 源节点列表（用于生成传感器事件）
            graph: 图实例（用于获取节点属性）
        """
        if type_list is None:
            type_list = ["external", "table"]
        # add the original events to the event_t
        self.event_t.extend(
            OrderedDict(
                (
                    elim_nume_error(_t + hp_idx * T_hp), _type) 
                    for _t, _type in self._original_events 
                    if _type in type_list
                ).items()
                )

        # 重新排序事件队列
        self.event_t.sort()
        self.remove_ood_events(curr_t)

    def get_next_event_time(self, curr_t):
        """返回下一个事件时间和类型"""
        # if not self.event_t:
        #     return float('inf'), 'finish'
        # 只返回大于当前时刻的事件
        # A good math format:
        # future_events = [e for e in self.event_t if e[0] > curr_t]
        # A high efficiency format:
        while self.event_t[0][0] <= curr_t:
            assert False, f"out of date event {self.event_t[0]} is detected"
        # We assume an event at inf is all ready inserted to the event_t
        # if not self.event_t:
        #     return float('inf'), 'finish'
        # next_timer_event = min(self.event_t, key=lambda x:x[0])
        return self.event_t[0][0]
    
    def confirm_next_event(self, curr_t):
        while self.event_t[0][0] <= curr_t:
            assert False, f"out of date event {self.event_t[0]} is detected"
        return self.event_t.pop(0)

    def remove_ood_events(self, curr_t):
        while self.event_t[0][0] <= curr_t:
            self.event_t.pop(0)

    def detect_empty(self, curr_t):
        self.remove_ood_events(curr_t)
        if self.event_t[0][0] == float('inf'):
            return True
        else:
            return False

class BaseProcessor:
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set=None, 
        stats_collector:StatisticsCollector=None
    ):
        # 统一的基础属性
        self.id = id
        self.base_pwr = base_pwr
        self.G_ptr = G
        self.cap = cap
        self.mapped_node = set()
        self.policy = 'N/A'
        # 存储原始映射模式，用于动态更新
        self._original_mapped_pattern = set(mapped_node) if mapped_node is not None else set()

        # 存储原始静态调度表，用于动态更新
        self.static_schedule_map = []
        self._original_static_schedule_map = None
        
        # 统计信息收集器
        if stats_collector:
            # init partition stats
            stats_collector.init_partition_stats(f"{id}", cap, base_pwr)
        self.stats_collector:StatisticsCollector = stats_collector

    def update_static_schedule_for_hyperperiod(self, hp_idx: int, T_hp: float):
        """
        根据超周期索引更新静态调度表，复制原始调度模式并添加时间偏移
        
        参数：
            hp_idx: 超周期索引
            T_hp: 超周期长度
        """
        if self._original_static_schedule_map is None:
            return
        
        # 静态调度表的扩展
        new_static_schedule = []
        time_offset = hp_idx * T_hp
        
        for slot_time, slot_config in self._original_static_schedule_map:
            # 为每个时间槽添加超周期偏移
            new_slot_time = elim_nume_error(slot_time + time_offset)
            
            # 为每个任务配置添加超周期后缀
            new_slot_config = {}
            for task_name, resource_count in slot_config.items():
                new_task_name = f"{task_name}_{hp_idx}"
                new_slot_config[new_task_name] = resource_count
            
            new_static_schedule.append((new_slot_time, new_slot_config))
        
        # 更新静态调度表
        self.static_schedule_map.extend(new_static_schedule)        
        print_if_verbose(f"\t[{self.id}] 更新静态调度表到超周期 {hp_idx}: {len(new_static_schedule)} 个时间槽")
    
    def update_prev_slot_schedule(self, curr_slot_index: int, T_hp: float):
        """
        update the static schedule map for the previous slot, when moving the curr_slot_index. 
        """
        if self.static_schedule_map is None:
            return
        t, cfg = self.static_schedule_map[curr_slot_index]
        # increase the hp_idx of the task name in the cfg
        new_cfg = {}
        for task_name, size in cfg.items():
            # match the current hp_idx
            match = re.search(r'_(-?[0-9]+)$', task_name)
            assert match is not None, "task_name should end with _hp_idx"
            hp_idx = int(match.group(1))
            new_task_name = re.sub(r'_(-?[0-9]+)$', f'_{hp_idx+1}', task_name)
            new_cfg[new_task_name] = size
        self.static_schedule_map[curr_slot_index] = (elim_nume_error(t+T_hp), new_cfg)

            
    def update_mapped_nodes_for_hyperperiod(self, hp_idx: int):
        """
        根据超周期索引更新mapped_node，复制原始映射模式
        例如：如果原始映射是 ['S1', 'S2']，超周期0会变成 ['S1_0', 'S2_0']
        """
        if not self._original_mapped_pattern:
            return
        
        for node in self._original_mapped_pattern:
            new_node = f"{node}_{hp_idx}"
            if new_node in self.G_ptr.nodes():
                self.mapped_node.add(new_node)

    def update_run(self, pred_t, curr_t) -> bool:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError
    
    def update_ready(self, curr_t) -> List:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError
    
    def sched(self, curr_t, **kwargs) -> float:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError
    
    def predict_next(self, curr_t) -> float:
        # 统一接口，子类实现具体逻辑
        raise NotImplementedError

    def __str__(self):
        return f"{self.id}, cap={self.cap}, base_pwr={self.base_pwr}, policy={self.policy}, mapped_ops={len(self.mapped_node)})"
    
    def __repr__(self):
        return self.__str__()

    def repr_info(self):
        if not VERBOSE_OUTPUT:
            return
        print(f"\n\tProcessor {self.id}: {self}")
        # Print sys_state if it exists
        if hasattr(self, 'sys_state'):
            print(f"\t\tstate: {self.sys_state}")
        # Print running if it exists
        if hasattr(self, 'running'):
            print(f"\t\tRunning_queue: {self.running}")
        # Print res_map if it exists
        if hasattr(self, 'res_map'):
            print(f"\t\tRes_map: {self.res_map}")
        # Print slack_map if it exists
        if hasattr(self, 'slack_map'):
            print(f"\t\tSlack_map: {self.slack_map}")

    def clear_timeout(self, node):
        if node in self.ready:
            del self.ready[node]

        if node in self.running:
            del self.running[node]
            if node in self.res_map: # also remove from resource map if it was running
                del self.res_map[node]
        
        self.G_ptr.mark_finish(node)
    
    def has_action(self) -> bool:
        """
        检查处理器是否有实际动作
        返回 True 如果处理器有运行中的任务、就绪任务、资源分配或状态变化
        """
        # 检查是否有运行中的任务
        if hasattr(self, 'running') and self.running:
            return True
        
        # 检查是否有就绪任务
        if hasattr(self, 'ready'):
            if isinstance(self.ready, dict) and self.ready:
                return True
            elif hasattr(self.ready, 'qsize') and self.ready.qsize() > 0:
                return True
        
        # 检查是否有资源分配
        if hasattr(self, 'res_map') and self.res_map:
            return True
        
        # 检查是否有状态变化
        if hasattr(self, 'sys_state') and self.sys_state == "R":
            return True
        
        return False


class Sen_p(BaseProcessor):
    """ Execution model of multi-sequential processor (MSSP), 
        where each processor only serves one task at a time,
        and scheduler will determine the which tasks are served, 
        and which processor is responsible for each task. 
    """
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set, stats_collector=None):
        super(Sen_p, self).__init__(id, cap, base_pwr, G, mapped_node, stats_collector)
        self.running = dict()
        self.ready = Queue(-1)
        self.policy = 'FCFS'

    def update_run(self, pred_t, curr_t) -> bool:
        for node in list(self.running.keys()):
            # rem_t  = self.running[node] - (curr_t - pred_t) * self.base_pwr
            rem_load, delta_load = update_task_progress(self.running[node], curr_t - pred_t, 1, self.base_pwr)
            if rem_load <= 0:
                self.G_ptr.mark_finish(node)
                self.running.pop(node)
                print_if_verbose(f"\t[{self.id}] src {node} arrives at {curr_t}")             
            else:
                self.running[node] = rem_load
        return False # No new task completion in Sen_p

    def update_ready(self, curr_t) -> List:
        """
        Scan mapped_nodes -> check if the node is ready -> 
        remove the node from the scan list -> 
        process the ready node depending their type and loads
        """
        # for node in list(self.G_ptr.srcs):
        for node in list(self.mapped_node):
            if time_eq(curr_t, self.G_ptr.nodes[node]["offset"]):
                load = cal_load(self.G_ptr.nodes[node]["exp_comp_t"], self.G_ptr.nodes[node]["base_size"])
                if load > 0:
                    self.ready.put((node, load))
                    print_if_verbose(f"\t[{self.id}] src {node} is triggered at {curr_t}")
                else:
                    self.G_ptr.mark_finish(node)
                    print_if_verbose(f"\t[{self.id}] src {node} arrives at {curr_t}")
                # self.G_ptr.srcs.remove(node)
                self.mapped_node.remove(node)
        return [] # No new ready tasks in Sen_p

    def sched(self, curr_t, **kwargs) -> float:
        """
        FCFS policy: serves new tasks until the last task is finished.
        Predicts the next event in this queue.
        """
        while not self.ready.empty() and len(self.running) < self.cap:
            node, rem_t = self.ready.get()
            self.running[node] = rem_t
            print_if_verbose(f"\t[{self.id}] sen starts task {node} at {self.G_ptr.nodes[node]['offset']}")

        # predict the next event in this queue
        duation_sen_p = float("inf")
        for node in self.running:
            duation_sen_p = min(duation_sen_p, self.running[node])
        if duation_sen_p == float("inf"):
            print_if_verbose(f"\t[{self.id}] No sensor event in future at {curr_t}")
        else:
            next_event_time = time_add(curr_t, duation_sen_p)
            print_if_verbose(f"\t[{self.id}] Next sensor event at {next_event_time}")
        return duation_sen_p

class Acc_p(BaseProcessor):
    def __init__(self, id, cap, base_pwr, G:MyGraph, mapped_node:Set, stats_collector=None):
        super(Acc_p, self).__init__(id, cap, base_pwr, G, mapped_node, stats_collector)

        # all keys in res_map should be in running
        # running: records the remaining load of task that is selected to be issued, including the system task "R"
        self.running = dict() 
        # res_map: records the actual resource allocation of each task, ** regardless system state**
        self.res_map = dict()
        self.ready = dict()
        self.starving = False
        self.sys_state = "S"
        self.sys_state_emu = ["S", "R"]
        self.slack_map = dict()
        # TODO: init the switch latency by cap and cap_sram, and DRAM_bw
        self.swt_lat: float
        # These will be bound by the factory function
        self.alloc_fn: Callable
        self.trigger_cond: Callable
        self.curr_slot_index = 0
        self.drop_timeout = True
    
    @property
    def drop(self):
        return not get_drop_disabled() and self.drop_timeout

    def iter_timeout(self, curr_t: float, hp_start_t: float):
        """Iterate over the timeout non-src/sink tasks.
        If DROP_DISABLED is False, remove the timeout tasks from ready and running queues.
        """
        # Use itertools.chain to safely iterate over multiple dictionaries
        for node, rem in chain(self.ready.copy().items(), self.running.copy().items()):
            if node == "R":
                continue
            _type = self.G_ptr.nodes[node]['type']
            if _type == "sink":
                continue
            ddl = self.G_ptr.ddl_map.get(node, float("inf"))
            # To avoid timeout tasks are repeatively yield, only ones whose ddl belong to this hyperperiod will be yielded
            if time_gt(curr_t, ddl) and time_gtq(ddl, hp_start_t):
                # if MISS_DISABLED, peacefully exit the simulation
                if MISS_DISABLED:
                    import sys; sys.exit(1)

                # if not get_drop_disabled():
                if self.drop:
                    self.clear_timeout(node)
                    print_if_verbose(f"\t[{self.id}] Task {node} is dropped at {curr_t}")
                yield (node, rem)


    def update_run(self, pred_t, curr_t) -> bool:
        """
        Updates the remaining workload for running tasks and checks for finishing events.
        All tasks depends on the system state, 
        the remining load is updated only if the system state is "S" 
        """
        # 保护：第一次调用时pred_t为-inf，直接返回
        if time_lt(pred_t, 0):
            return False
        
        new_complete_flag = False
        if self.sys_state == "R":
            assert "R" in self.running, \
                f"R should be in running at {curr_t}"
            assert "R" not in self.res_map, \
                f"R should not be in res_map at {curr_t}"
            rem_load, delta_load = update_task_progress(self.running["R"], curr_t - pred_t, 1, 1)
            if rem_load <= 0:
                self.sys_state = "S"
                print_if_verbose(f"\t[{self.id}] exist reallocation state at {curr_t}") 
                self.running.pop("R")
                print_if_verbose(f"\t[{self.id}] Task R finishes at {curr_t}") 
            else:
                self.running["R"] = rem_load
                
            # info collector, schedule-unrelated
            if self.stats_collector:
                # 记录realloc overhead
                task_list = list(self.res_map.keys())  # 假设ready队列中的任务受realloc影响
                # 内部会使用 cal_cost(delta_ld, cap, pwr) 来计算负载。
                self.stats_collector.record_realloc(self.id, elim_nume_error(curr_t - pred_t), task_list)
        else:
            # 先处理所有任务，累积实际使用的负载
            total_used_load = 0.0
            for node in list(self.res_map.keys()):
                # rem_t  = self.running[node] - (curr_t - pred_t) * self.res_map[node] * self.base_pwr
                rem_load, delta_load = update_task_progress(self.running[node], curr_t - pred_t, self.res_map[node], self.base_pwr)
                total_used_load += delta_load
                
                # info collector, schedule-unrelated
                if self.stats_collector:
                    self.stats_collector.record_compute_progress(node, elim_nume_error(curr_t - pred_t), delta_load)

                if rem_load <= 0:
                    # 检查是否超时
                    is_timeout = time_gt(curr_t, self.G_ptr.ddl_map[node])
                    if is_timeout:
                        print_if_verbose(f"\t[{self.id}] {node} is timeout at {curr_t}")
                    
                    # info collector, schedule-unrelated
                    if self.stats_collector:
                        # 任务完成的时候记录：完成时间-offset
                        self.stats_collector.record_task_finish(self.G_ptr, node, curr_t)
                    
                    self.G_ptr.mark_finish(node)
                    new_complete_flag |= True
                    self.running.pop(node)
                    self.res_map.pop(node)
                    print_if_verbose(f"\t[{self.id}] Task {node} finishes at {curr_t}") 
                else:
                    self.running[node] = rem_load
            
            # 循环结束后，记录 idle
            # idle = 总容量 - 实际使用的负载
            if self.stats_collector and curr_t > pred_t:
                dt = curr_t - pred_t
                total_capacity = dt * self.cap * self.base_pwr
                idle_load = total_capacity - total_used_load
                self.stats_collector.record_idle_capacity(idle_load, self.sys_state)
        return new_complete_flag

    def update_ready(self, curr_t) -> List:
        """
        Puts newly ready tasks into the ready queue.
        Scan mapped_nodes -> check if the node is ready -> 
        remove the node from the scan list -> 
        process the ready node depending their type and loads
        """
        new_ready_list = []
        # for node, n_pred in list(self.G_ptr.n_pred_map.items()):
        for node in list(self.mapped_node):
            if not node in self.G_ptr.n_pred_map:
                continue
            n_pred = self.G_ptr.n_pred_map[node]
            if n_pred == 0:
                if node in self.G_ptr.sinks:
                    if time_gt(curr_t, self.G_ptr.ddl_map[node]):
                        print_if_verbose(f"\t[{self.id}] {node} is timeout at {curr_t}")
                    # info collector, schedule-unrelated
                    if self.stats_collector:
                        # 任务完成的时候记录：完成时间-offset
                        self.stats_collector.record_e2e_finish(self.G_ptr, node, curr_t)

                    self.G_ptr.mark_finish(node)
                    print_if_verbose(f"\t[{self.id}] sink {node} finish at {curr_t}")
                elif node in self.G_ptr.ops:
                    # illegal check
                    if node in self.running or node in self.ready:
                        assert False, "task should not be in running or ready queue"
                    load = cal_load(self.G_ptr.nodes[node]["exp_comp_t"], self.G_ptr.nodes[node]["base_size"])
                    if load <= 0:
                        self.G_ptr.mark_finish(node)
                        print_if_verbose(f"\t[{self.id}] task {node} is skipped at {curr_t}")
                    else:
                        self.ready[node] = load                    
                        # info collector, schedule-unrelated
                        if self.stats_collector:
                            # record submitted load in this hyperperiod
                            self.stats_collector.record_period_load_arrival(self.ready[node])
                        
                        new_ready_list.append(node)
                        
                        # info collector, schedule-unrelated
                        # 记录任务开始统计（当任务进入ready队列时）
                        if self.stats_collector:
                            self.stats_collector.record_task_start(node, curr_t)
                        
                        print_if_verbose(f"\t[{self.id}] task {node} ready at {curr_t}")
                self.mapped_node.remove(node)
                self.G_ptr.mark_ready(node)
        return new_ready_list

    def sched(self, curr_t, new_comp, new_ready_list, **kwargs) -> float:
        """
        Allocation progress: 
            Check the event type: 
            If the event informs the completion of reallocation, 
            scheduler will directly use the cached allocation map. 
            Otherwise, scheduler will generate a new allocation map, 
            during which the allocation map in last round will be renamed as alloc_map_prev. 
            
            Two types of tasks: 
            - "system task" that stalls the accelerator
            - "user task" that can be executed by the accelerator
        """ 
        # artificial setting for testing not for modeling
        if REALLOC_DISABLED:
            realloc = False
        else:
            realloc = self.trigger_cond(new_comp, new_ready_list, curr_t)

        # The elegent way to handle the reallocation progress,
        # but will not display which task are going to be allocated in the allocation function.
        # TODO: Not examined yet.
        # if realloc: 
        #     for node in list(self.running.keys()):
        #         if node != "R":
        #             self.ready[node] = self.running.pop(node)
        #             print(f"\t[{self.id}] Task {node} preempted and moved to ready queue.")
        #     self.running["R"] = self.swt_lat
        #     self.res_map.clear()
        #     self.res_map["R"] = 1
        # elif "R" in self.running:
        #     pass
        # else:
        #     alloc_map_curr = self.alloc_fn(curr_t, False)
        #     self.update_queue(alloc_map_curr)
        #     self.res_map.update(alloc_map_curr) 
        
        # These lines belong to allocation progress. 
        # However, in order to maintain the running/ready queues, 
        # these two lines of logic are deliberately placed outside the conditional statement, which is not elegant.
        # TODO: double check
        if realloc:
            _t0 = time.perf_counter()
        alloc_map_curr = self.alloc_fn(curr_t, realloc)
        # info collector, schedule-unrelated
        # 记录任务开始统计（当任务进入ready队列时）
        if self.stats_collector and realloc:
            _elapsed = time.perf_counter() - _t0
            self.stats_collector.record_sched_overhead(self.id, _elapsed, self.swt_lat)
            # all running and incomming tasks undergo reallocation
            self.stats_collector.record_realloc_num(self.id, list(set(self.res_map.keys()) | set(alloc_map_curr.keys())))
        self.res_map.clear()
        self.res_map.update(alloc_map_curr) 
        self.update_queue(alloc_map_curr)


        # predict the next event in this queue
        duation_acc_p = self.predict_next(curr_t)
        # state display 
        if self.sys_state == "R":
            print_if_verbose(f"\t[{self.id}] Enter reallocation progress at {curr_t}")
            type_ = "reallocate"
        else:
            type_ = "finish" 
        # TODO: the event info is not accurate, should be removed in the future.
        # if duation_acc_p == float("inf"):
        #     print(f"\tNo accelerator event in future at {curr_t}")
        # else:
        #     print(f"\tNext accelerator {type_} event at {curr_t + duation_acc_p}")

        return duation_acc_p
    
    def update_queue(self, alloc_map_curr):
        """
        Moves tasks between running and ready queues based on the current allocation map.
        Tasks not in alloc_map_curr are preempted (moved to ready).
        Tasks in alloc_map_curr are moved to running.
        """
        for node in list(self.running.keys()):
            if node != "R" and node not in alloc_map_curr: 
                self.ready[node] = self.running.pop(node)
                print_if_verbose(f"\t[{self.id}] Task {node} preempted and moved to ready queue.")
            
        for node in list(self.ready.keys()):
            if node in alloc_map_curr: 
                self.running[node] = self.ready.pop(node)
                print_if_verbose(f"\t[{self.id}] Task {node} moved from ready to running queue.")

    def predict_next(self, curr_t) -> float:
        """Helper to predict the next event duration based on current resource map."""
        if self.sys_state == "R":
            duation_acc_p = sim_comp_time(self.running["R"], 1, 1)
        else:
            # predict the next event in this queue
            actual_running = [sim_comp_time(
                        self.running[pid], 
                        self.res_map[pid], 
                        self.base_pwr,
                        self.G_ptr.nodes[pid].get('exp_io_t', 0.0)
                    ) for pid in self.res_map  if self.res_map[pid] > 0]
            if not actual_running:
                return float("inf")
            duation_acc_p = min(actual_running)
        return duation_acc_p


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

