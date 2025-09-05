"""
任务执行时间和资源需求估计的工具函数
"""
import math
from typing import Dict, Any
from global_var import elim_nume_error
from sched.ref_alloc_search import find_legal

from scipy.stats import truncnorm
from scipy.stats import poisson
import numpy as np

time_unit = 1 
unit_align = True


def set_time_unit(timestep, int_slot):
    global time_unit
    if int_slot:
        time_unit = 1
        normalize_factor = timestep
    else:
        time_unit = timestep
        normalize_factor = 1
    return time_unit, normalize_factor

def get_var_t_fn(logical_graph, node, type):
    if type == "src":
        # follow truncated normal distribution
        half_len = logical_graph.nodes[node]['comp_ratio'] / logical_graph.nodes[node]['freq']
        ZScore = 3
        loc = half_len
        scale = half_len / ZScore
        a, b = -ZScore, ZScore
        var_t_fn = lambda rng: truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=rng).item()
    elif type == "op":
        # follow truncated poisson distribution
        lambda_ld = 1
        exp_comp_t = logical_graph.nodes[node]['exp_comp_t']
        k = logical_graph.nodes[node]['var_factor']
        if k == 1:
            var_t_fn = lambda rng: exp_comp_t
        else:
            ini_probs = np.zeros(k+1)
            for j in range(k+1):
                ini_probs[j] = poisson.pmf(j,lambda_ld)
            ini_probs = ini_probs/ini_probs.sum()
            var_t_fn = lambda rng: np.random.choice(k+1, p=ini_probs, replace=False) * exp_comp_t
    return var_t_fn


def update_task_progress(init_load: float, elapsed_time: float, res: float, base_pwr: float) -> tuple:
    """    
    Args:
        init_load: 初始进度, elapsed_time: 已用时间, res: 资源, base_pwr: 基础算力
    Returns:
        更新后的进度, 变化量
    """
    return elim_nume_error(init_load - elapsed_time * res * base_pwr), elim_nume_error(elapsed_time * res * base_pwr) 

def cal_cost(elapsed_time: float, res: float, base_pwr: float) -> float:
    return elim_nume_error(elapsed_time * res * base_pwr)

# Note that the 
def sim_comp_time(task_load: float, allocated_resources: int, base_power: float) -> float:
    """
    Args:
        task_load: 任务负载（剩余工作量）, allocated_resources: 分配的资源数量, base_power: 基础算力
    Returns:
        估计的执行时间
    """
    if allocated_resources <= 0 or base_power <= 0:
        return float('inf')
    execution_time = task_load / (allocated_resources * base_power)
    if unit_align:
        quant_fn = lambda x: normalize_time_to_unit(x, time_unit)
    else:
        quant_fn = lambda x: elim_nume_error(x)
    return quant_fn(execution_time)

def calculate_slack_time(deadline: float, current_time: float, reallocation_slack: float = 0) -> float:
    """
    计算任务的松弛时间
    
    Args:
        deadline: 截止时间
        current_time: 当前时间
        reallocation_slack: 重分配开销
    
    Returns:
        松弛时间
    """
    if unit_align:
        quant_fn = lambda x: normalize_time_to_unit(x, time_unit, mod='down')
    else:
        quant_fn = lambda x: elim_nume_error(x)
    return quant_fn(deadline - current_time - reallocation_slack) 

def estimate_resource_requirement(task_load: float, slack_time: float, base_power: float) -> int:
    """    
    Args:
        task_load: 任务负载（剩余工作量）
        slack_time: 可用时间窗口
        base_power: 基础算力
    Returns:
        估计所需的资源数量
    """
    if slack_time <= 0 or base_power <= 0:
        return 0
    return math.ceil(task_load / (slack_time * base_power))


def normalize_time_to_unit(time_value: float, time_unit: float, mod:str='up') -> float:
    """
    将时间值标准化到时间单位
    
    Args:
        time_value: 原始时间值
        time_unit: 时间单位
    
    Returns:
        标准化后的时间值
    """
    assert mod in ['round', 'up', 'down']
    assert time_unit <= 1
    n_bit = round(math.log10(1/time_unit))
    if mod == 'round':
        return round(time_value, n_bit)
    elif mod == 'up':
        return math.ceil(time_value / time_unit) * time_unit
    elif mod == 'down':
        return math.floor(time_value / time_unit) * time_unit
    else:
        raise ValueError(f"Invalid mode: {mod}")


def trasfer_realloc_as_task(BW_DRAM, cap, tile_buffer_size, time_norm_factor: float = 1.0):
    """
    将realloc作为任务，计算时间需要转换为计算量
    """
    return cap * tile_buffer_size /BW_DRAM / time_norm_factor


def get_task_load_and_base_size(node_attr: Dict[str, Any], time_norm_factor: float = 1.0) -> tuple:
    """
    从节点属性中提取任务负载和基础大小
    
    Args:
        node_attr: 节点属性字典
        normalize_factor: 标准化因子
    
    Returns:
        (exp_comp_t, base_size) 元组
    """
    if 'flops' in node_attr:
        exp_comp_t = node_attr['flops'] / time_norm_factor
    else:
        exp_comp_t = node_attr['exp_comp_t'] / time_norm_factor
    
    base_size = 1
    return exp_comp_t, base_size


def cal_load(exp_comp_t, base_size):
    """
    统一计算任务load的函数
    
    Args:
        node: 任务节点
        G_ptr: MyGraph实例    
    Returns:
        float: 任务的load值
    """
    return exp_comp_t * base_size


# 专门的时间比较函数
def time_eq(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) == elim_nume_error(time2)

def time_gt(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) > elim_nume_error(time2)

def time_gtq(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) >= elim_nume_error(time2)

def time_lt(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) < elim_nume_error(time2)

def time_ltq(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) <= elim_nume_error(time2)

def time_add(time1: float, time2: float) -> float:
    return elim_nume_error(time1 + time2)

def time_sub(time1: float, time2: float) -> float:
    return elim_nume_error(time1 - time2)
