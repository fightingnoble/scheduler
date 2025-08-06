"""
任务执行时间和资源需求估计的工具函数
"""
import math
from typing import Dict, Any
from global_var import elim_nume_error



def estimate_task_progress(init_load: float, elapsed_time: float, res: float, base_pwr: float) -> float:
    """
    估计任务的进度
    
    Args:
        init_load: 初始进度
        elapsed_time: 已用时间
        res: 资源
        base_pwr: 基础算力
    
    Returns:
        估计的进度
    """
    return elim_nume_error(init_load - elapsed_time * res * base_pwr)


def estimate_task_execution_time(task_load: float, allocated_resources: int, base_power: float) -> float:
    """
    估计任务执行时间
    
    Args:
        task_load: 任务负载（剩余工作量）
        allocated_resources: 分配的资源数量
        base_power: 基础算力
    
    Returns:
        估计的执行时间
    """
    if allocated_resources <= 0 or base_power <= 0:
        return float('inf')
    
    execution_time = task_load / (allocated_resources * base_power)
    return execution_time


def estimate_resource_requirement(task_load: float, slack_time: float, base_power: float) -> int:
    """
    估计任务所需的资源数量
    
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


def normalize_time_to_unit(time_value: float, time_unit: float) -> float:
    """
    将时间值标准化到时间单位
    
    Args:
        time_value: 原始时间值
        time_unit: 时间单位
    
    Returns:
        标准化后的时间值
    """
    return elim_nume_error(math.ceil(time_value / time_unit) * time_unit)


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
    return deadline - current_time - reallocation_slack 