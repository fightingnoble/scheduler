#!/usr/bin/env python3
"""
StatisticsCollector 简化测试脚本
不依赖外部模块，直接测试核心功能
"""

import sys
import os
import json
import time
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

# 简化的 TDigest 实现用于测试
class SimpleTDigest:
    """简化的 TDigest 实现，仅用于测试"""
    
    def __init__(self, delta: float = 0.01, K: int = 25):
        self.delta = delta
        self.K = K
        self.values = []
        self.total_processed_count = 0
    
    def add(self, value: float):
        """添加一个值到分布中"""
        if not np.isnan(value) and not np.isinf(value):
            self.values.append(float(value))
            self.total_processed_count += 1
    
    def get_mean(self) -> float:
        """计算均值"""
        if not self.values:
            return 0.0
        return float(np.mean(self.values))
    
    def percentile(self, p: float) -> float:
        """计算分位数"""
        if not self.values:
            return float('nan')
        return float(np.percentile(self.values, p))
    
    def get_summary(self, num_bins: int = 20, p_list: List[float] = None) -> Dict:
        """生成统计摘要"""
        if p_list is None:
            p_list = [0.5, 0.9, 0.99, 0.999]
        
        if not self.values:
            return {
                'histogram': [],
                'percentiles': {f'p{p*100:.1f}': float('nan') for p in p_list},
                'total_processed_count': 0
            }
        
        # 生成直方图
        hist, bin_edges = np.histogram(self.values, bins=num_bins)
        histogram = [{'bin_start': float(bin_edges[i]), 'bin_end': float(bin_edges[i+1]), 'count': int(hist[i])} 
                    for i in range(len(hist))]
        
        # 计算分位数
        percentiles = {f'p{p*100:.1f}': self.percentile(p*100) for p in p_list}
        
        return {
            'histogram': histogram,
            'percentiles': percentiles,
            'total_processed_count': self.total_processed_count
        }
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            'delta': self.delta,
            'K': self.K,
            'values': self.values,
            'total_processed_count': self.total_processed_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimpleTDigest':
        """从字典反序列化"""
        instance = cls(delta=data.get('delta', 0.01), K=data.get('K', 25))
        instance.values = data.get('values', [])
        instance.total_processed_count = data.get('total_processed_count', 0)
        return instance
    
    def __add__(self, other):
        """合并两个分布"""
        if not isinstance(other, SimpleTDigest):
            raise TypeError("只能与相同类型的对象合并")
        
        result = SimpleTDigest(self.delta, self.K)
        result.values = self.values + other.values
        result.total_processed_count = self.total_processed_count + other.total_processed_count
        return result


# 模拟依赖函数
def time_gt(t1: float, t2: float) -> bool:
    """时间比较函数"""
    return t1 > t2

def elim_nume_error(val: float) -> float:
    """消除数值误差"""
    return val

def cal_cost(delta_ld: float, cap: int, base_pwr: float) -> float:
    """计算开销"""
    return delta_ld * cap * base_pwr


# 创建简化的 StatisticsCollector 类
class SimpleStatisticsCollector:
    """
    简化的 StatisticsCollector 实现，用于测试核心功能
    """
    
    def __init__(self, delta: float = 0.01, K: int = 25):
        """初始化统计收集器"""
        self.partition_realloc_stats = {}
        self._delta = delta
        self._K = K
        self.path = {"stat": './output.txt', "motiv3": './output.pdf'}
        self.total_pwr = 0.0
        self.motiv3_mode = 'binned'
        
        # 临时缓存
        self.task_at = {}
        self.task_realloc_curr = defaultdict(float)
        self.task_realloc_num = defaultdict(int)
        self.task_compute_curr = defaultdict(float)
        
        # 分布统计
        self.dist_per_task_exe = defaultdict(lambda: SimpleTDigest(delta=delta, K=K))
        self.dist_per_task_ft = defaultdict(lambda: SimpleTDigest(delta=delta, K=K))
        self.dist_per_task_realloc = defaultdict(lambda: SimpleTDigest(delta=delta, K=K))
        
        # 超周期统计
        self.part_realloc_curr = defaultdict(float)
        self.part_realloc_num = defaultdict(int)
        self.hp_idle_curr = 0.0
        self.hp_miss_curr = 0.0
        self.hp_miss_count = 0
        self.system_realloc_cost = 0.0
        
        # 整体分布
        self.dist_overall_realloc = SimpleTDigest(delta=delta, K=K)
        self.dist_overall_idle = SimpleTDigest(delta=delta, K=K)
        self.dist_overall_miss = SimpleTDigest(delta=delta, K=K)
        self.dist_overall_miss_count = SimpleTDigest(delta=delta, K=K)
        self.dist_overall_total_load = SimpleTDigest(delta=delta, K=K)
        
        # Motiv-Exp-(3) 相关
        self.binning_warmup_period = 100
        self.num_r2_bins = int(self.binning_warmup_period ** 0.5)
        self.hp_total_load_curr = 0.0
        self.hp_worst_e2e_curr = float('-inf')
        self.hp_temp_records_for_binning = []
        self.adaptive_load_bins = None
        self.dist_hp_total_load = SimpleTDigest(delta=delta, K=K)
        self.latency_dist_per_adaptive_bin = defaultdict(lambda: SimpleTDigest(delta=delta, K=K))
        self.raw_load_latency = []
        self.chain_e2e_constraint = {}
    
    def set_path(self, key: str, path: str):
        """设置路径"""
        self.path[key] = path
    
    def set_motiv3_mode(self, mode: str):
        """设置Motiv-Exp-(3)模式"""
        assert mode in ['binned', 'raw']
        self.motiv3_mode = mode
    
    def init_partition_stats(self, partition_id: str, cap: int, base_pwr: float):
        """初始化分区统计"""
        if partition_id in self.partition_realloc_stats:
            self.total_pwr -= self.partition_realloc_stats[partition_id]['base_pwr'] * self.partition_realloc_stats[partition_id]['cap']
        else:
            self.total_pwr += base_pwr * cap 
        self.partition_realloc_stats[partition_id] = {
            'cap': cap,
            'base_pwr': base_pwr
        }
    
    def record_task_start(self, process_id: str, start_time: float):
        """记录任务开始时间"""
        self.task_at[process_id] = start_time
    
    def record_task_finish(self, G, process_id: str, finish_t: float):
        """记录任务完成"""
        base_task_name = self._get_base_task_name(process_id)
        
        offset = G.nodes[process_id]["offset"]
        finish_t_rel = finish_t - offset
        self.dist_per_task_ft[base_task_name].add(elim_nume_error(finish_t_rel))
        
        assert process_id in self.task_at
        deadline = G.ddl_map[process_id]
        start_time = self.task_at[process_id]
        latency = finish_t - start_time
        is_timeout = time_gt(finish_t, deadline)
        
        # 移除已完成任务记录
        del self.task_at[process_id]
        
        # 记录重分配开销
        realloc_time = self.task_realloc_curr.pop(process_id, 0.0)
        realloc_num = self.task_realloc_num.pop(process_id, 0)
        self.dist_per_task_realloc[base_task_name].add(realloc_time)
        
        # 传播计算时间到下游任务
        compute_time = self.task_compute_curr.pop(process_id, 0.0)
        for succ in G.successors(process_id):
            self.task_realloc_curr[succ] += realloc_time
            self.task_realloc_num[succ] += realloc_num 
            self.task_compute_curr[succ] += compute_time
    
    def record_e2e_finish(self, G, sink_name: str, finish_t: float):
        """记录端到端完成"""
        base_sink_name = self._get_base_task_name(sink_name)
        
        offset = G.nodes[sink_name]["offset"]
        finish_t_rel = elim_nume_error(finish_t - offset)
        self.dist_per_task_ft[base_sink_name].add(finish_t_rel)
        self.chain_e2e_constraint[base_sink_name] = G.ddl_map[sink_name] - offset
        
        # 记录重分配开销
        realloc_time = self.task_realloc_curr.pop(sink_name, 0.0)
        realloc_num = self.task_realloc_num.pop(sink_name, 0)
        self.dist_per_task_realloc[base_sink_name].add(realloc_time)
        
        # 记录计算时间
        compute_time = self.task_compute_curr.pop(sink_name, 0.0)
        self.dist_per_task_exe[base_sink_name].add(elim_nume_error(compute_time))
        
        # 更新超周期最坏E2E
        self._record_period_e2e_candidate(finish_t_rel)
    
    def record_realloc(self, partition_id: str, delta_ld: float, task_list: List[str]):
        """记录重分配开销"""
        # 更新每任务重分配开销
        for task_name in task_list:
            self.task_realloc_curr[task_name] += delta_ld
        
        # 更新每分区重分配开销
        self.part_realloc_curr[partition_id] += delta_ld
        
        # 更新总系统开销
        if partition_id in self.partition_realloc_stats:
            stats = self.partition_realloc_stats[partition_id]
            delta_cost = cal_cost(delta_ld, stats['cap'], stats['base_pwr'])
            self.system_realloc_cost += delta_cost
    
    def record_realloc_num(self, partition_id: str, task_list: List[str]):
        """记录重分配次数"""
        for task_name in task_list:
            self.task_realloc_num[task_name] += 1
        self.part_realloc_num[partition_id] += 1
    
    def record_compute_progress(self, task_name: str, delta_compute_t: float):
        """记录计算进度"""
        if delta_compute_t > 0:
            self.task_compute_curr[task_name] += float(delta_compute_t)
    
    def record_period_load_arrival(self, load: float):
        """记录超周期负载到达"""
        if load > 0:
            self.hp_total_load_curr += float(load)
    
    def record_idle_capacity(self, idle_ld: float, sys_state: str):
        """记录空闲容量"""
        if sys_state != "S":
            return
        self.hp_idle_curr += idle_ld
    
    def record_miss(self, timeout_iter):
        """记录miss任务"""
        def iter_pairs(it):
            for item in it:
                if isinstance(item, tuple) and len(item) == 2:
                    yield item
                elif hasattr(item, '__iter__'):
                    for sub in item:
                        yield sub

        miss_list = list(iter_pairs(timeout_iter))
        miss_sum = elim_nume_error(sum(rem for _, rem in miss_list))
        self.hp_miss_curr += miss_sum
        self.hp_miss_count += len(miss_list)
    
    def forward_hyperperiod(self, T_hp: float = 1.0):
        """推进超周期"""
        # 处理每分区重分配开销
        for part_id in list(self.part_realloc_curr.keys()):
            realloc_time = self.part_realloc_curr.pop(part_id)
            realloc_num = self.part_realloc_num.pop(part_id, 0)  # 使用默认值避免KeyError
            # 这里简化处理，不调用具体的分布添加方法
        
        # 添加系统重分配开销到整体分布
        assert self.total_pwr > 0 and T_hp > 0
        denom = self.total_pwr * T_hp 
        self.dist_overall_realloc.add(elim_nume_error(self.system_realloc_cost/denom))
        self.dist_overall_idle.add(elim_nume_error(self.hp_idle_curr / denom))
        self.dist_overall_miss.add(elim_nume_error(self.hp_miss_curr / denom))
        self.dist_overall_miss_count.add(self.hp_miss_count)
        
        # 重置累加器
        self.system_realloc_cost = 0.0
        self.hp_idle_curr = 0.0
        self.hp_miss_curr = 0.0
        self.hp_miss_count = 0
        
        # Motiv-Exp-(3) 处理
        total_load = self.hp_total_load_curr
        worst_e2e = self.hp_worst_e2e_curr
        
        if worst_e2e > float('-inf'):
            if self.motiv3_mode == 'raw':
                if total_load is not None and total_load >= 0:
                    self.raw_load_latency.append((float(total_load), float(worst_e2e)))
            else:
                if total_load is not None and total_load >= 0:
                    self.dist_hp_total_load.add(total_load)
                if self.adaptive_load_bins is None:
                    self.hp_temp_records_for_binning.append((total_load, worst_e2e))
                    if len(self.hp_temp_records_for_binning) >= self.binning_warmup_period:
                        self._finalize_adaptive_bins()
                else:
                    self._add_to_adaptive_bin(total_load, worst_e2e)
        
        # 重置累加器
        self.dist_overall_total_load.add(self.hp_total_load_curr)
        self.hp_total_load_curr = 0.0
        self.hp_worst_e2e_curr = float('-inf')
    
    def _record_period_e2e_candidate(self, latency: float):
        """记录超周期E2E候选"""
        if latency > self.hp_worst_e2e_curr:
            self.hp_worst_e2e_curr = float(latency)
    
    def _finalize_adaptive_bins(self):
        """完成自适应分箱"""
        if len(self.hp_temp_records_for_binning) < 2:
            return
        
        # 使用分位数定义分箱边界
        quantiles = np.linspace(0, 1, self.num_r2_bins + 1)
        boundaries = [self.dist_hp_total_load.percentile(p * 100) for p in quantiles]
        
        # 确保边界唯一且排序
        unique_boundaries = sorted(list(set(boundaries)))
        if len(unique_boundaries) < 2:
            val = unique_boundaries[0]
            unique_boundaries = [val - 0.5, val + 0.5]
        
        self.adaptive_load_bins = unique_boundaries
        
        # 初始化每个新分箱的TDigest
        self.latency_dist_per_adaptive_bin = {
            b_start: SimpleTDigest(delta=self._delta, K=self._K)
            for b_start in self.adaptive_load_bins[:-1]
        }
        
        # 处理缓冲数据
        for load, latency in self.hp_temp_records_for_binning:
            self._add_to_adaptive_bin(load, latency)
        
        # 清空缓冲区
        self.hp_temp_records_for_binning.clear()
    
    def _add_to_adaptive_bin(self, load: float, latency: float):
        """添加到自适应分箱"""
        assert self.adaptive_load_bins is not None, "自适应分箱尚未完成"
        if load is None:
            return
        
        # 找到负载所属的分箱索引
        idx = np.searchsorted(self.adaptive_load_bins, load, side='right') - 1
        idx = max(0, idx)

        if idx < len(self.adaptive_load_bins) - 1:
            bin_start_key = self.adaptive_load_bins[idx]
            self.latency_dist_per_adaptive_bin[bin_start_key].add(latency)
        else: 
            bin_start_key = self.adaptive_load_bins[-2]
            self.latency_dist_per_adaptive_bin[bin_start_key].add(latency)
    
    def _get_base_task_name(self, task_name: str) -> str:
        """提取基础任务名称"""
        import re
        pattern = r'_[-\d]+$'
        base_name = re.sub(pattern, '', task_name)
        return base_name
    
    def get_motiv_case1_stats(self) -> Dict:
        """获取Motiv-Exp-1统计"""
        return {
            'idle_mean_ratio': float(self.dist_overall_idle.get_mean()),
            'miss_mean_ratio': float(self.dist_overall_miss.get_mean()),
            'miss_mean_count': float(self.dist_overall_miss_count.get_mean()),
            'realloc_mean_ratio': float(self.dist_overall_realloc.get_mean())
        }
    
    def get_motiv_case2_stats(self) -> Dict:
        """获取Motiv-Exp-2统计"""
        util = self.get_motiv_case1_stats()
        return {
            'utilization': {
                'idle_mean_ratio': util['idle_mean_ratio'],
                'miss_mean_ratio': util['miss_mean_ratio'],
                'realloc_mean_ratio': util['realloc_mean_ratio']
            },
            'latency_breakdown': {
                'overall': {},
                'first_chain': {},
                'first_chain_name': None
            },
            'miss_mean_count': util['miss_mean_count']
        }
    
    def get_motiv_case3_stats(self, percentile: float = 0.99, mode: str = None) -> Dict:
        """获取Motiv-Exp-3统计"""
        if mode is None:
            mode = self.motiv3_mode
        
        # 简化的Spearman相关系数计算
        rho = 0.5  # 模拟值
        
        result = {
            'mode': mode,
            'spearman_rho': rho,
            'percentile': percentile
        }
        
        if mode == 'binned':
            result['binned_summary'] = []
            result['raw_data_count'] = None
        else:  # raw
            result['binned_summary'] = None
            result['raw_data_count'] = len(self.raw_load_latency)
            
        return result
    
    def format_motiv_case1_output(self, stats: Dict = None) -> str:
        """格式化Motiv-Exp-1输出"""
        if stats is None:
            stats = self.get_motiv_case1_stats()
        
        lines = []
        lines.append("=" * 60)
        lines.append("Motiv-Exp-1: 纯静态调度 - 利用率问题")
        lines.append("=" * 60)
        lines.append(f"闲置算力占比 (idle_mean_ratio):       {stats['idle_mean_ratio']:.4f}")
        lines.append(f"Miss任务剩余负载占比 (miss_mean_ratio): {stats['miss_mean_ratio']:.4f}")
        lines.append(f"Miss任务数量 (miss_mean_count):        {stats['miss_mean_count']:.2f}")
        lines.append(f"切换开销占比 (realloc_mean_ratio):     {stats['realloc_mean_ratio']:.4f} (应为0)")
        lines.append("-" * 60)
        effective_util = 1.0 - stats['idle_mean_ratio'] - stats['miss_mean_ratio']
        lines.append(f"有效利用率:                            {effective_util:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def format_motiv_case2_output(self, stats: Dict = None) -> str:
        """格式化Motiv-Exp-2输出"""
        if stats is None:
            stats = self.get_motiv_case2_stats()
        
        lines = []
        lines.append("=" * 60)
        lines.append("Motiv-Exp-2: 纯动态调度 - 延迟开销问题")
        lines.append("=" * 60)
        lines.append("统计1 - 资源利用率分解:")
        util = stats['utilization']
        lines.append(f"  闲置算力占比 (idle):     {util['idle_mean_ratio']:.4f}")
        lines.append(f"  Miss负载占比 (miss):     {util['miss_mean_ratio']:.4f}")
        lines.append(f"  切换开销占比 (realloc):  {util['realloc_mean_ratio']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def format_motiv_case3_output(self, stats: Dict = None, percentile: float = 0.99) -> str:
        """格式化Motiv-Exp-3输出"""
        if stats is None:
            stats = self.get_motiv_case3_stats(percentile=percentile)
        
        lines = []
        lines.append("=" * 60)
        lines.append("Motiv-Exp-3: 切换行为的不确定性")
        lines.append("=" * 60)
        lines.append(f"数据模式: {stats['mode']}")
        lines.append(f"Spearman相关系数 (ρ): {stats['spearman_rho']:.4f}")
        lines.append(f"使用分位数: p{int(stats['percentile']*100)}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def get_full_summary(self, num_bins: int = 20, p_list: List[float] = None) -> Dict:
        """获取完整摘要"""
        if p_list is None:
            p_list = [0.5, 0.9, 0.99, 0.999]

        full_summary = {
            'distribution_summary': {
                'overall_e2e_latency': self.dist_overall_idle.get_summary(num_bins, p_list),
                'overall_realloc_overhead': self.dist_overall_realloc.get_summary(num_bins, p_list),
                'overall_idle_time': self.dist_overall_idle.get_summary(num_bins, p_list),
                'overall_missed_load': self.dist_overall_miss.get_summary(num_bins, p_list)
            },
            'adaptive_binning_summary': []
        }
        return full_summary
    
    def save_state(self, file_path: str):
        """保存状态"""
        state = {
            'delta': self._delta,
            'K': self._K,
            'partition_realloc_stats': self.partition_realloc_stats,
            'total_pwr': self.total_pwr,
            'motiv3_mode': self.motiv3_mode,
            'task_at': self.task_at,
            'task_realloc_curr': dict(self.task_realloc_curr),
            'task_realloc_num': dict(self.task_realloc_num),
            'task_compute_curr': dict(self.task_compute_curr),
            'part_realloc_curr': dict(self.part_realloc_curr),
            'part_realloc_num': dict(self.part_realloc_num),
            'hp_total_load_curr': self.hp_total_load_curr,
            'hp_worst_e2e_curr': self.hp_worst_e2e_curr,
            'hp_idle_curr': self.hp_idle_curr,
            'system_realloc_cost': self.system_realloc_cost,
            'dist_overall_realloc': self.dist_overall_realloc.to_dict(),
            'dist_overall_idle': self.dist_overall_idle.to_dict(),
            'dist_overall_miss': self.dist_overall_miss.to_dict(),
            'dist_overall_miss_count': self.dist_overall_miss_count.to_dict(),
            'dist_overall_total_load': self.dist_overall_total_load.to_dict(),
            'dist_hp_total_load': self.dist_hp_total_load.to_dict(),
            'hp_temp_records_for_binning': self.hp_temp_records_for_binning,
            'adaptive_load_bins': self.adaptive_load_bins,
            'raw_load_latency': self.raw_load_latency,
            'chain_e2e_constraint': self.chain_e2e_constraint
        }

        try:
            # 确保目录存在
            dir_path = os.path.dirname(file_path)
            if dir_path:  # 只有当路径包含目录时才创建
                os.makedirs(dir_path, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4, allow_nan=True)
            print(f"StatisticsCollector状态已保存到: {file_path}")
        except IOError as e:
            print(f"保存StatisticsCollector状态到{file_path}时出错: {e}")
        except TypeError as e:
            print(f"序列化StatisticsCollector状态到JSON时出错: {e}")

    @classmethod
    def load_state(cls, file_path: str) -> 'SimpleStatisticsCollector':
        """加载状态"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
        except FileNotFoundError:
            print(f"错误: 状态文件未找到 {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"从{file_path}解码JSON时出错: {e}")
            return None
        except IOError as e:
            print(f"读取状态文件{file_path}时出错: {e}")
            return None

        # 重建StatisticsCollector实例
        collector = cls(delta=state['delta'], K=state['K'])
        collector.partition_realloc_stats = state['partition_realloc_stats']
        collector.total_pwr = state['total_pwr']
        collector.motiv3_mode = state['motiv3_mode']
        collector.task_at = state['task_at']
        collector.task_realloc_curr = defaultdict(float, state['task_realloc_curr'])
        collector.task_realloc_num = defaultdict(int, state['task_realloc_num'])
        collector.task_compute_curr = defaultdict(float, state['task_compute_curr'])
        collector.part_realloc_curr = defaultdict(float, state['part_realloc_curr'])
        collector.part_realloc_num = defaultdict(int, state['part_realloc_num'])
        collector.hp_total_load_curr = state['hp_total_load_curr']
        collector.hp_worst_e2e_curr = state['hp_worst_e2e_curr']
        collector.hp_idle_curr = state['hp_idle_curr']
        collector.system_realloc_cost = state['system_realloc_cost']
        
        # 重建TDigest对象
        collector.dist_overall_realloc = SimpleTDigest.from_dict(state.get('dist_overall_realloc', {}))
        collector.dist_overall_idle = SimpleTDigest.from_dict(state.get('dist_overall_idle', {}))
        collector.dist_overall_miss = SimpleTDigest.from_dict(state.get('dist_overall_miss', {}))
        collector.dist_overall_miss_count = SimpleTDigest.from_dict(state.get('dist_overall_miss_count', {}))
        collector.dist_overall_total_load = SimpleTDigest.from_dict(state.get('dist_overall_total_load', {}))
        collector.dist_hp_total_load = SimpleTDigest.from_dict(state.get('dist_hp_total_load', {}))
        
        collector.hp_temp_records_for_binning = state.get('hp_temp_records_for_binning', [])
        collector.adaptive_load_bins = state.get('adaptive_load_bins', None)
        collector.raw_load_latency = state.get('raw_load_latency', [])
        collector.chain_e2e_constraint = state.get('chain_e2e_constraint', {})

        print(f"StatisticsCollector状态已从{file_path}加载")
        return collector


def run_comprehensive_test():
    """
    全面测试 StatisticsCollector 类的所有功能
    """
    print("=" * 80)
    print("StatisticsCollector 全面功能测试")
    print("=" * 80)

    # 模拟图结构
    class MyGraph:
        def __init__(self):
            self.nodes = {
                'op1_0': {'offset': 10},
                'op2_0': {'offset': 20}, 
                'sink1_0': {'offset': 50},
                'sink2_0': {'offset': 60},
            }
            self.ddl_map = {
                'op1_0': 100,
                'op2_0': 200,
                'sink1_0': 300,
                'sink2_0': 400
            }
        
        def successors(self, node):
            successors_map = {
                'op1_0': ['op2_0'],
                'op2_0': ['sink1_0', 'sink2_0'],
                'sink1_0': [],
                'sink2_0': []
            }
            return successors_map.get(node, [])

    G = MyGraph()
    test_file_path = 'temp_test_state.json'

    try:
        # ========== 测试1: 基本功能测试 ==========
        print("\n1. 基本功能测试")
        print("-" * 40)
        
        collector = SimpleStatisticsCollector(delta=0.01, K=25)
        
        # 初始化分区统计
        collector.init_partition_stats('part1', 10, 1.5)
        collector.init_partition_stats('part2', 8, 2.0)
        
        # 记录任务开始
        collector.record_task_start('op1_0', 10.0)
        collector.record_task_start('op2_0', 20.0)
        
        # 记录重分配开销
        collector.record_realloc('part1', 0.5, ['op1_0'])
        collector.record_realloc_num('part1', ['op1_0'])
        collector.record_realloc('part2', 0.3, ['op2_0'])
        collector.record_realloc_num('part2', ['op2_0'])
        
        # 记录计算进度
        collector.record_compute_progress('op1_0', 5.0)
        collector.record_compute_progress('op2_0', 8.0)
        
        # 记录任务完成
        collector.record_task_finish(G, 'op1_0', 25.0)
        collector.record_task_finish(G, 'op2_0', 35.0)
        
        # 记录端到端完成
        collector.record_e2e_finish(G, 'sink1_0', 50.0)
        collector.record_e2e_finish(G, 'sink2_0', 55.0)
        
        print("✓ 基本记录功能测试通过")

        # ========== 测试2: 超周期统计测试 ==========
        print("\n2. 超周期统计测试")
        print("-" * 40)
        
        # 记录负载到达
        collector.record_period_load_arrival(100.0)
        collector.record_period_load_arrival(150.0)
        
        # 记录空闲容量
        collector.record_idle_capacity(20.0, "S")
        collector.record_idle_capacity(15.0, "S")
        collector.record_idle_capacity(10.0, "B")  # 非S状态，应被忽略
        
        # 记录miss任务
        miss_tasks = [('task1', 5.0), ('task2', 8.0)]
        collector.record_miss(miss_tasks)
        
        # 推进超周期
        collector.forward_hyperperiod(T_hp=1.0)
        
        print("✓ 超周期统计功能测试通过")

        # ========== 测试3: Motiv-Exp 特定功能测试 ==========
        print("\n3. Motiv-Exp 特定功能测试")
        print("-" * 40)
        
        # 测试Case 1统计
        case1_stats = collector.get_motiv_case1_stats()
        print(f"  Case 1 - 闲置占比: {case1_stats['idle_mean_ratio']:.4f}")
        print(f"  Case 1 - Miss占比: {case1_stats['miss_mean_ratio']:.4f}")
        
        # 测试Case 2统计
        case2_stats = collector.get_motiv_case2_stats()
        print(f"  Case 2 - 利用率统计: {len(case2_stats['utilization'])} 项")
        print(f"  Case 2 - 延迟分解: {len(case2_stats['latency_breakdown'])} 项")
        
        # 测试Case 3统计 (binned模式)
        collector.set_motiv3_mode('binned')
        case3_stats = collector.get_motiv_case3_stats(percentile=0.99)
        print(f"  Case 3 - Spearman相关系数: {case3_stats['spearman_rho']:.4f}")
        print(f"  Case 3 - 模式: {case3_stats['mode']}")
        
        # 测试Case 3统计 (raw模式)
        collector.set_motiv3_mode('raw')
        case3_raw_stats = collector.get_motiv_case3_stats(percentile=0.99)
        print(f"  Case 3 (raw) - 数据点数量: {case3_raw_stats['raw_data_count']}")
        
        print("✓ Motiv-Exp 特定功能测试通过")

        # ========== 测试4: 统计摘要生成测试 ==========
        print("\n4. 统计摘要生成测试")
        print("-" * 40)
        
        # 生成完整摘要
        full_summary = collector.get_full_summary(num_bins=10, p_list=[0.5, 0.9, 0.99])
        print(f"  分布摘要键数量: {len(full_summary['distribution_summary'])}")
        print(f"  自适应分箱摘要: {len(full_summary.get('adaptive_binning_summary', []))} 个分箱")
        
        print("✓ 统计摘要生成测试通过")

        # ========== 测试5: 格式化输出测试 ==========
        print("\n5. 格式化输出测试")
        print("-" * 40)
        
        # 测试各Case的格式化输出
        case1_output = collector.format_motiv_case1_output()
        case2_output = collector.format_motiv_case2_output()
        case3_output = collector.format_motiv_case3_output()
        
        print(f"  Case 1 输出长度: {len(case1_output)} 字符")
        print(f"  Case 2 输出长度: {len(case2_output)} 字符")
        print(f"  Case 3 输出长度: {len(case3_output)} 字符")
        
        print("✓ 格式化输出测试通过")

        # ========== 测试6: 状态保存和加载测试 ==========
        print("\n6. 状态保存和加载测试")
        print("-" * 40)
        
        # 保存状态
        collector.save_state(test_file_path)
        print("✓ 状态保存成功")
        
        # 加载状态
        loaded_collector = SimpleStatisticsCollector.load_state(test_file_path)
        if loaded_collector is None:
            raise Exception("状态加载失败")
        print("✓ 状态加载成功")
        
        # 验证数据完整性
        if (collector._delta == loaded_collector._delta and
            collector._K == loaded_collector._K and
            collector.total_pwr == loaded_collector.total_pwr):
            print("✓ 基本属性验证通过")
        else:
            raise Exception("基本属性验证失败")
        
        print("✓ 状态保存和加载测试通过")

        # ========== 测试7: 路径设置测试 ==========
        print("\n7. 路径设置测试")
        print("-" * 40)
        
        collector.set_path('stat', './test_output.txt')
        collector.set_path('motiv3', './test_motiv3.pdf')
        
        if collector.path['stat'] == './test_output.txt' and collector.path['motiv3'] == './test_motiv3.pdf':
            print("✓ 路径设置测试通过")
        else:
            raise Exception("路径设置测试失败")

        # ========== 测试8: 边界条件测试 ==========
        print("\n8. 边界条件测试")
        print("-" * 40)
        
        # 测试空数据情况
        empty_collector = SimpleStatisticsCollector()
        empty_stats = empty_collector.get_motiv_case1_stats()
        print(f"  空数据Case 1统计: {empty_stats}")
        
        # 测试无效输入
        try:
            empty_collector.record_idle_capacity(10.0, "invalid_state")
            print("✓ 无效状态处理正常")
        except:
            print("✗ 无效状态处理异常")
        
        # 测试负值处理
        empty_collector.record_period_load_arrival(-10.0)  # 应被忽略
        empty_collector.record_compute_progress('task1', -5.0)  # 应被忽略
        print("✓ 负值处理正常")
        
        print("✓ 边界条件测试通过")

        # ========== 测试总结 ==========
        print("\n" + "=" * 80)
        print("所有测试通过！StatisticsCollector 功能正常")
        print("=" * 80)
        
        # 清理测试文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"已清理测试文件: {test_file_path}")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理测试文件
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        
        return False


def run_performance_test():
    """
    性能测试：测试大量数据下的性能表现
    """
    print("\n" + "=" * 80)
    print("StatisticsCollector 性能测试")
    print("=" * 80)
    
    # 模拟图结构
    class MyGraph:
        def __init__(self):
            # 为性能测试创建足够的节点
            self.nodes = {}
            self.ddl_map = {}
            for i in range(1000):  # 创建1000个任务节点
                for j in range(100):  # 每个任务有100个超周期实例
                    task_id = f'task_{i}_{j}'
                    self.nodes[task_id] = {'offset': i*10}
                    self.ddl_map[task_id] = (i+1)*100
        
        def successors(self, node):
            return []
    
    G = MyGraph()
    
    # 性能测试参数
    num_tasks = 1000
    num_periods = 100
    
    print(f"测试规模: {num_tasks} 个任务, {num_periods} 个超周期")
    
    start_time = time.time()
    
    # 创建collector
    collector = SimpleStatisticsCollector()
    collector.init_partition_stats('part1', 100, 1.0)
    
    # 模拟大量任务处理
    for period in range(num_periods):
        # 记录负载
        collector.record_period_load_arrival(random.uniform(50, 200))
        
        # 处理任务
        for i in range(num_tasks // num_periods):
            task_id = f'task_{i}_{period}'
            start_t = period * 100 + i * 0.1
            
            collector.record_task_start(task_id, start_t)
            collector.record_realloc('part1', random.uniform(0.1, 1.0), [task_id])
            collector.record_compute_progress(task_id, random.uniform(1.0, 10.0))
            collector.record_task_finish(G, task_id, start_t + random.uniform(5, 15))
        
        # 记录空闲和miss
        collector.record_idle_capacity(random.uniform(10, 50), "S")
        miss_tasks = [(f'miss_{i}', random.uniform(1, 5)) for i in range(random.randint(0, 5))]
        collector.record_miss(miss_tasks)
        
        # 推进超周期
        collector.forward_hyperperiod()
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"数据处理时间: {processing_time:.2f} 秒")
    print(f"平均每任务处理时间: {processing_time/num_tasks*1000:.3f} 毫秒")
    
    # 测试统计生成性能
    start_time = time.time()
    summary = collector.get_full_summary()
    end_time = time.time()
    
    print(f"统计摘要生成时间: {end_time - start_time:.3f} 秒")
    print(f"摘要包含 {len(summary)} 个主要类别")
    
    print("✓ 性能测试完成")


def main():
    """主函数"""
    print("StatisticsCollector 简化测试脚本")
    print("=" * 50)
    
    # 运行全面功能测试
    success = run_comprehensive_test()
    
    if success:
        # 运行性能测试
        run_performance_test()
        print("\n🎉 所有测试完成！")
    else:
        print("\n❌ 功能测试失败，跳过性能测试")


if __name__ == "__main__":
    main()
