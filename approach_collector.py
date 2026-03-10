from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from approach_def import MyGraph
from typing import Dict, List, Iterable, Tuple
from collections import defaultdict
import re
import json
import os
from functools import reduce
from approach_Eq import time_gt, elim_nume_error, cal_cost
from ref_tdigest import TDigestStreamingHistogram # Assuming ref_tdigest.py is in the same directory
import numpy as np
import matplotlib.pyplot as plt

"""
Collection policy: 
    1. End-to-end, single-task latency: 
        Determine node type: sink, op
        Record upon task completion: finish_time - offset
    2. Per-partition/per-task chain realloc situation:
        Record realloc overhead each time remaining realloc tasks are updated.
        Maintain two data structures: per-task realloc overhead and per-hyperperiod overhead.
        Record per-hyperperiod overhead at the end of a period.
        Record per-task overhead at task completion, and accumulate overhead to downstream tasks in the table.
    3. required/idle/submitted resources each hyperperiod
        record required resource when tasks is released
        record idle resource when update remaining load
    4. task completion breakdown
        record execution time when update remaining load

        
Analysis stage statistics:
    1. Sum of (all partition realloc time * partition capacity * base_power).
    2. Histogram or quantile statistics of the distribution.

How to handle remaining tasks left in the queue.
"""

class StatisticsCollector:
    """
    Statistics collector, implementing streaming statistics based on T-Digest.
    Supports long-tailed distributions and high-quantile estimation.
    """
    
    def __init__(self, delta: float = 0.01, K: int = 25, **kwargs):
        """
        Initializes the statistics collector.
        
        Args:
            delta: T-Digest precision parameter, controlling relative error.
            K: T-Digest internal merging parameter.
        """
        # partition info
        self.partition_realloc_stats = {}
        self._delta = delta
        self._K = K
        self.summary = None
        self.p_list = None
        # Store delta and K for creating new TDigestStreamingHistogram objects if needed
        self.path = {"stat": './output.txt', "motiv3": './output.pdf'}
        self.total_pwr = 0.0
        self.task_cnt = 0
        # Motiv-Exp-(3) storage mode: 'binned' or 'raw'
        self.motiv3_mode = 'binned'
        self.motiv3_en = False
        
        # Parse stat_param from kwargs
        if 'motiv3_mode' in kwargs:
            self.set_motiv3_mode(kwargs['motiv3_mode'])
        if 'motiv3_en' in kwargs:
            self.motiv3_en = kwargs['motiv3_en']
        
        # Temp cache for Lat breakdown 
        # Per-task (instance) property
        # collected and reset at task completion
        # propagated to and aggregated at downstream
        self.task_at = {} # arrival time of each task

        # New logic for critical path tracking by user
        self.task_curr_stat = defaultdict(
            lambda: {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0}
        )
        # for each task, use a dict to store the stat of predecessors
        # task_pred_stat[task][pred] -> {'realloc','compute','realloc_num','e2e_lat'}
        self.task_pred_stat = defaultdict(
            lambda: defaultdict(lambda: {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0, 'e2e_lat': 0.0})
        )

        # distribution for latency breakdown
        self.dist_per_task_exe = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )
        self.dist_per_task_ft = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )
        self.dist_per_task_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )

        # Temp cache for usage
        # collected and reset at the end of each hyperperiod
        self.part_realloc_curr = defaultdict(float) # submitted realloc overhead 
        self.part_realloc_num = defaultdict(int) # submitted realloc number
        self.hp_idle_curr = 0.0 # idle compute
        self.hp_miss_curr = 0.0 # missed load (absolute), normalized at period end
        self.hp_used_curr = 0.0 # used load (absolute), normalized at period end
        self.hp_miss_count = 0 # number of missed tasks per period
        self.system_realloc_cost = 0.0  # System-level total reallocation cost
        # distribution for usage
        self.dist_per_part_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )
        self.dist_overall_realloc = TDigestStreamingHistogram(delta=delta, K=K) # normlized by total power
        self.dist_overall_idle = TDigestStreamingHistogram(delta=delta, K=K) 
        self.dist_overall_miss = TDigestStreamingHistogram(delta=delta, K=K)
        self.dist_overall_used = TDigestStreamingHistogram(delta=delta, K=K)
        self.dist_overall_miss_count = TDigestStreamingHistogram(delta=delta, K=K)  # number of missed tasks per period
        self.dist_overall_total_load = TDigestStreamingHistogram(delta=delta, K=K)  # total load per period
        self.dist_per_part_realloc_count = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )

        # For Motiv-Exp-(3): Adaptive binning for load vs. worst E2E latency relationship
        self.binning_warmup_period = 100  # Number of samples to learn the distribution from
        self.num_r2_bins = int(self.binning_warmup_period ** 0.5) # Number of bins for the histograms in the summary.
        self.hp_total_load_curr = 0.0 # submitted load
        self.hp_worst_e2e_curr = float('-inf') # worst e2e latency
        self.hp_temp_records_for_binning = [] # buffer for learning the distribution 

        self.adaptive_load_bins = None  # Will store list of bin boundaries, e.g., [0, 1000, 2500, ...]
        self.dist_hp_total_load = TDigestStreamingHistogram(delta=delta, K=K)
        self.latency_dist_per_adaptive_bin = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        ) # Keys will be the start of each bin
        # Raw storage for Motiv-Exp-(3) if needed
        self.raw_load_latency: List[Tuple[float, float]] = []
        # Per-chain E2E constraint (relative deadline) for normalization
        self.chain_e2e_constraint: Dict[str, float] = {}

    def set_path(self, key: str, path: str):
        self.path[key] = path
    
    def set_task_cnt(self, task_cnt: int):
        self.task_cnt = task_cnt

    def set_motiv3_mode(self, mode: str):
        """Set Motiv-Exp-(3) collection mode: 'binned' or 'raw'"""
        assert mode in ['binned', 'raw']
        self.motiv3_mode = mode

    def init_partition_stats(
            self, partition_id: str, cap: int, base_pwr: float
        ):
        """Initializes statistics for a partition."""
        if partition_id in self.partition_realloc_stats:
            self.total_pwr -= self.partition_realloc_stats[partition_id]['base_pwr'] * self.partition_realloc_stats[partition_id]['cap']
        else:
            self.total_pwr += base_pwr * cap 
        self.partition_realloc_stats[partition_id] = {
            'cap': cap,
            'base_pwr': base_pwr
        }
    
    def record_task_start(self, process_id: str, start_time: float):
        """Records the start time of a task."""
        self.task_at[process_id] = start_time

    def record_task_finish(self, G: MyGraph, process_id: str, finish_t: float):
        """
        Records task completion: finish_time - offset.
        Determines node type: sink, op.
        """
        base_task_name = self._get_base_task_name(process_id)
        
        offset = G.nodes[process_id]["offset"]
        finish_t_rel = finish_t - offset
        self.dist_per_task_ft[base_task_name].add(elim_nume_error(finish_t_rel))

        assert process_id in self.task_at
        deadline = G.ddl_map[process_id] # Assuming ddl_map is available in G
        start_time = self.task_at[process_id]
        latency = finish_t - start_time
        is_timeout = time_gt(finish_t, deadline) # Assuming time_gt is defined
        
        # Remove completed task record
        del self.task_at[process_id]

        # Pop this task's own stats
        my_stat = self.task_curr_stat.pop(process_id, {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0})
        compute_time = my_stat['compute']
        realloc_time = my_stat['realloc']
        realloc_num = my_stat['realloc_num']

        # --- Critical Path Propagation Logic ---
        pred_stat = self.task_pred_stat.pop(process_id, {})
        # 1. Find the critical path predecessor based on e2e_lat (finish time relative to offset)
        if not pred_stat:
            # Source node case: its predecessor is the conceptual start
            crit_pred_stat = {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0, 'e2e_lat': start_time - offset}
        else:
            crit_pred_stat = max(pred_stat.values(), key=lambda x: x['e2e_lat'])
        
        # 2. Update the stat of current task's path
        path_stat = {
            'compute': crit_pred_stat['compute'] + compute_time,
            'realloc': crit_pred_stat['realloc'] + realloc_time,
            'realloc_num': crit_pred_stat['realloc_num'] + realloc_num,
            'e2e_lat': finish_t_rel,
        }

        # For non-sink nodes, dists are for path-based accumulation
        self.dist_per_task_realloc[base_task_name].add(path_stat['realloc'])
        self.dist_per_task_exe[base_task_name].add(elim_nume_error(path_stat['compute']))

        # 3. Propagate path stat to all successors
        for succ in G.successors(process_id):
            self.task_pred_stat[succ][process_id] = path_stat
    
    def record_e2e_finish(self, G: MyGraph, sink_name: str, finish_t: float):
        """
        Records end-to-end completion: full latency from source to sink task.
        """
        base_sink_name = self._get_base_task_name(sink_name)
        
        offset = G.nodes[sink_name]["offset"]
        finish_t_rel = elim_nume_error(finish_t - offset)
        # record finish time
        self.dist_per_task_ft[base_sink_name].add(finish_t_rel)
        # cache chain constraint if available (relative deadline from src to sink)
        self.chain_e2e_constraint[base_sink_name] = G.ddl_map[sink_name] - offset
        
        # --- Critical Path Final Calculation for Sink ---
        pred_stat = self.task_pred_stat.pop(sink_name, {})
        if not pred_stat:
            # This can happen if a source is directly connected to a sink
            crit_pred_stat = {'realloc': 0.0, 'compute': 0.0}
        else:
            crit_pred_stat = max(pred_stat.values(), key=lambda x: x['e2e_lat'])

        path_compute = crit_pred_stat['compute']
        path_realloc = crit_pred_stat['realloc']
        
        # record realloc overhead
        self.dist_per_task_realloc[base_sink_name].add(path_realloc)
        # record compute time
        self.dist_per_task_exe[base_sink_name].add(elim_nume_error(path_compute))

        # update per-hyperperiod worst e2e
        self._record_period_e2e_candidate(finish_t_rel)

    def record_realloc(self, partition_id: str, delta_ld: float, task_list: List[str]):
        """
        Records reallocation overhead each time the remaining realloc tasks are updated.
        Maintains per-task realloc overhead and per-hyperperiod overhead.
        """
        # Update per-task realloc overhead (own time)
        for task_name in task_list:
            self.task_curr_stat[task_name]['realloc'] += delta_ld
        
        # Update per-part realloc overhead
        self.part_realloc_curr[partition_id] += delta_ld
        
        # Update total system cost
        if partition_id in self.partition_realloc_stats:
            stats = self.partition_realloc_stats[partition_id]
            delta_cost = cal_cost(delta_ld, stats['cap'], stats['base_pwr'])
            self.system_realloc_cost += delta_cost

    def record_realloc_num(self, partition_id: str, task_list: List[str]):
        """
        Records reallocation overhead each time the remaining realloc tasks are updated.
        Maintains per-task realloc overhead and per-hyperperiod overhead.
        """
        # Update per-task realloc overhead
        for task_name in task_list:
            self.task_curr_stat[task_name]['realloc_num'] += 1
        
        # Update per-part realloc overhead
        self.part_realloc_num[partition_id] += 1


    # ---------------- New minimal APIs for experiments ----------------
    def record_compute_progress(self, task_name: str, delta_compute_t: float, delta_load: float=None):
        """Accumulate compute time for a task instance (delta time domain)."""
        if delta_compute_t > 0:
            self.task_curr_stat[task_name]['compute'] += float(delta_compute_t)
        if delta_load is not None:
            self.hp_used_curr += delta_load

    def record_period_load_arrival(self, load: float):
        """Accumulate total submitted load within current hyperperiod."""
        if load > 0:
            self.hp_total_load_curr += float(load)

    def record_idle_capacity(self, idle_ld: float, sys_state: str):
        """Record idle capacity over a time slice. Idle counts only when device is in 'S' (available) state.
        idle = max(0, cap - allocated) * delta_t when sys_state=='S'; otherwise 0.
        """
        if sys_state != "S":
            return
        self.hp_idle_curr += idle_ld

    def record_miss(self, timeout_iter:Iterable[Tuple[str, float]]):
        """Accumulate remaining load of overdue (missed) tasks at hyperperiod boundary.
        Only supports dict-like ready/running containers where values are remaining load.
        Returns the accumulated amount just added.
        """
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


    def forward_hyperperiod(self, T_hp: float=1.0):
        """
        Records per-hyperperiod overhead at the end of a period.
        """        
        # Process per-part realloc overhead
        for part_id in list(self.part_realloc_curr.keys()):
            realloc_time = self.part_realloc_curr.pop(part_id)
            realloc_num = self.part_realloc_num.pop(part_id)
            self.dist_per_part_realloc[part_id].add(
                elim_nume_error(realloc_time/T_hp)
            )
            self.dist_per_part_realloc_count[part_id].add(realloc_num)
        
        # Add system_realloc_cost to the overall distribution
        
        # Normalize by total_pwr * T_hp to get ratios per period
        assert self.total_pwr > 0 and T_hp > 0
        denom = self.total_pwr * T_hp 
        
        self.dist_overall_realloc.add(elim_nume_error(self.system_realloc_cost/denom))
        self.dist_overall_idle.add(elim_nume_error(self.hp_idle_curr / denom))
        self.dist_overall_miss.add(elim_nume_error(self.hp_miss_curr / denom))
        self.dist_overall_used.add(elim_nume_error(self.hp_used_curr / denom))
        self.dist_overall_miss_count.add(self.hp_miss_count)
            
        # Reset accumulators for next period
        self.system_realloc_cost = 0.0
        self.hp_idle_curr = 0.0
        self.hp_miss_curr = 0.0
        self.hp_used_curr = 0.0
        self.hp_miss_count = 0

        # --- Motiv-Exp-(3) Adaptive Binning ---
        if self.motiv3_en:
            total_load = self.hp_total_load_curr
            worst_e2e = self.hp_worst_e2e_curr
            
            # We only add Motiv-Exp-(3) stats if a valid E2E latency was recorded in this period
            if worst_e2e > float('-inf'):
                if self.motiv3_mode == 'raw':
                    # Save raw (load, worst_e2e) pair for this period
                    if total_load is not None and total_load >= 0:
                        self.raw_load_latency.append((float(total_load), float(worst_e2e)))
                else:
                    # 1. Always learn the load distribution
                    if total_load is not None and total_load >= 0:
                        self.dist_hp_total_load.add(total_load)
                    # 2. Handle data based on whether bins are finalized
                    if self.adaptive_load_bins is None: # Learning phase
                        self.hp_temp_records_for_binning.append((total_load, worst_e2e))
                        # Check if learning period is over
                        if len(self.hp_temp_records_for_binning) >= self.binning_warmup_period:
                            self._finalize_adaptive_bins()
                    else: # Bins are finalized, do online binning
                        self._add_to_adaptive_bin(total_load, worst_e2e)

        # reset accumulators for next period
        self.dist_overall_total_load.add(self.hp_total_load_curr)
        self.hp_total_load_curr = 0.0
        self.hp_worst_e2e_curr = float('-inf')

    # ---------------- internal APIs ----------------
    def _record_period_e2e_candidate(self, latency: float):
        """Update worst (max) E2E latency within current hyperperiod for completed chains."""
        if latency > self.hp_worst_e2e_curr:
            self.hp_worst_e2e_curr = float(latency)

    def _finalize_adaptive_bins(self):
        """Uses the learned load distribution to create adaptive bins and processes the warmup data."""
        if len(self.hp_temp_records_for_binning) < 2:
            # Not enough data to create meaningful bins
            return

        # 1. Define bin boundaries using quantiles from the TDigest distribution.
        # The percentile function expects values in [0, 100].
        quantiles = np.linspace(0, 1, self.num_r2_bins + 1)
        boundaries = [self.dist_hp_total_load.percentile(p * 100) for p in quantiles]

        # Ensure boundaries are unique and sorted
        unique_boundaries = sorted(list(set(boundaries)))
        if len(unique_boundaries) < 2:
            # If all data points are the same, create a single bin around it
            val = unique_boundaries[0]
            unique_boundaries = [val - 0.5, val + 0.5]
        
        self.adaptive_load_bins = unique_boundaries

        # 2. Initialize TDigests for each new bin
        self.latency_dist_per_adaptive_bin = {
            b_start: TDigestStreamingHistogram(delta=self._delta, K=self._K)
            for b_start in self.adaptive_load_bins[:-1]
        }

        # 3. Process the buffered data
        for load, latency in self.hp_temp_records_for_binning:
            self._add_to_adaptive_bin(load, latency)
        
        # 4. Clear the buffer
        self.hp_temp_records_for_binning.clear()

    def _add_to_adaptive_bin(self, load: float, latency: float):
        """Adds a latency value to the correct adaptive bin based on the load."""
        # This function should only be called after bins are finalized.
        assert self.adaptive_load_bins is not None, "Adaptive bins have not been finalized yet."
        if load is None:
            return
        # Find the index of the bin this load falls into
        # bisect_right gives an insertion point, which corresponds to the correct bin
        idx = np.searchsorted(self.adaptive_load_bins, load, side='right') - 1
        
        # Ensure index is within bounds (can happen for loads smaller than the first boundary)
        idx = max(0, idx)

        if idx < len(self.adaptive_load_bins) -1:
            bin_start_key = self.adaptive_load_bins[idx]
            self.latency_dist_per_adaptive_bin[bin_start_key].add(latency)
        else: 
            bin_start_key = self.adaptive_load_bins[-2]
            self.latency_dist_per_adaptive_bin[bin_start_key].add(latency)


    def _get_base_task_name(self, task_name: str) -> str:
        """
        Extracts the base task name, removing hyperperiod numbering.
        E.g., 'task1_0' -> 'task1', 'S1_2' -> 'S1'.
        """
        pattern = r'_[-\d]+$'
        base_name = re.sub(pattern, '', task_name)
        return base_name

    def get_adaptive_load_latency_summary(self, p_list: List[float] = None) -> List[Dict]:
        """
        Processes the adaptive binned data for Motiv-Exp-(3) and returns a summary.
        
        Args:
            p_list: List of percentiles to calculate for latency distribution in each bin.
        
        Returns:
            A list of dictionaries, where each dict represents a load bin and contains stats.
        """
        if self.adaptive_load_bins is None:
            # Bins were never finalized, maybe not enough data
            return []

        if p_list is None:
            p_list = [0.5, 0.9, 0.99, 0.999]
        
        summary_list = []
        for i, bin_start in enumerate(self.adaptive_load_bins[:-1]):
            tdigest = self.latency_dist_per_adaptive_bin[bin_start]
            count = tdigest.total_processed_count
            bin_end = self.adaptive_load_bins[i+1]
            
            # We can still provide stats for empty bins if desired
            # if count > 0:
            percentiles = {f'p{p*100:.1f}': tdigest.percentile(p*100) for p in p_list} if count > 0 else {f'p{p*100:.1f}': float('nan') for p in p_list}
            mean = tdigest.get_mean() if count > 0 else float('nan')

            summary_list.append({
                'load_bin_start': bin_start,
                'load_bin_end': bin_end,
                'sample_count': count,
                'latency_mean': mean,
                **percentiles
            })
        return summary_list

    def get_raw_load_latency(self) -> List[Tuple[float, float]]:
        """Return a copy of raw (load, worst_e2e) samples for Motiv-Exp-(3) when in raw mode."""
        return list(self.raw_load_latency)

    def get_summary(self, num_bins: int = 20, p_list: List[float] = [0.5, 0.9, 0.99, 0.999]) -> Dict:
        """
        Retrieves a statistical summary, including histograms and quantiles of distributions.
        Gets per-chain e2e latency/realloc overhead by filtering out the sink tasks in dist_per_task_ft and dist_per_task_realloc.
        Gets overall e2e distribution by merging the dist of all non-sink tasks (chains).
        
        Args:
            num_bins: Number of bins for the histograms in the summary.
            p_list: List of percentiles (as floats, e.g., 0.5 for 50%) to calculate for each distribution.
        
        Returns:
            A dictionary containing the summarized statistics.
        """
        summary = {
            'per_task_finish_time': {},
            'per_task_realloc_overhead': {},
            'per_chain_execution_time': {},
            'per_chain_e2e_latency': {},
            'overall_e2e_latency': {},

            'per_part_realloc_overhead': {},
            'per_part_realloc_count': {},
            'per_chain_realloc_overhead': {},
            'overall_realloc_overhead': {},
            'overall_idle_time': {},
            'overall_missed_load': {},
            'overall_missed_count': {},
            'overall_total_load': {},
            'overall_idle_ratio': {},
            'overall_miss_ratio': {}
        }
        
        # 1. Per-task finish time distribution and per-chain E2E latency
        all_sink_ft_dists = []
        for task_name, dist in self.dist_per_task_ft.items():
            if 'sink' in task_name.lower(): # Case-insensitive check for 'sink'
                summary['per_chain_e2e_latency'][task_name] = dist.get_summary(num_bins, p_list)
                all_sink_ft_dists.append(dist)
            else:
                summary['per_task_finish_time'][task_name] = dist.get_summary(num_bins, p_list)
        
        # 2. Per-task realloc overhead distribution and per-chain realloc overhead
        for task_name, dist in self.dist_per_task_realloc.items():
            if 'sink' in task_name.lower(): # Case-insensitive check for 'sink'
                summary['per_chain_realloc_overhead'][task_name] = dist.get_summary(num_bins, p_list)
            else:
                summary['per_task_realloc_overhead'][task_name] = dist.get_summary(num_bins, p_list)
        
        # 2.5 Per-chain compute time (for breakdown analysis)
        for task_name, dist in self.dist_per_task_exe.items():
            if 'sink' in task_name.lower():
                summary['per_chain_execution_time'][task_name] = dist.get_summary(num_bins, p_list)
        
        # 3. Per-part realloc overhead distribution
        for part_id, dist in self.dist_per_part_realloc.items():
            summary['per_part_realloc_overhead'][part_id] = dist.get_summary(num_bins, p_list)
        # 3.1 Per-part realloc count distribution
        for part_id, dist in self.dist_per_part_realloc_count.items():
            summary['per_part_realloc_count'][part_id] = dist.get_summary(num_bins, p_list)
        
        # 4. Overall reallocation overhead
        summary['overall_realloc_overhead'] = self.dist_overall_realloc.get_summary(num_bins, p_list)
        summary['overall_idle_time'] = self.dist_overall_idle.get_summary(num_bins, p_list)
        summary['overall_missed_load'] = self.dist_overall_miss.get_summary(num_bins, p_list)
        summary['overall_used_load'] = self.dist_overall_used.get_summary(num_bins, p_list)
        summary['overall_missed_count'] = self.dist_overall_miss_count.get_summary(num_bins, p_list)
        summary['overall_total_load'] = self.dist_overall_total_load.get_summary(num_bins, p_list)
        
        # 5. Overall E2E latency (merging all non-sink task finish time distributions)
        # Only perform reduction if there are distributions to merge
        if all_sink_ft_dists:
            merged_e2e_dist = reduce(
                lambda x, y: x + y, 
                all_sink_ft_dists
            )
            summary['overall_e2e_latency'] = merged_e2e_dist.get_summary(num_bins, p_list)
        else:
            summary['overall_e2e_latency'] = {
                'histogram': [],
                'percentiles': {f'p{p*100:.1f}': float('nan') for p in p_list},
                'total_processed_count': 0
            }


        self.summary = summary
        self.p_list = p_list
        return summary

    def get_utilization_avg_ratio(self) -> Dict:
        """Return mean ratios for utilization-related metrics across periods.
        - idle_mean_ratio: mean of dist_overall_idle 
        - miss_mean_ratio: mean of dist_overall_miss 
        - realloc_mean_ratio: mean of dist_overall_realloc
        - used_mean_ratio: mean of dist_overall_used
        - miss_mean_count: mean number of missed tasks per period
        (already normalized by total_pwr*T_hp)
        """
        return {
            'idle_mean_ratio': float(self.dist_overall_idle.get_mean()),
            'miss_mean_ratio': float(self.dist_overall_miss.get_mean()),
            'realloc_mean_ratio': float(self.dist_overall_realloc.get_mean()),
            'used_mean_ratio': float(self.dist_overall_used.get_mean()),
            'miss_mean_count': float(self.dist_overall_miss_count.get_mean()/self.task_cnt)
        }
    
    def get_realloc_info(self) -> Dict:
        """Return the realloc info.
        """
        return {
            'realloc_mean_ratio': float(self.dist_overall_realloc.get_mean()),
            'realloc_mean_count': sum(dist.get_mean() for dist in self.dist_per_part_realloc_count.values())
        }
    
    # ============ Case-Specific Interfaces ============
    
    def get_motiv_case1_stats(self) -> Dict:
        """[Motiv-Exp-1] 纯静态调度 - 利用率问题
        
        统计指标：
        - idle_mean_ratio: 闲置算力占比
        - miss_mean_ratio: miss任务剩余负载占比  
        - miss_mean_count: miss任务数量
        - realloc_mean_ratio: 切换开销占比（纯静态应为0）
        
        Returns:
            dict: {
                'idle_mean_ratio': float,
                'miss_mean_ratio': float, 
                'miss_mean_count': float,
                'realloc_mean_ratio': float  # 验证静态方法无切换开销
            }
        """
        return self.get_utilization_avg_ratio()
    
    def get_motiv_case2_stats(self) -> Dict:
        """[Motiv-Exp-2] 纯动态调度 - 延迟开销问题
        
        统计1 - 资源利用率分解：
        - idle_mean_ratio, miss_mean_ratio, realloc_mean_ratio
        
        统计2 - 端到端延迟分解（相对于约束）：
        - overall_vs_constraint: 所有链合并的 exec/realloc/wait 占比
        - first_chain_vs_constraint: 第一条链的 exec/realloc/wait 占比
        - miss_mean_count: miss任务数量
        
        Returns:
            dict: {
                'utilization': {...},
                'latency_breakdown': {
                    'overall': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
                    'first_chain': {'exec_ratio', 'realloc_ratio', 'wait_ratio'},
                    'first_chain_name': str
                },
                'miss_mean_count': float
            }
        """
        util = self.get_utilization_avg_ratio()
        breakdown = self.get_latency_breakdown_avg_ratio()
        
        # 获取第一条链的名称和breakdown
        per_chain = breakdown.get('per_chain_vs_constraint', {})
        first_chain_name = list(per_chain.keys())[0] if per_chain else None
        first_chain_breakdown = per_chain.get(first_chain_name, {}) if first_chain_name else {}
        
        return {
            'utilization': {
                'idle_mean_ratio': util['idle_mean_ratio'],
                'miss_mean_ratio': util['miss_mean_ratio'],
                'realloc_mean_ratio': util['realloc_mean_ratio']
            },
            'latency_breakdown': {
                'overall': breakdown.get('overall_vs_constraint', {}),
                'first_chain': first_chain_breakdown,
                'first_chain_name': first_chain_name
            },
            'miss_mean_count': util['miss_mean_count']
        }
    
    def get_motiv_case3_stats(self, percentile: float = 0.99, mode: str = None) -> Dict:
        """[Motiv-Exp-3] 切换行为的不确定性
        
        统计指标：
        - spearman_rho: Spearman秩相关系数（单调相关性）
        - binned_summary: 自适应分箱的负载-延迟摘要（仅binned模式）
        - raw_data: 原始数据点列表（仅raw模式）
        
        Args:
            percentile: 用于计算相关性的分位数（默认0.99，即p99）
            mode: 'raw' or 'binned'，不指定则使用self.motiv3_mode
            
        Returns:
            dict: {
                'mode': 'raw' | 'binned',
                'spearman_rho': float,
                'percentile': float,
                'binned_summary': [...] | None,  # binned模式
                'raw_data': [(load, latency), ...] | None  # raw模式
            }
        """
        if mode is None:
            mode = self.motiv3_mode
            
        rho = self.get_spearman_correlation(percentile=percentile)
        
        result = {
            'mode': mode,
            'spearman_rho': rho,
            'percentile': percentile
        }
        
        if mode == 'binned':
            result['binned_summary'] = self.get_adaptive_load_latency_summary([0.5, 0.9, percentile, 0.999])
            result['raw_data'] = None
        else:  # raw
            result['binned_summary'] = None
            result['raw_data'] = self.get_raw_load_latency()
            
        return result
    
    @staticmethod
    def plot_motiv_case1(data_points: List[Dict], save_path: str, show: bool = False):
        """[Motiv-Exp-1] 绘制利用率-可靠性权衡图（三柱状图+附轴）
        
        绘制不同预留分位数下的idle_ratio、miss_ratio、realloc_ratio对比柱状图。
        主轴：百分比对数轴显示三个ratio
        附轴：线性轴显示miss_count
        
        Args:
            data_points: 数据点列表，每个点为一个dict，包含:
                - 'idle_mean_ratio': float
                - 'miss_mean_ratio': float
                - 'realloc_mean_ratio': float
                - 'miss_mean_count': float
                - 'label': str (用于x轴标签，如'p50', 'p60')
            save_path: 保存路径
            show: 是否显示图形
            
        Example:
            # 扫描多个配置
            results = []
            for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
                collector = run_simulation(exec_t_comp_ratioA=ratio)
                stats = collector.get_motiv_case1_stats()
                results.append({
                    'idle_mean_ratio': stats['idle_mean_ratio'],
                    'miss_mean_ratio': stats['miss_mean_ratio'],
                    'realloc_mean_ratio': stats['realloc_mean_ratio'],
                    'miss_mean_count': stats['miss_mean_count'],
                    'label': f'p{int(ratio*100)}'
                })
            collector.plot_motiv_case1(data_points=results)
        """
        # ========== 统一颜色方案 ==========
        # Idle:      C7 (灰色) - 空闲算力
        # Miss:      C3 (红色) - 未完成算力（警示色）
        # Realloc:   C1 (橙色) - 重调度开销
        # Effective: C0 (蓝色) - 有效计算
        # 折线:      C4 (紫色) - Miss Rate
        COLOR_IDLE = 'C7'
        COLOR_MISS = 'C3'
        COLOR_REALLOC = 'C1'
        COLOR_EFFECTIVE = 'C0'
        COLOR_LINE = 'C4'

        # 提取数据并按 percentile 排序（确保 p50 < p60 < ... < p99）
        def _extract_percentile(label: str) -> float:
            """从 label 提取 percentile 数值用于排序"""
            import re
            match = re.search(r'p(\d+(?:\.\d+)?)', label)
            return float(match.group(1)) if match else 0

        sorted_points = sorted(data_points, key=lambda p: _extract_percentile(p.get('label', 'p0')))

        labels = [p.get('label', f'config{i}') for i, p in enumerate(sorted_points)]
        idle_ratios = [p['idle_mean_ratio'] for p in sorted_points]
        miss_ratios = [p['miss_mean_ratio'] for p in sorted_points]
        realloc_ratios = [p['realloc_mean_ratio'] for p in sorted_points]
        miss_counts = [p['miss_mean_count'] for p in sorted_points]
        # 计算 effective (有效算力 = 1 - idle - miss - realloc)
        effective_ratios = [max(0, 1 - idle_ratios[i] - miss_ratios[i] - realloc_ratios[i])
                          for i in range(len(labels))]

        # 创建双轴图 - 调高图片以容纳表格
        fig, ax1 = plt.subplots(figsize=(4, 2.2))
        ax2 = ax1.twinx()  # 创建附轴

        x_pos = np.arange(len(labels))
        width = 0.25

        # 主轴：绘制三个ratio的柱状图（对数轴）- 使用统一颜色
        bars1 = ax1.bar(x_pos - width, idle_ratios, width,
                       label='Idle', color=COLOR_IDLE, alpha=0.8)
        bars2 = ax1.bar(x_pos, miss_ratios, width,
                       label='Miss', color=COLOR_MISS, alpha=0.8)
        bars3 = ax1.bar(x_pos + width, realloc_ratios, width,
                       label='Realloc', color=COLOR_REALLOC, alpha=0.8)

        # 附轴：绘制miss_count的线图
        line = ax2.plot(x_pos, miss_counts, 'o-', color=COLOR_LINE, linewidth=1.5,
                       markersize=4, label='Miss Rate', alpha=0.8)

        # 移除柱状图标记（对数轴上位置难以控制）
        # 改为在图下方添加数据表格

        # 在折线图点上方标注数值（提高精度）
        for i, count in enumerate(miss_counts):
            # 动态格式：根据数值大小选择精度
            if count >= 10:
                fmt = f'{count:.1f}'
            elif count >= 1:
                fmt = f'{count:.2f}'
            else:
                fmt = f'{count:.3f}'
            ax2.text(i, count, fmt, ha='center', va='bottom',
                    fontsize=6, color=COLOR_LINE, fontweight='bold')

        # 设置主轴（对数轴）
        ax1.set_xlabel('Reservation Percentile', fontsize=9)
        ax1.set_ylabel('Ops Ratio (Log)', fontsize=9, color='black')
        ax1.set_yscale('log')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, fontsize=7)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax1.tick_params(axis='y', labelcolor='black', labelsize=7)

        # 增加对数轴刻度密度
        from matplotlib.ticker import LogLocator
        ax1.yaxis.set_major_locator(LogLocator(numticks=10))
        ax1.yaxis.set_minor_locator(LogLocator(subs='auto', numticks=20))

        # 设置附轴（线性轴）
        ax2.set_ylabel('Miss Rate', fontsize=9, color=COLOR_LINE)
        ax2.tick_params(axis='y', labelcolor=COLOR_LINE, labelsize=7)

        # 移除参考线

        # 合并图例：柱状图 + 折线图
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_handles = lines1 + lines2
        all_labels = labels1 + labels2
        ax1.legend(all_handles, all_labels, loc='upper left', fontsize=6, framealpha=0.7)

        ax1.set_title('Cyc.: Utilization-Reliability Tradeoff', fontsize=9, pad=8)

        # 调整布局，为底部表格留出空间
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # 为表格留出空间（调小以显示横轴）

        # 添加底部数据表格（分两栏显示，每栏：Percentile | Idle | Miss）
        # 不包含 Realloc 列（值都是0），不包含中间空列
        def _fmt_ratio(val):
            if val < 0.001:
                return f'{val:.1e}'
            elif val < 0.01:
                return f'{val:.3f}'
            else:
                return f'{val:.2f}'

        # 将数据分成两栏
        n = len(labels)
        half = (n + 1) // 2  # 向上取整

        # 构造两栏数据（不包含 Realloc，不包含空列）
        table_data = []
        for i in range(half):
            row = [
                labels[i],
                _fmt_ratio(idle_ratios[i]),
                _fmt_ratio(miss_ratios[i]),
            ]
            # 第二栏数据（如果存在）
            if i + half < n:
                row.extend([
                    labels[i + half],
                    _fmt_ratio(idle_ratios[i + half]),
                    _fmt_ratio(miss_ratios[i + half]),
                ])
            else:
                row.extend(['', '', ''])
            table_data.append(row)

        # 创建表格（两栏，共6列：第一栏3列 + 第二栏3列，无分隔列）
        # 向下移动表格，增加行高
        col_labels = ['', 'Idle', 'Miss', '', 'Idle', 'Miss']
        col_widths = [0.10, 0.12, 0.12, 0.10, 0.12, 0.12]
        table = plt.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='bottom',
            cellLoc='center',
            colWidths=col_widths,
            bbox=[0.08, -0.65, 0.84, 0.28]  # [left, bottom, width, height] - 向上移动表格
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.scale(1, 1.3)  # 增加行高

        # 设置表头样式
        for j in range(len(col_labels)):
            cell = table[(0, j)]
            if col_labels[j]:  # 非空表头
                cell.set_facecolor('#e8e8e8')
                cell.set_text_props(fontweight='bold')
            else:
                cell.set_facecolor('white')  # 分隔列为白色
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Case 1 权衡图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)
    
    @staticmethod
    def _cluster_case2_data(data_points: List[Dict], group_order: List = None):
        """辅助函数：将 Case 2 数据按 (tiles, load_factor) 分簇，簇内按 chains 分组
        
        按 group_order 指定的顺序组织数据。
        
        Args:
            data_points: 包含 tiles, load_factor, chains 等字段的数据点列表
            group_order: [cluster_keys_list, chains_list]，指定簇和簇内的顺序
                        例如：[[(400, 0.5), (400, 1.0), (200, 0.5)], [1, 4, 9]]
        
        Returns:
            clusters: OrderedDict {(tiles, load): OrderedDict{chains: data_point}}
        """
        from collections import OrderedDict
        
        # 先将数据点按 (tiles, load, chains) 索引化
        data_index = {}
        for p in data_points:
            tiles = p.get('tiles', 0)
            load_factor = p.get('load_factor', 0)
            chains = p.get('chains', 0)
            key = (tiles, load_factor, chains)
            data_index[key] = p
            
        cluster_keys_order = group_order[0]
        chains_order = group_order[1]
        
        # 按指定顺序重建 clusters
        clusters = OrderedDict()
        for cluster_key in cluster_keys_order:
            clusters[cluster_key] = OrderedDict()
            for chains in chains_order:
                full_key = (cluster_key[0], cluster_key[1], chains)
                if full_key in data_index:
                    clusters[cluster_key][chains] = data_index[full_key]
        
        return clusters
    
    @staticmethod
    def plot_motiv_case2(data_points: List[Dict], plot_type: str, 
                        save_path: str, show: bool = False, group_order: List = None):
        """[Motiv-Exp-2] 绘制可扩展性分析图
        
        支持两种图表类型：
        1. 'breakdown': 延迟分解的堆叠柱状图（exec/realloc/wait占比）
        2. 'utilization': 资源利用率的堆叠柱状图（effective/idle/miss/realloc）
        
        Args:
            data_points: 数据点列表，每个点为一个dict，包含:
                - 'tiles': int (硬件tile数)
                - 'load_factor': float (负载倍数)
                - 'chains': int (任务链数)
                - 'label': str (x轴标签，例如'300T-4C')
                - 'utilization': dict (需要包含idle/miss/realloc_mean_ratio)
                - 'latency_breakdown': dict (需要包含overall的exec/realloc/wait_ratio)
                - 'miss_mean_count': float (miss任务数量)
            plot_type: 'breakdown' 或 'utilization'
            save_path: 保存路径
            show: 是否显示图形
            
        Example:
            # 扫描多个配置
            results = []
            tiles_loads = [(400, 1.0), (400, 0.5), (200, 1.0)]
            chains_list = [1, 4, 9]
            for tiles, load in tiles_loads:
                for chains in chains_list:
                    collector = run_simulation(num_tiles=tiles, load_factor=load, num_chains=chains)
                    stats = collector.get_motiv_case2_stats()
                    results.append({
                        'tiles': tiles,
                        'load_factor': load,
                        'chains': chains,
                        'label': f'{tiles}T-{load:.1f}×-{chains}C',
                        **stats
                    })
            StatisticsCollector.plot_motiv_case2(data_points=results, plot_type='breakdown')
        """
        import matplotlib.pyplot as plt
        
        # 根据图表类型设置尺寸
        if plot_type == 'breakdown':
            fig, ax = plt.subplots(figsize=(4, 2.2))
        else:  # utilization
            fig, ax = plt.subplots(figsize=(4, 2.2))
        
        if plot_type == 'breakdown':
            # 延迟分解堆叠柱状图 + miss num ratio 曲线（附轴）
            # ========== 统一颜色方案 ==========
            COLOR_WAITING = 'C2'    # 绿色 - 等待时间
            COLOR_REALLOC = 'C1'    # 橙色 - 重调度/调度开销
            COLOR_EXECUTION = 'C0'  # 蓝色 - 执行时间
            COLOR_LINE = 'C4'       # 紫色 - Miss Rate 折线

            # 创建附轴
            ax2 = ax.twinx()

            # 使用分簇逻辑
            clusters = StatisticsCollector._cluster_case2_data(data_points, group_order)

            # 从排序好的 clusters 中提取 labels 和 chains
            cluster_labels = [f'{t}T-{l:.1f}×' for t, l in clusters.keys()]
            chain_values = list(next(iter(clusters.values())).keys()) if clusters else []

            num_clusters = len(clusters)
            bars_per_cluster = max(len(v) for v in clusters.values()) if clusters else 0

            x_pos = np.arange(num_clusters)
            total_width = 0.8
            width = total_width / bars_per_cluster

            # 绘制堆叠柱状图（按 chain 分组）
            # 堆叠顺序：Waiting (底部) → Realloc (中间) → Execution (顶部)
            hatch_patterns = ['', '///', 'xxx', '\\\\\\']
            for i_cluster, (cluster_key, cluster_data) in enumerate(clusters.items()):

                x_cluster = []
                y_cluster = []

                for i_bar, chains in enumerate(cluster_data.keys()):
                    p = cluster_data[chains]
                    x_offset = x_pos[i_cluster] + (i_bar - (bars_per_cluster - 1) / 2) * width
                    breakdown = p.get('latency_breakdown', {}).get('overall', {})
                    miss_num = p.get('miss_mean_count', 0)
                    temp = {
                        'exec': breakdown.get('exec_ratio', 0),
                        'realloc': breakdown.get('realloc_ratio', 0),
                        'wait': breakdown.get('wait_ratio', 0),
                        'miss_num': miss_num,
                        'hatch': hatch_patterns[i_bar % len(hatch_patterns)]
                    }
                    bot = 0
                    # Execution (底部)
                    ax.bar(x_offset, temp['exec'], width=width, bottom=bot,
                        label='Execution' if i_cluster == 0 and i_bar == 0 else '',
                        color=COLOR_EXECUTION, alpha=0.8, hatch=temp['hatch'])
                    bot += temp['exec']
                    # Realloc (中间)
                    ax.bar(x_offset, temp['realloc'], width=width, bottom=bot,
                        label='Scheduling' if i_cluster == 0 and i_bar == 0 else '',
                        color=COLOR_REALLOC, alpha=0.85, hatch=temp['hatch'])
                    bot += temp['realloc']
                    # Waiting (顶部)
                    ax.bar(x_offset, temp['wait'], width=width, bottom=bot,
                        label='Waiting' if i_cluster == 0 and i_bar == 0 else '',
                        color=COLOR_WAITING, alpha=0.8, hatch=temp['hatch'])
                    bot += temp['wait']

                    if not np.isnan(temp['miss_num']):
                        x_cluster.append(x_offset)
                        y_cluster.append(temp['miss_num'])
                # 绘制 miss num ratio 曲线（簇内连接）
                if len(x_cluster) >= 2:
                    ax2.plot(x_cluster, y_cluster, 'o-', color=COLOR_LINE, linewidth=2,
                            markersize=7, alpha=0.8)
                elif len(x_cluster) == 1:
                    ax2.plot(x_cluster, y_cluster, 'o', color=COLOR_LINE,
                            markersize=7, alpha=0.8)

            ax2.plot([], [], 'o-', color=COLOR_LINE, linewidth=1.5, markersize=4,
                    label='Miss Rate', alpha=0.8)

            # 主轴设置
            ax.set_ylabel(r'Lat. Ratio w.r.t. $\mathcal{D}_{\mathrm{e2e}}$', fontsize=9, color='black')
            ax.set_title('Tp-driven: Latency Breakdown vs Scale', fontsize=9, pad=8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cluster_labels, rotation=0, fontsize=7)
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5)
            ax.tick_params(axis='y', labelcolor='black', labelsize=7)

            # 附轴设置
            ax2.set_ylabel('Miss Rate', fontsize=9, color=COLOR_LINE)
            ax2.tick_params(axis='y', labelcolor=COLOR_LINE, labelsize=7)
            ax.set_xlabel('Scale Configurations', fontsize=9)

            # 图例：两列布局
            # 第一列：Waiting, Scheduling, Execution（与堆叠顺序一致）
            # 第二列：1 chains, 4 chains, 9 chains
            from matplotlib.patches import Rectangle

            handles_col1 = []
            labels_col1 = ['Waiting', 'Scheduling', 'Execution']
            colors_col1 = [COLOR_WAITING, COLOR_REALLOC, COLOR_EXECUTION]
            for lbl, clr in zip(labels_col1, colors_col1):
                handles_col1.append(Rectangle((0, 0), 1, 1, facecolor=clr, edgecolor='black', alpha=0.8))

            handles_col2 = []
            labels_col2 = [f'{ch} chains' for ch in chain_values]
            for i, ch in enumerate(chain_values):
                hatch_pattern = hatch_patterns[i % len(hatch_patterns)]
                handles_col2.append(Rectangle((0, 0), 1, 1, facecolor='lightgray',
                                             edgecolor='black', hatch=hatch_pattern, alpha=0.8))

            # 合并两列图例
            all_handles = handles_col1 + handles_col2
            all_labels = labels_col1 + labels_col2
            ax.legend(all_handles, all_labels,
                     loc='upper left', fontsize=6, framealpha=0.7, ncol=2)
            
        elif plot_type == 'utilization':
            # 分簇柱状图 + miss ops ratio 曲线（附轴）
            # 簇：(tiles, load_factor)；簇内：chains

            # ========== 统一颜色方案（与 Case 1 一致）==========
            # Idle:      C7 (灰色) - 空闲算力
            # Miss:      C3 (红色) - 未完成算力（警示色）
            # Realloc:   C1 (橙色) - 重调度开销
            # Effective: C0 (蓝色) - 有效计算
            # 折线:      C3 (红色) - Ops Ratio (Miss)
            COLOR_IDLE = 'C7'
            COLOR_MISS = 'C3'
            COLOR_REALLOC = 'C1'
            COLOR_EFFECTIVE = 'C0'
            COLOR_LINE = 'C3'

            # 1. 使用分簇逻辑
            clusters = StatisticsCollector._cluster_case2_data(data_points, group_order)

            # 从排序好的 clusters 中提取 labels 和 chains
            cluster_labels = [f'{t}T-{l:.1f}×' for t, l in clusters.keys()]
            chain_values = list(next(iter(clusters.values())).keys()) if clusters else []

            # 2. 计算位置
            num_clusters = len(clusters)
            bars_per_cluster = max(len(v) for v in clusters.values()) if clusters else 0

            x_pos = np.arange(num_clusters)
            total_width = 0.8
            width = total_width / bars_per_cluster

            # 3. 创建附轴用于 miss op ratio
            ax2 = ax.twinx()

            # 为附轴腾出空间
            fig.subplots_adjust(right=0.88)

            # 4. 绘制堆叠柱状图（使用统一颜色）
            hatch_patterns = ['', '///', 'xxx', '\\\\\\']
            for i_cluster, (cluster_key, cluster_data) in enumerate(clusters.items()):

                x_cluster_line = []
                y_cluster_line = []

                for i_bar, chains in enumerate(cluster_data.keys()):
                    p = cluster_data[chains]
                    x_offset = x_pos[i_cluster] + (i_bar - (bars_per_cluster - 1) / 2) * width

                    util = p.get('utilization', {})
                    idle = util.get('idle_mean_ratio', 0.0)
                    realloc = util.get('realloc_mean_ratio', 0.0)
                    miss_ops = util.get('miss_mean_ratio', 0.0)
                    effective = max(0.0, 1.0 - idle - realloc)
                    hatch = hatch_patterns[i_bar % len(hatch_patterns)]

                    # 绘制柱状图（使用统一颜色）
                    bot = 0
                    ax.bar(x_offset, realloc, width=width, bottom=bot,
                           label='Realloc' if i_cluster == 0 and i_bar == 0 else '',
                           color=COLOR_REALLOC, alpha=0.85, hatch=hatch)
                    bot += realloc
                    ax.bar(x_offset, effective, width=width, bottom=bot,
                           label='Effective' if i_cluster == 0 and i_bar == 0 else '',
                           color=COLOR_EFFECTIVE, alpha=0.8, hatch=hatch)
                    bot += effective
                    ax.bar(x_offset, idle, width=width, bottom=bot,
                           label='Idle' if i_cluster == 0 and i_bar == 0 else '',
                           color=COLOR_IDLE, alpha=0.7, hatch=hatch)

                    # 收集曲线数据
                    if not np.isnan(miss_ops):
                        x_cluster_line.append(x_offset)
                        y_cluster_line.append(miss_ops)

                # 5. 绘制 miss ops ratio 曲线在附轴上（簇内连接）
                if len(x_cluster_line) >= 2:
                    ax2.plot(x_cluster_line, y_cluster_line, 'o-', color=COLOR_LINE, linewidth=2,
                             markersize=7, alpha=0.8)
                elif len(x_cluster_line) == 1:
                    ax2.plot(x_cluster_line, y_cluster_line, 'o', color=COLOR_LINE,
                             markersize=7, alpha=0.8)

            ax2.plot([], [], 'o-', color=COLOR_LINE, linewidth=2, markersize=7,
                     label='Ops Ratio (Miss)', alpha=0.8)

            # 6. 主轴设置
            ax.set_xlabel('Scale Configurations', fontsize=9)
            ax.set_ylabel('Tile Util.', fontsize=9, color='black')
            ax.set_title('Tp.-driven: Resource Utilization vs Scale', fontsize=9, pad=8)
            ax.set_ylim([0, 1.1])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cluster_labels, rotation=0, fontsize=7) #  ha='right',
            ax.tick_params(axis='y', labelcolor='black', labelsize=7)

            # 7. 附轴 ax2 设置（使用统一颜色）
            ax2.set_ylabel('Ops Ratio (Miss)', fontsize=9, color=COLOR_LINE)
            ax2.tick_params(axis='y', labelcolor=COLOR_LINE, labelsize=7)

            # 8. 图例：两列布局
            # 第一列：Realloc, Idle, Effective
            # 第二列：1 chains, 4 chains, 9 chains
            from matplotlib.patches import Rectangle

            handles_col1 = []
            labels_col1 = ['Realloc', 'Idle', 'Effective']
            colors_col1 = [COLOR_REALLOC, COLOR_IDLE, COLOR_EFFECTIVE]
            for lbl, clr in zip(labels_col1, colors_col1):
                handles_col1.append(Rectangle((0, 0), 1, 1, facecolor=clr, edgecolor='black', alpha=0.8))

            handles_col2 = []
            labels_col2 = [f'{ch} chains' for ch in chain_values]
            for i, ch in enumerate(chain_values):
                hatch_pattern = hatch_patterns[i % len(hatch_patterns)]
                handles_col2.append(Rectangle((0, 0), 1, 1, facecolor='lightgray',
                                             edgecolor='black', hatch=hatch_pattern, alpha=0.8))

            # 合并两列图例
            all_handles = handles_col1 + handles_col2
            all_labels = labels_col1 + labels_col2
            ax.legend(all_handles, all_labels,
                     loc='upper left', fontsize=6, framealpha=0.7, ncol=2)
        
        else:
            raise NotImplementedError(f"plot_type '{plot_type}' is not supported for case 2.")
        
        # xlabel、xticks、图例都已在各分支内设置
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        if plot_type != 'utilization':
            fig.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Case 2 {plot_type}图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        plt.close(fig)

    @staticmethod
    def plot_motiv_legend(save_path: str = None, show: bool = False):
        """生成 Fig 6 的统一图例图片（三列布局）

        三列显示：
        - Workload: 1 chain, 4 chains, 9 chains (hatch patterns)
        - 计算量 (Ops Ratio): Idle, Miss, Realloc, Effective
        - 延迟 (Latency): Execution, Scheduling, Waiting, Miss Rate (line)

        Args:
            save_path: 保存路径
            show: 是否显示图形
        """
        # ========== 统一颜色方案 ==========
        COLOR_IDLE = 'C7'
        COLOR_MISS = 'C3'
        COLOR_REALLOC = 'C1'
        COLOR_EFFECTIVE = 'C0'
        COLOR_EXECUTION = 'C0'
        COLOR_WAITING = 'C5'
        COLOR_LINE = 'C4'

        # Hatch patterns for chains
        hatch_patterns = ['', '///', 'xxx']

        # 创建竖向图例（适合放在三图左侧）
        fig, ax = plt.subplots(figsize=(1.8, 2.8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        from matplotlib.patches import Rectangle, Patch
        from matplotlib.lines import Line2D

        # ========== 列1: Workload (chains hatch patterns) ==========
        col1_x = 0.5
        y_start = 9.0
        y_step = 1.2

        # 列标题
        ax.text(col1_x + 0.3, y_start + 0.3, 'Workload', fontsize=8, va='bottom', fontweight='bold')

        # 1 chain, 4 chains, 9 chains
        chain_labels = ['1 chain', '4 chains', '9 chains']
        for i, (label, hatch) in enumerate(zip(chain_labels, hatch_patterns)):
            y = y_start - (i + 1) * y_step
            rect = Rectangle((col1_x, y - 0.3), 0.8, 0.6,
                            facecolor='lightgray', edgecolor='black',
                            hatch=hatch, alpha=0.8)
            ax.add_patch(rect)
            ax.text(col1_x + 1.0, y, label, fontsize=7, va='center')

        # ========== 列2: 计算量 (Ops Ratio) ==========
        col2_x = 3.5
        y_start = 9.0

        # 列标题
        ax.text(col2_x + 0.3, y_start + 0.3, 'Ops Ratio', fontsize=8, va='bottom', fontweight='bold')

        ops_items = [
            ('Idle', COLOR_IDLE),
            ('Miss', COLOR_MISS),
            ('Realloc', COLOR_REALLOC),
            ('Effective', COLOR_EFFECTIVE),
        ]
        for i, (label, color) in enumerate(ops_items):
            y = y_start - (i + 1) * y_step
            rect = Rectangle((col2_x, y - 0.3), 0.8, 0.6,
                            facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            ax.text(col2_x + 1.0, y, label, fontsize=7, va='center')

        # ========== 列3: 延迟 (Latency) ==========
        col3_x = 6.5
        y_start = 9.0

        # 列标题
        ax.text(col3_x + 0.3, y_start + 0.3, 'Latency', fontsize=8, va='bottom', fontweight='bold')

        lat_items = [
            ('Execution', COLOR_EXECUTION, 'rect'),
            ('Scheduling', COLOR_REALLOC, 'rect'),
            ('Waiting', COLOR_WAITING, 'rect'),
            ('Miss Rate', COLOR_LINE, 'line'),
        ]
        for i, (label, color, style) in enumerate(lat_items):
            y = y_start - (i + 1) * y_step
            if style == 'rect':
                rect = Rectangle((col3_x, y - 0.3), 0.8, 0.6,
                                facecolor=color, edgecolor='black', alpha=0.8)
                ax.add_patch(rect)
            else:  # line
                line = Line2D([col3_x, col3_x + 0.8], [y, y],
                             color=color, linewidth=2, marker='o', markersize=5)
                ax.add_line(line)
            ax.text(col3_x + 1.0, y, label, fontsize=7, va='center')

        # 调整布局
        plt.tight_layout(pad=0.5)

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
            print(f"图例图片已保存到: {save_path}")

        if show:
            plt.show()

        plt.close(fig)

        return fig


    # ============ Case-Specific Formatted Output ============
    
    def format_motiv_case1_output(self, stats: Dict = None) -> str:
        """[Motiv-Exp-1] 格式化输出
        
        Args:
            stats: get_motiv_case1_stats()的返回值，不提供则自动调用
        """
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
        # 优先使用 used_mean_ratio；若不存在则回退为 1-idle-miss-realloc
        used = stats.get('used_mean_ratio', None)
        if used is None:
            used = 1.0 - stats['idle_mean_ratio'] - stats['miss_mean_ratio'] - stats['realloc_mean_ratio']
        lines.append(f"有效利用率(used_mean_ratio):           {used:.4f}")
        # 一致性校验：realloc+idle+used 是否约等于 1
        trio_sum = stats['realloc_mean_ratio'] + stats['idle_mean_ratio'] + used
        lines.append(f"一致性校验: realloc+idle+used =        {trio_sum:.4f}  {'(≈1 ✅)' if abs(trio_sum-1.0) < 0.02 else '(偏离1 ⚠)'}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def format_motiv_case2_output(self, stats: Dict = None) -> str:
        """[Motiv-Exp-2] 格式化输出
        
        Args:
            stats: get_motiv_case2_stats()的返回值，不提供则自动调用
        """
        if stats is None:
            stats = self.get_motiv_case2_stats()
        
        lines = []
        lines.append("=" * 60)
        lines.append("Motiv-Exp-2: 纯动态调度 - 延迟开销问题")
        lines.append("=" * 60)
        
        # 统计1: 资源利用率分解
        util = stats['utilization']
        lines.append("统计1 - 资源利用率分解:")
        lines.append(f"  闲置算力占比 (idle):     {util['idle_mean_ratio']:.4f}")
        lines.append(f"  Miss负载占比 (miss):     {util['miss_mean_ratio']:.4f}")
        lines.append(f"  切换开销占比 (realloc):  {util['realloc_mean_ratio']:.4f}")
        effective_util = 1.0 - util['idle_mean_ratio'] - util['miss_mean_ratio'] - util['realloc_mean_ratio']
        lines.append(f"  有效利用率:              {effective_util:.4f}")
        lines.append("")
        
        # 统计2: 端到端延迟分解
        breakdown = stats['latency_breakdown']
        lines.append("统计2 - 端到端延迟分解 (相对于约束):")
        lines.append("  Overall (所有链合并):")
        overall = breakdown['overall']
        if overall:
            lines.append(f"    执行时间占比 (exec):    {overall.get('exec_ratio', 0):.4f}")
            lines.append(f"    调度开销占比 (realloc): {overall.get('realloc_ratio', 0):.4f}")
            lines.append(f"    等待时间占比 (wait):    {overall.get('wait_ratio', 0):.4f}")
            total_ratio = overall.get('exec_ratio', 0) + overall.get('realloc_ratio', 0) + overall.get('wait_ratio', 0)
            lines.append(f"    总和:                   {total_ratio:.4f} {'(>1表示超时)' if total_ratio > 1 else ''}")
        else:
            lines.append("    (无数据)")
        
        lines.append("")
        first_chain_name = breakdown.get('first_chain_name')
        if first_chain_name:
            lines.append(f"  First Chain ({first_chain_name}):")
            first_chain = breakdown['first_chain']
            if first_chain:
                lines.append(f"    执行时间占比 (exec):    {first_chain.get('exec_ratio', 0):.4f}")
                lines.append(f"    调度开销占比 (realloc): {first_chain.get('realloc_ratio', 0):.4f}")
                lines.append(f"    等待时间占比 (wait):    {first_chain.get('wait_ratio', 0):.4f}")
                total_ratio = first_chain.get('exec_ratio', 0) + first_chain.get('realloc_ratio', 0) + first_chain.get('wait_ratio', 0)
                lines.append(f"    总和:                   {total_ratio:.4f} {'(>1表示超时)' if total_ratio > 1 else ''}")
            else:
                lines.append("    (无数据)")
        
        lines.append("")
        lines.append(f"Miss任务数量 (miss_mean_count): {stats['miss_mean_count']:.2f}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def format_motiv_case3_output(self, stats: Dict = None, percentile: float = 0.99) -> str:
        """[Motiv-Exp-3] 格式化输出
        
        Args:
            stats: get_motiv_case3_stats()的返回值，不提供则自动调用
            percentile: 分位数（用于自动调用时）
        """
        if stats is None:
            stats = self.get_motiv_case3_stats(percentile=percentile)
        
        lines = []
        lines.append("=" * 60)
        lines.append("Motiv-Exp-3: 切换行为的不确定性")
        lines.append("=" * 60)
        lines.append(f"数据模式: {stats['mode']}")
        lines.append(f"Spearman相关系数 (ρ): {stats['spearman_rho']:.4f}")
        lines.append(f"使用分位数: p{int(stats['percentile']*100)}")
        lines.append("")
        
        if stats['mode'] == 'raw':
            lines.append(f"原始数据点数量: {stats['raw_data_count']}")
        else:
            binned_summary = stats.get('binned_summary', [])
            if binned_summary:
                lines.append(f"自适应分箱数量: {len(binned_summary)}")
                lines.append("")
                lines.append("负载分箱摘要 (前5个和后5个):")
                lines.append(f"{'Load Range':^20} | {'Count':>7} | {'p50':>9} | {'p90':>9} | {'p99':>9}")
                lines.append("-" * 70)
                
                # 显示前5个
                for i, record in enumerate(binned_summary[:5]):
                    load_range = f"[{record['load_bin_start']:.1f}, {record['load_bin_end']:.1f})"
                    count = record['sample_count']
                    p50 = record.get('p50.0', float('nan'))
                    p90 = record.get('p90.0', float('nan'))
                    p99 = record.get('p99.0', float('nan'))
                    lines.append(f"{load_range:^20} | {count:7d} | {p50:9.2f} | {p90:9.2f} | {p99:9.2f}")
                
                if len(binned_summary) > 10:
                    lines.append(f"{'...':^20} | {'...':>7} | {'...':>9} | {'...':>9} | {'...':>9}")
                    
                # 显示后5个
                for record in binned_summary[-5:]:
                    load_range = f"[{record['load_bin_start']:.1f}, {record['load_bin_end']:.1f})"
                    count = record['sample_count']
                    p50 = record.get('p50.0', float('nan'))
                    p90 = record.get('p90.0', float('nan'))
                    p99 = record.get('p99.0', float('nan'))
                    lines.append(f"{load_range:^20} | {count:7d} | {p50:9.2f} | {p90:9.2f} | {p99:9.2f}")
            else:
                lines.append("(无分箱数据)")
        
        lines.append("=" * 60)
        lines.append("提示: 使用 plot_motiv_case3() 绘制负载-延迟关系图")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def export_motiv_case_results(self, case: int, save_path: str = None, verbose: bool = True, **kwargs):
        """统一导出指定Motiv-Exp的结果
        
        Args:
            case: 1, 2, or 3，对应三个实验
            save_path: 保存路径，不提供则使用默认路径
            verbose: 是否打印到控制台
            **kwargs: 传递给特定case的额外参数
                - case 3: percentile, fit, iqr_band等绘图参数
        """
        if case == 1:
            stats = self.get_motiv_case1_stats()
            output = self.format_motiv_case1_output(stats)
            default_path = self.path.get('stat', './motiv_case1_output.txt')
            
        elif case == 2:
            stats = self.get_motiv_case2_stats()
            output = self.format_motiv_case2_output(stats)
            default_path = self.path.get('stat', './motiv_case2_output.txt')
            
        elif case == 3:
            percentile = kwargs.get('percentile', 0.99)
            stats = self.get_motiv_case3_stats(percentile=percentile)
            output = self.format_motiv_case3_output(stats, percentile=percentile)
            default_path = self.path.get('stat', './motiv_case3_output.txt')
            
            # 同时生成图表
            fit = kwargs.get('fit', 'none')
            iqr_band = kwargs.get('iqr_band', (0.25, 0.75))
            show = kwargs.get('show', False)
            self.plot_motiv_case3(percentile=percentile, iqr_band=iqr_band, 
                                 fit=fit, show=show)
        else:
            raise ValueError(f"Invalid case number: {case}. Must be 1, 2, or 3.")
        
        # 打印到控制台
        if verbose:
            print(output)
        
        # 保存到文件
        if save_path is None:
            save_path = default_path
            
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(output)
                if verbose:
                    print(f"\n结果已保存到: {save_path}")
            except Exception as e:
                print(f"保存文件失败: {e}")
        
        return stats
    
    def print_motiv_case_summary(self, case: int, **kwargs):
        """快捷打印指定case的统计摘要（不保存文件）
        
        Args:
            case: 1, 2, or 3
            **kwargs: 传递给格式化函数的参数
        """
        if case == 1:
            stats = self.get_motiv_case1_stats()
            print(self.format_motiv_case1_output(stats))
        elif case == 2:
            stats = self.get_motiv_case2_stats()
            print(self.format_motiv_case2_output(stats))
        elif case == 3:
            percentile = kwargs.get('percentile', 0.99)
            stats = self.get_motiv_case3_stats(percentile=percentile)
            print(self.format_motiv_case3_output(stats, percentile=percentile))
        else:
            raise ValueError(f"Invalid case number: {case}. Must be 1, 2, or 3.")

    def get_latency_breakdown_avg_ratio(self) -> Dict:
        """Compute average ratios of execution, realloc(scheduling), and waiting.
        提供两套口径：
          - vs_e2e: mean(component)/mean(e2e)（兼容老口径）
          - vs_constraint: mean(component)/constraint（统一衡量标准，优先用于比较）
        等待占比作为补项：max(0, 1 - exec - realloc)。
        TDigest.get_mean() returns 0 for empty distributions, never NaN.
        """
        per_chain_vs_e2e = {}
        per_chain_vs_constraint = {}
        sink_names = [name for name in self.dist_per_task_ft.keys() if 'sink' in name.lower()]

        # Per-chain
        for name in sink_names:
            # Get means (returns 0 if empty, never NaN)
            e2e_mean = self.dist_per_task_ft[name].get_mean()
            exec_mean = self.dist_per_task_exe[name].get_mean() if name in self.dist_per_task_exe else 0.0
            realloc_mean = self.dist_per_task_realloc[name].get_mean() if name in self.dist_per_task_realloc else 0.0
            
            # vs_e2e (only compute if e2e_mean > 0 to avoid division by zero)
            if e2e_mean > 0:
                exec_ratio = float(exec_mean) / float(e2e_mean)
                realloc_ratio = float(realloc_mean) / float(e2e_mean)
                wait_ratio = max(0.0, 1.0 - exec_ratio - realloc_ratio)
                per_chain_vs_e2e[name] = {'exec_ratio': exec_ratio, 'realloc_ratio': realloc_ratio, 'wait_ratio': wait_ratio}
            
            # vs_constraint：基于"均值-再归一"的口径，先在绝对时间域取均值，再归一到约束
            if name in self.chain_e2e_constraint and self.chain_e2e_constraint[name] > 0:
                cons = float(self.chain_e2e_constraint[name])
                exec_ratio_c = float(exec_mean) / cons
                realloc_ratio_c = float(realloc_mean) / cons
                wait_ratio_c = max(0.0, float(e2e_mean) / cons - exec_ratio_c - realloc_ratio_c)
                per_chain_vs_constraint[name] = {'exec_ratio': exec_ratio_c, 'realloc_ratio': realloc_ratio_c, 'wait_ratio': wait_ratio_c}

        # Overall (merged)
        overall_vs_e2e = {'exec_ratio': 0.0, 'realloc_ratio': 0.0, 'wait_ratio': 0.0}
        overall_vs_constraint = {'exec_ratio': 0.0, 'realloc_ratio': 0.0, 'wait_ratio': 0.0}
        sink_e2e_dists = [self.dist_per_task_ft[name] for name in sink_names]
        sink_exec_dists = [self.dist_per_task_exe[name] for name in sink_names if name in self.dist_per_task_exe]
        sink_realloc_dists = [self.dist_per_task_realloc[name] for name in sink_names if name in self.dist_per_task_realloc]
        if sink_e2e_dists:
            merged_e2e = reduce(lambda a, b: a + b, sink_e2e_dists)
            e2e_mean_o = merged_e2e.get_mean()
            exec_mean_o = reduce(lambda a, b: a + b, sink_exec_dists).get_mean() if sink_exec_dists else 0.0
            realloc_mean_o = reduce(lambda a, b: a + b, sink_realloc_dists).get_mean() if sink_realloc_dists else 0.0
            
            # vs_e2e
            if e2e_mean_o > 0:
                exec_ratio_o = float(exec_mean_o) / float(e2e_mean_o)
                realloc_ratio_o = float(realloc_mean_o) / float(e2e_mean_o)
                wait_ratio_o = max(0.0, 1.0 - exec_ratio_o - realloc_ratio_o)
                overall_vs_e2e = {'exec_ratio': exec_ratio_o, 'realloc_ratio': realloc_ratio_o, 'wait_ratio': wait_ratio_o}
            
            # vs_constraint (use mean constraint across chains if available)：先取均值再做补项，避免比值均值引入负等待
            cons_vals = [self.chain_e2e_constraint[n] for n in sink_names if n in self.chain_e2e_constraint and self.chain_e2e_constraint[n] > 0]
            if cons_vals:
                cons_o = float(np.mean(cons_vals))
                exec_ratio_oc = float(exec_mean_o) / cons_o
                realloc_ratio_oc = float(realloc_mean_o) / cons_o
                wait_ratio_oc = max(0.0, float(e2e_mean_o) / cons_o - exec_ratio_oc - realloc_ratio_oc)
                overall_vs_constraint = {'exec_ratio': exec_ratio_oc, 'realloc_ratio': realloc_ratio_oc, 'wait_ratio': wait_ratio_oc}

        return {
            'overall_vs_e2e': overall_vs_e2e,
            'per_chain_vs_e2e': per_chain_vs_e2e,
            'overall_vs_constraint': overall_vs_constraint,
            'per_chain_vs_constraint': per_chain_vs_constraint
        }
    
    def get_weighted_pearson_corr_binned(self) -> float:
        """
        Calculates the weighted Pearson correlation coefficient from the adaptively binned data.
        This reflects the linear relationship between load (bin centers) and the mean latency within each bin.

        Returns:
            The weighted correlation coefficient, or NaN if it cannot be computed.
        """
        if self.adaptive_load_bins is None or len(self.adaptive_load_bins) < 2:
            return float('nan')

        x_centers = []
        y_means = []
        weights = []

        # 1. Extract data triples (bin_center, mean_latency, count)
        for i, bin_start in enumerate(self.adaptive_load_bins[:-1]):
            tdigest = self.latency_dist_per_adaptive_bin[bin_start]
            count = tdigest.total_processed_count
            if count > 0:
                bin_end = self.adaptive_load_bins[i + 1]
                x_centers.append((bin_start + bin_end) / 2)
                y_means.append(tdigest.get_mean())
                weights.append(count)

        # 2. Use numpy to calculate the weighted correlation coefficient
        if len(x_centers) < 2:
            return float('nan')

        x = np.array(x_centers)
        y = np.array(y_means)
        w = np.array(weights)

        avg_x = np.average(x, weights=w)
        avg_y = np.average(y, weights=w)

        cov_xy = np.average((x - avg_x) * (y - avg_y), weights=w)
        std_x = np.sqrt(np.average((x - avg_x)**2, weights=w))
        std_y = np.sqrt(np.average((y - avg_y)**2, weights=w))

        if std_x > 0 and std_y > 0:
            correlation = cov_xy / (std_x * std_y)
        else:
            correlation = float('nan')
        
        return correlation

    def _spearman_from_arrays(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman rho from raw arrays (unweighted)."""
        if x.size < 2:
            return float('nan')
        # ranks with average method
        order_x = np.argsort(x, kind='mergesort')
        ranks_x = np.empty_like(x, dtype=float)
        i = 0
        while i < x.size:
            j = i + 1
            while j < x.size and x[order_x[j]] == x[order_x[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            ranks_x[order_x[i:j]] = avg_rank
            i = j
        order_y = np.argsort(y, kind='mergesort')
        ranks_y = np.empty_like(y, dtype=float)
        i = 0
        while i < y.size:
            j = i + 1
            while j < y.size and y[order_y[j]] == y[order_y[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            ranks_y[order_y[i:j]] = avg_rank
            i = j
        # Pearson on ranks
        rx = (ranks_x - ranks_x.mean()) / (ranks_x.std() + 1e-12)
        ry = (ranks_y - ranks_y.mean()) / (ranks_y.std() + 1e-12)
        return float(np.clip((rx * ry).mean(), -1.0, 1.0))

    def get_spearman_correlation(self, percentile: float = 0.99) -> float:
        """Unified Spearman interface: use raw samples if mode=='raw', else binned pXX curve."""
        if self.motiv3_mode == 'raw':
            if not self.raw_load_latency:
                return float('nan')
            arr = np.asarray(self.raw_load_latency, dtype=float)
            x = arr[:, 0]
            y = arr[:, 1]
            return self._spearman_from_arrays(x, y)
        return self.get_spearman_correlation_binned(percentile=percentile)

    # ---------------- Spearman correlation and plotting (Motiv-Exp-3) ----------------
    def _weighted_ranks(self, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Compute weighted ranks in [0,1] for values with weights.
        Ties share the same average rank.
        """
        order = np.argsort(values, kind='mergesort')
        v_sorted = values[order]
        w_sorted = weights[order]
        cum_w = np.cumsum(w_sorted)
        total_w = cum_w[-1]
        # group by equal values
        ranks = np.empty_like(values, dtype=float)
        i = 0
        while i < len(v_sorted):
            j = i + 1
            while j < len(v_sorted) and v_sorted[j] == v_sorted[i]:
                j += 1
            w_group = w_sorted[i:j].sum()
            w_before = cum_w[i] - w_sorted[i]
            # average rank position within the group: midpoint of the weight block
            rank_val = (w_before + 0.5 * w_group) / total_w
            ranks[order[i:j]] = rank_val
            i = j
        return ranks

    def get_spearman_correlation_binned(self, percentile: float = 0.99) -> float:
        """Compute weighted Spearman correlation on adaptively binned data.
        y uses the given percentile (e.g., 0.99 for p99) from each bin; x is bin center.
        Weights are bin sample counts.
        """
        if self.adaptive_load_bins is None or len(self.adaptive_load_bins) < 2:
            return float('nan')
        xs = []
        ys = []
        ws = []
        for i, bin_start in enumerate(self.adaptive_load_bins[:-1]):
            td = self.latency_dist_per_adaptive_bin[bin_start]
            cnt = td.total_processed_count
            if cnt <= 0:
                continue
            bin_end = self.adaptive_load_bins[i+1]
            xs.append((bin_start + bin_end) / 2.0)
            ys.append(td.percentile(percentile * 100))
            ws.append(cnt)
        if len(xs) < 2:
            return float('nan')
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        w = np.asarray(ws, dtype=float)
        rx = self._weighted_ranks(x, w)
        ry = self._weighted_ranks(y, w)
        # weighted Pearson on ranks
        avg_rx = np.average(rx, weights=w)
        avg_ry = np.average(ry, weights=w)
        cov = np.average((rx - avg_rx) * (ry - avg_ry), weights=w)
        std_rx = np.sqrt(np.average((rx - avg_rx) ** 2, weights=w))
        std_ry = np.sqrt(np.average((ry - avg_ry) ** 2, weights=w))
        if std_rx == 0 or std_ry == 0:
            return float('nan')
        return cov / (std_rx * std_ry)

    @staticmethod
    def _calculate_rmse_from_trend(raw_data: List[Tuple[float, float]]) -> float:
        """
        Calculates the RMSE of raw data points from a simple linear regression trend line.
        This provides a stable measure of vertical dispersion around the linear trend.
        """
        if not raw_data or len(raw_data) < 2:
            return float('nan')
        
        arr = np.asarray(raw_data, dtype=float)
        x = arr[:, 0]
        y_actual = arr[:, 1]
        
        # Fit a linear trend line (degree 1 polynomial)
        coef = np.polyfit(x, y_actual, deg=1)
        y_pred = np.polyval(coef, x)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_actual - y_pred)**2))
        return float(rmse)

    @staticmethod
    def plot_motiv_case3(collector_base: 'StatisticsCollector' = None, 
                         collector_exp: 'StatisticsCollector' = None,
                         percentile: float = 0.99, iqr_band: Tuple[float, float] = (0.25, 0.75), 
                         fit: str = 'none', save_path: str = None, show: bool = False):
        """[Motiv-Exp-3] 绘制负载-延迟关系图（统一入口）
        
        根据collector的motiv3_mode，自动选择绘制raw散点图或binned曲线图。
        如果提供了collector_exp，则会在同一张图上绘制两组数据进行对比。

        Args:
            collector_base: 基线/第一组数据的collector
            collector_exp: (可选) 实验/第二组数据的collector
            percentile: e.g., 0.99 for p99 curve in binned mode
            iqr_band: tuple of (p_low, p_high) for binned mode IQR band
            fit: 'none' | 'wls' | 'lowess' — trend line type
            save_path: if provided, save the figure to this path
            show: if True, display via plt.show()
        """
        if not collector_base:
            print("Warning: plot_motiv_case3 requires at least a base collector.")
            return

        mode = collector_base.motiv3_mode
        
        if mode == 'raw':
            data_groups = []
            # Baseline group
            stats_base = collector_base.get_motiv_case3_stats(percentile=percentile, mode='raw')
            data_groups.append({
                'raw_data': stats_base.get('raw_data', []),
                'spearman_rho': stats_base.get('spearman_rho', float('nan')),
                'label': 'w/o realloc',
                'fit': 'wls',
                'color': 'C0'
            })
            
            # Experiment group (optional)
            if collector_exp:
                stats_exp = collector_exp.get_motiv_case3_stats(percentile=percentile, mode='raw')
                data_groups.append({
                    'raw_data': stats_exp.get('raw_data', []),
                    'spearman_rho': stats_exp.get('spearman_rho', float('nan')),
                    'label': 'w/ realloc',
                    'fit': fit,
                    'color': 'C1'
                })

            StatisticsCollector.plot_load_latency_raw(data_groups=data_groups, save_path=save_path, show=show)
            
        else: # binned mode
            # Binned mode plotting currently only supports single plots, not comparison
            stats = collector_base.get_motiv_case3_stats(percentile=percentile, mode='binned')
            StatisticsCollector.plot_load_latency_binned(
                binned_summary=stats.get('binned_summary', []),
                spearman_rho=stats.get('spearman_rho', float('nan')),
                percentile=percentile,
                iqr_band=iqr_band,
                fit=fit,
                save_path=save_path,
                show=show
            )

    @staticmethod
    def plot_load_latency_binned(binned_summary: List[Dict], spearman_rho: float, 
                                 percentile: float = 0.99, iqr_band: Tuple[float, float] = (0.25, 0.75), 
                                 fit: str = 'none', save_path: str = None, show: bool = False):
        """Plot binned load vs. latency percentile with optional IQR band and Spearman rho.
        
        Args:
            binned_summary: List of bin data dicts with keys: load_bin_start, load_bin_end, 
                           sample_count, p50.0, p90.0, p99.0, etc.
            spearman_rho: Spearman correlation coefficient
            percentile: e.g., 0.99 for p99 curve
            iqr_band: tuple of (p_low, p_high) to draw vertical error band, e.g., (0.25, 0.75)
            fit: 'none' | 'wls' | 'lowess' — do trend on binned (xs, pXX), weights=bin counts
            save_path: if provided, save the figure to this path
            show: if True, display via plt.show()
        """
        
        if not binned_summary:
            print("Warning: No binned data to plot")
            return
            
        xs = []
        ys = []
        ws = []
        y_low = []
        y_high = []
        p_low, p_high = iqr_band
        p_key = f'p{percentile*100:.1f}'
        p_low_key = f'p{p_low*100:.1f}'
        p_high_key = f'p{p_high*100:.1f}'
        
        for bin_data in binned_summary:
            cnt = bin_data.get('sample_count', 0)
            if cnt <= 0:
                continue
            bin_start = bin_data['load_bin_start']
            bin_end = bin_data['load_bin_end']
            xs.append((bin_start + bin_end) / 2.0)
            ys.append(bin_data.get(p_key, 0))
            ws.append(cnt)
            y_low.append(bin_data.get(p_low_key, 0))
            y_high.append(bin_data.get(p_high_key, 0))
        
        if len(xs) == 0:
            print("Warning: No valid binned data to plot")
            return
            
        rho = spearman_rho
        fig, ax = plt.subplots(figsize=(7, 4))
        # plot IQR band
        ax.fill_between(xs, y_low, y_high, color='C0', alpha=0.15, label=f'IQR p{int(p_low*100)}-p{int(p_high*100)}')
        # plot percentile points with size ~ weights
        sizes = np.array(ws, dtype=float)
        sizes = 50 * (sizes / sizes.max()) ** 0.5
        ax.scatter(xs, ys, s=sizes, color='C0', alpha=0.8, label=f'p{int(percentile*100)}')
        # optional trend on binned data
        if fit in ('wls', 'lowess') and len(xs) >= 3:
            xs_np = np.asarray(xs, dtype=float)
            ys_np = np.asarray(ys, dtype=float)
            ws_np = np.asarray(ws, dtype=float)
            if fit == 'wls':
                # weighted linear fit on binned curve
                coef = np.polyfit(xs_np, ys_np, deg=1, w=np.sqrt(ws_np))
                xx = np.linspace(min(xs_np), max(xs_np), 200)
                yy = np.polyval(coef, xx)
                ax.plot(xx, yy, color='C3', linewidth=2, label='WLS trend (binned)')
            else:
                # lowess-like via running median on binned points
                nb = min(50, max(10, int(np.sqrt(len(xs_np)))))
                bins = np.linspace(min(xs_np), max(xs_np), nb + 1)
                xc = 0.5 * (bins[:-1] + bins[1:])
                med = np.full(nb, np.nan)
                for i in range(nb):
                    m = (xs_np >= bins[i]) & (xs_np < bins[i+1] if i < nb - 1 else xs_np <= bins[i+1])
                    if np.any(m):
                        med[i] = np.median(ys_np[m])
                mm = ~np.isnan(med)
                if mm.sum() >= 3:
                    ax.plot(xc[mm], med[mm], color='C2', lw=2, label='LOWESS-like (binned)')
        ax.set_xlabel('Load (bin center)')
        ax.set_ylabel(f'Latency (p{int(percentile*100)})')
        ax.set_title(f'Load vs Latency (Spearman rho={rho:.3f})')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"Case 3 Binned 图已保存到: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_load_latency_raw(data_groups: List[Dict], save_path: str = None, show: bool = False):
        """Plot raw (load, worst_e2e) with scatter and optional trend on raw points for multiple groups.
        
        Args:
            data_groups: List of dicts, where each dict represents a data group and contains:
                - 'raw_data': List of (load, worst_e2e) tuples
                - 'spearman_rho': Spearman correlation coefficient
                - 'label': Name of the data group (e.g., 'baseline', 'experiment')
                - 'fit': 'none' | 'wls' | 'lowess' (optional)
                - 'color': a valid matplotlib color (optional)
            save_path: if provided, save the figure to this path
            show: if True, display via plt.show()
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not data_groups:
            print("Warning: No data groups to plot")
            return
            
        fig, ax = plt.subplots(figsize=(4, 2.2))
        
        for i, group in enumerate(data_groups):
            raw_data = group.get('raw_data')
            if not raw_data:
                continue
            
            arr = np.asarray(raw_data, dtype=float)
            x = arr[:, 0]
            y = arr[:, 1]
            rho = group.get('spearman_rho', float('nan'))
            rmse = group.get('rmse', float('nan'))
            
            base_label = group.get('label', f'Group {i}')
            # 将 ρ 和 RMSE 添加到散点图的标签中（简化）
            scatter_label = f"{base_label}\n(ρ={rho:.2f},RMSE={rmse:.3f})"
            
            fit = group.get('fit', 'none')
            color = group.get('color', f'C{i}')
            
            # Scatter plot for raw data with metrics in label
            ax.scatter(x, y, s=8, alpha=0.4, color=color, linewidths=0, label=scatter_label)
            
            # Optional trend on raw data
            if fit in ('wls', 'lowess') and x.size >= 3:
                xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
                if fit == 'wls':
                    coef = np.polyfit(x, y, deg=1)
                    yy = np.polyval(coef, xx)
                    ax.plot(xx, yy, color=color, linestyle='--', linewidth=1.5, label=f'{base_label} trend')
                else: # lowess
                    nb = min(100, max(20, int(np.sqrt(x.size))))
                    bins = np.linspace(float(np.min(x)), float(np.max(x)), nb + 1)
                    xc = 0.5 * (bins[:-1] + bins[1:])
                    med = np.full(nb, np.nan)
                    for j in range(nb):
                        m = (x >= bins[j]) & (x < bins[j+1] if j < nb - 1 else x <= bins[j+1])
                        if np.any(m):
                            med[j] = np.median(y[m])
                    mm = ~np.isnan(med)
                    if mm.sum() >= 3:
                        ax.plot(xc[mm], med[mm], color=color, linestyle='-', lw=1.5, label=f'{base_label} lowess trend')
        
        ax.set_xlabel('Load', fontsize=9)
        ax.set_ylabel('Worst E2E Lat.', fontsize=9)
        ax.set_title('Case 3: Load vs Latency\nUncertainty', fontsize=9, pad=8)
        ax.legend(loc='best', fontsize=6, framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)
        fig.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"Case 3 Raw Comparison图已保存到: {save_path}")
        if show:
            plt.show()
        plt.close(fig)

    def get_full_summary(self, num_bins: int = 20, p_list: List[float] = None) -> Dict:
        """
        Generates a comprehensive summary dictionary containing all statistics.
        This is the new top-level function for getting all results at once.
        """
        if p_list is None:
            p_list = [0.5, 0.9, 0.99, 0.999]

        full_summary = {
            'distribution_summary': self.get_summary(num_bins, p_list),
            'adaptive_binning_summary': self.get_adaptive_load_latency_summary(p_list)
        }
        return full_summary

    def _format_summary_for_print(self, summary_data: Dict, p_list: List[float]) -> str:
        """
        Formats the full summary dictionary into a human-readable string.
        """
        formatted_output = []
        p_list_str = f"Percentiles calculated: {[f'{p*100:.1f}%' for p in p_list]}"
        
        # --- Section 1: Distribution Summary ---
        dist_summary = summary_data.get('distribution_summary', {})
        if dist_summary:
            formatted_output.append("--- Distribution Statistics Summary ---")
            formatted_output.append(p_list_str + "\n")

            for category, data in dist_summary.items():
                formatted_output.append(f"=== {category.replace('_', ' ').title()} ===")
                if not data:
                    formatted_output.append("  No data available.\n")
                    continue

                # Check if it's a single distribution summary (e.g., overall_realloc)
                if isinstance(data, dict) and 'histogram' in data and 'percentiles' in data:
                    count = int(data.get('total_processed_count', 0))
                    formatted_output.append(f"  Total processed count: {count}")
                    if count > 0:
                        formatted_output.append("  Percentiles:")
                        for p_key, p_val in data['percentiles'].items():
                            formatted_output.append(f"    {p_key}: {p_val:.4f}")
                    formatted_output.append("")
                # Check if it's a dictionary of distributions (e.g., per_task_...)
                elif isinstance(data, dict):
                    for key, dist_summary_item in data.items():
                        formatted_output.append(f"  --- {key.replace('_', ' ').title()} ---")
                        count = int(dist_summary_item.get('total_processed_count', 0))
                        formatted_output.append(f"    Total processed count: {count}")
                        if count > 0:
                            formatted_output.append("    Percentiles:")
                            for p_key, p_val in dist_summary_item['percentiles'].items():
                                formatted_output.append(f"      {p_key}: {p_val:.4f}")
                        formatted_output.append("")
            formatted_output.append("\n")

        # --- Section 2: Periodic Summary (Motiv-Exp-1) ---
        periodic_summary = summary_data.get('periodic_summary', [])
        if periodic_summary:
            formatted_output.append("--- Periodic Summary (first 5 and last 5 periods) ---")
            # Create a header
            header = f"{'Period':>6} | {'Total Load':>12} | {'Worst E2E':>12} | {'Idle Compute':>14} | {'Miss Remaining':>16}"
            formatted_output.append(header)
            formatted_output.append("-" * len(header))
            
            def format_row(i, record):
                return (f"{i:<6d} | {record['total_load']:>12.2f} | {record['worst_e2e']:>12.2f} | "
                        f"{record['idle_compute']:>14.2f} | {record['miss_remaining']:>16.2f}")

            for i, record in enumerate(periodic_summary[:5]):
                formatted_output.append(format_row(i, record))
            
            if len(periodic_summary) > 10:
                formatted_output.append("...")
            
            start_idx = max(5, len(periodic_summary) - 5)
            for i in range(start_idx, len(periodic_summary)):
                formatted_output.append(format_row(i, periodic_summary[i]))
            formatted_output.append("\n")
        
        # --- Section 3: Adaptive Binning Summary (Motiv-Exp-3) ---
        adaptive_summary = summary_data.get('adaptive_binning_summary', [])
        if adaptive_summary:
            formatted_output.append(f"--- Adaptive Load-Latency Binning Summary ---")
            formatted_output.append(p_list_str + "\n")
            
            # Create header
            p_headers = " | ".join([f"{p_key:>9}" for p_key in adaptive_summary[0] if p_key.startswith('p')])
            header = f"{'Load Range':>18} | {'Count':>7} | {'Mean Latency':>13} | {p_headers}"
            formatted_output.append(header)
            formatted_output.append("-" * len(header))

            for record in adaptive_summary:
                load_range = f"[{record['load_bin_start']:<8.2f}, {record['load_bin_end']:<8.2f})"
                p_values = " | ".join([f"{record[p_key]:>9.4f}" for p_key in record if p_key.startswith('p')])
                row = (f"{load_range:>18} | {record['sample_count']:>7d} | {record['latency_mean']:>13.4f} | {p_values}")
                formatted_output.append(row)
            formatted_output.append("\n")

        return "\n".join(formatted_output)

    def export_summary(self, num_bins: int = 20, p_list: List[float] = None, verbose: bool = False):
        """
        Prints the full statistical summary to console and optionally to a file.
        
        Args:
            num_bins: Number of bins for the histograms in the summary.
            p_list: List of percentiles (as floats, e.g., 0.5 for 50%) to calculate.
            output_path: Optional file path to save the formatted summary.
        """
        if p_list is None:
            p_list = [0.5, 0.9, 0.99, 0.999]

        # 1. Generate the new comprehensive summary
        full_summary = self.get_full_summary(num_bins, p_list)
        if self.motiv3_mode == 'raw':
            self.plot_load_latency_raw(save_path=self.path["motiv3"])
        else:
            self.plot_load_latency_binned(percentile=0.99, save_path=self.path["motiv3"])
        # 2. Format the comprehensive summary
        formatted_summary = self._format_summary_for_print(full_summary, p_list)
        
        if verbose:
            print(formatted_summary)

        if self.path["stat"]:
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.path['stat']), exist_ok=True)
                with open(self.path['stat'], 'w') as f:
                    f.write(formatted_summary)
                print(f"\nFormatted summary successfully saved to: {self.path['stat']}")
            except IOError as e:
                print(f"Error saving formatted summary to {self.path['stat']}: {e}")

    def save_state(self, file_path: str):
        """
        Saves the entire state of the StatisticsCollector object to a JSON file.
        This includes all internal caches and TDigestStreamingHistogram objects.
        
        Args:
            file_path: The file path to save the state.
        """
        state = {
            'delta': self._delta,
            'K': self._K,
            'total_pwr': self.total_pwr,
            'partition_realloc_stats': self.partition_realloc_stats,
            
            'task_at': self.task_at, # Regular dict, directly serializable
            
            # New critical path data
            'task_curr_stat': dict(self.task_curr_stat),
            'task_pred_stat': {k: dict(v) for k, v in self.task_pred_stat.items()},

            'part_realloc_curr': dict(self.part_realloc_curr), # Convert defaultdict to dict
            'part_realloc_num': dict(self.part_realloc_num),   # Convert defaultdict to dict

            'hp_total_load_curr': self.hp_total_load_curr,
            'hp_worst_e2e_curr': self.hp_worst_e2e_curr,
            'hp_idle_curr': self.hp_idle_curr,
            'system_realloc_cost': self.system_realloc_cost,

            'dist_per_task_exe': {k: v.to_dict() for k, v in self.dist_per_task_exe.items()},
            'dist_per_task_ft': {k: v.to_dict() for k, v in self.dist_per_task_ft.items()},
            'dist_per_task_realloc': {k: v.to_dict() for k, v in self.dist_per_task_realloc.items()},
            'dist_per_part_realloc': {k: v.to_dict() for k, v in self.dist_per_part_realloc.items()},
            'dist_per_part_realloc_count': {k: v.to_dict() for k, v in self.dist_per_part_realloc_count.items()},
            'dist_overall_realloc': self.dist_overall_realloc.to_dict(),
            'dist_overall_idle': self.dist_overall_idle.to_dict(),
            'dist_overall_miss': self.dist_overall_miss.to_dict(),
            'dist_overall_miss_count': self.dist_overall_miss_count.to_dict(),
            'dist_overall_total_load': self.dist_overall_total_load.to_dict(),
            
            # new fields
            # for motiv-exp-3 (adaptive)
            'dist_hp_total_load': self.dist_hp_total_load.to_dict(),
            'hp_temp_records_for_binning': self.hp_temp_records_for_binning,
            'adaptive_load_bins': self.adaptive_load_bins,
            'latency_dist_per_adaptive_bin': {k: v.to_dict() for k, v in self.latency_dist_per_adaptive_bin.items()},
            
            # summary and p_list are transient, can be re-generated on load if needed
            'summary': None, 
            'p_list': None
        }

        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4, allow_nan=True)
            print(f"\nStatisticsCollector state successfully saved to: {file_path}")
        except IOError as e:
            print(f"Error saving StatisticsCollector state to {file_path}: {e}")
        except TypeError as e:
            print(f"Error serializing StatisticsCollector state to JSON. Error: {e}")

    @classmethod
    def load_state(cls, file_path: str) -> StatisticsCollector:
        """
        Loads the state of a StatisticsCollector object from a JSON file.
        
        Args:
            file_path: The file path to load the state from.
        
        Returns:
            A new StatisticsCollector instance with the loaded state.
        """
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
        except FileNotFoundError:
            print(f"Error: State file not found at {file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return None
        except IOError as e:
            print(f"Error reading state file {file_path}: {e}")
            return None

        # Reconstruct the StatisticsCollector instance
        # Initialize with base delta and K from saved state
        collector = cls(delta=state['delta'], K=state['K'])

        collector.total_pwr = state['total_pwr']
        collector.partition_realloc_stats = state['partition_realloc_stats']
        collector.task_at = state['task_at']
        
        # Restore critical path data
        collector.task_curr_stat = defaultdict(lambda: {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0}, state.get('task_curr_stat', {}))
        collector.task_pred_stat = defaultdict(
            lambda: defaultdict(lambda: {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0, 'e2e_lat': 0.0}),
            {k: defaultdict(lambda: {'realloc': 0.0, 'compute': 0.0, 'realloc_num': 0, 'e2e_lat': 0.0}, v)
             for k, v in state.get('task_pred_stat', {}).items()}
        )
        
        collector.part_realloc_curr = defaultdict(float, state['part_realloc_curr'])
        collector.part_realloc_num = defaultdict(int, state['part_realloc_num'])
        collector.system_realloc_cost = state['system_realloc_cost']

        # Reconstruct TDigestStreamingHistogram objects from their serialized states
        collector.dist_per_task_ft = defaultdict(
            lambda: TDigestStreamingHistogram(delta=state['delta'], K=state['K']), 
            {k: TDigestStreamingHistogram.from_dict(v) for k, v in state['dist_per_task_ft'].items()}
        )
        collector.dist_per_task_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=state['delta'], K=state['K']),
            {k: TDigestStreamingHistogram.from_dict(v) for k, v in state['dist_per_task_realloc'].items()}
        )
        collector.dist_per_part_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=state['delta'], K=state['K']),
            {k: TDigestStreamingHistogram.from_dict(v) for k, v in state.get('dist_per_part_realloc', {}).items()}
        )
        collector.dist_per_part_realloc_count = defaultdict(
            lambda: TDigestStreamingHistogram(delta=state['delta'], K=state['K']),
            {k: TDigestStreamingHistogram.from_dict(v) for k, v in state.get('dist_per_part_realloc_count', {}).items()}
        )
        collector.dist_overall_realloc = TDigestStreamingHistogram.from_dict(state.get('dist_overall_realloc', {}))
        collector.dist_overall_idle = TDigestStreamingHistogram.from_dict(state.get('dist_overall_idle', {}))
        collector.dist_overall_miss = TDigestStreamingHistogram.from_dict(state.get('dist_overall_miss', {}))
        collector.dist_overall_miss_count = TDigestStreamingHistogram.from_dict(state.get('dist_overall_miss_count', {}))
        collector.dist_overall_total_load = TDigestStreamingHistogram.from_dict(state.get('dist_overall_total_load', {}))
        
        # Load state for Motiv-Exp-(3)
        collector.dist_hp_total_load = TDigestStreamingHistogram.from_dict(state.get('dist_hp_total_load', {}))
        collector.hp_temp_records_for_binning = state.get('hp_temp_records_for_binning', [])
        collector.adaptive_load_bins = state.get('adaptive_load_bins', None)
        
        # Handle dict with float keys after JSON serialization (keys become strings)
        binned_data_from_json = state.get('latency_dist_per_adaptive_bin', {})
        collector.latency_dist_per_adaptive_bin = {
            float(k): TDigestStreamingHistogram.from_dict(v) for k, v in binned_data_from_json.items()
        }

        # summary and p_list are transient and can be re-generated
        collector.summary = None 
        collector.p_list = None

        print(f"StatisticsCollector state successfully loaded from: {file_path}")
        return collector

    def load_save_check(self):
        test_file_path = './temp_state.json'

        # 2. 将状态保存到文件 (Save the state to a file)
        print("--- 保存状态 (Saving state) ---")
        self.save_state(test_file_path)

        # 3. 加载状态到新实例 (Load the state into a new instance)
        print("--- 加载状态 (Loading state) ---")
        loaded_collector = StatisticsCollector.load_state(test_file_path)

        # 4. 验证状态 (Verify the state)
        if loaded_collector is None:
            print("--- 测试失败：加载失败 (Test failed: Load failed) ---")
            return

        print("--- 验证数据完整性 (Verifying data integrity) ---")

        # 比较基本属性 (Compare basic attributes)
        if (self._delta != loaded_collector._delta or
            self._K != loaded_collector._K or
            self.system_realloc_cost != loaded_collector.system_realloc_cost):
            print("--- 测试失败：基本属性不匹配 (Test failed: Basic attributes do not match) ---")
            return

        # 比较字典 (Compare dictionaries)
        if (self.partition_realloc_stats != loaded_collector.partition_realloc_stats or
            self.task_at != loaded_collector.task_at or
            self.part_realloc_curr != loaded_collector.part_realloc_curr or
            self.part_realloc_num != loaded_collector.part_realloc_num or
            # Deep comparison for nested defaultdicts could be complex,
            # for now, let's just check the top-level keys
            set(self.task_curr_stat.keys()) != set(loaded_collector.task_curr_stat.keys()) or
            set(self.task_pred_stat.keys()) != set(loaded_collector.task_pred_stat.keys())
            ):
            print("--- 测试失败：字典数据不匹配 (Test failed: Dictionary data do not match) ---")
            return

        # 比较 TDigestStreamingHistogram 对象
        # Compare TDigestStreamingHistogram objects by their serialized representation
        overall_dists = ['dist_overall_realloc', 'dist_overall_idle', 'dist_overall_miss', 
                        'dist_overall_miss_count', 'dist_overall_total_load']
        for dist_name in overall_dists:
            if (getattr(self, dist_name).to_dict() != getattr(loaded_collector, dist_name).to_dict()):
                print(f"--- 测试失败：{dist_name} 不匹配 (Test failed: {dist_name} mismatch) ---")
                return

        # 比较 defaultdict 中的 TDigestStreamingHistogram 对象
        # Compare TDigestStreamingHistogram objects in defaultdicts
        for dist_type in ['dist_per_task_ft', 'dist_per_task_realloc', 'dist_per_part_realloc', 'dist_per_part_realloc_count']:
            orig_dist_dict = getattr(self, dist_type)
            loaded_dist_dict = getattr(loaded_collector, dist_type)

            if set(orig_dist_dict.keys()) != set(loaded_dist_dict.keys()):
                print(f"--- 测试失败：{dist_type} 的键不匹配 (Test failed: {dist_type} keys mismatch) ---")
                return

            for key in orig_dist_dict:
                orig_td = orig_dist_dict[key].to_dict()
                loaded_td = loaded_dist_dict[key].to_dict()
                if orig_td != loaded_td:
                    print(f"--- 测试失败：{dist_type}[{key}] 数据不匹配 (Test failed: {dist_type}[{key}] data mismatch) ---")
                    return

        # 比较 defaultdict 中的 TDigestStreamingHistogram 对象
        # Compare TDigestStreamingHistogram objects in defaultdicts
        for dist_type in ['dist_per_task_exe', 'latency_dist_per_adaptive_bin']:
            orig_dist_dict = getattr(self, dist_type)
            loaded_dist_dict = getattr(loaded_collector, dist_type)

            if set(orig_dist_dict.keys()) != set(loaded_dist_dict.keys()):
                print(f"--- 测试失败：{dist_type} 的键不匹配 (Test failed: {dist_type} keys mismatch) ---")
                return

            for key in orig_dist_dict:
                orig_td = orig_dist_dict[key].to_dict()
                loaded_td = loaded_dist_dict[key].to_dict()
                if orig_td != loaded_td:
                    print(f"--- 测试失败：{dist_type}[{key}] 数据不匹配 (Test failed: {dist_type}[{key}] data mismatch) ---")
                    return

        # 如果所有检查都通过 (If all checks pass)
        print("--- 测试成功：所有状态均已正确保存和加载 (Test successful: All state was correctly saved and loaded) ---")

        # 清理测试文件 (Clean up test file)
        os.remove(test_file_path)
        print(f"--- 已删除测试文件: {test_file_path} (Removed test file) ---")


if __name__ == "__main__":
    from test_approach_collector import run_all_tests

    # 运行全面功能测试
    success = run_all_tests()
    
