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
    
    def __init__(self, delta: float = 0.01, K: int = 25):
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
        self.output_path = None
        self.total_pwr = 0.0
        
        # Temp cache for Lat breakdown 
        # Per-task (instance) property
        # collected and reset at task completion
        # propagated to and aggregated at downstream
        self.task_at = {} # arrival time of each task
        self.task_realloc_curr = defaultdict(float) # realloc overhead of each task
        self.task_realloc_num = defaultdict(int) # realloc number of each task
        self.task_compute_curr = defaultdict(float) # compute time of each task
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
        self.system_realloc_cost = 0.0  # System-level total reallocation cost
        # distribution for usage
        self.dist_per_part_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )
        self.dist_overall_realloc = TDigestStreamingHistogram(delta=delta, K=K)
        self.dist_overall_idle = TDigestStreamingHistogram(delta=delta, K=K) 
        self.dist_overall_miss = TDigestStreamingHistogram(delta=delta, K=K)

        # For Motiv-Exp-(3): Adaptive binning for load vs. worst E2E latency relationship
        self.binning_warmup_period = 100  # Number of samples to learn the distribution from
        self.hp_total_load_curr = 0.0 # submitted load
        self.hp_worst_e2e_curr = float('-inf') # worst e2e latency
        self.hp_temp_records_for_binning = [] # buffer for learning the distribution 

        self.adaptive_load_bins = None  # Will store list of bin boundaries, e.g., [0, 1000, 2500, ...]
        self.dist_hp_total_load = TDigestStreamingHistogram(delta=delta, K=K)
        self.latency_dist_per_adaptive_bin = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        ) # Keys will be the start of each bin

    def set_output_path(self, output_path: str):
        self.output_path = output_path

    def init_partition_stats(self, partition_id: str, cap: int, base_pwr: float):
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

        # Record per-task overhead at task completion, and accumulate overhead to downstream tasks
        realloc_time = self.task_realloc_curr.pop(process_id, 0.0) # Use .pop with default to avoid KeyError
        realloc_num = self.task_realloc_num.pop(process_id, 0) # Use .pop with default to avoid KeyError
        self.dist_per_task_realloc[base_task_name].add(realloc_time)
        # propagate compute time downstream
        compute_time = self.task_compute_curr.pop(process_id, 0.0)
        for succ in G.successors(process_id):
            self.task_realloc_curr[succ] += realloc_time
            self.task_realloc_num[succ] += realloc_num 
            self.task_compute_curr[succ] += compute_time
    
    def record_e2e_finish(self, G: MyGraph, sink_name: str, finish_t: float):
        """
        Records end-to-end completion: full latency from source to sink task.
        """
        base_sink_name = self._get_base_task_name(sink_name)
        
        offset = G.nodes[sink_name]["offset"]
        finish_t_rel = elim_nume_error(finish_t - offset)
        # record finish time
        self.dist_per_task_ft[base_sink_name].add(finish_t_rel)
        # record realloc overhead
        realloc_time = self.task_realloc_curr.pop(sink_name, 0.0) # Use .pop with default to avoid KeyError
        realloc_num = self.task_realloc_num.pop(sink_name, 0) # Use .pop with default to avoid KeyError
        self.dist_per_task_realloc[base_sink_name].add(realloc_time)
        # record compute time
        compute_time = self.task_compute_curr.pop(sink_name, 0.0)
        self.dist_per_task_exe[base_sink_name].add(elim_nume_error(compute_time))

        # update per-hyperperiod worst e2e
        self._record_period_e2e_candidate(finish_t_rel)

    def record_realloc(self, partition_id: str, delta_ld: float, task_list: List[str]):
        """
        Records reallocation overhead each time the remaining realloc tasks are updated.
        Maintains per-task realloc overhead and per-hyperperiod overhead.
        """
        # Update per-task realloc overhead
        for task_name in task_list:
            self.task_realloc_curr[task_name] += delta_ld
        
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
            self.task_realloc_num[task_name] += 1
        
        # Update per-part realloc overhead
        self.part_realloc_num[partition_id] += 1


    # ---------------- New minimal APIs for experiments ----------------
    def record_compute_progress(self, task_name: str, delta_compute_t: float):
        """Accumulate compute time for a task instance (delta time domain)."""
        if delta_compute_t > 0:
            self.task_compute_curr[task_name] += float(delta_compute_t)

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

        miss_sum = elim_nume_error(sum(rem for _, rem in iter_pairs(timeout_iter)))
        self.dist_overall_miss.add(miss_sum)


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
        
        # Add system_realloc_cost to the overall distribution
        
        self.dist_overall_realloc.add(elim_nume_error(self.system_realloc_cost/self.total_pwr/T_hp))
        self.system_realloc_cost = 0.0
        self.dist_overall_idle.add(self.hp_idle_curr)
        self.hp_idle_curr = 0.0

        # --- Motiv-Exp-(3) Adaptive Binning ---
        total_load = self.hp_total_load_curr
        worst_e2e = self.hp_worst_e2e_curr
        
        # We only add to binning stats if a valid E2E latency was recorded in this period
        if worst_e2e > float('-inf'):
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

        # 1. Define bin boundaries using quantiles (e.g., 5 bins)
        quantiles = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        boundaries = [self.dist_hp_total_load.percentile(p) for p in quantiles]
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
            mean = self._td_mean(tdigest) if count > 0 else float('nan')

            summary_list.append({
                'load_bin_start': bin_start,
                'load_bin_end': bin_end,
                'sample_count': count,
                'latency_mean': mean,
                **percentiles
            })
        return summary_list

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
            'per_chain_realloc_overhead': {},
            'overall_realloc_overhead': {},
            'overall_idle_time': {},
            'overall_missed_load': {}
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
        
        # 4. Overall reallocation overhead
        summary['overall_realloc_overhead'] = self.dist_overall_realloc.get_summary(num_bins, p_list)
        summary['overall_idle_time'] = self.dist_overall_idle.get_summary(num_bins, p_list)
        summary['overall_missed_load'] = self.dist_overall_miss.get_summary(num_bins, p_list)
        
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
    
    def get_load_latency_correlation(self) -> float:
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
                y_means.append(self._td_mean(tdigest))
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

    def _td_mean(self, td: TDigestStreamingHistogram, num_bins: int = 32) -> float:
        """Approximate mean from TDigest by histogram midpoints weighting."""
        hist = td.get_histogram_data(num_bins=num_bins)
        if not hist:
            return float('nan')
        total = sum(c for _, _, c in hist)
        if total <= 0:
            return float('nan')
        weighted_sum = sum(((a + b) / 2.0) * c for a, b, c in hist)
        return weighted_sum / total

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
        
        # 2. Format the comprehensive summary
        formatted_summary = self._format_summary_for_print(full_summary, p_list)
        
        if verbose:
            print(formatted_summary)

        if self.output_path:
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                with open(self.output_path, 'w') as f:
                    f.write(formatted_summary)
                print(f"\nFormatted summary successfully saved to: {self.output_path}")
            except IOError as e:
                print(f"Error saving formatted summary to {self.output_path}: {e}")

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
            'partition_realloc_stats': self.partition_realloc_stats,
            
            'task_at': self.task_at, # Regular dict, directly serializable
            'task_realloc_curr': dict(self.task_realloc_curr), # Convert defaultdict to dict
            'task_realloc_num': dict(self.task_realloc_num),   # Convert defaultdict to dict
            'task_compute_curr': dict(self.task_compute_curr),
            
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
            'dist_overall_realloc': self.dist_overall_realloc.to_dict(),
            'dist_overall_idle': self.dist_overall_idle.to_dict(),
            'dist_overall_miss': self.dist_overall_miss.to_dict(),
            
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

        collector.partition_realloc_stats = state['partition_realloc_stats']
        collector.task_at = state['task_at']
        collector.task_realloc_curr = defaultdict(float, state['task_realloc_curr'])
        collector.task_realloc_num = defaultdict(int, state['task_realloc_num'])
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
        collector.dist_overall_realloc = TDigestStreamingHistogram.from_dict(state.get('dist_overall_realloc', {}))
        collector.dist_overall_idle = TDigestStreamingHistogram.from_dict(state.get('dist_overall_idle', {}))
        collector.dist_overall_miss = TDigestStreamingHistogram.from_dict(state.get('dist_overall_miss', {}))
        
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
            self.task_realloc_curr != loaded_collector.task_realloc_curr or
            self.task_realloc_num != loaded_collector.task_realloc_num or
            self.part_realloc_curr != loaded_collector.part_realloc_curr or
            self.part_realloc_num != loaded_collector.part_realloc_num):
            print("--- 测试失败：字典数据不匹配 (Test failed: Dictionary data do not match) ---")
            return

        # 比较 TDigestStreamingHistogram 对象
        # Compare TDigestStreamingHistogram objects by their serialized representation
        if (self.dist_overall_realloc.to_dict() != loaded_collector.dist_overall_realloc.to_dict()):
            print("--- 测试失败：dist_overall_realloc 不匹配 (Test failed: dist_overall_realloc mismatch) ---")
            return

        # 比较 defaultdict 中的 TDigestStreamingHistogram 对象
        # Compare TDigestStreamingHistogram objects in defaultdicts
        for dist_type in ['dist_per_task_ft', 'dist_per_task_realloc', 'dist_per_part_realloc']:
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


def run_test():
    """
    测试 StatisticsCollector 类的状态保存和加载功能。
    Test the state saving and loading functionality of the StatisticsCollector class.
    """
    print("--- 启动测试 (Starting test) ---")

    class MyGraph:
        def __init__(self):
            self.nodes = {
                'op1_0': {'offset': 10, 'ddl_map': 100},
                'op2_0': {'offset': 20, 'ddl_map': 200},
                'sink_0': {'offset': 50, 'ddl_map': 300},
            }
            self.succs = {
                'op1_0': ['op2_0'],
                'op2_0': ['sink_0'],
                'sink_0': [],
            }
            self.ddl_map = {
                'op1_0': 100,
                'op2_0': 200,
                'sink_0': 300
            }

    def time_gt(t1, t2):
        return t1 > t2

    def elim_nume_error(val):
        return val
    
    test_file_path = 'temp_state.json'
    G = MyGraph()

    # 1. 创建并填充一个 StatisticsCollector 实例 (Create and populate a StatisticsCollector instance)
    original_collector = StatisticsCollector()
    original_collector.init_partition_stats('part1', 10, 1.5)

    print("--- 填充原始数据 (Populating original data) ---")
    original_collector.record_task_start('op1_0', 10.0)
    original_collector.record_realloc('part1', 0.5, ['op1_0'])
    original_collector.record_task_finish(G, 'op1_0', 25.0)

    original_collector.record_task_start('op2_0', 30.0)
    original_collector.record_realloc('part1', 0.8, ['op2_0'])
    original_collector.record_task_finish(G, 'op2_0', 45.0)

    original_collector.forward_hyperperiod()

    original_collector.record_e2e_finish(G, 'sink_0', 50.0)

    original_collector.load_save_check()

if __name__ == "__main__":
    run_test()