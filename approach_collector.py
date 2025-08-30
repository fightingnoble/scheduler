from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from approach_util import MyGraph
from typing import Dict, List, Set
from collections import defaultdict
import re
import json
import os
from functools import reduce
from task_estimation import time_gt, elim_nume_error
from ref_tdigest import TDigestStreamingHistogram # Assuming ref_tdigest.py is in the same directory

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
        
        # temp cache
        # arrival time of each task
        self.task_at = {}
        self.task_realloc_curr = defaultdict(float)
        self.task_realloc_num = defaultdict(int)
        self.part_realloc_curr = defaultdict(float)
        self.part_realloc_num = defaultdict(int)
        self.system_realloc_cost = 0.0  # System-level total reallocation cost
        
        # dict for distribution 
        # per-task ft: per-task
        # realloc time: per-task, per-part
        self.dist_per_task_ft = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )

        self.dist_per_task_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )

        self.dist_per_part_realloc = defaultdict(
            lambda: TDigestStreamingHistogram(delta=delta, K=K)
        )
        self.dist_overall_realloc = TDigestStreamingHistogram(delta=delta, K=K) # This should be a single TDigest, not a defaultdict

        self.summary = None
        self.p_list = None
        # Store delta and K for creating new TDigestStreamingHistogram objects if needed
        self._delta = delta
        self._K = K
    
    def init_partition_stats(self, partition_id: str, cap: int, base_pwr: float):
        """Initializes statistics for a partition."""
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
        for succ in G.successors(process_id):
            self.task_realloc_curr[succ] += realloc_time
            self.task_realloc_num[succ] += realloc_num 
    
    def record_e2e_finish(self, G: MyGraph, sink_name: str, finish_t: float):
        """
        Records end-to-end completion: full latency from source to sink task.
        """
        base_sink_name = self._get_base_task_name(sink_name)
        
        offset = G.nodes[sink_name]["offset"]
        self.dist_per_task_ft[base_sink_name].add(elim_nume_error(finish_t - offset))
        
        realloc_time = self.task_realloc_curr.pop(sink_name, 0.0) # Use .pop with default to avoid KeyError
        realloc_num = self.task_realloc_num.pop(sink_name, 0) # Use .pop with default to avoid KeyError
        self.dist_per_task_realloc[base_sink_name].add(realloc_time)
    
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
            delta_cost = delta_ld * stats['cap'] * stats['base_pwr']
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


    def forward_hyperperiod(self):
        """
        Records per-hyperperiod overhead at the end of a period.
        """        
        # Process per-part realloc overhead
        for part_id in list(self.part_realloc_curr.keys()):
            realloc_time = self.part_realloc_curr.pop(part_id)
            realloc_num = self.part_realloc_num.pop(part_id)
            self.dist_per_part_realloc[part_id].add(realloc_time)
        
        # Add system_realloc_cost to the overall distribution
        self.dist_overall_realloc.add(self.system_realloc_cost)
        self.system_realloc_cost = 0.0

    def _get_base_task_name(self, task_name: str) -> str:
        """
        Extracts the base task name, removing hyperperiod numbering.
        E.g., 'task1_0' -> 'task1', 'S1_2' -> 'S1'.
        """
        pattern = r'_[-\d]+$'
        base_name = re.sub(pattern, '', task_name)
        return base_name
    
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
            'per_part_realloc_overhead': {},
            'per_chain_e2e_latency': {},
            'per_chain_realloc_overhead': {},
            'overall_e2e_latency': {},
            'overall_realloc_overhead': {}
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
        
        # 3. Per-part realloc overhead distribution
        for part_id, dist in self.dist_per_part_realloc.items():
            summary['per_part_realloc_overhead'][part_id] = dist.get_summary(num_bins, p_list)
        
        # 4. Overall reallocation overhead
        summary['overall_realloc_overhead'] = self.dist_overall_realloc.get_summary(num_bins, p_list)
        
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
                'percentiles': {f'p{p*100:.1f}': float('nan') for p in p_list}
            }


        self.summary = summary
        self.p_list = p_list
        return summary
    
    def _format_summary_for_print(self, summary_data: Dict, p_list: List[float]) -> str:
        """
        Formats the summary dictionary into a human-readable string.
        """
        formatted_output = []
        formatted_output.append("--- Statistics Summary ---")
        formatted_output.append(f"Percentiles calculated: {[f'{p*100:.1f}%' for p in p_list]}\n")

        for category, data in summary_data.items():
            formatted_output.append(f"=== {category.replace('_', ' ').title()} ===")
            if not data:
                formatted_output.append("  No data available.\n")
                continue

            if isinstance(data, dict) and 'histogram' in data and 'percentiles' in data: # It's a single distribution summary
                formatted_output.append("  Percentiles:")
                for p_key, p_val in data['percentiles'].items():
                    formatted_output.append(f"    {p_key}: {p_val:.4f}")
                
                formatted_output.append("  Histogram (first 3 & last 3 bins):")
                if data['histogram']:
                    for i, (start, end, count) in enumerate(data['histogram'][:3]):
                        formatted_output.append(f"    Bin {i+1}: [{start:.4f}, {end:.4f}): Count={count:.2f}")
                    if len(data['histogram']) > 6:
                        formatted_output.append("    ...")
                    for i, (start, end, count) in enumerate(data['histogram'][-3:]):
                        if len(data['histogram']) - 3 + i >= 3: # Avoid printing same bins if less than 6 total
                            formatted_output.append(f"    Bin {len(data['histogram']) - 3 + i + 1}: [{start:.4f}, {end:.4f}): Count={count:.2f}")
                else:
                    formatted_output.append("    No histogram data.")
                formatted_output.append("") # Newline for spacing
            elif isinstance(data, dict): # It's a dictionary of distribution summaries
                for key, dist_summary in data.items():
                    formatted_output.append(f"  --- {key.replace('_', ' ').title()} ---")
                    if not dist_summary or ('histogram' not in dist_summary and 'percentiles' not in dist_summary):
                        formatted_output.append("    No data available.\n")
                        continue

                    formatted_output.append("    Percentiles:")
                    for p_key, p_val in dist_summary['percentiles'].items():
                        formatted_output.append(f"      {p_key}: {p_val:.4f}")
                    
                    formatted_output.append("    Histogram (first 3 & last 3 bins):")
                    if dist_summary['histogram']:
                        for i, (start, end, count) in enumerate(dist_summary['histogram'][:3]):
                            formatted_output.append(f"      Bin {i+1}: [{start:.4f}, {end:.4f}): Count={count:.2f}")
                        if len(dist_summary['histogram']) > 6:
                            formatted_output.append("      ...")
                        for i, (start, end, count) in enumerate(dist_summary['histogram'][-3:]):
                            if len(dist_summary['histogram']) - 3 + i >= 3: # Avoid printing same bins if less than 6 total
                                formatted_output.append(f"      Bin {len(dist_summary['histogram']) - 3 + i + 1}: [{start:.4f}, {end:.4f}): Count={count:.2f}")
                    else:
                        formatted_output.append("      No histogram data.")
                    formatted_output.append("") # Newline for spacing
            formatted_output.append("\n") # Newline after each main category

        return "\n".join(formatted_output)

    def export_summary(self, num_bins: int = 20, p_list: List[float] = None, output_path: str = None):
        """
        Prints the statistical summary to console and optionally to a file.
        
        Args:
            num_bins: Number of bins for the histograms in the summary.
            p_list: List of percentiles (as floats, e.g., 0.5 for 50%) to calculate.
            output_path: Optional file path to save the formatted summary.
        """
        if p_list is None:
            p_list = [0.5, 0.9, 0.99, 0.999]

        # Ensure summary is generated
        if self.summary is None or self.p_list != p_list: # Re-generate if p_list changed
            self.get_summary(num_bins, p_list)
        
        formatted_summary = self._format_summary_for_print(self.summary, p_list)
        
        print(formatted_summary)

        if output_path:
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(formatted_summary)
                print(f"\nFormatted summary successfully saved to: {output_path}")
            except IOError as e:
                print(f"Error saving formatted summary to {output_path}: {e}")

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
            'part_realloc_curr': dict(self.part_realloc_curr), # Convert defaultdict to dict
            'part_realloc_num': dict(self.part_realloc_num),   # Convert defaultdict to dict
            'system_realloc_cost': self.system_realloc_cost,
            'dist_per_task_ft': {k: v.to_dict() for k, v in self.dist_per_task_ft.items()},
            'dist_per_task_realloc': {k: v.to_dict() for k, v in self.dist_per_task_realloc.items()},
            'dist_per_part_realloc': {k: v.to_dict() for k, v in self.dist_per_part_realloc.items()},
            'dist_overall_realloc': self.dist_overall_realloc.to_dict(),
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
            {k: TDigestStreamingHistogram.from_dict(v) for k, v in state['dist_per_part_realloc'].items()}
        )
        collector.dist_overall_realloc = TDigestStreamingHistogram.from_dict(state['dist_overall_realloc'])
        
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