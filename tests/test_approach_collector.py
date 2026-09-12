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
import pytest

# Import the class to be tested
from approach_collector import StatisticsCollector

# ================= Mocks and Fixtures =================

class MockMyGraph:
    """A mock graph class to simulate the behavior of MyGraph for testing."""
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
    
    def successors(self, node: str) -> List[str]:
        """Returns successors for a given node."""
        successors_map = {
            'op1_0': ['op2_0'],
            'op2_0': ['sink1_0', 'sink2_0'],
            'sink1_0': [],
            'sink2_0': []
        }
        return successors_map.get(node, [])

@pytest.fixture
def mock_graph() -> MockMyGraph:
    """Pytest fixture to provide a MockMyGraph instance."""
    return MockMyGraph()

@pytest.fixture
def collector() -> StatisticsCollector:
    """Pytest fixture to provide a clean StatisticsCollector instance for each test."""
    return StatisticsCollector(delta=0.01, K=25)

@pytest.fixture(autouse=True)
def mock_plotting(monkeypatch):
    """Auto-used fixture to mock matplotlib plotting functions to avoid GUI popups."""
    def mock_show(*args, **kwargs):
        pass
    # Mock savefig as well to prevent file creation during most tests
    def mock_savefig(*args, **kwargs):
        pass
    
    # Use try...except to avoid errors if matplotlib is not installed
    try:
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", mock_show)
        monkeypatch.setattr(plt.Figure, "savefig", mock_savefig)
    except ImportError:
        pass


# ================= Test Cases =================

def test_basic_functionality(collector: StatisticsCollector, mock_graph: MockMyGraph):
    """Test basic recording functionalities."""
    print("\n--- Testing Basic Functionality ---")
    
    # 1. Initialize partitions
    collector.init_partition_stats('part1', 10, 1.5)
    collector.init_partition_stats('part2', 8, 2.0)
    assert collector.total_pwr == (10 * 1.5) + (8 * 2.0)
    
    # 2. Record task starts
    collector.record_task_start('op1_0', 10.0)
    collector.record_task_start('op2_0', 20.0)
    assert 'op1_0' in collector.task_at and 'op2_0' in collector.task_at
    
    # 3. Record reallocation
    collector.record_realloc('part1', 0.5, ['op1_0'])
    collector.record_realloc_num('part1', ['op1_0'])
    collector.record_realloc('part2', 0.3, ['op2_0'])
    collector.record_realloc_num('part2', ['op2_0'])
    assert collector.task_realloc_curr['op1_0'] == 0.5
    assert collector.task_realloc_num['op2_0'] == 1
    assert collector.part_realloc_curr['part1'] == 0.5
    
    # 4. Record compute progress
    collector.record_compute_progress('op1_0', 5.0)
    collector.record_compute_progress('op2_0', 8.0)
    assert collector.task_compute_curr['op1_0'] == 5.0
    
    # 5. Record task finish and check propagation
    collector.record_task_finish(mock_graph, 'op1_0', 25.0)
    assert 'op1_0' not in collector.task_at # should be cleared
    assert collector.task_realloc_curr['op2_0'] == 0.5 + 0.3 # propagated + original
    
    collector.record_task_finish(mock_graph, 'op2_0', 35.0)
    assert collector.task_realloc_curr['sink1_0'] > 0
    
    # 6. Record E2E finish
    collector.record_e2e_finish(mock_graph, 'sink1_0', 50.0)
    collector.record_e2e_finish(mock_graph, 'sink2_0', 55.0)
    
    assert collector.dist_per_task_ft['sink1'].total_processed_count > 0
    print("✓ Basic functionality test passed.")


def test_hyperperiod_statistics(collector: StatisticsCollector):
    """Test statistics aggregation at hyperperiod boundaries."""
    print("\n--- Testing Hyperperiod Statistics ---")
    collector.init_partition_stats('part1', 100, 1.0) # total_pwr = 100
    
    # 1. Record periodic data
    collector.record_period_load_arrival(100.0)
    collector.record_period_load_arrival(150.0)
    assert collector.hp_total_load_curr == 250.0
    
    collector.record_idle_capacity(20.0, "S")
    collector.record_idle_capacity(15.0, "S")
    collector.record_idle_capacity(10.0, "B")  # Should be ignored
    assert collector.hp_idle_curr == 35.0
    
    miss_tasks = [('task1', 5.0), ('task2', 8.0)]
    collector.record_miss(miss_tasks)
    assert collector.hp_miss_curr == 13.0
    assert collector.hp_miss_count == 2
    
    # 2. Forward hyperperiod
    collector.forward_hyperperiod(T_hp=1.0)
    
    # 3. Verify distributions
    assert collector.dist_overall_total_load.get_mean() == 250.0
    assert collector.dist_overall_idle.get_mean() == 35.0 / 100.0
    assert collector.dist_overall_miss.get_mean() == 13.0 / 100.0
    assert collector.dist_overall_miss_count.get_mean() == 2
    
    # 4. Verify accumulators are reset
    assert collector.hp_total_load_curr == 0.0
    assert collector.hp_idle_curr == 0.0
    assert collector.hp_miss_curr == 0.0
    
    print("✓ Hyperperiod statistics test passed.")


def test_motiv_exp_specific_stats(collector: StatisticsCollector, mock_graph: MockMyGraph):
    """Test the specific statistics gathering for Motiv-Exp cases."""
    print("\n--- Testing Motiv-Exp Specific Statistics ---")
    collector.init_partition_stats('p1', 100, 1) # total_pwr = 100, denom=100
    
    # Populate data for one period
    collector.hp_idle_curr = 10
    collector.hp_miss_curr = 5
    collector.hp_miss_count = 1
    collector.system_realloc_cost = 2
    
    # Add data for latency breakdown
    collector.record_task_start('op1_0', 0)
    collector.record_compute_progress('op1_0', 30)
    collector.record_realloc('p1', 10, ['op1_0'])
    collector.record_realloc_num('p1', ['op1_0'])  # Add this missing call
    collector.record_task_finish(mock_graph, 'op1_0', 50) # op1 propagates to op2
    
    # Record E2E finish with proper task start time and compute time
    collector.record_task_start('sink1_0', 0)  # Start time for E2E calculation
    collector.record_compute_progress('sink1_0', 30)  # Add compute time for sink task
    collector.record_realloc('p1', 10, ['sink1_0'])  # Add realloc overhead for sink task
    collector.record_realloc_num('p1', ['sink1_0'])  # Add realloc number for sink task
    collector.record_e2e_finish(mock_graph, 'sink1_0', 100) # e2e latency = 100-0=100
    
    collector.forward_hyperperiod(T_hp=1.0)

    # Test Case 1 stats
    case1_stats = collector.get_motiv_case1_stats()
    assert np.isclose(case1_stats['idle_mean_ratio'], 10/100)
    assert np.isclose(case1_stats['miss_mean_ratio'], 5/100)
    
    # Test Case 2 stats
    case2_stats = collector.get_motiv_case2_stats()
    breakdown = case2_stats['latency_breakdown']['overall']
    # Constraint is ddl-offset = 300-50 = 250
    assert np.isclose(breakdown['exec_ratio'], 30/250)
    assert np.isclose(breakdown['realloc_ratio'], 10/250)
    
    # Test Case 3 stats (binned)
    collector.set_motiv3_mode('binned')
    collector.binning_warmup_period = 1 # to finalize bins quickly
    collector.forward_hyperperiod() # a second period to finalize bins
    case3_stats = collector.get_motiv_case3_stats(percentile=0.99)
    assert case3_stats['mode'] == 'binned' and 'spearman_rho' in case3_stats
    
    # Test Case 3 stats (raw)
    collector.set_motiv3_mode('raw')
    collector.hp_total_load_curr = 100
    collector.hp_worst_e2e_curr = 10
    collector.forward_hyperperiod()
    case3_raw_stats = collector.get_motiv_case3_stats()
    assert case3_raw_stats['mode'] == 'raw' and case3_raw_stats['raw_data_count'] == 1
    
    print("✓ Motiv-Exp specific stats test passed.")


def test_summary_generation(collector: StatisticsCollector):
    """Test generation of various statistical summaries."""
    print("\n--- Testing Summary Generation ---")
    
    # 1. Generate full summary on an empty collector
    full_summary_empty = collector.get_full_summary()
    assert isinstance(full_summary_empty, dict)
    assert len(full_summary_empty['distribution_summary']) > 0
    assert len(full_summary_empty['adaptive_binning_summary']) == 0 # Bins not finalized
    
    # 2. Add some data
    collector.init_partition_stats('p1', 100, 1.0)
    collector.hp_idle_curr = 10.0  # Set idle capacity
    collector.forward_hyperperiod()  # This will add 0.1 to dist_overall_idle
    
    # 3. Generate summary with data
    full_summary = collector.get_full_summary()
    dist_summary = full_summary['distribution_summary']
    assert dist_summary['overall_idle_time']['total_processed_count'] == 1
    
    # 4. Test other summary types
    breakdown = collector.get_latency_breakdown_avg_ratio()
    assert isinstance(breakdown, dict)
    util_ratios = collector.get_utilization_avg_ratio()
    assert np.isclose(util_ratios['idle_mean_ratio'], 0.1)
    
    print("✓ Summary generation test passed.")


def test_formatted_output(collector: StatisticsCollector):
    """Test the formatted string output for each Motiv case."""
    print("\n--- Testing Formatted Output ---")
    
    case1_output = collector.format_motiv_case1_output()
    case2_output = collector.format_motiv_case2_output()
    case3_output = collector.format_motiv_case3_output()
    
    assert "Motiv-Exp-1" in case1_output
    assert "Motiv-Exp-2" in case2_output
    assert "Motiv-Exp-3" in case3_output
    
    print("✓ Formatted output test passed.")


def test_save_and_load_state(collector: StatisticsCollector, tmp_path):
    """Test saving the collector's state to a file and loading it back."""
    print("\n--- Testing Save and Load State ---")
    
    # 1. Populate collector with data
    collector.init_partition_stats('part1', 10, 1.5)
    collector.record_task_start('op1_0', 10.0)
    collector.forward_hyperperiod()
    
    test_file_path = tmp_path / "temp_test_state.json"
    
    # 2. Save the state
    collector.save_state(str(test_file_path))
    assert os.path.exists(test_file_path)
    
    # 3. Load the state into a new instance
    loaded_collector = StatisticsCollector.load_state(str(test_file_path))
    assert loaded_collector is not None
    
    # 4. Verify data integrity
    assert collector._delta == loaded_collector._delta
    assert collector.total_pwr == loaded_collector.total_pwr
    assert collector.dist_overall_total_load.get_mean() == loaded_collector.dist_overall_total_load.get_mean()
    
    print("✓ Save and load state test passed.")


def test_edge_cases(collector: StatisticsCollector):
    """Test edge cases like empty data and invalid inputs."""
    print("\n--- Testing Edge Cases ---")
    
    # 1. Test empty data handling
    empty_collector = StatisticsCollector()
    empty_stats = empty_collector.get_motiv_case1_stats()
    assert empty_stats['idle_mean_ratio'] == 0.0, "Should be 0 for empty collector"
    
    # 2. Test invalid system state for idle capacity (should be ignored)
    empty_collector.record_idle_capacity(10.0, "invalid_state")
    assert empty_collector.hp_idle_curr == 0.0
    
    # 3. Test negative value handling (should be ignored)
    empty_collector.record_period_load_arrival(-10.0)
    assert empty_collector.hp_total_load_curr == 0.0
    empty_collector.record_compute_progress('task1', -5.0)
    assert empty_collector.task_compute_curr['task1'] == 0.0
    
    print("✓ Edge cases test passed.")


def run_all_tests():
    """Function to run all tests using pytest.main."""
    print("=" * 80)
    print("Running StatisticsCollector Comprehensive Test Suite")
    print("=" * 80)
    
    # We need to run pytest on this file itself.
    # __file__ gives the path to the current file.
    result = pytest.main([__file__])
    
    if result == pytest.ExitCode.OK:
        print("\n" + "=" * 80)
        print("All tests passed successfully!")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print(f"Test suite failed with exit code: {result}")
        print("=" * 80)
        return False

if __name__ == "__main__":
    # This allows running the test script directly
    run_all_tests()
