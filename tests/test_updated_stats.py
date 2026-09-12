#!/usr/bin/env python3
"""
测试更新后的统计功能
"""

import pytest

from approach.approach_collector import StatisticsCollector


class _TestGraph:
    def __init__(self, process_ids):
        self.nodes = {process_id: {"offset": 0.0} for process_id in process_ids}
        self.ddl_map = {process_id: 10.0 for process_id in process_ids}

    @staticmethod
    def successors(_process_id):
        return ()


def test_updated_statistics():
    print("测试更新后的统计功能...")
    
    # 创建统计收集器
    stats = StatisticsCollector()
    
    # 初始化分区统计
    stats.init_partition_stats("acc_p0", 100, 1.0)
    stats.init_partition_stats("acc_p1", 200, 2.0)
    
    # 模拟一些数据
    # 任务完成时间
    process_ids = [
        process_id
        for i in range(10)
        for process_id in (f"task1_{i}", f"sink1_{i}")
    ]
    graph = _TestGraph(process_ids)
    stats.set_task_cnt(len(process_ids))
    for i in range(10):
        for process_id, finish_time in (
            (f"task1_{i}", 1.0 + i * 0.1),
            (f"sink1_{i}", 2.0 + i * 0.1),
        ):
            stats.record_task_start(process_id, 0.0)
            stats.record_task_finish(graph, process_id, finish_time)
    
    # 重分配开销
    for i in range(5):
        stats.record_realloc("acc_p0", 0.1 + i * 0.01, ["task1"])
        stats.record_realloc_num("acc_p0", ["task1"])
        stats.record_realloc("acc_p1", 0.2 + i * 0.02, ["task2"])
        stats.record_realloc_num("acc_p1", ["task2"])
    
    # 周期推进
    stats.forward_hyperperiod()
    
    # 测试不同的分位数列表
    p_list1 = [0.5, 0.9, 0.99]
    p_list2 = [0.25, 0.5, 0.75, 0.95]
    
    summary1 = stats.get_full_summary(num_bins=10, p_list=p_list1)
    summary2 = stats.get_full_summary(num_bins=15, p_list=p_list2)
    assert summary1["distribution_summary"]
    assert summary2["distribution_summary"]
    
    print("测试完成！")
    assert stats.dist_per_task_ft["task1"].total_processed_count == 10
    assert stats.dist_per_task_ft["sink1"].total_processed_count == 10
    assert stats.dist_overall_realloc.total_processed_count == 1


@pytest.mark.xfail(
    strict=True,
    reason="export_summary calls plot_load_latency_binned with its retired signature",
)
def test_export_summary_uses_current_plot_signature(tmp_path):
    stats = StatisticsCollector()
    stats.init_partition_stats("acc_p0", 1, 1.0)
    stats.set_task_cnt(1)
    stats.set_path("stat", str(tmp_path / "summary.txt"))
    stats.set_path("motiv3", str(tmp_path / "load-latency.pdf"))

    stats.export_summary()
