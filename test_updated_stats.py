#!/usr/bin/env python3
"""
测试更新后的统计功能
"""

from approach_collector import StatisticsCollector
from ref_tdigest import TDigestStreamingHistogram

def test_updated_statistics():
    print("测试更新后的统计功能...")
    
    # 创建统计收集器
    stats = StatisticsCollector()
    
    # 初始化分区统计
    stats.init_partition_stats("acc_p0", 100, 1.0)
    stats.init_partition_stats("acc_p1", 200, 2.0)
    
    # 模拟一些数据
    # 任务完成时间
    for i in range(10):
        stats.record_task_finish(None, f"task1_{i}", 1.0 + i * 0.1)
        stats.record_task_finish(None, f"sink1_{i}", 2.0 + i * 0.1)
    
    # 重分配开销
    for i in range(5):
        stats.record_realloc("acc_p0", 0.1 + i * 0.01, ["task1"])
        stats.record_realloc("acc_p1", 0.2 + i * 0.02, ["task2"])
    
    # 周期推进
    stats.forward_hyperperiod()
    
    # 测试不同的分位数列表
    p_list1 = [0.5, 0.9, 0.99]
    p_list2 = [0.25, 0.5, 0.75, 0.95]
    
    print("\n测试分位数列表 1:", p_list1)
    stats.export_summary(num_bins=10, p_list=p_list1)
    
    print("\n测试分位数列表 2:", p_list2)
    stats.export_summary(num_bins=15, p_list=p_list2)
    
    print("测试完成！")

if __name__ == "__main__":
    test_updated_statistics() 