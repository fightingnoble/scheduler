#!/usr/bin/env python3
"""
测试动态映射功能的脚本
"""

from approach.approach_def import MyGraph, Sen_p, Acc_p
from approach_plot import instantiate_processors, update_processors_for_hyperperiod

def test_dynamic_mapping():
    """测试动态映射功能"""
    print("=== 测试动态映射功能 ===")
    
    # 创建简单的测试图
    srcs = ['S1', 'S2', 'S3']
    ops = ['O1', 'O2']
    sinks = ['T1']
    
    # 创建节点属性
    src_attr = {
        'S1': {'type': 'src', 'offset': 0.0, 'exp_comp_t': 0.01, 'base_size': 1},
        'S2': {'type': 'src', 'offset': 0.02, 'exp_comp_t': 0.01, 'base_size': 1},
        'S3': {'type': 'src', 'offset': 0.04, 'exp_comp_t': 0.01, 'base_size': 1}
    }
    
    task_attr = {
        'O1': {'type': 'op', 'exp_comp_t': 0.05, 'base_size': 1, 'ert': 0.0, 'ddl': 0.2},
        'O2': {'type': 'op', 'exp_comp_t': 0.03, 'base_size': 1, 'ert': 0.0, 'ddl': 0.15}
    }
    
    sink_attr = {
        'T1': {'type': 'sink', 'exp_comp_t': 0.01, 'base_size': 1}
    }
    
    # 创建图
    G = MyGraph(srcs, ops, sinks, task_attr, src_attr, sink_attr)
    
    # 添加边
    G.add_edge('S1', 'O1')
    G.add_edge('S2', 'O1')
    G.add_edge('S3', 'O2')
    G.add_edge('O1', 'T1')
    G.add_edge('O2', 'T1')
    
    # 初始化前驱计数
    G.init_rng_fn_list(G.srcs, G.ops)
    
    print(f"初始图状态:")
    print(f"  节点: {list(G.nodes())}")
    print(f"  srcs: {G.srcs}")
    print(f"  ops: {G.ops}")
    print(f"  sinks: {G.sinks}")
    
    # 创建处理器
    from approach.approach_sched import PartitionConfig
    partition_cfg = PartitionConfig(
        num_partitions=1,
        cap_list=[2],
        base_pwr_list=[1.0],
        mapped_node_list=[['O1', 'O2', 'T1']],
        G=G
    )
    
    processors, _ = instantiate_processors(G, partition_cfg, [], policy="glb")
    
    print(f"\n初始处理器状态:")
    for i, proc in enumerate(processors):
        print(f"  处理器 {i}: {proc.id}, mapped_node: {proc.mapped_node}")
    
    # 测试超周期复制
    print(f"\n=== 执行超周期复制 ===")
    G.duplicate_for_hyperperiod(0, 0, 0.1)
    
    print(f"复制后的图状态:")
    print(f"  节点: {list(G.nodes())}")
    print(f"  srcs: {G.srcs}")
    print(f"  ops: {G.ops}")
    print(f"  sinks: {G.sinks}")
    
    # 更新处理器映射
    update_processors_for_hyperperiod(G, processors, 0)
    
    print(f"\n更新后的处理器状态:")
    for i, proc in enumerate(processors):
        print(f"  处理器 {i}: {proc.id}, mapped_node: {proc.mapped_node}")
    
    # 测试第二个超周期
    print(f"\n=== 执行第二个超周期复制 ===")
    G.duplicate_for_hyperperiod(1, 0, 0.1)
    
    print(f"第二个超周期后的图状态:")
    print(f"  节点: {list(G.nodes())}")
    print(f"  srcs: {G.srcs}")
    print(f"  ops: {G.ops}")
    print(f"  sinks: {G.sinks}")
    
    # 更新处理器映射
    update_processors_for_hyperperiod(G, processors, 1)
    
    print(f"\n第二个超周期后的处理器状态:")
    for i, proc in enumerate(processors):
        print(f"  处理器 {i}: {proc.id}, mapped_node: {proc.mapped_node}")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_dynamic_mapping() 