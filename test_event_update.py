#!/usr/bin/env python3
"""
测试动态事件更新功能的脚本
"""

from approach_util import MyGraph, GlobalEvent_t
from approach_plot import instantiate_processors, update_processors_for_hyperperiod

def test_dynamic_event_update():
    """测试动态事件更新功能"""
    print("=== 测试动态事件更新功能 ===")
    
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
    
    # 创建初始事件队列
    initial_events = []
    for node in G.srcs:
        t = G.nodes[node]['offset']
        initial_events.append((t, "external"))
        print(f"初始事件: {node} 在时间 {t}")
    
    # 创建事件队列
    event_queue = GlobalEvent_t(initial_events)
    
    print(f"\n初始事件队列:")
    for event in event_queue.event_t:
        print(f"  {event}")
    
    # 测试超周期复制和事件更新
    T_hp = 0.1
    original_srcs = ['S1', 'S2', 'S3']
    
    print(f"\n=== 执行超周期复制和事件更新 ===")
    
    # 复制第一个超周期
    G.duplicate_for_hyperperiod(0, 0, T_hp)
    G.update_events_for_hyperperiod(0, T_hp, event_queue, original_srcs)
    
    print(f"第一个超周期后的事件队列:")
    for event in event_queue.event_t:
        print(f"  {event}")
    
    # 复制第二个超周期
    G.duplicate_for_hyperperiod(1, 0, T_hp)
    G.update_events_for_hyperperiod(1, T_hp, event_queue, original_srcs)
    
    print(f"第二个超周期后的事件队列:")
    for event in event_queue.event_t:
        print(f"  {event}")
    
    # 复制第三个超周期
    G.duplicate_for_hyperperiod(2, 0, T_hp)
    G.update_events_for_hyperperiod(2, T_hp, event_queue, original_srcs)
    
    print(f"第三个超周期后的事件队列:")
    for event in event_queue.event_t:
        print(f"  {event}")
    
    print(f"\n=== 测试完成 ===")

if __name__ == "__main__":
    test_dynamic_event_update() 