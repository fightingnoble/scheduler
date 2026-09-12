#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MyGraph.duplicate_for_hyperperiod 函数
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approach_def import MyGraph

def test_duplicate_for_hyperperiod():
    """测试超周期复制功能"""
    
    # 创建测试图
    srcs = {
        'src1': ['op1', 'op2'],
        'src2': ['op3']
    }
    
    ops = {
        'op1': ['sink1'],
        'op2': ['sink2'],
        'op3': ['sink1']
    }
    
    sinks = ['sink1', 'sink2']
    
    task_attr = {
        'op1': {'exp_comp_t': 10.0, 'ert': 50.0, 'ddl': 100.0, 'var_factor': [0, 1]},
        'op2': {'exp_comp_t': 15.0, 'ert': 60.0, 'ddl': 120.0, 'var_factor': [0, 1]},
        'op3': {'exp_comp_t': 20.0, 'ert': 70.0, 'ddl': 140.0, 'var_factor': [0, 1]}
    }
    
    src_attr = {
        'src1': {'exp_comp_t': 5.0, 'offset': 0.0, 'comp_ratio': .5, 'freq': 1.0},
        'src2': {'exp_comp_t': 8.0, 'offset': 10.0, 'comp_ratio': .2, 'freq': 1.0}
    }
    
    sink_attr = {
        'sink1': {'ddl': 200.0},
        'sink2': {'ddl': 250.0}
    }
    
    # 创建图实例
    G = MyGraph(srcs, ops, sinks, task_attr, src_attr, sink_attr)
    
    print("原始图信息:")
    print(f"节点数量: {len(G.nodes())}")
    print(f"边数量: {len(G.edges())}")
    print(f"srcs: {G.srcs}")
    print(f"ops: {G.ops}")
    print(f"sinks: {G.sinks}")
    print(f"ddl_map: {G.ddl_map}")
    print(f"ert_map: {G.ert_map}")
    print(f"offset_map: {G.offset_map}")
    print()
    
    # 复制第一个超周期
    print("复制第一个超周期 (hp_idx=1, T_hp=0.1):")
    G.duplicate_for_hyperperiod(hp_idx=1, seed=1234, T_hp=0.1)
    
    print(f"复制后节点数量: {len(G.nodes())}")
    print(f"复制后边数量: {len(G.edges())}")
    print(f"复制后srcs: {G.srcs}")
    print(f"复制后ops: {G.ops}")
    print(f"复制后sinks: {G.sinks}")
    print(f"复制后ddl_map: {G.ddl_map}")
    print(f"复制后ert_map: {G.ert_map}")
    print(f"复制后offset_map: {G.offset_map}")
    print()
    
    # 检查时间偏移
    print("检查时间偏移:")
    for node in G.nodes():
        if node.endswith('_1'):  # 复制的节点
            if 'offset' in G.nodes[node]:
                print(f"{node}: offset = {G.nodes[node]['offset']}")
            if 'ert' in G.nodes[node]:
                print(f"{node}: ert = {G.nodes[node]['ert']}")
            if 'ddl' in G.nodes[node]:
                print(f"{node}: ddl = {G.nodes[node]['ddl']}")
    
    # 复制第二个超周期
    print("\n复制第二个超周期 (hp_idx=2, T_hp=0.1):")
    G.duplicate_for_hyperperiod(hp_idx=2, seed=5678, T_hp=0.1)
    
    print(f"最终节点数量: {len(G.nodes())}")
    print(f"最终边数量: {len(G.edges())}")
    print(f"最终srcs: {G.srcs}")
    print(f"最终ops: {G.ops}")
    print(f"最终sinks: {G.sinks}")

if __name__ == "__main__":
    test_duplicate_for_hyperperiod() 