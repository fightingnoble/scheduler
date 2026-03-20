#!/usr/bin/env python3
"""
调试脚本：验证在 full mode 下，physical graph 中 sink 节点的前驱数量
"""

import math
from collections import defaultdict

class SimpleGraph:
    """简单的图结构用于测试"""
    def __init__(self):
        self.nodes = set()
        self.edges = []  # (pred, succ)
        self.pred_dict = defaultdict(list)  # succ -> [pred1, pred2, ...]
    
    def add_node(self, node):
        self.nodes.add(node)
    
    def add_edge(self, pred, succ):
        self.edges.append((pred, succ))
        self.pred_dict[succ].append(pred)
    
    def predecessors(self, node):
        return self.pred_dict.get(node, [])

def build_node_relationship_test(G, freq_A:int, freq_B:int, S_A:int, S_B:int, A_n:str, B_n:str, dispatch_mode:str='repeat'):
    """简化的 build_node_relationship 函数用于测试"""
    for i in range(S_A):
        A_node = f"{A_n}_{i}"
        G.add_node(A_node)
        for j in range(S_B):
            B_node = f"{B_n}_{j}"
            G.add_node(B_node)
            for k in range(math.ceil(freq_A / S_A)):
                A_data_idx = i * math.ceil(freq_A / S_A) + k
                for l in range(math.ceil(freq_B / S_B)):
                    B_data_idx = j * math.ceil(freq_B / S_B) + l
                    if freq_A > freq_B:
                        if A_data_idx == int(B_data_idx / freq_B * freq_A):
                            G.add_edge(A_node, B_node)
                    else:
                        assert dispatch_mode in ['interleave', 'repeat']
                        if dispatch_mode == 'interleave':
                            if A_data_idx == B_data_idx % freq_A:
                                G.add_edge(A_node, B_node)
                        elif dispatch_mode == 'repeat':
                            if A_data_idx == int(B_data_idx / freq_B * freq_A):
                                G.add_edge(A_node, B_node)
    return G

# 模拟场景
print("=" * 80)
print("测试场景：在 full mode 下，logical graph 中 op1 -> sink_0")
print("=" * 80)

# 假设参数
f_gcd = 1
op1_freq = 30
op1_factor = 3  # freq_division_factor
op1_copy_n = 1  # thread_scaling_factor
sink_freq = 30
sink_factor = 30  # 在 full mode 下，factor = freq
sink_copy_n = 1

print(f"\n逻辑图中：")
print(f"  op1: freq={op1_freq}, factor={op1_factor}, copy_n={op1_copy_n}")
print(f"  sink_0: freq={sink_freq}, factor={sink_factor}, copy_n={sink_copy_n}")
print(f"  逻辑图边: op1 -> sink_0 (sink_0 有 1 个前驱)")

# 创建 physical graph
physical_graph = SimpleGraph()

# 模拟 creat_physical_graph 中第 438-443 行的逻辑
for pred_copy_j in range(op1_copy_n):
    for succ_copy_j in range(sink_copy_n):   
        build_node_relationship_test(
            physical_graph, 
            op1_freq, sink_freq, 
            op1_factor, sink_factor,
            f"op1_{pred_copy_j}",
            f"sink_0_{succ_copy_j}", 
            'repeat'
        )

print(f"\n物理图中：")
op1_nodes = [n for n in physical_graph.nodes if n.startswith('op1')]
sink_nodes = [n for n in physical_graph.nodes if n.startswith('sink_0')]
print(f"  op1 的物理节点：{sorted(op1_nodes)}")
print(f"  sink_0 的物理节点数量：{len(sink_nodes)}")

# 检查每个 sink 节点的前驱数量
print(f"\n检查每个 sink 物理节点的前驱数量（只显示前10个和违规的）：")
violation_count = 0
for idx, sink_node in enumerate(sorted(sink_nodes)):
    preds = physical_graph.predecessors(sink_node)
    pred_count = len(preds)
    if pred_count != 1:
        print(f"  ❌ {sink_node}: {pred_count} 个前驱 {preds}")
        violation_count += 1
    elif idx < 10:  # 只显示前10个正常的
        print(f"  ✓ {sink_node}: {pred_count} 个前驱")

print(f"\n{'='*80}")
if violation_count > 0:
    print(f"❌ 发现问题！有 {violation_count} 个 sink 节点违反了'唯一前驱'约束")
    print(f"\n问题原因分析：")
    print(f"  在 full mode 下，sink 节点会按 freq 被复制成 {sink_factor} 个物理节点")
    print(f"  但是 op1 只按 factor={op1_factor} 被复制成 {op1_factor} 个物理节点")
    print(f"  build_node_relationship 会建立多对多的连接关系")
    print(f"  导致部分 sink 物理节点有多个前驱节点")
else:
    print(f"✓ 所有 sink 节点都满足'唯一前驱'约束")
print(f"{'='*80}")

