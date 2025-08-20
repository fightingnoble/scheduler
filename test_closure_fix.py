#!/usr/bin/env python3
"""
测试闭包变量捕获问题修复的脚本
"""

import numpy as np
from scipy.stats import truncnorm, poisson

def test_closure_problem():
    """测试闭包变量捕获问题"""
    print("=== 测试闭包变量捕获问题 ===")
    
    # 模拟原始的问题代码
    print("1. 测试有问题的代码（闭包捕获引用）:")
    rng_fn_list_bad = {}
    
    # 模拟节点数据
    nodes_data = {
        'node1': {'exp_comp_t': 0.1, 'var_factor': 1},
        'node2': {'exp_comp_t': 0.2, 'var_factor': 1},
        'node3': {'exp_comp_t': 0.3, 'var_factor': 1}
    }
    
    # 有问题的代码：直接使用变量引用
    for node, data in nodes_data.items():
        exp_comp_t = data['exp_comp_t']
        var_t_fn = lambda rng: exp_comp_t  # 问题：捕获的是引用
        rng_fn_list_bad[node] = var_t_fn
    
    print("   有问题的lambda函数结果:")
    for node, fn in rng_fn_list_bad.items():
        result = fn(None)
        print(f"     {node}: {result}")
    
    # 修复后的代码：使用默认参数冻结值
    print("\n2. 测试修复后的代码（使用默认参数）:")
    rng_fn_list_good = {}
    
    for node, data in nodes_data.items():
        exp_comp_t = data['exp_comp_t']
        var_t_fn = lambda rng, exp_comp_t=exp_comp_t: exp_comp_t  # 修复：使用默认参数
        rng_fn_list_good[node] = var_t_fn
    
    print("   修复后的lambda函数结果:")
    for node, fn in rng_fn_list_good.items():
        result = fn(None)
        print(f"     {node}: {result}")
    
    # 测试更复杂的情况（poisson分布）
    print("\n3. 测试poisson分布的情况:")
    rng_fn_list_poisson = {}
    
    for node, data in nodes_data.items():
        exp_comp_t = data['exp_comp_t']
        k = data['var_factor']
        
        if k == 1:
            var_t_fn = lambda rng, exp_comp_t=exp_comp_t: exp_comp_t
        else:
            lambda_ld = 1
            ini_probs = np.zeros(k+1)
            for j in range(k+1):
                ini_probs[j] = poisson.pmf(j, lambda_ld)
            ini_probs = ini_probs/ini_probs.sum()
            var_t_fn = lambda rng, exp_comp_t=exp_comp_t, ini_probs=ini_probs: np.random.choice(k+1, p=ini_probs, replace=False) * exp_comp_t
        
        rng_fn_list_poisson[node] = var_t_fn
    
    print("   Poisson分布的lambda函数结果:")
    for node, fn in rng_fn_list_poisson.items():
        result = fn(None)
        print(f"     {node}: {result}")
    
    print("\n=== 测试完成 ===")

def test_truncated_normal():
    """测试截断正态分布的情况"""
    print("\n=== 测试截断正态分布 ===")
    
    rng_fn_list_normal = {}
    
    # 模拟src节点的数据
    src_nodes_data = {
        'src1': {'comp_ratio': 0.1, 'freq': 1.0},
        'src2': {'comp_ratio': 0.2, 'freq': 1.0},
        'src3': {'comp_ratio': 0.3, 'freq': 1.0}
    }
    
    for node, data in src_nodes_data.items():
        half_len = data['comp_ratio'] / data['freq']
        ZScore = 3
        loc = half_len
        scale = half_len / ZScore
        a, b = -ZScore, ZScore
        
        # 使用默认参数冻结所有变量的值
        var_t_fn = lambda rng, half_len=half_len, scale=scale, a=a, b=b: truncnorm.rvs(a, b, loc=half_len, scale=scale, random_state=rng)
        rng_fn_list_normal[node] = var_t_fn
    
    print("   截断正态分布的lambda函数结果:")
    for node, fn in rng_fn_list_normal.items():
        # 使用固定的随机种子进行测试
        rng = np.random.RandomState(42)
        result = fn(rng)
        print(f"     {node}: {result:.6f}")

if __name__ == "__main__":
    test_closure_problem()
    test_truncated_normal() 