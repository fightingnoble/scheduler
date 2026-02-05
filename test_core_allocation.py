import numpy as np
import copy
from collections import OrderedDict

# ==============================================================================
# 新的向量化实现 (来自 utils.py)
# ==============================================================================
from utils import vectorized_core_allocation

# ==============================================================================
# 用于测试的 Mock 对象
# ==============================================================================
class MockBin:
    def __init__(self, num_resources, bin_id):
        self.num_resources = num_resources
        self.id = bin_id

    def __repr__(self):
        return f"Bin(id={self.id}, cores={self.num_resources})"

# ==============================================================================
# 旧的实现 (从 sim_main.py 和 utils.py 复制过来用于对比)
# ==============================================================================
def core_distr(rsc_map, score_dict, curr_aval_rsc, order_fn=lambda x:x[1], sort=True):
    sorted_score_dict = OrderedDict(sorted(score_dict.items(), key=order_fn) if sort else score_dict.items())
    cum_score_reverse = np.cumsum(list(reversed(sorted_score_dict.values())))
    if cum_score_reverse[-1] == 0: # 避免除零
        return
    cum_size = [curr_aval_rsc * s / cum_score_reverse[-1] for s in cum_score_reverse]
    cum_size[-1] = curr_aval_rsc
    for i, pid in enumerate(reversed(sorted_score_dict.keys())):
        if i == 0:
            size = int(cum_size[0])
            rsc_map[pid] += size
            cum_size[0] = size
        else:
            size = int(cum_size[i] - cum_size[i - 1])
            rsc_map[pid] += size
            cum_size[i] = size + cum_size[i - 1]

def override_total_cores_old(bin_list, target_total_cores):
    if target_total_cores <= 0:
        return
    if len(bin_list) == 1:
        bin_list[0].num_resources = target_total_cores
        return
    current_total = sum(b.num_resources for b in bin_list)
    if current_total == 0:
        avg = max(1, target_total_cores // len(bin_list))
        for b in bin_list:
            b.num_resources = avg
        diff = target_total_cores - sum(b.num_resources for b in bin_list)
        idx = 0
        while diff > 0:
            bin_list[idx % len(bin_list)].num_resources += 1
            idx += 1
            diff -= 1
        return
    scaled = []
    for b in bin_list:
        scaled.append(max(1, int(round(b.num_resources / current_total * target_total_cores))))
    diff = target_total_cores - sum(scaled)
    idx = 0
    while diff != 0:
        if diff > 0:
            scaled[idx % len(scaled)] += 1
            diff -= 1
        else:
            if scaled[idx % len(scaled)] > 1:
                scaled[idx % len(scaled)] -= 1
                diff += 1
        idx += 1
    for i, b in enumerate(bin_list):
        b.num_resources = scaled[i]

def apply_forced_num_cores_old(bin_list, estimated_num_cores, target):
    if target == estimated_num_cores:
        return target
    if len(bin_list) == 1:
        bin_list[0].num_resources = target
    else:
        score_dict = {_bin.id: _bin.num_resources for _bin in bin_list}
        rsc_map = {_bin.id: _bin.num_resources for _bin in bin_list}
        curr_aval_rsc = target - estimated_num_cores
        core_distr(rsc_map, score_dict, curr_aval_rsc)
        
        # 修正因取整导致的误差
        current_sum = sum(rsc_map.values())
        diff = target - current_sum
        if diff != 0:
            # 简单地加到第一个 bin 上
            keys = list(rsc_map.keys())
            if keys:
                rsc_map[keys[0]] += diff

        for _bin in bin_list:
            _bin.num_resources = rsc_map[_bin.id]
    return target


# ==============================================================================
# 测试函数
# ==============================================================================
def run_test(test_name, initial_cores, target_cores):
    print(f"\n{'='*20} {test_name} {'='*20}")
    print(f"Initial: {[c for _, c in initial_cores]}, Target: {target_cores}")
    
    # --- 旧 override_total_cores ---
    bins1 = [MockBin(c, i) for i, c in initial_cores]
    override_total_cores_old(bins1, target_cores)
    res1 = [b.num_resources for b in bins1]
    
    # --- 旧 apply_forced_num_cores ---
    bins2 = [MockBin(c, i) for i, c in initial_cores]
    estimated = sum(c for _,c in initial_cores)
    apply_forced_num_cores_old(bins2, estimated, target_cores)
    res2 = [b.num_resources for b in bins2]
    
    # --- 新 vectorized_core_allocation ---
    bins3 = [MockBin(c, i) for i, c in initial_cores]
    vectorized_core_allocation(bins3, target_cores)
    res3 = [b.num_resources for b in bins3]
    
    print(f"  Old override: {res1}, Sum: {sum(res1)}")
    print(f"  Old apply   : {res2}, Sum: {sum(res2)}")
    print(f"  New vectorized: {res3}, Sum: {sum(res3)}")
    
    assert sum(res1) == target_cores
    assert sum(res2) == target_cores
    assert sum(res3) == target_cores
    print("All sums match target.")

if __name__ == "__main__":
    # 测试用例 1: 基本缩放
    run_test(
        "Basic Scaling", 
        initial_cores=[(0, 10), (1, 20), (2, 70)], 
        target_cores=150
    )
    
    # 测试用例 2: 从 0 开始分配
    run_test(
        "Allocation from Zero", 
        initial_cores=[(0, 0), (1, 0), (2, 0)], 
        target_cores=10
    )
    
    # 测试用例 3: 包含 0 核心的 bin
    run_test(
        "Bins with Zero Cores", 
        initial_cores=[(0, 10), (1, 0), (2, 90)], 
        target_cores=50
    )
    
    # 测试用例 4: 缩减核心数
    run_test(
        "Scaling Down", 
        initial_cores=[(0, 50), (1, 100), (2, 150)], 
        target_cores=100
    )
    
    # 测试用例 5: 无法被整除的场景
    run_test(
        "Complex Ratio", 
        initial_cores=[(0, 1), (1, 1), (2, 1)], 
        target_cores=10
    )
    
    # 测试用例 6: 大量 bins
    run_test(
        "Many Bins", 
        initial_cores=[(i, 1) for i in range(10)], 
        target_cores=109
    )

    # 测试用例 7: 确保每个 bin 至少一个核心
    run_test(
        "Constraint Min 1 Core",
        initial_cores=[(0, 1000), (1, 1), (2, 1)],
        target_cores=10
    )
