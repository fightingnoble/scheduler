import math
import bisect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Union
import numpy as np

# 定义并联模式的类型
ParallelMode = Literal["list", "lwb", "upb", "range", "none"]
# 定义请求资源和分配结果的类型，兼容标量和Numpy数组
ReqRsc = Union[int, np.ndarray]
AllocResult = Union[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]

@dataclass
class TaskConstraints:
    """
    用于存储任务资源约束的数据结构。
    """
    parallel_mode: ParallelMode = "none"
    core_list: List[int] = field(default_factory=list)
    core_min: int = 1
    core_max: int = 1

    def __post_init__(self):
        """
        在对象初始化后，确保 core_list 是有序且唯一的，便于后续处理。
        """
        if self.core_list:
            self.core_list = sorted(list(set(self.core_list)))

def _get_intersection_endpoints(
    constraints: TaskConstraints, 
    curr_aval_rsc: int
) -> Tuple[Optional[int], Optional[int]]:
    """
    计算任务自身约束区间与系统可用资源区间的交集端点。
    不处理 'list' 模式。

    Args:
        constraints (TaskConstraints): 任务的约束配置。
        curr_aval_rsc (int): 当前系统最大可用资源。

    Returns:
        一个元组 (min_endpoint, max_endpoint)。如果交集为空，则 min_endpoint > max_endpoint。
    """
    # 1. 确定任务自身的约束区间 [task_min, task_max]
    # 我们将 lwb, upb, range 统一转换为 [min_val, max_val] 区间
    # 对于没有上限的模式（lwb, none），其上限就是系统可用资源上限
    task_min, task_max = 1, curr_aval_rsc
    
    mode = constraints.parallel_mode
    if mode in ["lwb", "range"]:
        task_min = constraints.core_min
    if mode in ["upb", "range"]:
        task_max = constraints.core_max

    # 2. 确定系统可用资源区间 [system_min, system_max]
    system_min, system_max = 1, curr_aval_rsc

    # 3. 计算交集 [final_min, final_max]
    final_min = max(task_min, system_min)
    final_max = min(task_max, system_max)
    
    return final_min, final_max

def _find_best_fit(
    req_rsc: int,
    options: Optional[List[int]] = None,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None
) -> Optional[int]:
    """
    根据优先级规则从可行解中找到最优解。
    规则:
    1. 理想值本身
    2. 大于理想值的最小值 (Ceiling)
    3. 最接近理想值的其他值 (此时必然是小于理想值的最大值, Floor)

    Args:
        req_rsc: 理想资源数。
        options: 离散的可行解列表 (用于 list 模式)。
        min_val, max_val: 连续的可行解区间端点。

    Returns:
        最优解，如果无解则返回 None。
    """
    # --- Case 1: 处理离散的列表 (list mode) ---
    if options is not None:
        if not options:
            return None
        
        # 使用二分查找找到插入点
        idx = bisect.bisect_left(options, req_rsc)

        # 规则1: 理想值正好在选项中
        if idx < len(options) and options[idx] == req_rsc:
            return req_rsc

        # 规则2: 存在大于理想值的选项，选其中最小的
        if idx < len(options):
            return options[idx]

        # 规则3: 所有选项都小于理想值，选择最大的那个
        return options[-1]

    # --- Case 2: 处理连续的区间 ---
    if min_val is not None and max_val is not None:
        if min_val > max_val:
            return None # 区间无效

        # 规则1: 理想值落在区间内
        if min_val <= req_rsc <= max_val:
            return req_rsc
        
        # 规则2: 理想值小于区间下限，选下限值
        if req_rsc < min_val:
            return min_val
            
        # 规则3: 理想值大于区间上限，选上限值
        if req_rsc > max_val:
            return max_val
    
    return None


def _allocate_scalar_resource(
    constraints: TaskConstraints,
    curr_aval_rsc: int,
    req_rsc_size: int
) -> Tuple[int, str]:
    """
    为单个标量请求分配资源的核心逻辑。
    """
    # --- 步骤 2 & 3: 求交集并寻找最优解 ---
    final_rsc = None
    mode = constraints.parallel_mode

    if mode == "list":
        # 对 list 模式，先过滤出在可用资源范围内
        valid_options = [opt for opt in constraints.core_list if opt <= curr_aval_rsc]
        final_rsc = _find_best_fit(req_rsc_size, options=valid_options)
    else:
        # 对其他模式，计算相交区间的端点
        min_endpoint, max_endpoint = _get_intersection_endpoints(constraints, curr_aval_rsc)
        final_rsc = _find_best_fit(req_rsc_size, min_val=min_endpoint, max_val=max_endpoint)

    if final_rsc is None:
        return 0, "N/A"

    # --- 步骤 4: 确定约束来源 ---
    if final_rsc == req_rsc_size:
        return final_rsc, "none"

    # 为了判断约束来源，我们计算一下如果没有系统可用资源限制，任务会选择什么解
    # 假设有无限可用资源，重新计算最优解
    unconstrained_rsc = None
    if mode == "list":
        unconstrained_rsc = _find_best_fit(req_rsc_size, options=constraints.core_list)
    else:
        # 假设可用资源非常大
        inf_min, inf_max = _get_intersection_endpoints(constraints, float('inf'))
        unconstrained_rsc = _find_best_fit(req_rsc_size, min_val=inf_min, max_val=inf_max)

    if final_rsc != unconstrained_rsc:
        # 最终解和无系统约束解不一致，说明是可用资源起了决定性作用
        return final_rsc, "available"
    else:
        # 约束来自任务本身
        if mode == "list":
            return final_rsc, "list"
        if final_rsc < req_rsc_size:
            return final_rsc, "upb" # 被迫下调，肯定是上限导致
        if final_rsc > req_rsc_size:
            return final_rsc, "lwb" # 被迫上调，肯定是下限导致

    return final_rsc, "unknown" # 理论上不应到达这里

vectorized_alloc = np.vectorize(
    _allocate_scalar_resource,
    otypes=[np.int64, np.object_]
)

def find_legal(
    constraints: Union[TaskConstraints, List[TaskConstraints]],
    curr_aval_rsc: Union[int, List[int], np.ndarray],
    req_rsc_size: ReqRsc
) -> AllocResult:
    """
    为任务分配资源，兼容标量和Numpy数组输入。

    Args:
        constraints (Union[TaskConstraints, List[TaskConstraints]]): 
            任务的约束配置。可以是单个对象或对象列表。
        curr_aval_rsc (Union[int, List[int], np.ndarray]): 
            当前系统最大可用资源。可以是单个值或列表/数组。
        req_rsc_size (ReqRsc): 
            直接给定的理想资源数。可以是标量或Numpy数组。

    Returns:
        AllocResult: (分配的资源数, 应用的约束类型)。
    """
    req_rsc_size = np.maximum(1, req_rsc_size).astype(int)

    is_scalar_req = np.isscalar(req_rsc_size)
    
    if is_scalar_req:
        if not isinstance(constraints, TaskConstraints) or not np.isscalar(curr_aval_rsc):
            raise ValueError("For scalar req_rsc_size, constraints and curr_aval_rsc must also be scalars.")
        return _allocate_scalar_resource(constraints, curr_aval_rsc, req_rsc_size)
    else:  # array request
        n = len(req_rsc_size)
        
        # 如果 constraints 或 curr_aval_rsc 是标量，则广播它们以匹配请求数组的长度
        constraints_arr = [constraints] * n if isinstance(constraints, TaskConstraints) else constraints
        curr_aval_rsc_arr = np.full(n, curr_aval_rsc) if np.isscalar(curr_aval_rsc) else curr_aval_rsc

        # 验证长度是否匹配
        if len(constraints_arr) != n or len(curr_aval_rsc_arr) != n:
            raise ValueError("Length of constraints, curr_aval_rsc, and req_rsc_size must match for array input.")
            
        return vectorized_alloc(constraints_arr, curr_aval_rsc_arr, req_rsc_size)


if __name__ == '__main__':
    print("--- 开始资源分配测试 ---")
    
    # 场景1: range 模式, 理想值落在区间内
    constraints_range = TaskConstraints(parallel_mode="range", core_min=4, core_max=16)
    result = find_legal(constraints_range, curr_aval_rsc=20, req_rsc_size=8)
    print(f"场景1 (Range, 理想值in): {result}")
    assert result == (8, 'none')

    # 场景2: range 模式, 理想值太高，受 core_max 约束
    result = find_legal(constraints_range, curr_aval_rsc=20, req_rsc_size=25)
    print(f"场景2 (Range, 理想值high): {result}")
    assert result == (16, 'upb')

    # 场景3: range 模式, 理想值太高，受 curr_aval_rsc 约束
    result = find_legal(constraints_range, curr_aval_rsc=10, req_rsc_size=25)
    print(f"场景3 (Range, 可用资源low): {result}")
    assert result == (10, 'available')

    # 场景4: range 模式, 理想值太低，受 core_min 约束
    result = find_legal(constraints_range, curr_aval_rsc=20, req_rsc_size=2)
    print(f"场景4 (Range, 理想值low): {result}")
    assert result == (4, 'lwb')

    # 场景5: list 模式, 寻找大于理想值的最小值
    constraints_list = TaskConstraints(parallel_mode="list", core_list=[2, 4, 8, 16, 32])
    result = find_legal(constraints_list, curr_aval_rsc=32, req_rsc_size=9)
    print(f"场景5 (List, 向上取整): {result}")
    assert result == (16, 'list')

    # 场景6: list 模式, 理想值超过列表最大值
    result = find_legal(constraints_list, curr_aval_rsc=32, req_rsc_size=40)
    print(f"场景6 (List, 超过最大值): {result}")
    assert result == (32, 'list')

    # 场景7: list 模式, 受可用资源限制
    result = find_legal(constraints_list, curr_aval_rsc=10, req_rsc_size=20)
    print(f"场景7 (List, 受可用资源限制): {result}")
    assert result == (8, 'available')
    
    # 场景8: 无可行解 (可用资源 < 列表最小值)
    result = find_legal(constraints_list, curr_aval_rsc=1, req_rsc_size=4)
    print(f"场景8 (List, 无解): {result}")
    assert result == (0, 'N/A')
    
    # 场景9: 无可行解 (任务下限 > 任务上限)
    constraints_invalid = TaskConstraints(parallel_mode="range", core_min=10, core_max=5)
    result = find_legal(constraints_invalid, curr_aval_rsc=20, req_rsc_size=8)
    print(f"场景9 (Range, 无效区间): {result}")
    assert result == (0, 'N/A')

    # 场景10: lwb 模式, 可用资源是瓶颈
    constraints_lwb = TaskConstraints(parallel_mode="lwb", core_min=8)
    result = find_legal(constraints_lwb, curr_aval_rsc=12, req_rsc_size=20)
    print(f"场景10 (LWB, 受可用资源限制): {result}")
    assert result == (12, 'available')

    constraints_range = TaskConstraints(parallel_mode="range", core_min=4, core_max=16)
    assert find_legal(constraints_range, curr_aval_rsc=20, req_rsc_size=8) == (8, 'none')
    assert find_legal(constraints_range, curr_aval_rsc=20, req_rsc_size=25) == (16, 'upb')
    constraints_list = TaskConstraints(parallel_mode="list", core_list=[2, 4, 8, 16, 32])
    assert find_legal(constraints_list, curr_aval_rsc=10, req_rsc_size=20) == (8, 'available')
    print("--- 原有标量测试通过 ---")

    # 场景12: Numpy array input, 标量 constraints 和 curr_aval_rsc (广播)
    print("\n--- Numpy Array Test (Broadcasting) ---")
    req_array = np.array([2, 8, 9, 25, 40])
    constraints_mix = TaskConstraints(parallel_mode="range", core_min=4, core_max=16)
    rsc_array, constr_array = find_legal(constraints_mix, curr_aval_rsc=20, req_rsc_size=req_array)
    
    print(f"场景12 (Numpy Input): req={req_array}")
    print(f"                    rsc={rsc_array}")
    print(f"                    constr={constr_array}")

    expected_rsc = np.array([4, 8, 9, 16, 16])
    expected_constr = np.array(['lwb', 'none', 'none', 'upb', 'upb'])
    
    np.testing.assert_array_equal(rsc_array, expected_rsc)
    np.testing.assert_array_equal(constr_array, expected_constr)
    print("--- Numpy Array Broadcasting Test Passed ---")

    # 场景13: 另一个广播测试
    print("\n--- Numpy Pre-calculated Req Test (Broadcasting) ---")
    pre_calculated_req_array = np.array([1, 9, 17])
    rsc_array_f, constr_array_f = find_legal(constraints_range, 
                                                     curr_aval_rsc=20, 
                                                     req_rsc_size=pre_calculated_req_array)
    print(f"场景13 (Numpy Pre-calculated): req={pre_calculated_req_array}")
    print(f"                             rsc={rsc_array_f}")
    print(f"                             constr={constr_array_f}")
    expected_rsc_f = np.array([4, 9, 16])
    expected_constr_f = np.array(['lwb', 'none', 'upb'])
    np.testing.assert_array_equal(rsc_array_f, expected_rsc_f)
    np.testing.assert_array_equal(constr_array_f, expected_constr_f)
    print("--- Numpy Pre-calculated Req Broadcasting Test Passed ---")

    # 场景14: curr_aval_rsc 和 constraints 都是列表
    print("\n--- Element-wise Array Test ---")
    req_array_ew = np.array([5, 10, 25])
    aval_rsc_ew = [10, 8, 30]
    constraints_ew = [
        TaskConstraints(parallel_mode="range", core_min=2, core_max=8),
        TaskConstraints(parallel_mode="list", core_list=[4, 8, 16]),
        TaskConstraints(parallel_mode="lwb", core_min=30)
    ]
    rsc_array_ew, constr_array_ew = find_legal(constraints_ew, aval_rsc_ew, req_array_ew)
    
    print(f"场景14 (Element-wise): req={req_array_ew}")
    print(f"                     avail={aval_rsc_ew}")
    print(f"                     rsc={rsc_array_ew}")
    print(f"                     constr={constr_array_ew}")
    
    expected_rsc_ew = np.array([5, 8, 30])
    expected_constr_ew = np.array(['none', 'available', 'lwb'])
    np.testing.assert_array_equal(rsc_array_ew, expected_rsc_ew)
    np.testing.assert_array_equal(constr_array_ew, expected_constr_ew)
    print("--- Element-wise Array Test Passed ---")

    print("\n--- 所有测试用例通过 ---")
