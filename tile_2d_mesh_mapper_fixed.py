import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import heapq
import time

class Tile2DMeshMapper:
    """
    2D-mesh物理资源映射器
    支持最小化移动距离的连续资源分配
    """
    
    def __init__(self, mesh_width: int, mesh_height: int):
        """
        初始化2D-mesh映射器
        
        Args:
            mesh_width: mesh宽度
            mesh_height: mesh高度
        """
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.total_tiles = mesh_width * mesh_height
        
        # 当前时刻的分配状态
        self.current_allocation: Dict[str, Set[Tuple[int, int]]] = {}
        # 物理tile占用状态 (x, y) -> task_id
        self.tile_occupancy: Dict[Tuple[int, int], str] = {}
        
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_connected_tiles(self, start_pos: Tuple[int, int], size: int) -> List[List[Tuple[int, int]]]:
        """
        获取从start_pos开始的size个连续tile的所有可能组合
        优先返回矩形区域，然后是连通区域
        """
        candidates = []
        
        # 1. 尝试矩形区域（最优）
        for w in range(1, min(size + 1, self.mesh_width + 1)):
            h = (size + w - 1) // w  # 向上取整
            if h <= self.mesh_height:
                # 检查从start_pos开始的w*h矩形是否可用
                rect_tiles = []
                valid = True
                for i in range(h):
                    for j in range(w):
                        x, y = start_pos[0] + j, start_pos[1] + i
                        if (0 <= x < self.mesh_width and 
                            0 <= y < self.mesh_height and
                            (x, y) not in self.tile_occupancy):
                            rect_tiles.append((x, y))
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(rect_tiles) >= size:
                    # 取前size个tile
                    candidates.append(rect_tiles[:size])
        
        # 2. 尝试连通区域（BFS搜索）
        if not candidates:
            visited = set()
            queue = [(start_pos, [start_pos])]
            
            while queue and len(candidates) < 10:  # 限制搜索数量
                current, path = queue.pop(0)
                
                if len(path) == size:
                    candidates.append(path)
                    continue
                
                # 四个方向的邻居
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = current[0] + dx, current[1] + dy
                    neighbor = (nx, ny)
                    
                    if (0 <= nx < self.mesh_width and 
                        0 <= ny < self.mesh_height and
                        neighbor not in visited and
                        neighbor not in self.tile_occupancy):
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        return candidates
    
    def calculate_movement_cost(self, task_id: str, new_tiles: Set[Tuple[int, int]], 
                               prev_allocation: Dict[str, Set[Tuple[int, int]]]) -> int:
        """
        计算任务移动成本
        
        Args:
            task_id: 任务ID
            new_tiles: 新分配的tile集合
            prev_allocation: 前一时刻的分配
            
        Returns:
            cost: 移动成本
        """
        if task_id not in prev_allocation:
            return 0  # 新任务
        
        old_tiles = prev_allocation[task_id]
        new_tiles_set = set(new_tiles)
        
        # 计算差集大小
        removed = len(old_tiles - new_tiles_set)
        added = len(new_tiles_set - old_tiles)
        return max(removed, added)
    
    def calculate_priority_cost(self, task_id: str, required_tiles: int, 
                               prev_allocation: Dict[str, Set[Tuple[int, int]]]) -> int:
        """
        计算任务优先级成本（用于排序）
        
        Args:
            task_id: 任务ID
            required_tiles: 需要的tile数量
            prev_allocation: 前一时刻的分配
            
        Returns:
            cost: 优先级成本
        """
        if task_id not in prev_allocation:
            return 0  # 新任务，优先级最高
        
        old_tiles = prev_allocation[task_id]
        old_count = len(old_tiles)
        
        # 计算资源需求变化
        if required_tiles > old_count:
            # 需要增加tile，成本较高
            return required_tiles - old_count
        elif required_tiles < old_count:
            # 需要减少tile，成本较低
            return old_count - required_tiles
        else:
            # 数量不变，成本最低
            return 0
    
    def find_optimal_allocation(self, task_requirements: Dict[str, int], 
                               prev_allocation: Dict[str, Set[Tuple[int, int]]]) -> Dict[str, Set[Tuple[int, int]]]:
        """
        找到最优的资源分配方案
        
        Args:
            task_requirements: 任务需求 {task_id: tile_count}
            prev_allocation: 前一时刻的分配 {task_id: set of (x,y)}
            
        Returns:
            新的分配方案
        """
        start_time = time.time()
        
        # 清理已结束的任务
        for task_id in list(self.current_allocation.keys()):
            if task_id not in task_requirements:
                for pos in self.current_allocation[task_id]:
                    self.tile_occupancy.pop(pos, None)
                self.current_allocation.pop(task_id)
        
        # 贪心算法：优先处理移动成本最低的任务
        allocation_result = {}
        available_tiles = set()
        
        # 收集所有可用tile
        for x in range(self.mesh_width):
            for y in range(self.mesh_height):
                if (x, y) not in self.tile_occupancy:
                    available_tiles.add((x, y))
        
        # 按优先级排序任务（移动成本低的优先）
        task_priorities = []
        for task_id, required_tiles in task_requirements.items():
            # 修正：使用正确的优先级计算
            cost = self.calculate_priority_cost(task_id, required_tiles, prev_allocation)
            task_priorities.append((cost, task_id, required_tiles))
        
        # 按移动成本排序
        task_priorities.sort()
        
        # 分配资源
        for _, task_id, required_tiles in task_priorities:
            best_allocation = None
            min_cost = float('inf')
            
            # 如果是现有任务，优先尝试保持原位置
            if task_id in prev_allocation:
                old_tiles = prev_allocation[task_id]
                if len(old_tiles) >= required_tiles:
                    # 可以保持原位置，选择前required_tiles个
                    candidate = set(list(old_tiles)[:required_tiles])
                    if all(tile in available_tiles for tile in candidate):
                        best_allocation = candidate
                        min_cost = 0
            
            # 如果无法保持原位置，寻找最优新位置
            if best_allocation is None:
                # 尝试从每个可用位置开始分配
                for start_pos in available_tiles:
                    candidates = self.get_connected_tiles(start_pos, required_tiles)
                    
                    for candidate in candidates:
                        candidate_set = set(candidate)
                        if all(tile in available_tiles for tile in candidate_set):
                            # 计算移动成本
                            cost = self.calculate_movement_cost(task_id, candidate_set, prev_allocation)
                            if cost < min_cost:
                                min_cost = cost
                                best_allocation = candidate_set
                
                # 如果还是没找到，使用任意可用tile
                if best_allocation is None:
                    available_list = list(available_tiles)
                    if len(available_list) >= required_tiles:
                        best_allocation = set(available_list[:required_tiles])
            
            if best_allocation:
                allocation_result[task_id] = best_allocation
                # 更新占用状态
                for tile in best_allocation:
                    self.tile_occupancy[tile] = task_id
                    available_tiles.discard(tile)
                self.current_allocation[task_id] = best_allocation
        
        execution_time = (time.time() - start_time) * 1_000_000  # 转换为微秒
        print(f"映射算法执行时间: {execution_time:.2f} μs")
        
        return allocation_result
    
    def get_allocation_summary(self) -> Dict:
        """获取分配摘要"""
        return {
            'current_allocation': self.current_allocation,
            'tile_occupancy': self.tile_occupancy,
            'utilization': len(self.tile_occupancy) / self.total_tiles
        }

# 使用示例
def example_usage():
    """使用示例"""
    # 创建10x10的2D-mesh
    mapper = Tile2DMeshMapper(10, 10)
    
    # t-1时刻的分配
    prev_allocation = {
        'A': {(0, 0), (0, 1), (1, 0), (1, 1)},
        'B': {(2, 0), (2, 1)},
        'C': {(3, 0), (3, 1), (3, 2)}
    }
    
    # t时刻的需求（A减少1个tile，B保持不变，C结束，新增D）
    task_requirements = {
        'A': 3,  # 从4个减少到3个
        'B': 2,  # 保持不变
        'D': 2   # 新任务
    }
    
    # 执行映射
    new_allocation = mapper.find_optimal_allocation(task_requirements, prev_allocation)
    
    print("新的分配方案:")
    for task_id, tiles in new_allocation.items():
        print(f"{task_id}: {sorted(tiles)}")
    
    print(f"资源利用率: {mapper.get_allocation_summary()['utilization']:.2%}")

# 测试修正后的逻辑
def test_fixed_logic():
    """测试修正后的逻辑"""
    print("=== 测试修正后的逻辑 ===")
    
    mapper = Tile2DMeshMapper(8, 8)
    
    # 测试用例1：任务A从4个tile减少到3个
    prev_allocation = {'A': {(0,0), (0,1), (1,0), (1,1)}}
    task_requirements = {'A': 3}
    
    # 计算优先级成本
    cost = mapper.calculate_priority_cost('A', 3, prev_allocation)
    print(f"任务A从4个tile减少到3个的优先级成本: {cost}")
    
    # 测试用例2：任务B从2个tile增加到3个
    prev_allocation = {'B': {(2,0), (2,1)}}
    task_requirements = {'B': 3}
    
    cost = mapper.calculate_priority_cost('B', 3, prev_allocation)
    print(f"任务B从2个tile增加到3个的优先级成本: {cost}")
    
    # 测试用例3：新任务C
    prev_allocation = {}
    task_requirements = {'C': 2}
    
    cost = mapper.calculate_priority_cost('C', 2, prev_allocation)
    print(f"新任务C的优先级成本: {cost}")

if __name__ == "__main__":
    test_fixed_logic()
    print("\n")
    example_usage() 