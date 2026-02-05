from __future__ import annotations
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from approach_Eq import Variation

class BinPackConfig(dict):
    """
    binpack_cfg 包装器，继承自 dict，提供属性访问模式。
    旨在解决字典访问缺少类型检查和 IDE 补全的问题，同时保持对旧代码的完全兼容。
    由于继承自 dict，它在任何需要字典的地方都可以直接使用。
    """

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None):
        if cfg_dict is None:
            super().__init__()
        else:
            super().__init__(cfg_dict)

    # ========== 类型安全的属性访问 (New Interface) ==========
    
    @property
    def algorithm(self) -> str:
        """算法选择: 'guided', 'scratch', 'mem_plan', 'full'"""
        return self.get("algorithm", "coalescing")

    @algorithm.setter
    def algorithm(self, value: str):
        self["algorithm"] = value

    @property
    def sort(self) -> str:
        """Bin 排序方式: 'EAT', 'barycenter', 'bf', 'reverse_bf'"""
        return self.get("sort", "EAT")

    @property
    def mode(self) -> str:
        """插入模式: 'non-block', 'block'"""
        return self.get("mode", "non-block")

    @property
    def bin_sel_mod(self) -> str:
        """Bin 选择模式: 'search', 'pre_defined'"""
        return self.get("bin_sel_mod", "search")

    @property
    def reservation_policy(self) -> str:
        """预留策略: 'manual', 'auto'"""
        return self.get("reservation_policy", "manual")

    @property
    def affinity_en(self) -> bool:
        """是否启用亲和性"""
        return self.get("affinity_en", True)

    @property
    def affinity_level(self) -> int:
        """亲和性层级"""
        return self.get("affinity_level", 2)

    @property
    def preempt_en(self) -> bool:
        """是否允许抢占"""
        return self.get("preempt_en", True)

    @property
    def partial_alloc_en(self) -> bool:
        """是否允许部分分配"""
        return self.get("partial_alloc_en", False)

    @property
    def quantum_check_en(self) -> bool:
        """是否启用量子化检查"""
        return self.get("quantum_check_en", False)

    @property
    def mapping(self) -> Dict[int, int]:
        """任务到 Bin 的静态映射 (仅用于 pre_defined 模式)"""
        return self.get("mapping", {})

    @mapping.setter
    def mapping(self, value: Dict[int, int]):
        self["mapping"] = value

    @property
    def quantile(self) -> float:
        """资源估算分位数"""
        return self.get("quantile", 0.99)

    @quantile.setter
    def quantile(self, value: float):
        self["quantile"] = value

    @property
    def var_dist_map(self) -> Dict[str, Variation]:
        """任务负载分布图"""
        return self.get("var_dist_map", {})

    @var_dist_map.setter
    def var_dist_map(self, value: Dict[str, Variation]):
        self["var_dist_map"] = value

    @property
    def core_size(self) -> str:
        """核心大小模式: 'specified', 'induced'"""
        return self.get("core_size", "specified")

    @core_size.setter
    def core_size(self, value: str):
        self["core_size"] = value

    @property
    def exec_t_comp_ratioB(self) -> float:
        """repack 使用的分位数"""
        return self.get("exec_t_comp_ratioB", -1.0)

    @exec_t_comp_ratioB.setter
    def exec_t_comp_ratioB(self, value: float):
        self["exec_t_comp_ratioB"] = value

    @staticmethod
    def generate_template() -> Dict[str, Any]:
        """
        生成包含所有可用参数及其默认值的模板字典。
        可用于生成新的 JSON 配置文件。
        """
        return {
            "algorithm": "guided",
            "sort": "EAT",
            "mode": "non-block",
            "bin_sel_mod": "search",
            "reservation_policy": "manual",
            "affinity_en": True,
            "affinity_level": 2,
            "preempt_en": True,
            "partial_alloc_en": False,
            "quantum_check_en": False,
            "core_size": "specified",
            # 以下参数通常由运行时注入，但在模板中列出以供参考
            "quantile": 0.99,
            "exec_t_comp_ratioB": -1.0,
        }

    def __repr__(self):
        return f"BinPackConfig({super().__repr__()})"
