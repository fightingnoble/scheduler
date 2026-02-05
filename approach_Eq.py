"""
任务执行时间和资源需求估计的工具函数
"""
import math
from typing import Dict, Any, Union
from global_var import elim_nume_error
from sched.ref_alloc_search import find_legal

from scipy.stats import truncnorm as _scipy_truncnorm, truncexpon as _scipy_truncexpon, expon as _scipy_expon, norm as _scipy_norm
from scipy.stats import poisson
import numpy as np

time_unit = 1 
unit_align = True


def set_time_unit(timestep, int_slot):
    global time_unit
    if int_slot:
        time_unit = 1
        normalize_factor = timestep
    else:
        time_unit = timestep
        normalize_factor = 1
    return time_unit, normalize_factor



# ------------------------------
# Variation quantile utilities
# ------------------------------
# required parameters: 
# task: flops(i.e., cpu time), Data_size(i.e. io_time)
# hw-compute: processing power, 
# hw-io: system bandwidth, latency curve w.r.t. injection rate, controlled bandwidth utilization

# latency: 
# acc tasks: ld_var* flops/processing power + data_size/bandwidth + io_jitter
# sen tasks: distribution of execution time 


def norm_inv_cdf(p: float) -> float:
    return float(_scipy_norm.ppf(p))

def exp_quantile(p: float, scale: float) -> float:
    """
        Exp distribution has analytical solution for quantile.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    return -float(scale) * math.log(1.0 - float(p))

def discrete_quantile(values, probs, p: float) -> float:
    import numpy as np
    v = np.asarray(values, dtype=float)
    w = np.asarray(probs, dtype=float)
    if v.size == 0 or w.size == 0 or v.size != w.size:
        raise ValueError("values/probs invalid")
    w = w / w.sum()
    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cdf = np.cumsum(w)
    k = int(np.searchsorted(cdf, p, side='left'))
    k = min(max(k, 0), v.size - 1)
    return float(v[k])

def get_truncnorm_para(range_max, trunc_ratio, ZScore=3, loc=0):
    """Prepare the parameters for scipy's normal or truncnorm.rvs        
        This simulator assume the maximum/minimum observed jitter falls within ZScore * scale away from the mean, following 
        a truncated normal distribution. The default ZScore is 3, which means we only consider 3sigma away from the mean. 
    Args:
        range_max: the maximum range of the jitter
        trunc_ratio: the truncation ratio w.r.t. the range_max.
        ZScore: Defines the confidence interval coverage. Defaults to 3.
    Returns:
        loc, scale, half_len, a, b
    """
    # Step 1: translate input parameters into ditribution features
    # half_len: truncate the normal distribution to [loc-half_len, loc+half_len]
    half_len = range_max * trunc_ratio 

    # 特殊处理：当range_max为0时，无变化，返回退化分布参数
    if half_len == 0:
        # 返回一个非常小的scale值，使得分布几乎退化到loc点
        # 这样可以避免分位数计算时的问题
        scale = 1e-10  # 非常小的标准差
        a, b = -1e-6, 1e-6  # 非常小的截断范围
        return loc, scale, half_len, a, b

    # Step 2: translate the distribution features into api parameters
    # a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
    # Here, we expect it is truncated at specified confidence interval,
    # a, b = -ZScore, ZScore 
    a, b = -ZScore, ZScore 

    # determine the standard deviation
    # scale =  (myclip_a - loc) / scale = half_len / ZScore
    scale=half_len/ZScore

    return loc,scale,half_len,a,b


def get_truncexpon_param(ref_value, trunc_ratio:float, lamda_exec:float=100.):
    """
    Prepare the parameters for scipy's expon or truncexpon.rvs
    """
    # Step 1: translate input parameters into ditribution features
    # TODO: deduce by true workload
    loc = ref_value # the lower latency bound of the execution, and the shiftd value of the exponential 
    scope = ref_value * trunc_ratio # the truncated range [loc, loc+scope]
    scale=1/lamda_exec # mean value of the exponential

    # 特殊处理：当ref_value为0时，无变化，返回退化分布参数
    if scope == 0:
        # 返回一个非常小的b值，使得分布几乎退化到loc点
        # 这样可以避免分位数计算时的问题
        b = 1e-10  # 非常小的截断范围
        return scope, loc, scale, b

    b = scope/scale # the truncated range of the vanilla exponential [0, b]
    return scope,loc,scale,b

def get_discrete_param(var_factor_list, exp_comp_t, lambda_ld):
    values = np.array(var_factor_list, dtype=float) * float(exp_comp_t)
    if len(var_factor_list) == 1:
        probs = np.array([1.0], dtype=float)
    else:
        pmf = np.array([poisson.pmf(j, lambda_ld) for j in range(len(var_factor_list))], dtype=float)
        pmf = pmf / pmf.sum()
        probs = pmf
    return probs, values


def truncnorm_closure(a, b, loc, scale, rng):
    return lambda rng=rng, processing_power=1.0: _scipy_truncnorm.rvs(a, b, loc=loc, scale=scale, random_state=rng)
def norm_closure(loc, scale, rng):
    return lambda rng=rng, processing_power=1.0: _scipy_norm.rvs(loc=loc, scale=scale, random_state=rng)


def truncexpon_closure(b, loc, scale, rng):
    return lambda rng=rng: _scipy_truncexpon.rvs(b, loc=loc, scale=scale, random_state=rng)
def expon_closure(loc, scale, rng):
    return lambda rng=rng: _scipy_expon.rvs(loc=loc, scale=scale, random_state=rng)
def descrete_closure(values, probs, rng):
    return lambda rng=rng: float(rng.choice(values, p=probs, replace=False))


class Variation:
    def get_var_fn(self):
        return self.var_fn

    def quantile(self, q: float) -> float:
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the distribution to a dictionary."""
        raise NotImplementedError


class IOVarDist(Variation):
    def __init__(self, b, loc, scale,
                 seed: Union[None, int, np.random.Generator, np.random.RandomState] = None, 
                 truncate: bool = True):
        self.b = b
        self.loc = loc
        self.scale = scale
        # scale is the parameter for exponential distribution (1/lambda)
        self.truncate = truncate        
        self.generator = np.random.default_rng(seed)

        if self.truncate:
            self.var_fn = truncexpon_closure(self.b, self.loc, self.scale, self.generator)
        else:
            self.var_fn = expon_closure(self.loc, self.scale, self.generator)

    def quantile(self, q: float) -> float:
        # 特殊处理：当b非常小时，分布几乎退化到loc点
        if self.b < 1e-9:
            return self.loc
        
        if self.truncate:
            xq = _scipy_truncexpon.ppf(q, self.b, loc=self.loc, scale=self.scale)
        else:
            xq = _scipy_expon.ppf(q, loc=self.loc, scale=self.scale)
        return xq 

    def cdf(self, x: float) -> float:
        if self.truncate:
            return float(_scipy_truncexpon.cdf(x, self.b, loc=self.loc, scale=self.scale))
        else:
            return float(_scipy_expon.cdf(x, loc=self.loc, scale=self.scale))

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__dist_type__': 'ExecVarDist',
            'b': self.b, 'loc': self.loc, 'scale': self.scale, 'truncate': self.truncate
        }

class LoadVarDist(Variation): 
    """
    基于 get_var_t_fn(op) 所用的离散分布（var_factor_list + Poisson 概率）构造：
    - sample(rng): 复用与 get_var_t_fn 一致的采样逻辑
    - quantile(q): 使用离散分布分位数（通过线性索引，不插值），可替换为 TDigest 连续化
    """
    def __init__(self, values, probs,
                 seed: Union[None, int, np.random.Generator, np.random.RandomState] = None):
        self.values = np.asarray(values, dtype=float)
        self.probs = np.asarray(probs, dtype=float)
        self.generator = np.random.default_rng(seed)

        if self.values.size == 1:
            single_value = float(self.values[0])
            self.var_fn = lambda rng: single_value
        else:
            # 使用嵌套函数创建闭包
            self.var_fn = descrete_closure(self.values, self.probs, self.generator)

    def quantile(self, q: float) -> float:
        return discrete_quantile(self.values, self.probs, q)

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__dist_type__': 'LoadVarDist',
            'values': self.values.tolist(),
            'probs': self.probs.tolist()
        }

class SenVarDist(Variation):
    def __init__(self, a, b, loc, scale, truncate:bool=True, 
    seed:Union[None, int, np.random.Generator, np.random.RandomState]=None):
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b
        self.truncate = truncate
        self.generator = np.random.default_rng(seed)
        if truncate:
            self.var_fn = truncnorm_closure(self.a, self.b, self.loc, self.scale, self.generator)
        else:
            self.var_fn = norm_closure(self.loc, self.scale, self.generator)

    def quantile(self, q: float, processing_power: float = 1.0) -> float:
        # 特殊处理：当scale非常小时，分布几乎退化到loc点
        if self.scale < 1e-9:
            return self.loc
        
        if self.truncate:
            return _scipy_truncnorm.ppf(q, self.a, self.b, loc=self.loc, scale=self.scale)
        else:
            return _scipy_norm.ppf(q, loc=self.loc, scale=self.scale)

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__dist_type__': 'SenVarDist',
            'a': self.a, 'b': self.b, 'loc': self.loc, 'scale': self.scale, 'truncate': self.truncate
        }

class AccVarDist(Variation):
    """
    加速任务的联合延迟分布，组合负载变化和执行时间变化：
    - 延迟 = ld_var * flops/processing_power + exec_var
    - 复用 LoadVarDist 和 ExecVarDist 的采样逻辑
    """
    def __init__(self, load_dist: LoadVarDist, exec_dist: IOVarDist,
                 seed: Union[None, int, np.random.Generator, np.random.RandomState] = None):
        self.load_dist = load_dist
        self.exec_dist = exec_dist
        self.generator = np.random.default_rng(seed)
        
        # 创建联合采样函数
        def joint_var_fn(rng=self.generator, processing_power: float = 1.0):
            ld_var = self.load_dist.var_fn(rng)
            exec_var = self.exec_dist.var_fn(rng) 
            return ld_var / processing_power + exec_var
        
        self.var_fn = joint_var_fn

    def quantile(self, q: float, processing_power: float) -> float:
        """
        联合分布的分位数计算（近似）
        使用独立假设的分位数近似
        """
        # 简单近似：假设两个分布独立，使用卷积的分位数近似
        # 对于更精确的结果，可以使用蒙特卡洛采样
        ld_q = self.load_dist.quantile(q) / processing_power
        exec_q = self.exec_dist.quantile(q)
        return ld_q + exec_q

    def to_dict(self) -> Dict[str, Any]:
        return {
            '__dist_type__': 'AccVarDist',
            'load_dist': self.load_dist.to_dict(),
            'exec_dist': self.exec_dist.to_dict()
        }

# 工厂函数，用于从字典反序列化回对象
def dist_from_dict(data: Dict[str, Any]) -> Variation:
    dist_type = data.pop('__dist_type__')
    if dist_type == 'SenVarDist':
        return SenVarDist(**data)
    elif dist_type == 'ExecVarDist':
        return IOVarDist(**data)
    elif dist_type == 'LoadVarDist':
        # 需要将 list 转回 numpy array
        data['values'] = np.array(data['values'])
        data['probs'] = np.array(data['probs'])
        return LoadVarDist(**data)
    elif dist_type == 'AccVarDist':
        # 递归调用
        data['load_dist'] = dist_from_dict(data['load_dist'])
        data['exec_dist'] = dist_from_dict(data['exec_dist'])
        return AccVarDist(**data)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def init_var_dist(args, logical_graph):
    for _node, _type in logical_graph.nodes(data="type"):
        var_dist = None
        if _type == "sink":
            continue
        elif _type == "src":
            # 统一在这里计算截断正态分布参数
            range_max = 1 / logical_graph.nodes[_node]['freq']
            trunc_ratio = args.jitter_sim_para['scale']
            zscore = args.jitter_sim_para['zscore']
            loc, scale, myclip_b, a, b = get_truncnorm_para(range_max, trunc_ratio, zscore)
            var_dist = SenVarDist(a, b, loc, scale, truncate=True, seed=None)
        elif _type == "op":
            # TODO: 这里不统一，新版本用的exp_comp_t, 老版本用的flops（应该是计算量）
            exp_comp_t = logical_graph.nodes[_node]['flops']
            var_factor_list = logical_graph.nodes[_node]['var_factor']
            lambda_ld = 1.0            
            # 1. 计算负载(计算)分布参数
            probs, values = get_discrete_param(var_factor_list, exp_comp_t, lambda_ld)
            load_dist = LoadVarDist(values, probs)
            
            # 2. 计算执行(访存)时间变化分布参数
            trunc_ratio_exec = args.exec_var_para['scale']
            lambda_exec = args.exec_var_para['lambda_exp']
            exp_io_t = logical_graph.nodes[_node]['exp_io_t']
            _, loc_e, scale_e, b_e = get_truncexpon_param(exp_io_t, trunc_ratio_exec, lamda_exec=lambda_exec)
            exec_dist = IOVarDist(b=b_e, loc=loc_e, scale=scale_e, seed=None, truncate=True)
            
            # 3. 组合成 AccVarDist, processing_power 在运行时提供
            var_dist = AccVarDist(load_dist, exec_dist)
        
        if var_dist:
            # 序列化分布对象并附加到节点
            logical_graph.nodes[_node]['dist_info'] = var_dist.to_dict()
            logical_graph.nodes[_node]['var_dist'] = var_dist


def update_task_progress(init_load: float, elapsed_time: float, res: float, base_pwr: float) -> tuple:
    """    
    Args:
        init_load: 初始进度, elapsed_time: 已用时间, res: 资源, base_pwr: 基础算力
    Returns:
        更新后的进度, 变化量
    """
    return elim_nume_error(init_load - elapsed_time * res * base_pwr), elim_nume_error(elapsed_time * res * base_pwr) 

def cal_cost(elapsed_time: float, res: float, base_pwr: float) -> float:
    return elim_nume_error(elapsed_time * res * base_pwr)

# Note that the 
def sim_comp_time(task_load: float, allocated_resources: int, base_power: float, exp_io_t: float = 0.0) -> float:
    """
    Args:
        task_load: 任务负载（剩余工作量）
        allocated_resources: 分配的资源数量
        base_power: 基础算力
        exp_io_t: 固定访存/传输时间
    Returns:
        估计的执行时间
    """
    if allocated_resources <= 0 or base_power <= 0:
        return float('inf')
    compute_time = task_load / (allocated_resources * base_power)
    execution_time = compute_time + exp_io_t
    if unit_align:
        quant_fn = lambda x: normalize_time_to_unit(x, time_unit)
    else:
        quant_fn = lambda x: elim_nume_error(x)
    return quant_fn(execution_time)

def calculate_slack_time(deadline: float, current_time: float, reallocation_slack: float = 0) -> float:
    """
    计算任务的松弛时间
    
    Args:
        deadline: 截止时间
        current_time: 当前时间
        reallocation_slack: 重分配开销
    
    Returns:
        松弛时间
    """
    if unit_align:
        quant_fn = lambda x: normalize_time_to_unit(x, time_unit, mod='down')
    else:
        quant_fn = lambda x: elim_nume_error(x)
    return quant_fn(deadline - current_time - reallocation_slack) 

def estimate_resource_requirement(task_load: float, slack_time: float, base_power: float, exp_io_t: float = 0.0) -> int:
    """    
    Args:
        task_load: 任务负载（剩余工作量）
        slack_time: 可用时间窗口
        base_power: 基础算力
        exp_io_t: 需保留的固定访存/传输时间
    Returns:
        估计所需的资源数量
    """
    eff_slack = slack_time - float(exp_io_t)
    if eff_slack <= 0 or base_power <= 0:
        return 0
    return math.ceil(task_load / (eff_slack * base_power))


def normalize_time_to_unit(time_value: float, time_unit: float, mod:str='up') -> float:
    """
    将时间值标准化到时间单位
    
    Args:
        time_value: 原始时间值
        time_unit: 时间单位
    
    Returns:
        标准化后的时间值
    """
    assert mod in ['round', 'up', 'down']
    assert time_unit <= 1
    n_bit = round(math.log10(1/time_unit))
    if mod == 'round':
        return round(time_value, n_bit)
    elif mod == 'up':
        return math.ceil(time_value / time_unit) * time_unit
    elif mod == 'down':
        return math.floor(time_value / time_unit) * time_unit
    else:
        raise ValueError(f"Invalid mode: {mod}")


def trasfer_realloc_as_task(BW_DRAM, cap, tile_buffer_size, time_norm_factor: float = 1.0):
    """
    将realloc作为任务，计算时间需要转换为计算量
    """
    return cap * tile_buffer_size /BW_DRAM / time_norm_factor


def get_task_load_and_base_size(node_attr: Dict[str, Any], time_norm_factor: float = 1.0) -> tuple:
    """
    从节点属性中提取任务负载和基础大小
    
    Args:
        node_attr: 节点属性字典
        normalize_factor: 标准化因子
    
    Returns:
        (exp_comp_t, base_size) 元组
    """
    if 'flops' in node_attr:
        exp_comp_t = node_attr['flops'] / time_norm_factor
    else:
        exp_comp_t = node_attr['exp_comp_t'] / time_norm_factor
    
    base_size = 1
    return exp_comp_t, base_size


def cal_load(exp_comp_t, base_size):
    """
    统一计算任务load的函数
    
    Args:
        node: 任务节点
        G_ptr: MyGraph实例    
    Returns:
        float: 任务的load值
    """
    return exp_comp_t * base_size


# 专门的时间比较函数
def time_eq(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) == elim_nume_error(time2)

def time_gt(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) > elim_nume_error(time2)

def time_gtq(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) >= elim_nume_error(time2)

def time_lt(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) < elim_nume_error(time2)

def time_ltq(time1: float, time2: float) -> bool:
    return elim_nume_error(time1) <= elim_nume_error(time2)

def time_add(time1: float, time2: float) -> float:
    return elim_nume_error(time1 + time2)

def time_sub(time1: float, time2: float) -> float:
    return elim_nume_error(time1 - time2)
