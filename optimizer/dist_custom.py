import torch
import torch.distributions as dist
from torch.distributions import Distribution, Normal, Uniform
from pyro.distributions.torch_distribution import TorchDistributionMixin

class TruncatedNormal(Distribution, TorchDistributionMixin):
    """
    截断正态分布，支持重参数化采样和自动求导。
    
    参数：
        loc (Tensor): 原正态分布的均值 μ。
        scale (Tensor): 原正态分布的标准差 σ（σ > 0）。
        lower (Tensor): 截断区间下限 a。
        upper (Tensor): 截断区间上限 b（a < b）。
    """
    arg_constraints = {
        "loc": dist.constraints.real,
        "scale": dist.constraints.positive,
        "lower": dist.constraints.real,
        "upper": dist.constraints.real,
    }
    support = dist.constraints.interval(-float('inf'), float('inf'))  # 理论支持，但实际有效区间是 [lower, upper]

    def __init__(self, loc, scale, lower, upper, validate_args=None):
        # 参数校验
        if (scale <= 0).any():
            raise ValueError(f"scale 必须大于 0，当前值：{scale}")
        if (lower >= upper).any():
            raise ValueError(f"lower 必须小于 upper，当前 lower={lower}, upper={upper}")
        
        self.loc = loc
        self.scale = scale
        self.lower = lower
        self.upper = upper
        
        # 原正态分布（未截断）
        self.base_dist = Normal(loc=loc, scale=scale)
        
        # 计算截断区间的标准化 CDF 值（用于归一化）
        with torch.no_grad():
            phi_a = self.base_dist.cdf(lower)  # Φ((a-μ)/σ)
            phi_b = self.base_dist.cdf(upper)  # Φ((b-μ)/σ)
            self.z = phi_b - phi_a  # 归一化常数的倒数（1/z）
        
        # 确保 z > 0（截断区间有效）
        if (self.z <= 0).any():
            raise ValueError(f"截断区间 [{lower}, {upper}] 与原正态分布无交集，导致 z ≤ 0")
        
        super().__init__(batch_shape=torch.broadcast_shapes(loc.shape, scale.shape, lower.shape, upper.shape), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """扩展分布的批量形状"""
        new = self._get_checked_instance(TruncatedNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.lower = self.lower.expand(batch_shape)
        new.upper = self.upper.expand(batch_shape)
        new.base_dist = self.base_dist.expand(batch_shape)
        new.z = self.z.expand(batch_shape)
        super(TruncatedNormal, new).__init__(batch_shape, validate_args=self.validate_args)
        return new

    def sample(self, sample_shape=torch.Size()):
        """采样（非重参数化，用于兼容）"""
        return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """重参数化采样（可导）"""
        # 生成均匀分布样本 [0, 1)
        u = Uniform(low=0.0, high=1.0).sample(sample_shape + self.batch_shape)
        
        # 计算截断区间的标准化 CDF 范围
        a_norm = (self.lower - self.loc) / self.scale  # (a-μ)/σ
        b_norm = (self.upper - self.loc) / self.scale  # (b-μ)/σ
        
        # 标准正态分布的逆 CDF（通过误差函数近似）
        # 标准正态 CDF: Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        # 逆 CDF: Φ^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        p = u * self.z + (self.base_dist.cdf(torch.tensor(a_norm, device=u.device)) if a_norm.numel() == 1 else self.base_dist.cdf(a_norm))
        x = torch.erfinv(2 * p - 1) * (2**0.5) * self.scale + self.loc
        
        # 确保样本落在 [lower, upper] 区间内（由于数值误差可能超出）
        x = torch.clamp(x, min=self.lower, max=self.upper)
        return x

    def log_prob(self, value):
        """计算对数概率密度"""
        # 原正态分布的 log_prob
        log_prob_base = self.base_dist.log_prob(value)
        
        # 归一化项：-log(z)
        log_z = torch.log(self.z)
        
        # 截断区间外的样本概率为 -inf
        mask = (value >= self.lower) & (value <= self.upper)
        log_prob = log_prob_base - log_z
        log_prob = torch.where(mask, log_prob, -float('inf'))
        
        return log_prob

    def cdf(self, value):
        """计算累积分布函数（CDF）"""
        # 原正态分布的 CDF
        cdf_base = self.base_dist.cdf(value)
        
        # 截断后的 CDF：(Φ((x-μ)/σ) - Φ(a/σ)) / (Φ(b/σ) - Φ(a/σ))
        a_norm = (self.lower - self.loc) / self.scale
        b_norm = (self.upper - self.loc) / self.scale
        cdf_trunc = (cdf_base - self.base_dist.cdf(a_norm)) / self.z
        
        # 区间外的 CDF 设为 0 或 1
        cdf_trunc = torch.where(value < self.lower, 0.0, cdf_trunc)
        cdf_trunc = torch.where(value > self.upper, 1.0, cdf_trunc)
        return cdf_trunc

    @property
    def mean(self):
        """计算截断正态分布的均值（理论公式）"""
        a_norm = (self.lower - self.loc) / self.scale
        b_norm = (self.upper - self.loc) / self.scale
        phi_a = torch.erf(a_norm / (2**0.5))
        phi_b = torch.erf(b_norm / (2**0.5))
        numerator = torch.exp(-0.5 * a_norm**2) - torch.exp(-0.5 * b_norm**2)
        denominator = self.z * (2 * torch.pi)**0.5 * self.scale
        return self.loc + (numerator / denominator)

    @property
    def variance(self):
        """计算截断正态分布的方差（理论公式）"""
        a_norm = (self.lower - self.loc) / self.scale
        b_norm = (self.upper - self.loc) / self.scale
        phi_a = torch.erf(a_norm / (2**0.5))
        phi_b = torch.erf(b_norm / (2**0.5))
        term1 = (a_norm * torch.exp(-0.5 * a_norm**2) - b_norm * torch.exp(-0.5 * b_norm**2)) / self.z
        term2 = (torch.exp(-0.5 * a_norm**2) - torch.exp(-0.5 * b_norm**2)) / (self.z**2)
        return self.scale**2 * (term1 - term2)

if __name__ == '__main__':
    # 测试参数
    loc = torch.tensor(0.0)    # 均值 μ=0
    scale = torch.tensor(1.0)  # 标准差 σ=1
    lower = torch.tensor(-1.0) # 截断下限 a=-1
    upper = torch.tensor(1.0)  # 截断上限 b=1

    # 初始化截断正态分布
    trunc_norm = TruncatedNormal(loc=loc, scale=scale, lower=lower, upper=upper)

    # 生成样本（10000 个）
    samples = trunc_norm.rsample(sample_shape=(10000,))

    # 验证样本是否落在 [lower, upper] 内
    assert (samples >= lower).all() and (samples <= upper).all(), "样本超出截断区间"

    # 验证均值接近理论值（截断正态均值公式）
    theory_mean = trunc_norm.mean.item()
    sample_mean = samples.mean().item()
    print(f"理论均值: {theory_mean:.4f}, 样本均值: {sample_mean:.4f}")  # 应接近 0.4602（当 μ=0, σ=1, a=-1, b=1 时）

    # 验证 log_prob 正确性（取区间内一点 x=0）
    x = torch.tensor(0.0)
    log_p = trunc_norm.log_prob(x).item()
    # 理论 log_prob = Φ((0+1)/1) - Φ((-1+1)/1) 的 log 倒数 - 原正态 log_prob(0)
    phi_1 = torch.erf(torch.tensor(1 / (2**0.5)))  # Φ(1) ≈ 0.8413
    phi_0 = torch.erf(torch.tensor(0 / (2**0.5)))  # Φ(0) = 0.5
    z = phi_1 - phi_0  # ≈ 0.3413
    theory_log_p = (torch.log(phi_1 - phi_0) - torch.log(torch.exp(-torch.tensor(0.5 * 0**2)) / (2 * torch.pi)**0.5))  # 原正态 log_prob(0) = -0.5*log(2π) ≈ -0.9189
    theory_log_p = (torch.log(phi_1 - phi_0) - (-0.9189))  # 截断后 log_prob = 原 log_prob + log(z)（注意符号）
    print(f"理论 log_prob: {theory_log_p:.4f}, 实际 log_prob: {log_p:.4f}")  # 应接近一致