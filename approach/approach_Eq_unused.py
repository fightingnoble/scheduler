"""Unused standalone quantile helpers preserved from approach_Eq."""

import math
from scipy.stats import norm as _scipy_norm


def norm_inv_cdf(p: float) -> float:
    return float(_scipy_norm.ppf(p))

def exp_quantile(p: float, scale: float) -> float:
    """
        Exp distribution has analytical solution for quantile.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    return -float(scale) * math.log(1.0 - float(p))
