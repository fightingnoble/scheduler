"""Unused experiment helper preserved without rewriting in B23."""

def round_to_step(val: int, step: int) -> int:
    """将值舍入到最近的 step 倍数。"""
    return round(val / step) * step
