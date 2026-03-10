# 仿真数值精度与公式分离设计

> **状态**: 草稿 (2026-02)
>
> **相关文件**:
> - `global_var.py`: 数值精度常量和 `elim_nume_error`
> - `approach_Eq.py`: 公式模块 (Equations)
> - `approach_def.py`, `approach_sim.py`, `approach_sched.py`: 仿真逻辑

## 1. 设计理念

### 1.1 公式与仿真逻辑分离

**目的**: 方便后期调整计算公式，无需修改仿真核心逻辑

**架构**:
```
┌─────────────────────────────────────────────────────────────┐
│                    仿真逻辑层                                │
│  (approach_def.py, approach_sim.py, approach_sched.py)     │
│                                                             │
│   调用 ───────────────────────────────► 返回结果           │
│                                                             │
│   time_eq(t1, t2)                      bool                │
│   time_gt(t1, t2)                      bool                │
│   time_add(t1, t2)                     float               │
│   sim_comp_time(load, res, pwr)        float               │
│   estimate_resource_requirement(...)   int                 │
│   calculate_slack_time(...)            float               │
│   update_task_progress(...)            (float, float)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    公式模块层                                │
│                      (approach_Eq.py)                       │
│                                                             │
│   内部调用 ─────────────────────────► 返回结果              │
│                                                             │
│   elim_nume_error(x)                   float               │
│   normalize_time_to_unit(t, unit)      float               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    精度常量层                                │
│                     (global_var.py)                         │
│                                                             │
│   numerical_tol_bit = 12                                    │
│   numerical_error_tol_abs = 1e-12                           │
│   elim_nume_error = lambda x: round(x, 12)                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 分离的好处

1. **公式可替换**: 修改 `approach_Eq.py` 中的公式，仿真逻辑不变
2. **精度可调**: 修改 `global_var.py` 中的常量，全局生效
3. **测试友好**: 公式可以独立测试
4. **维护简单**: 公式集中管理，避免散落在各处

## 2. 数值精度处理

### 2.1 核心问题

浮点数运算存在精度问题，导致：
- 时间比较失败：`0.1 + 0.2 != 0.3`
- 边界条件错误：`t >= T_hp` 本应为真却为假
- 累积误差：多次运算后误差放大

### 2.2 解决方案：统一舍入

**定义位置**: `global_var.py:17-20`

```python
# 精度常量
numerical_error_tol_abs = 1e-12   # 绝对误差容限
numerical_tol_bit = 12            # 小数位数
numerical_error_tol_rel = 0.01    # 相对误差容限

# 核心舍入函数
elim_nume_error = lambda x: round(x, numerical_tol_bit)
# 即：round(x, 12)，保留12位小数
```

**原理**: 所有浮点数运算后，统一舍入到12位小数，消除浮点误差。

### 2.3 时间运算封装

**定义位置**: `approach_Eq.py:486-506`

```python
# 所有时间运算都经过 elim_nume_error
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
```

**使用示例**:
```python
# 错误写法（可能有精度问题）
if curr_t >= next_hp_boundary:
    ...

# 正确写法（使用封装函数）
if time_gtq(curr_t, next_hp_boundary):
    ...
```

### 2.4 其他需要舍入的场景

| 场景 | 处理方式 | 代码位置 |
|------|----------|----------|
| 任务进度更新 | `elim_nume_error(init_load - elapsed * res * base_pwr)` | `approach_Eq.py:362` |
| 执行时间计算 | `normalize_time_to_unit()` 或 `elim_nume_error()` | `approach_Eq.py:382-386` |
| 松弛时间计算 | `normalize_time_to_unit()` 或 `elim_nume_error()` | `approach_Eq.py:400-404` |
| 节点时间属性 | `elim_nume_error(node['offset'])` | `approach_initiator.py` |

## 3. 时间单位对齐

### 3.1 全局时间单位

**定义位置**: `approach_Eq.py:13-14`

```python
time_unit = 1        # 时间单位（秒）
unit_align = True    # 是否对齐到时间单位
```

**设置函数**: `approach_Eq.py:17-25`

```python
def set_time_unit(timestep, int_slot):
    global time_unit
    if int_slot:
        time_unit = 1
        normalize_factor = timestep
    else:
        time_unit = timestep
        normalize_factor = 1
    return time_unit, normalize_factor
```

### 3.2 时间标准化函数

**定义位置**: `approach_Eq.py:422-443`

```python
def normalize_time_to_unit(time_value: float, time_unit: float, mod: str = 'up') -> float:
    """
    将时间值标准化到时间单位

    Args:
        time_value: 原始时间值
        time_unit: 时间单位
        mod: 舍入模式 ('round', 'up', 'down')

    Returns:
        标准化后的时间值
    """
    assert mod in ['round', 'up', 'down']
    assert time_unit <= 1
    n_bit = round(math.log10(1 / time_unit))

    if mod == 'round':
        return round(time_value, n_bit)
    elif mod == 'up':
        return math.ceil(time_value / time_unit) * time_unit
    elif mod == 'down':
        return math.floor(time_value / time_unit) * time_unit
```

**使用场景**:
- `mod='up'`: 计算执行时间（向上取整，保证资源足够）
- `mod='down'`: 计算松弛时间（向下取整，保证不会超时）

## 4. 核心公式汇总

### 4.1 任务执行时间

**公式**: `approach_Eq.py:368-386`

```python
def sim_comp_time(task_load: float, allocated_resources: int,
                  base_power: float, exp_io_t: float = 0.0) -> float:
    """
    计算任务执行时间

    公式: execution_time = task_load / (resources * base_power) + exp_io_t
    """
    if allocated_resources <= 0 or base_power <= 0:
        return float('inf')

    compute_time = task_load / (allocated_resources * base_power)
    execution_time = compute_time + exp_io_t

    # 精度处理
    if unit_align:
        return normalize_time_to_unit(execution_time, time_unit)
    else:
        return elim_nume_error(execution_time)
```

**物理含义**:
- `task_load`: 计算负载 (FLOPS)
- `allocated_resources * base_power`: 有效算力
- `exp_io_t`: 固定访存/传输时间

### 4.2 资源需求估计

**公式**: `approach_Eq.py:406-419`

```python
def estimate_resource_requirement(task_load: float, slack_time: float,
                                  base_power: float, exp_io_t: float = 0.0) -> int:
    """
    估计在给定松弛时间内完成任务所需的资源

    公式: resources = ceil(task_load / ((slack_time - exp_io_t) * base_power))
    """
    eff_slack = slack_time - float(exp_io_t)
    if eff_slack <= 0 or base_power <= 0:
        return 0
    return math.ceil(task_load / (eff_slack * base_power))
```

**物理含义**:
- 资源数量向上取整，保证满足时间约束

### 4.3 松弛时间计算

**公式**: `approach_Eq.py:388-404`

```python
def calculate_slack_time(deadline: float, current_time: float,
                         reallocation_slack: float = 0) -> float:
    """
    计算任务的松弛时间（可用时间窗口）

    公式: slack = deadline - current_time - reallocation_slack
    """
    if unit_align:
        return normalize_time_to_unit(
            deadline - current_time - reallocation_slack,
            time_unit, mod='down'
        )
    else:
        return elim_nume_error(deadline - current_time - reallocation_slack)
```

### 4.4 任务进度更新

**公式**: `approach_Eq.py:355-362`

```python
def update_task_progress(init_load: float, elapsed_time: float,
                         res: float, base_pwr: float) -> tuple:
    """
    更新任务进度

    公式:
        new_load = init_load - elapsed_time * resources * base_power
        delta_load = elapsed_time * resources * base_power

    Returns:
        (剩余负载, 已完成负载)
    """
    return (elim_nume_error(init_load - elapsed_time * res * base_pwr),
            elim_nume_error(elapsed_time * res * base_pwr))
```

### 4.5 负载计算

**公式**: `approach_Eq.py:473-483`

```python
def cal_load(exp_comp_t, base_size):
    """
    计算任务负载

    公式: load = exp_comp_t * base_size
    """
    return exp_comp_t * base_size
```

### 4.6 重调度开销

**公式**: `approach_Eq.py:446-450`

```python
def trasfer_realloc_as_task(BW_DRAM, cap, tile_buffer_size, time_norm_factor=1.0):
    """
    将重调度开销转换为计算负载

    公式: load = cap * tile_buffer_size / BW_DRAM / time_norm_factor

    物理含义: 重调度需要刷新的缓冲区大小 / DRAM带宽
    """
    return cap * tile_buffer_size / BW_DRAM / time_norm_factor
```

## 5. 公式修改指南

### 5.1 修改执行时间公式

**场景**: 考虑新的开销因素

**步骤**:
1. 修改 `approach_Eq.py:sim_comp_time()`
2. 添加新参数（如有需要）
3. 更新调用处传递新参数

**示例**:
```python
# 修改前
def sim_comp_time(task_load, allocated_resources, base_power, exp_io_t=0.0):
    compute_time = task_load / (allocated_resources * base_power)
    return compute_time + exp_io_t

# 修改后：添加通信开销
def sim_comp_time(task_load, allocated_resources, base_power,
                  exp_io_t=0.0, comm_overhead=0.0):
    compute_time = task_load / (allocated_resources * base_power)
    return compute_time + exp_io_t + comm_overhead
```

### 5.2 修改精度参数

**场景**: 需要更高或更低的精度

**步骤**:
1. 修改 `global_var.py` 中的常量
2. 全局生效，无需修改其他代码

```python
# 提高精度
numerical_tol_bit = 15  # 从 12 改为 15

# 降低精度（提升性能）
numerical_tol_bit = 9   # 从 12 改为 9
```

### 5.3 添加新的时间运算

**场景**: 需要新的时间操作

**步骤**:
1. 在 `approach_Eq.py` 中添加新函数
2. 内部使用 `elim_nume_error`
3. 在仿真代码中调用

```python
# 添加时间乘法
def time_mul(time: float, factor: float) -> float:
    return elim_nume_error(time * factor)

# 添加时间除法
def time_div(time: float, divisor: float) -> float:
    return elim_nume_error(time / divisor)
```

## 6. 精度常量参考

| 常量 | 值 | 用途 |
|------|-----|------|
| `numerical_tol_bit` | 12 | 通用舍入位数 |
| `numerical_error_tol_abs` | 1e-12 | 绝对误差容限 |
| `numerical_error_tol_rel` | 0.01 | 相对误差容限 |
| `time1u_error_tol_bit` | 6 | 微秒级时间精度 |
| `time1n_error_tol_bit` | 9 | 纳秒级时间精度 |
| `flop1u_error_tol_bit` | 7 | 微秒级 FLOPS 精度 |
| `flop1n_error_tol_bit` | 10 | 纳秒级 FLOPS 精度 |

## 7. 最佳实践

### 7.1 DO

- ✅ 始终使用 `time_*` 函数进行时间运算
- ✅ 所有浮点结果都经过 `elim_nume_error`
- ✅ 时间比较使用 `time_eq`, `time_gt` 等
- ✅ 修改公式时只改 `approach_Eq.py`

### 7.2 DON'T

- ❌ 直接使用 `==`, `>=` 比较浮点时间
- ❌ 在仿真代码中直接写计算公式
- ❌ 跳过 `elim_nume_error` 处理浮点数
- ❌ 在多处重复相同的计算逻辑

## 8. 典型问题排查

### 8.1 时间比较失败

**症状**: `if t1 >= t2:` 本应为真但为假

**原因**: 浮点精度误差

**解决**: 使用 `if time_gtq(t1, t2):`

### 8.2 累积误差

**症状**: 多次运算后结果明显偏离

**原因**: 未统一舍入

**解决**: 每次运算后调用 `elim_nume_error()`

### 8.3 边界条件错误

**症状**: 超周期边界检测失败

**原因**: 时间对齐不一致

**解决**: 使用 `normalize_time_to_unit()` 统一对齐
