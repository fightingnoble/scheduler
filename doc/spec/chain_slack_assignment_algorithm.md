# 链式松弛时间分配算法

## 摘要

本文档描述了一种针对实时端到端应用链的资源分配算法，该算法通过概率分布建模任务执行时间的不确定性，并基于分位数约束进行松弛时间分配。算法提供了两种求解策略：基于混合整数二次规划（MIQP）的精确求解器和基于约束驱动的启发式求解器。

## 1. 问题定义

### 1.1 系统模型

考虑一个由 $n$ 个任务组成的端到端应用链 $\mathcal{C} = \{T_1, T_2, \ldots, T_n\}$，其中：
- 每个任务 $T_i$ 具有随机执行时间，由概率分布 $\mathcal{D}_i$ 建模
- 任务间存在严格的串行依赖关系
- 系统具有端到端延迟约束 $D_{e2e}$
- 每个任务可分配 $c_i \geq 1$ 个计算核心
- 系统总核心数上限为 $C_{total}$

### 1.2 延迟模型

任务的执行延迟由两部分组成：

**传感器任务（SenVarDist）**：
$$L_i = \mathcal{D}_i^{quantile}(q)$$

**计算任务（AccVarDist）**：
$$L_i = \frac{W_i^{quantile}(q)}{c_i \cdot P} + I_i^{quantile}(q)$$

其中：
- $W_i^{quantile}(q)$ 是任务 $T_i$ 在分位数 $q$ 下的计算负载
- $I_i^{quantile}(q)$ 是任务 $T_i$ 在分位数 $q$ 下的I/O延迟
- $P$ 是单核心计算性能（FLOPS_PER_CORE）
- $c_i$ 是分配给任务 $T_i$ 的核心数

### 1.3 约束条件

1. **端到端延迟约束**：
   $$\sum_{i=1}^{n} L_i \leq D_{e2e}$$

2. **核心数约束**：
   - 上界约束：$c_i \leq c_i^{max}$
   - 下界约束：$c_i \geq c_i^{min}$
   - 区间约束：$c_i^{min} \leq c_i \leq c_i^{max}$
   - 离散约束：$c_i \in \{c_1, c_2, \ldots, c_k\}$

3. **系统资源约束**：
   $$\max_{i=1}^{n} c_i \leq C_{total}$$

## 2. 算法设计

### 2.1 精确求解器（GurobiRscSlackEstim）

#### 2.1.1 数学模型

**决策变量**：
- $c_i \in \mathbb{Z}^+$：分配给任务 $T_i$ 的核心数
- $l_i \in \mathbb{R}^+$：任务 $T_i$ 的执行延迟
- $c_{max} \in \mathbb{Z}^+$：所有任务使用的最大核心数

**目标函数**：
$$\min c_{max}$$

**约束条件**：

1. 端到端延迟约束：
   $$\sum_{i=1}^{n} l_i \leq D_{e2e}$$

2. 延迟-核心关系约束：
   - 传感器任务：$l_i \geq \mathcal{D}_i^{quantile}(q), \quad c_i = 1$
   - 计算任务：$(l_i - I_i^{quantile}(q)) \cdot c_i \geq \frac{W_i^{quantile}(q)}{P}$

3. 核心数约束：
   - 上界：$c_i \leq c_i^{max}$
   - 下界：$c_i \geq c_i^{min}$
   - 区间：$c_i^{min} \leq c_i \leq c_i^{max}$
   - 离散：$c_i = \sum_{j=1}^{k} x_{ij} \cdot c_j, \quad \sum_{j=1}^{k} x_{ij} = 1, \quad x_{ij} \in \{0,1\}$

4. 系统资源约束：
   $$c_{max} = \max_{i=1}^{n} c_i, \quad c_{max} \leq C_{total}$$

#### 2.1.2 求解策略

该模型是一个混合整数二次规划（MIQP）问题，使用Gurobi求解器进行优化。关键特性包括：

- **二次约束处理**：利用Gurobi的二次约束能力处理延迟-核心的非线性关系
- **二进制变量**：对于离散核心数约束，引入二进制选择变量
- **数值稳定性**：通过误差容忍度处理浮点数精度问题

### 2.2 启发式求解器（HeuriRscSlackEstim）

#### 2.2.1 算法框架

启发式算法采用"约束驱动的理想均分"策略，核心思想是：

1. **理想分配**：在无约束情况下，按计算负载比例分配松弛时间
2. **约束处理**：识别并处理资源约束冲突
3. **迭代优化**：通过移除约束冲突节点，重新分配剩余资源

#### 2.2.2 算法步骤

**步骤1：可行性预检查**
```python
min_latency = sum(dist.quantile(q, max_cores) for dist, max_cores in zip(node_var_dists, max_cores_list))
if min_latency > e2e_lat:
    raise ValueError("Infeasible problem")
```

**步骤2：负载分离**
- 计算负载：$W_{total} = \sum_{i} W_i^{quantile}(q)$
- 固定延迟：$I_{total} = \sum_{i} I_i^{quantile}(q)$
- 可用松弛：$S_{available} = D_{e2e} - I_{total}$

**步骤3：理想核心数计算**
$$c_{ideal} = \lceil \frac{W_{total}}{S_{available} \cdot P} \rceil$$

**步骤4：约束冲突检测与处理**

对于每个任务 $T_i$：
1. 计算约束下的最优核心数：$c_i^* = \text{find\_legal}(constraints_i, W_i, c_{ideal})$
2. 识别约束类型：
   - 上界约束：$c_i^* = c_i^{max}$
   - 下界约束：$c_i^* = c_i^{min}$

**步骤5：迭代优化**

```python
while not converged:
    # 计算当前分配的总延迟
    total_latency = sum(compute_latency(c_i, W_i, I_i) for i in tasks)
    slack_gap = total_latency - S_available
    
    if abs(slack_gap) <= threshold:
        # 收敛：重新分配剩余松弛时间
        redistribute_slack()
        break
    elif slack_gap > 0:
        # 延迟超限：移除上界约束任务
        remove_upper_bounded_tasks()
    else:
        # 延迟不足：移除下界约束任务
        remove_lower_bounded_tasks()
```

#### 2.2.3 算法特性

- **时间复杂度**：$O(n \cdot k)$，其中 $k$ 是迭代次数
- **空间复杂度**：$O(n)$
- **近似比**：在约束合理的情况下，通常能获得接近最优的解
- **鲁棒性**：通过阈值机制处理数值误差

## 3. 解决方案验证

### 3.1 验证函数（check_solution_validity）

为确保求解结果的正确性，算法实现了统一的解决方案验证机制：

**验证项目**：
1. **端到端延迟约束**：$\sum_{i} L_i \leq D_{e2e}$
2. **任务级约束**：每个任务的核心数满足其约束条件
3. **延迟-核心关系**：验证延迟计算的一致性
4. **系统资源约束**：$\max_i c_i \leq C_{total}$

**验证流程**：
```python
def check_solution_validity(solution, node_var_dists, e2e_lat, quantile, constraints, tot_cores):
    # 1. 端到端延迟检查
    total_latency = sum(solution[i][1] for i in range(len(solution)))
    if total_latency > e2e_lat:
        return False
    
    # 2. 任务级约束检查
    for i, (cores, latency, _) in solution.items():
        if not validate_task_constraints(cores, constraints[i]):
            return False
        if not validate_latency_core_relation(cores, latency, node_var_dists[i], quantile):
            return False
    
    # 3. 系统资源检查
    max_cores = max(solution[i][0] for i in range(len(solution)))
    if max_cores > tot_cores:
        return False
    
    return True
```

## 4. 算法比较

| 特性 | 精确求解器 | 启发式求解器 |
|------|------------|--------------|
| 求解质量 | 最优解 | 近似解 |
| 时间复杂度 | 指数级 | 多项式级 |
| 内存需求 | 高 | 低 |
| 适用场景 | 小规模问题 | 大规模问题 |
| 约束处理 | 完整支持 | 简化处理 |

## 5. 实现细节

### 5.1 数值稳定性

- 使用 `elim_nume_error()` 函数处理浮点数精度问题
- 引入误差容忍度 `time1n_error_tol_abs` 和 `flop1n_error_tol_abs`
- 在约束比较中使用数值容差机制

### 5.2 约束处理

- 统一的 `TaskConstraints` 数据结构
- 支持四种约束模式：上界、下界、区间、离散
- 通过 `find_legal()` 函数处理复杂约束逻辑

### 5.3 性能优化

- 精确求解器使用Gurobi的商业优化引擎
- 启发式求解器采用贪心策略减少计算复杂度
- 统一的验证机制确保结果正确性

## 6. 结论

本文提出的链式松弛时间分配算法通过概率分布建模和分位数约束，有效解决了实时系统中的资源分配问题。两种求解策略各有优势，精确求解器保证最优性，启发式求解器提供高效性，为不同应用场景提供了灵活的选择。

算法的统一验证机制确保了解决方案的正确性，而模块化的设计便于扩展和维护。该算法为实时系统的资源管理提供了理论基础和实践指导。
