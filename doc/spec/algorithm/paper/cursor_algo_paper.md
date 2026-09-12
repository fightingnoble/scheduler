这是一个专为其他智能 Agent（或后续开发任务）设计的**“时空引导式混合分配算法（Guided Hybrid Allocation Algorithm）”**核心上下文汇总与知识库。它凝练了我们在多轮讨论中所澄清的概念、核心建模思想、数学形式化定义以及具体的代码接线，能够帮助其他 Agent 在数秒内建立起完整的业务与数学模型背景。

---

# 🚀 时空引导式混合分配算法 (Guided Hybrid Allocation Algorithm) 知识上下文

本协议/上下文旨在解决以下核心挑战：**如何在具有执行变异性（Execution Time Variability）的多核瓦片式（Tile-based）硬件加速器上，联合求解任务有向无环图（DAG）的“空间”与“时间”双维度分配，以在满足严格端到端（E2E）延迟约束的前提下最大化硬件利用率。**

---

## 1. 核心问题定义与经典问题抽象

### 1.1 联合分配问题 (Joint Spatio-Temporal Allocation)
*   **挑战**：传统的实时调度要么单独优化时域（如时间片预留 Cyclic 调度），要么单独优化空域（如贪心的 Gang 调度）。ADS（自动驾驶系统）任务在多核瓦片式加速器上面临**“时空双重耦合”**。每个任务节点必须同时决策：
    1.  **何时执行 (Temporal)**：开始时间 $t_{start, v}$，分配的运行预算时长 $l_v$（子截止时间为 $t_{end, v} = t_{start, v} + l_v$）。
    2.  **分配多少瓦片 (Spatial - Core Count)**：运行该任务所需的瓦片（Core）数量 $c_v$。
    3.  **在何处执行 (Spatial - Placement)**：映射到物理处理器分区（Bin） $x_s$。
*   **非线性耦合**：由于延迟与分配的核心数成反比（$l_i \propto 1/c_i$），使得问题空间呈现非凸性，伴随着高度的约束耦合（不同任务链交织共享节点）。

### 1.2 经典问题抽象 (Theoretical Abstraction)
1.  **时间-成本权衡问题 (Time-Cost Trade-off, TCT)**：
    *   在项目管理（PERT/CPM）中，可以通过追加成本（Crashing，对应**增加核心数**）来缩短单个活动的时间（对应**降低任务延迟**），目标是在满足项目总截止日期（**E2E Deadline**）下使成本最小。本算法是 TCT 问题在具有多重路径依赖和峰值成本最小化（$\min \max c_i$）目标下的泛化变体。
2.  **混合整数非线性规划 (MINLP) / 混合整数二次约束规划 (MIQCP)**：
    *   由于核心数 $c_i \in \mathbb{Z}^+$ 为整数，而延迟 $l_i$ 为连续变量，且两者满足双线性非凸约束 $(l_i - I_i^{(q)}) \cdot c_i \ge \frac{W_i^{(q)}}{P}$。第一阶段本质是一个 **MIQCP** 优化问题。
3.  **二维装箱问题 (2D Bin Packing) 的泛化变体**：
    *   将物理处理器按时间和瓦片数（空间）划分为多个 Bin 容器（大小为 $|B_s| \times T_{hp}$），任务则是具有可变长宽（$c_v$ 宽，$l_v$ 高）的待装箱 Items。

---

## 2. 三阶段引导式分解框架 (The Three-Stage Guided Framework)

为了降低全局优化的复杂度，使其在工程尺度上可解，我们将联合优化解耦为三个逻辑递进的阶段。每一步通过“固定”部分决策变量，来“收窄”后一步的搜索空间。

```
┌────────────────────────────────────────────────────────┐
│  Phase 1: 理想时序属性推导 (Chain-by-Chain Slack)       │
│  - 目标: 确定逻辑上最优的时间-资源折中 (l_i, c_i)        │
└───────────────────────────┬────────────────────────────┘
                            │ (l_i, c_i) 引导
┌───────────────────────────▼────────────────────────────┐
│  Phase 2: 物理空间划分 (Spatial Partitioning)           │
│  - 目标: 利用任务亲和度(Affinity)，把节点归类到 Bins 中  │
└───────────────────────────┬────────────────────────────┘
                            │ 任务-Bin 映射与分区容量 引导
┌───────────────────────────▼────────────────────────────┐
│  Phase 3: 分区内时域调度 (Intra-Bin Temporal Packing)   │
│  - 目标: 利用 optimistic quantile (q_B) 紧凑排程        │
└────────────────────────────────────────────────────────┘
```

### 📋 阶段对比矩阵

| 阶段 | 固定（输入） | 搜索（输出） | 核心数学模型 |
| :--- | :--- | :--- | :--- |
| **Phase 1: 理想时序** | 逻辑 DAG 拓扑、E2E 约束 $D_{e2e}$ | **局部延迟预算 $l_i$**<br>**理想瓦片数 $c_i$** | **MIQCP (Gurobi)**<br>链式贪心排序与迭代解耦 |
| **Phase 2: 物理空间** | Phase 1 的理想形状限制 $(c_i, l_i)$ | **任务-Bin 映射 $x_{im}$**<br>**分区容量 $S_m$** | **多目标规划 (MOP / ILP)**<br>亲和度最大化（Gurobi 聚类） |
| **Phase 3: 时域调度** | Bin 物理划分与容量 $S_m$ | **精细启动偏移 $t_{start, v}$** | **启发式装箱 (FFD)**<br>乐观排程 + 动态回退机制 |

---

## 3. 各阶段数学建模与算法实现细节

### 3.1 Phase 1: 理想时序属性推导 (MIQCP 建模)
*   **数学公式**：
    $$
    \begin{aligned}
    \min & \quad \max_{i \in \text{chain}} c_i \\
    \text{s.t.} & \quad s_{\text{last}} + l_{\text{last}} \le D_{e2e} \\
    & \quad s_i \ge \max_{j \in \text{pred}(i)} (s_j + l_j) \\
    & \quad l_i \ge L_i(q, c_i) \quad (\forall i) \\
    & \quad c_i \in \mathbb{Z}^+, \ c_i^{\min} \le c_i \le c_i^{\max} \quad (\text{or } c_i \in c_i^{\text{compiled}})
    \end{aligned}
    $$
*   **链式分解（Chain-by-Chain）解决相交耦合的贪心机制**：
    1.  **排序**：将图分解为多条链，并使用优先级函数排序（硬截止时间链优先；同类链按 $\frac{\text{Total Workload}}{D_{e2e}}$ 降序——即“最紧急、资源压力最大”的优先）。
    2.  **顺序传播与扣减**：遍历排好序的链。如果节点 $i$ 已在先前的链中被分配了 $(c_i, l_i)$：
        *   该节点被视为“固定成本”，在当期链中不参与优化。
        *   剩余延迟预算自动扣减：$D^{\text{rem}} = D_{e2e} - \sum_{i \in \text{Assigned}} l_i$。
        *   调用子求解器（Gurobi）**仅为链上剩余未分配的节点**求解。
*   **代码参考**：`sched/slack_estim.py` 中的 `rsc_slack_estim`、`GurobiDistributeSlack` 与 `gurobi_MP_chain_assign.py`。

### 3.2 Phase 2: 物理空间划分与亲和度定义 (ILP 建模)
*   **亲和度（Affinity）定义**：为了最小化分区内任务由于高并发带来的调度冲突并优化数据传输开销，我们通过聚类求解器（如 `ClusterGurobiSolverSemi2D`）引入两类亲和度：
    1.  **任务-任务亲和度 (Task-to-Task Affinity, $\beta_{ik}$)**：反映任务 $i$ 与 $k$ 频繁通信或数据交互度，高分促使共置于一个 Bin 中。
    2.  **任务-Bin 亲和度 (Task-to-Bin Affinity, $\alpha_{im}$)**：反映任务 $i$ 映射至特定硬件 Bin $m$ 的特定偏好。
*   **总亲和度函数**：
    $$
    A = \sum_{i \in V} \sum_{m \in B} \alpha_{im} x_{im} + \sum_{i,k \in V, i \neq k} \sum_{m \in B} \beta_{ik} x_{im} x_{km}
    $$
*   **优化目标**：最小化总容量、最大化总亲和度、均衡各 Bin 的算力利用率：
    $$
    \min \quad w_1 \cdot \sum_{m \in B} S_m - w_2 \cdot A + w_3 \cdot (\max_m U_m - \min_m U_m)
    $$
*   **保守尺寸（Pessimistic Sizing）**：在 Phase 2 确定 Bin 的核心容量 $S_m$ 时，采用**保守的高分位数 $q_A = 0.95$** 下的任务负载进行容量规划，以提供充足的物理资源安全余量。
*   **代码参考**：`sched/packing_solver/gurobi_semi2Dclst_mapping.py`。

### 3.3 Phase 3: 分区内时域调度 (FFD 启发式)
*   **乐观排程 (Optimistic Repacking)**：在 Bin 容量 $S_m$ 和任务到 Bin 映射已固定的情况下，采用**激进的低分位数 $q_B = 0.5$** 估计任务延迟。利用类似首适应递减（First-Fit Decreasing, FFD）的启发式算法将任务紧密打包排程，提高典型场景利用率。
*   **动态回退（Dynamic Fallback）**：运行时若任务实际耗时超过 $q_B$ 乐观预算，由于 Phase 2 按照保守分位数 $q_A$ 预留了物理瓦片容量，多余的空闲瓦片将无缝作为“安全垫”被动态调度，实现“静态时间表（快乐路径） + 动态资源池（安全回退）”的混合调度。
*   **代码参考**：`sched/global_sched.py` 的 repacking 逻辑。

---

## 4. 关键实现与建模澄清

### 4.1 统一的执行时间与访存延迟解耦模型
在执行时间和资源需求估计中，延迟模型已被重构为：
$$
L_i = T_{\text{compute}} + T_{\text{memory}} = \frac{\text{Load}_i}{C_i \times \text{Power}_{\text{base}}} + \text{exp\_io\_t}_i
$$
*   其中 $\text{exp\_io\_t}_i$ 代表固定的访存与 NoC 传输延迟，与分配的瓦片数 $C_i$ 无关（基于 Shifted Exponential 移位指数分布，其 `loc` 偏移不为零）。
*   在超周期复制（`duplicate_for_hyperperiod`）时，系统在 `var_en` 下对 `AccVarDist` 会**同时进行计算负载（Compute Load）与访存延迟（Memory IO Time）的物理拆分采样**，并将两者分别赋回 `exp_comp_t` 和 `exp_io_t` 属性中，确保状态在跨程序（`sim_main` 与 `main_approach`）JSON 序列化前后一致。

### 4.2 无损的分布函数序列化
由于随机采样函数无法通过 JSON 序列化，所有随机变化分布均被改造为**参数化序列化对象**：
1.  **序列化**：通过继承自 `Variation` 并在每个子类（`SenVarDist`, `ExecVarDist`, `LoadVarDist`, `AccVarDist`）中实现 `to_dict()`，将生成随机采样器所需的“参数配方”（如 $\alpha, \beta, \text{loc}, \text{scale}$）存入图节点的 `dist_info` 字段并落盘。
2.  **反序列化与复建**：仿真端 `MyGraph` 初始化时，通过全局工厂函数 `dist_from_dict(dist_info)` 解析“配方”，并在本地完美重建采样函数塞入 `self.var_dist_map` 并附加回节点的 `var_dist` 属性中，打通了两端的数据壁垒。
3.  **代码参考**：`approach/approach_Eq.py`（分布类、`to_dict`、`dist_from_dict`）、`approach/approach_def.py`（`_rebuild_distributions` 与 `duplicate_for_hyperperiod`）。

---

## 📂 核心代码库涉及文件雷达图

```
📌 scheduler/
   ├── task/
   │   └── task_cfg.py          <-- 构图入口，触发 init_var_dist (Phase 1)
   ├── sched/
   │   ├── slack_estim.py       <-- 链式资源估算分配 (Phase 1)
   │   ├── global_sched.py      <-- 二阶段聚类与三阶段Repacking (Phase 2 & 3)
   │   └── packing_solver/
   │       ├── chain_slack_assign.py             <-- Phase 1 求解器
   │       └── gurobi_semi2Dclst_mapping.py      <-- Phase 2 亲和力求解器
   └── approach/approach_Eq.py           <-- 采样分布、序列化/反序列化、T_compute+T_memory核心延迟公式
```