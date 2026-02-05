
## 目的与方案汇总：
这篇文档旨在，比较提出的混合静态和动态的调度方法，相对于纯静态的周期调度和纯动态的调度方法，在资源利用率，切换开销，任务出错概率，最小资源需求方面的表现。

## 三种方法的比较
- 静态：
  - 核心思想：静态的指定所有任务的资源分配。
  - 流程：将资源划分为若干分区，任务静态的绑定到这些分区；在每个分区内部，每个周期的时间片被静态的划分成若干个区间，每个分区内的任务，在对应的区间内串行的执行。为了保证任务能够满足延迟约束，分区的大小和每个任务分配的时间片长度，都需要满足对应的分位数延迟约束。
  - 优势：得益于加速器double buffer的设计和确定的切换行为，计算密集的任务kernel的启动延迟，可以大部分被隐藏。
  - 劣势：这种per-task 的预留调度方式过于悲观。导致资源利用率低。
  - 代码实现： deduce_cfg2 强制所有的任务都保证对应特定概率的分位数（exec_t_comp_ratioA）的延迟，并保证他们的总和满足端到端延迟约束。分配阶段，coalesce_alloc_cluster, num_bin = -1；执行阶段，run_simulation, policy = 'cyc'。


- 动态:
  - 核心思想：不再静态的指定,切换的时间,以及执行的位置。
  - 流程：通过优先级队列以及充调度策略，动态的决定队列中的哪些任务可以执行，以及执行的任务各自分配多少资源。
  - 优势：能够灵活的所有tile上空闲的时间片。
  - 劣势：启动kernel所需的延迟，难以被掩盖。
  - 代码实现： 分配阶段，无；执行阶段，run_simulation, policy = 'glb'。


- 静态+动态：
    - 核心思想：任务per-task静态的预留方式过于悲观，资源应该以任务组而不是per-task，为粒度预留资源，在组内动态的分配资源可以进一步提高资源利用率，这样即使用更宽松的分位数延迟进行预留，也能满足延迟。
    - 流程：
    - Split的功能（coleasing_alloc_cluster）：相对于纯静态，使用更少的分组数量，允许分组内的任务在空间维度上共享资源。
    - repack（push_task_into_bins_new）则是，在固定分区的前提下，在组内为每个任务分配一个时间区间。如果执行落在该区间内，则可以隐藏启动开销，如果不在，则需要动态的分配资源。相对于纯动态，时间区间的分配会更加激进：使用更加宽松的分位数exec_t_comp_ratioB进行预留，因此，ERT和ddl会比静态的方法更早。
    - 分配阶段，采用guided 算法，包含三个阶段：
      - (1) 计算资源和时间片的初始分派；(2) Spatial Partitioning (Task-to-Bin Clustering)；(3) Temporal Scheduling (Intra-Bin Time Window Assignment)
      - 第一个流程在deduce_cfg2中实现，文档在doc/chain_slack_assignment_algorithm.md 中。后两个流程由 perform_bin_packing 中的guided 算法，分两轮（coalesce_alloc_cluster/push_task_into_bins_new）实现。文档在doc/guided_hybrid_allocation_algorithm.md 中。

## 简称
纯静态的周期调度，纯动态的调度方法和 混合静态和动态的调度方法，分别对应简写：cyc，dyn，reserv。
后面消融实验中，跳过repack的步骤，只进行bin-split 的case，对应简写：cyc(S)
以及跳过bin-split的步骤，只进行repack的case，对应简写：pglb

## 统计方式
- 延迟分解（Distribution by TDigest）
  - per_task_finish_time: 每个 op 基名的完成时间分布（finish_time − offset）。
  - per_chain_e2e_latency: 每条链（sink 基名）的端到端延迟分布。
  - per_chain_execution_time: 每条链的计算时间分布（运行累积的 compute time）。
  - per_chain_realloc_overhead: 每条链累计的调度/切换开销分布（realloc time）。
  - 等待时间不单独存储，按需计算：wait = e2e − execution − realloc。

- 资源与开销（Per-period/Overall）
  - per_part_realloc_overhead: 分区级每周期的重分配开销（时间）分布（在 forward_hyperperiod 时归一到 T_hp）。
  - overall_realloc_overhead: 系统级重分配成本分布（system_realloc_cost / total_pwr / T_hp）。
  - overall_idle_time: 每周期闲置算力累计分布（仅在 S 状态计入）。
  - overall_missed_load: 每周期超时未完成任务的剩余负载分布（在周期边界统计）。
  - 其他计数：task_realloc_num（任务级次数）、part_realloc_num（分区级次数）。

- Motiv-Exp-(3) 负载-延迟（Raw 或 Binned）
  - Raw 模式：raw_load_latency = [(total_load, worst_e2e)]（每周期一对）；用于小样本直观散点与原始 Spearman 相关。
  - Binned 模式：
    - 自适应分箱：dist_hp_total_load 学习负载分布的分位数作为边界；
    - latency_dist_per_adaptive_bin：每个负载 bin 的最差 E2E 延迟 TDigest（可取 p50/p90/p99、IQR 等）；
    - 相关性：
      - 加权 Spearman（单调相关，默认用各 bin 的 p99 与 bin 中心、权重=样本数）；
      - 加权 Pearson（线性相关，y 为 bin 内延迟近似均值）。
  - 可视化：
    - plot_load_latency_raw(fit='wls'|'lowess'|'none')：原始散点 + 可选拟合；
    - plot_load_latency_binned(percentile=0.99, iqr_band=(0.25,0.75), fit='wls'|'lowess'|'none')：pXX 曲线 + IQR 带 + 趋势线。

- 汇总与导出
  - get_summary(): 返回上述分布的 histogram + percentiles + total_processed_count。
  - get_adaptive_load_latency_summary(): 返回各负载 bin 的 pXX、样本数与均值（近似）。
  - get_full_summary(): 聚合分布摘要与 Motiv-Exp-(3) 摘要。
  - export_summary(): 输出文本摘要至 path['stat']，并按需绘制 Motiv-Exp-(3) 图至 path['motiv3']。

- 关键配置
  - set_motiv3_mode('raw'|'binned')：控制 Motiv-Exp-(3) 采集方式（原始点或分箱）。
  - set_path(key, path)：设置输出路径，key ∈ {'stat','motiv3'}。

## 脚本步骤

**Step 1**：
- 对应纯静态方法cyc，以及resev 方法三步中的第一步
- 计算资源和时间片的初始分派。
- 参数：使用num_bin = -1, exec_t_comp_ratioA \in [0.5, 0.99], exec_t_comp_ratioB == None，进行bin_split。

**step 2-1**：
- 对应分箱 cyc(S)，enable time sharing within each bin。
- 将纯静态方法cyc作为初始分配（加载step1的结果进入repack），使用 \(exec_t_comp_ratioB < exec_t_comp_ratioA) 进行repack。

**Step 2-2**：
- 对应分箱的动态方法 pglb，以及对应resev 方法三步中的第二步：合并分箱，实现Spatial Partitioning (Task-to-Bin Clustering)。
- 基于step1 结果（按照step1 配置 from scratch），设置\( num_bin \in [1, num_bin_max] \)，进行bin_split。

**step 3**：
- 对应resev 三步方法中的第三步，enable constrained sharing among and along chains within each bin by adjusting the expected time intervals and corresponding tile amount within each bin。
- 加载step2-2的结果，使用 \(exec_t_comp_ratioB < exec_t_comp_ratioA) 进行repack。

**step 4**：
- 随机测试，模拟运行时的调度，以及负载变化。
- run_simulation, 包含"cyc, glb, pglb, reserv"四种方法。开始之前，需要加载step1，step2-1，step2-2，step3的结果。

## 两种方法的比较

**总体目标**：通过三个motivation实验，从不同角度揭示纯静态和纯动态方法的局限性，为混合方法的必要性提供实验依据。

**三个实验的逻辑关联**：
- **Motiv-Exp-(1)**：静态方法的"利用率-可靠性权衡"（trade-off维度）
- **Motiv-Exp-(2)**：动态方法的"可扩展性瓶颈"（规模维度）
- **Motiv-Exp-(3)**：切换开销的"不确定性问题"（随机性维度）

- 闲置算力占比（`dist_overall_idle`）：负载不足时的未分配容量
- miss任务剩余负载占比（`dist_overall_miss`）：超时未完成的负载
- 切换开销占比（`dist_overall_realloc`）：调度器重分配的时间成本

### Motiv-Exp-(1) 纯静态的调度-利用率问题
证明静态方法，难以同时满足低miss rate 和 高利用率。存在"**利用率-可靠性权衡**"：保守预留导致低miss但高idle；激进预留导致高利用率但高miss。

【实验】
- 脚本：基于 step 1 得到的静态分配方案，使用cyc进行随机测试
- 扫描：修改预留的分位数 \(exec\_t\_comp\_ratioA \in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]\)
- 统计： 
  - 闲置算力占比（`dist_overall_idle`），miss任务剩余负载占比（`dist_overall_miss`），miss任务数量（`dist_overall_miss_count`）
  - 获取方式：`collector.get_utilization_avg_ratio()`
    - 'idle_mean_ratio', 'miss_mean_ratio', 'miss_mean_count'
- 期望看到：
  - 随着预留分位数增加：`idle_ratio ↑`，`miss_ratio ↓`，`miss_count ↓`
  - 切换开销恒为0（纯静态无运行时切换）


### Motiv-Exp-(2) 纯动态的调度-延迟开销问题
说明动态方法存在"**可扩展性瓶颈**"：随着硬件和任务规模增长，切换开销和排队延迟激增，导致延迟组件失衡，miss rate上升。

【实验】
- 脚本：无需调度信息，直接使用glb方法进行随机测试
- 扫描：调整硬件和任务规模
  - 硬件tile数：\([300, 500]\)
  - 任务链数量：\([1, 4]\)
  - 任务负载倍数：\([0.5, 1]\)（相对于基线workload）
- 统计 1 - 资源利用率分解： 
  - 闲置算力占比（`dist_overall_idle`）：负载不足时的未分配容量，miss任务剩余负载占比（`dist_overall_miss`），切换开销占比（`dist_overall_realloc`）
  - 获取方式：`collector.get_utilization_avg_ratio()`
    - 'idle_mean_ratio', 'miss_mean_ratio', 'realloc_mean_ratio'.
  - 期望看到：随着规模增加，`realloc_ratio ↑`，`miss_ratio ↑`，`idle_ratio` 在高负载时应趋近0

- 统计 2 - 端到端延迟分解（相对于端到端约束）：
  - 第一条链执行/调度/等待时间相对于端到端约束的占比 在多个周期内的平均值
  - 所有链合并的执行/调度/等待时间相对于端到端约束的占比 在多个周期内的平均值
  - miss任务数量（`dist_overall_miss_count`）
  - 获取方式：
    - 延迟分解：`collector.get_latency_breakdown_avg_ratio()`
      - 'overall_vs_constraint', 'per_chain_vs_constraint'
        - 'exec_ratio', 'realloc_ratio', 'wait_ratio'
    - miss count：`collector.get_utilization_avg_ratio()['miss_mean_count']`
  - 期望看到：随着规模增加
    - `exec_ratio ↓` 需要更快的完成计算，以补偿延迟开销 
    - `realloc_ratio ↑`（调度开销随任务数增长）
    - `wait_ratio ↑`（排队延迟随竞争加剧）
    - 三者之和可能 > 1（表示超时：实际延迟超过约束）

> **Takeaway**: 纯静态的方法和纯动态的方法，都难以在延迟，利用率，切换开销之间取得平衡。Motiv-Exp-(1) 显示静态方法为降低miss需大幅牺牲利用率；Motiv-Exp-(2) 显示动态方法在高负载下等待和调度开销激增，延迟组件失衡。

### Motiv-Exp-(3) 切换行为带来的不确定性

**理论预期**：根据排队论，在理想调度下，队列服务时间（最差E2E延迟）与总负载应呈现强单调相关性（负载越大→延迟越大）。

**问题假设**：切换行为的随机性（依赖系统状态、任务到达时刻）会破坏这种相关性，导致：
1. 相同负载下，延迟分布变宽（散布增大）
2. 负载-延迟的单调关系减弱（相关系数下降）
3. Worst case不再简单对应于peak load时刻

【实验设计 - 对照实验】
- 脚本：无需调度信息，直接使用glb方法进行随机测试
- **对照组设置**：
  1. **基线组（无开销）**：禁用调度开销（`realloc_overhead = 0`），理想调度
  2. **实验组（有开销）**：启用调度开销（真实切换代价），存在随机性
- 统计指标：
  - 每周期记录：`(total_load, worst_e2e_latency)` 对
  - 计算Spearman秩相关系数 ρ（单调相关性）
  - 绘制散点图或binned percentile曲线
  - 获取方式：
    - `collector.set_motiv3_mode('raw')` 或 `'binned'`
    - `collector.get_spearman_correlation(percentile=0.99)`
    - `collector.plot_load_latency_raw()` 或 `plot_load_latency_binned()`
- 期望结果：
  - **基线组**：ρ > 0.85（强正相关），散点紧密沿趋势线
  - **实验组**：ρ < 0.6（中弱相关），散点分散
  - **对比**：Δρ = ρ_基线 - ρ_实验 > 0.25，证明切换开销引入显著不确定性
  - 可视化：实验组散点图明显比基线组"更宽"，IQR带更大

#### 相关性的指标的选择：
选择  Spearman（秩相关）衡量“单调趋势”，对非线性关系与离群值更稳健，更符合我们“负载增加是否总体趋向延迟更大”的问题设定；特别是我们观察 p99（尾部）时，这一点更重要。

#### 大批量数据的收据收集方法和绘图方法（两种：raw和binned）

大概仿真1W个超周期，可以降低记录精度，以获取效率更好的方法。

1.  计算方法（与实现接口对应）：
  - Raw（点数较少，使用原始周期点对 (load, worst_e2e)）：
    - Spearman：对 x 与 y 分别求秩（并列取平均秩），对秩做 Pearson 相关。实现：`StatisticsCollector.get_spearman_correlation()`。

  - Binned（点数较多，使用自适应分箱）：
  将负载分段，然后在每个负载分段下的最差端到端延迟分布也分段。

    - 定义：对第 i 个负载分箱，取
      - x_i = 该分箱的负载中心；
      - y_i = 该分箱内的延迟分位数（默认 p99，也可选 p95/p50）；
      - w_i = 该分箱样本数。
    - 加权 Spearman：先对 {x_i} 与 {y_i} 计算“加权秩”（并列取权重块的平均秩），再对秩向量做加权 Pearson。实现：`get_spearman_correlation_binned(percentile)`。

2. 可视化
  - Raw：散点 +（可选）趋势线（`fit='wls'|'lowess'`），接口：`plot_load_latency_raw(...)`，标题给出 Spearman ρ。
  - Binned：p99 曲线 + IQR 带 +（可选）趋势线（`fit='wls'|'lowess'`），接口：`plot_load_latency_binned(...)`，标题给出 Spearman ρ。


## 提出的方法的优势：消融试验和 端到端性能 

本章节定义了下面两个实验：
- 消融比较
    - 隔离多大的缓解了调度开销的问题，共享多大的缓解了利用率不均衡的挑战（时间维度，高负载和低负载的差值，延迟的改善）"
- 端到端的比较
    - 固定资源和时间约束：吞吐；固定资源和任务规模：出错概率；固定任务规模和时间约束：最小资源需求"

### 消融比较


消融试验基于我们提出的guided 分配算法。

消融试验将依次应用上面的三个阶段：
依次应用step1, step2-1, step2-2, step3，得到cyc，cyc(S)，pglb，reserv 的调度信息。

消融比较1-(cyc(S) vs cyc)：
证明引入时间维度的共享，能够平衡利用率和出错概率。
【试验】
- 固定资源，固定任务规模，扫描exec_t_comp_ratioB \in [0.5, 0.9]：
绘制出错概率和资源利用率随着 exec_t_comp_ratioB 的变化的柱状图。

消融比较2-(pglb vs glb)：
证明引入Spatial Partitioning (Task-to-Bin Clustering) 能平衡利用率和切换开销。
【试验】
和cyc 保持相同的资源和任务规模：
- 扫描：分箱的数量，i.e., \( num_bin \in [1, num_bin_max] \)
- 绘制切换开销随着分箱数量的变化。

消融比较3-(reserv vs pglb)：
证明调整和分箱和repack，能够保证，即使随着硬件和任务规模的增长，资源利用率和切换开销维持在一个较优的范围内。
- 扫描：任务规模，i.e., \( n_task \in [1, n_task_max] \)
- 绘制切换开销&资源利用率 随着任务，资源规模的变化的柱状图。

### 端到端的比较

验证我们提出的hybrid 分配算法，在端到端性能上（吞吐，出错概率，最小资源需求）的提升。分别固定（资源数量，任务规模，时间约束）中的两个，查看第三个的性能。

【试验】
- 禁止超时，超时终止仿真。
  - 固定资源和时间约束，查找不超时最大吞吐。
  - 固定资源和任务规模：查找drop任务情况下能满足的最小延迟约束。
  - 固定任务规模和时间约束：查找最小资源需求。




  - **权衡曲线**：画出idle_ratio (x) 和 miss_ratio (y) 的曲线。
