# Motivation Experiment 2: Dynamic Scheduling Scalability Analysis

## Figure Description

**Latency Breakdown (Fig. 2a)**: Stacked bar chart illustrates the decomposition of end-to-end latency relative to timing constraint $\mathcal{D}_{\mathrm{e2e}}$ across different hardware-workload configurations. Each bar comprises three components: execution ratio (computation time), realloc ratio (scheduling overhead), and waiting ratio (queueing delay). The auxiliary axis plots miss rate per task type, revealing deadline violation trends as system scale increases.

**Resource Utilization (Fig. 2b)**: Clustered stacked bar chart demonstrates processing power allocation under varying configurations. Clusters represent hardware scale (tile count) and load intensity combinations, with bars within each cluster differentiated by task chain count using hatch patterns. Each bar stacks three components: realloc overhead (bottom, red), effective utilization (middle, blue), and idle capacity (top, yellow). The auxiliary axis depicts miss ops ratio—the fraction of uncompleted workload due to timeout—connected within clusters to highlight intra-configuration trends while maintaining inter-cluster separation.

## Statistical Methodology

Metrics are computed as hyperperiod averages using TDigest streaming histograms. Latency components are normalized by $\mathcal{D}_{\mathrm{e2e}}$ to enable cross-configuration comparison; ratios exceeding unity indicate constraint violations. Resource utilization excludes missed operations (as they consume no power), with the identity $\text{realloc} + \text{effective} + \text{idle} = 1$ enforced. Miss rate is task-type-normalized to reflect per-task-class timeout probability.

### **图表描述**：

1. **Latency Breakdown (Fig. 2a)**：
   - 堆叠柱状图展示延迟分解（exec/realloc/wait）相对于时间约束的占比
   - 附轴显示 miss rate（超时任务比例）

2. **Resource Utilization (Fig. 2b)**：
   - 分簇堆叠柱状图，簇按硬件规模和负载分组
   - 簇内用 hatch 图案区分不同 chain 数量
   - 堆叠展示资源分配（realloc/effective/idle）
   - 附轴显示 miss ops ratio（未完成负载占比），簇内连接

### **统计方法**：
- 使用 TDigest 流式直方图计算超周期平均值
- 延迟归一化到 $\mathcal{D}_{\mathrm{e2e}}$，便于跨配置比较
- 利用率满足恒等式 realloc + effective + idle = 1
- Miss rate 按任务类型归一化
