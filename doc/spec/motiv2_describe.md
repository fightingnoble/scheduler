# Motivation Experiment 2: Dynamic Scheduling Scalability Analysis

## Figure Description

**Latency Breakdown (Fig. 2a)**: Stacked bar chart illustrates the decomposition of end-to-end latency relative to timing constraint $\mathcal{D}_{\mathrm{e2e}}$ across different hardware-workload configurations. Each bar comprises three components stacked bottom-to-top: execution ratio (computation time), realloc ratio (scheduling overhead), and waiting ratio (queueing delay). The auxiliary axis plots miss rate, revealing deadline violation trends as system scale increases.

**Resource Utilization (Fig. 2b)**: Clustered stacked bar chart demonstrates processing power allocation under varying configurations. Clusters represent hardware scale (tile count) and load intensity combinations, with bars within each cluster differentiated by task chain count using hatch patterns. Each bar stacks three components bottom-to-top: effective utilization, realloc overhead, and idle capacity. The auxiliary axis depicts miss ops ratio connected within clusters to highlight intra-configuration trends while maintaining inter-cluster separation.

## Visual Specification

| Property | Value |
|----------|-------|
| Figure size | (4, 2.2) inches |
| Implementation | `StatisticsCollector.plot_motiv_case2()` |
| Output | `case2_breakdown.pdf`, `case2_utilization.pdf` |

### Latency Breakdown (plot_type='breakdown')

**Axes:**
- X-axis: Clusters by (tiles, load_factor), inner grouping by chains
- Y-left: Latency / $\mathcal{D}_{\mathrm{e2e}}$, linear scale (ratios >1 indicate constraint violations)
- Y-right: Miss Rate (line)

**Stacked bars** (bottom to top):

| Layer | Metric | Color | Matplotlib |
|-------|--------|-------|------------|
| Bottom | `exec_ratio` (execution) | Blue | `C0` |
| Middle | `realloc_ratio` (scheduling) | Orange | `C1` |
| Top | `wait_ratio` (waiting) | Green | `C2` |

**Miss Rate line:**

| Element | Metric | Color | Style |
|---------|--------|-------|-------|
| Miss Rate | `miss_mean_count` | Purple | `C4`, `o-` |

### Resource Utilization (plot_type='utilization')

**Axes:**
- X-axis: Same cluster/inner layout as breakdown
- Y-left: Resource utilization ratio, linear scale (identity: effective + realloc + idle = 1)
- Y-right: Miss Ops Ratio (line, connected within clusters)

**Stacked bars** (bottom to top):

| Layer | Metric | Color | Matplotlib |
|-------|--------|-------|------------|
| Bottom | effective (1 - idle - miss - realloc) | Blue | `C0` |
| Middle | `realloc_mean_ratio` | Orange | `C1` |
| Top | `idle_mean_ratio` | Gray | `C7` |

**Miss Ops Ratio line:**

| Element | Metric | Color | Style |
|---------|--------|-------|-------|
| Miss Ops | `miss_mean_ratio` | Red | `C3`, connected within clusters |

### Data Sources

- `collector.get_motiv_case2_stats()` returns `utilization` dict and `latency_breakdown` dict
- `collector.get_utilization_avg_ratio()` for idle/miss/realloc means

## Statistical Methodology

Metrics are computed as hyperperiod averages using TDigest streaming histograms. Latency components are normalized by $\mathcal{D}_{\mathrm{e2e}}$ to enable cross-configuration comparison. Resource utilization excludes missed operations (as they consume no power), with the identity $\text{realloc} + \text{effective} + \text{idle} = 1$ enforced. Miss rate is task-type-normalized to reflect per-task-class timeout probability.

## Unified Color Scheme (Cross-Figure Reference)

| Category | Color | Matplotlib | Used in |
|----------|-------|------------|---------|
| Execution / Effective | Blue | `C0` | Latency breakdown, Utilization |
| Scheduling / Realloc | Orange | `C1` | All figures |
| Waiting | Green | `C2` | Latency breakdown |
| Miss (capacity) | Red | `C3` | Utilization (bar + line) |
| Miss Rate (line) | Purple | `C4` | Motiv-1, Breakdown |
| Idle | Gray | `C7` | Motiv-1, Utilization |
