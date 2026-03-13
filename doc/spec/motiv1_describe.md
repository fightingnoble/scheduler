# Motivation Experiment 1: Static Scheduling Utilization-Reliability Tradeoff

## Figure Description

**Utilization-Reliability Tradeoff (Fig. 1)**: Grouped bar chart with logarithmic primary axis illustrates the fundamental tradeoff inherent in static cyclic scheduling. Each percentile point displays three side-by-side bars representing capacity breakdown: idle ratio, miss ratio, and realloc ratio. The auxiliary linear axis overlays miss rate as a connected line, with cyc reference points projected as horizontal dashed lines when used in ablation context.

## Visual Specification

| Property | Value |
|----------|-------|
| Figure size | (4, 2.2) inches |
| DPI | 150 |
| Implementation | `StatisticsCollector.plot_motiv_case1()` |
| Output | `case1_motiv1_style.pdf` |

### Axes

- **X-axis**: Reservation percentile labels (e.g., p50, p60, ..., p99)
- **Y-left (primary)**: Ops Ratio, **logarithmic** scale — displays three grouped bars
- **Y-right (secondary)**: Miss Rate, linear scale — line plot

### Visual Elements

**Three grouped bars** (side-by-side, width=0.25 each, on log-scale primary axis):

| Bar | Metric | Color | Matplotlib |
|-----|--------|-------|------------|
| Idle | `idle_mean_ratio` | Gray | `C7` |
| Miss | `miss_mean_ratio` | Red | `C3` |
| Realloc | `realloc_mean_ratio` | Orange | `C1` |

**Miss Rate line** (linear secondary axis):

| Element | Metric | Color | Style |
|---------|--------|-------|-------|
| Miss Rate | `miss_mean_count` | Purple | `C4`, `o-`, lw=1.5 |

**Reference lines** (ablation mode, when `cyc_ref` data present):
- Gray horizontal dashed lines on Y-right, labeled `cyc p{N}` at right margin

### Data Sources

- `collector.get_motiv_case1_stats()` returns `idle_mean_ratio`, `miss_mean_ratio`, `miss_mean_count`, `realloc_mean_ratio`
- Bottom data table: two-column layout showing exact Idle/Miss values per percentile

## Expected Trends (Static Scheduling)

- Idle ratio increases with percentile (p50 → p99): more conservative reservation wastes more resources
- Miss ratio decreases with percentile: higher reservation guarantees fewer deadline violations
- Realloc ratio near zero: static scheduling has no runtime switching overhead
- Miss Rate line trends downward monotonically
