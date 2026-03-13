# Ablation Experiments 2 & 3: Isolation and Reservation in Parallel Execution

Both experiments share a unified visualization framework: two figures per experiment (overhead + tradeoff), using identical clustered layout with three representative load configurations.

## Common Visual Framework

### Load Configurations (3 clusters, all chains averaged)

| Cluster | Tiles | Load Factor | Label | Color |
|---------|-------|-------------|-------|-------|
| Low | 400 | 0.5 | Low (400T-0.5×) | Blue `#1f77b4` |
| Mid | 400 | 1.0 | Mid (400T-1.0×) | Orange `#ff7f0e` |
| High | 200 | 1.0 | High (200T-1.0×) | Red `#d62728` |

### Clustered Bar Layout

- **Outer grouping (clusters):** 3 load configurations, labeled at cluster centers
- **Inner grouping (within each cluster):** Scan variable values (num_bins for Exp 2, ratioB for Exp 3), shown as gray sub-labels below bars
- **Bar alpha gradient:** Progressively more opaque within each cluster (0.25 + 0.15 × inner_index)

### Shared Properties

| Property | Value |
|----------|-------|
| Figure size | (7.0, 3.2) inches |
| DPI | 300 |
| Grid | alpha=0.2, dashed, y-axis only |
| Title fontsize | 10 |
| Implementation | `_plot_abla_overhead()`, `_plot_abla_tradeoff()` in `abla_exp_runner.py` |

---

## Ablation Experiment 2: Isolation Effect (pglb vs glb)

### Fig. A2a — Switching Overhead

Demonstrates that spatial partitioning reduces reallocation overhead while keeping reallocation count roughly unchanged.

| Property | Value |
|----------|-------|
| Output | `case2/case2_overhead.pdf` |
| X inner variable | `num_bins` ∈ [1, 2, 4, 8] |

**Axes:**
- X-axis: 3 load clusters × 4 num_bins within each
- Y-left: Realloc Count (bars) — number of reallocation events
- Y-right: Realloc Ratio (lines) — reallocation overhead as fraction of total capacity

**Visual elements:**

| Element | Metric | Color | Style |
|---------|--------|-------|-------|
| Bars (Y-left) | `realloc_mean_count` | Load color, alpha gradient | Grouped, value-annotated |
| Lines (Y-right) | `realloc_mean_ratio` | Load color | `o-`, lw=1.5 |

**Legend:** One entry per load config (Low/Mid/High), upper right.

### Fig. A2b — Latency Breakdown & Miss Rate

Shows the tradeoff: more bins reduce scheduling overhead but can increase idle/miss under tight resources.

| Property | Value |
|----------|-------|
| Output | `case2/case2_tradeoff.pdf` |
| X inner variable | `num_bins` ∈ [1, 2, 4, 8] |

**Axes:**
- X-axis: 3 load clusters × 4 num_bins
- Y-left: Latency / $\mathcal{D}_{\mathrm{e2e}}$, linear — stacked bars
- Y-right: Miss Rate — line

**Stacked bars** (bottom to top, unified color scheme):

| Layer | Metric | Color | Matplotlib |
|-------|--------|-------|------------|
| Bottom | `exec_ratio` | Blue | `C0` |
| Middle | `realloc_ratio` | Orange | `C1` |
| Top | `wait_ratio` | Green | `C2` |

**Miss Rate lines:**

| Element | Metric | Color | Style |
|---------|--------|-------|-------|
| Miss Rate | `miss_mean_count` | Load color | `s-`, lw=1.5 |

**Two legends:** Left—latency components (Exec/Realloc/Wait patches); Right—load configs.

### Expected Trends (Exp 2)

- Realloc ratio drops sharply as num_bins increases (glb 7–20% → bins=8 under 2%)
- Realloc count remains roughly unchanged across bin counts
- Idle ratio increases with more bins (less flexible resource sharing)
- Miss rate may increase under high load with aggressive partitioning

---

## Ablation Experiment 3: Reservation in Parallel Execution (reserv vs pglb)

Fixed `num_bins=8` (largest available). Scans `ratioB` to show non-monotonic U-shape behavior — fundamentally different from Exp 1's monotonic trend.

### Fig. A3a — Switching Overhead

| Property | Value |
|----------|-------|
| Output | `case3/case3_overhead.pdf` |
| X inner variable | `exec_t_comp_ratioB` ∈ [0.5, 0.6, 0.7, 0.8, 0.9, 0.99] |

**Axes:** Same structure as A2a, but inner variable is ratioB instead of num_bins.

**Additional element:** pglb baseline shown as horizontal dotted lines per load config on Y-right (realloc ratio axis), providing direct comparison between reservation-enabled (reserv) and reservation-disabled (pglb) configurations.

### Fig. A3b — Latency Breakdown & Miss Rate

| Property | Value |
|----------|-------|
| Output | `case3/case3_tradeoff.pdf` |
| X inner variable | `exec_t_comp_ratioB` ∈ [p50, p60, ..., p99] |

**Axes and visual elements:** Identical structure to A2b.

### Expected Trends (Exp 3)

- **U-shape (distinct from Exp 1):** As ratioB becomes more aggressive:
  - Realloc count and ratio first decrease then increase
  - Miss rate first improves then worsens
- This non-monotonic behavior arises because in parallel execution, overly aggressive reservation causes excessive preemption, unlike serial execution where more aggressive reservation monotonically improves reliability
- pglb baseline reference lines highlight the improvement window where reservation adds value

---

## Unified Color Scheme (All Figures)

Consistent across Motivation and Ablation experiments:

| Category | Semantic | Color | Matplotlib | Used in |
|----------|----------|-------|------------|---------|
| Execution / Effective | Useful compute | Blue | `C0` | Breakdown bars, Utilization bars |
| Scheduling / Realloc | Overhead | Orange | `C1` | All figures with realloc |
| Waiting | Queue delay | Green | `C2` | Breakdown bars |
| Miss (capacity bar) | Overdue load | Red | `C3` | Utilization bars/line |
| Miss Rate (line) | Deadline violation | Purple | `C4` | Motiv-1, Abla-1, Breakdown |
| Idle | Unused capacity | Gray | `C7` | Motiv-1, Utilization |

### Load Configuration Colors (Ablation Exp 2 & 3 only)

| Load Level | Color | Hex |
|------------|-------|-----|
| Low (400T-0.5×) | Blue | `#1f77b4` |
| Mid (400T-1.0×) | Orange | `#ff7f0e` |
| High (200T-1.0×) | Red | `#d62728` |

## Data Sources

| Experiment | Worker | Stats Method | Cache File |
|------------|--------|-------------|------------|
| Motiv-1 / Abla-1 | `_case1_worker` | `get_motiv_case1_stats()` | `case1_summary.json` |
| Motiv-2 / Abla-2 | `_case2_worker` | `get_motiv_case2_stats()` + `get_realloc_info()` | `case2_summary.json` |
| Abla-3 | `_case3_worker` | `get_motiv_case2_stats()` + `get_realloc_info()` | `case3_summary.json` |

Note: All plots support `--use_plot_cache` for rapid re-rendering from cached JSON without re-running simulations.
