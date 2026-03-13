# Ablation Experiment 1: Reservation in Serial Execution (cyc-S vs cyc)

## Figures

Two figures are produced for this experiment:

### Fig. A1a — Utilization-Reliability Tradeoff (Motiv-1 Style)

Reuses `StatisticsCollector.plot_motiv_case1()` with cyc-S data as the primary series and cyc reference points projected as horizontal dashed lines on the miss rate axis. This directly compares soft reservation (cyc-S) against hard isolation (cyc) at identical resource budgets.

| Property | Value |
|----------|-------|
| Figure size | (4, 2.2) inches |
| DPI | 150 |
| Implementation | `StatisticsCollector.plot_motiv_case1()` (shared with Motiv-1) |
| Output | `case1/case1_motiv1_style.pdf` |

**Axes:**
- X-axis: Soft reservation percentile `exec_t_comp_ratioB` (p50–p99)
- Y-left (log): Ops Ratio — three grouped bars (Idle `C7`, Miss `C3`, Realloc `C1`)
- Y-right (linear): Miss Rate — purple line (`C4`)

**Reference lines:** Gray horizontal dashed lines showing cyc baseline at ratioA = 0.5, 0.7, 0.99, labeled `cyc p{N}` at right margin.

### Fig. A1b — Satisfaction Projection

Single-axis line plot showing how latency satisfaction rate (1 − miss_mean_count) improves as soft reservation becomes more aggressive, with cyc baselines projected as horizontal reference lines.

| Property | Value |
|----------|-------|
| Figure size | (7.0, 3.2) inches |
| DPI | 300 |
| Implementation | `AblaExp1Runner._plot_satisfy_projection()` |
| Output | `case1/case1_satisfy_projection.pdf` |

**Axes:**
- X-axis: `exec_t_comp_ratioB` (float 0.5–0.99)
- Y-axis: Latency Satisfaction Rate [0, 1], linear

**Visual elements:**

| Element | Metric | Color | Style |
|---------|--------|-------|-------|
| cyc-S line | 1 − `miss_mean_count` | Purple (`C4`) | `o-`, lw=1.5 |
| cyc reference | per-ratioA satisfaction | Gray | `:`, alpha=0.6 |

**Labels:** cyc reference annotated as `cyc p{N}` at right margin (x=0.905 in axis coordinates).

## Expected Trends

- cyc-S miss rate is lower than cyc at the same resource budget (ratioA=0.7)
- cyc-S satisfaction rate improves monotonically as ratioB becomes more aggressive (smaller values)
- Idle ratio for cyc-S is lower than cyc (~20% vs ~33%), confirming better utilization through soft reservation
- Realloc ratio for cyc-S is small but nonzero (soft reservation introduces minimal switching)
