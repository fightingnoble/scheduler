# REVIEW_PACKET_PHASE1

Post-review note: this English packet is the original Phase 1 proposal. The current user-approved state is recorded in `REVIEW_PACKET_PHASE1.zh.md` and `cleanup/requests/user-decisions-20260612.md`.

Current user decision summary:

- Approved: `P1-GIT-001`, `P1-GIT-002`, `P1-KEEP-001`, `P1-DEP-001`, `P1-REUSE-001`, `P1-SYMBOL-001`, `P1-NODE-001` for tracked Node manifests only.
- Keep review: `P1-DEP-002`, `P1-REVIEW-DOCS-001`, `P1-REVIEW-RUNTIME-001`.
- Not approved: all `P1-REMOVE-*` batches, `P1-CACHE-001`, and `P1-CACHE-002`.
- Scope constraint: do not operate directly on `/home/zhangchg/git_repo/scheduler`; ignore untracked files.

Phase 1 audit only. No cleanup, source edits, resets, rebases, merges, or commits were performed on `test_pipeline`.

## Audit Scope

- Source branch: `test_pipeline` at `d5bfda5`
- Archive tag: `archive/test_pipeline-20260612`
- Archive branch: `archive/test_pipeline-20260612-branch`
- Audit worktree: `/home/zhangchg/git_repo/scheduler-audit-20260612`
- Integration worktree: `/home/zhangchg/git_repo/scheduler-integration-20260612`
- Integration base used: `master` because `main`/`origin/main` do not exist locally

## Summary

- Files inventoried: 237
- KEEP: 67
- REMOVE_FROM_CLEAN candidates: 46
- REVIEW: 112
- REUSE_CANDIDATE: 12
- Symbol candidates recorded only: 1179
- Entry point `--help` probes pass in the required `conda activate gurobi` environment. The earlier base-Python failures are kept only as an environment caution.

## How To Respond

Reply with decision IDs only, plus approve/reject/keep-review. Examples:

```text
approve P1-GIT-001 P1-KEEP-001 P1-DEP-001
reject P1-NODE-001
keep-review P1-REVIEW-DOCS-001
```

## 1. MUST REVIEW

### P1-GIT-001

- Category: git-safety
- Recommended action: confirm master as integration base or provide the real main ref
- Path/dependency count: 0
- Scope sample: n/a
- Risk: medium
- Evidence: `cleanup/reports/mainline-diff-summary.md`
- If approved: continue Phase 2/3 planning against master-as-mainline

### P1-GIT-002

- Category: git-safety
- Recommended action: acknowledge original untracked files are outside archive refs and cleanup scope unless separately approved
- Path/dependency count: 0
- Scope sample: n/a
- Risk: low
- Evidence: `cleanup/reports/original-worktree-status.txt`
- If approved: do not import untracked original worktree files into audit

### P1-KEEP-001

- Category: keep-seed
- Recommended action: accept KEEP seed for current clean main runtime surface
- Path/dependency count: 67
- Scope sample: allocator_agent.py, approach_Eq.py, approach_collector.py, approach_def.py, approach_initiator.py, approach_sched.py, approach_setup.py, approach_sim.py, cfgs/Bp_guided.json, cfgs/Bp_scratch.json, cfgs/var_sim_cfg.json, example/bm4.py, ... (+55)
- Risk: medium
- Evidence: `cleanup/reachable-files.csv, cleanup/reachability-graph.md`
- If approved: Phase 2 does not remove these paths

### P1-REUSE-001

- Category: reuse-candidates
- Recommended action: classify non-entrypoint but useful experiment/optimizer/mapper files as REUSE_CANDIDATE, not immediate removal
- Path/dependency count: 12
- Scope sample: model/noc.py, optimizer/delta_ver.py, optimizer/dist_custom.py, optimizer/ops.py, optimizer/scheduler_base.py, sched/packing_solver/gurobi_semi2Dclst_mapping.py, sched/packing_solver/gurobi_semi2Dclst_mapping2.py, scripts/e2e_exp_runner.py, scripts/e2e_hyperparam.py, scripts/repack_sweep.py, tile_2d_mesh_mapper_fixed.py, unused_fun.py
- Risk: medium
- Evidence: `cleanup/reuse-candidates.csv`
- If approved: exclude from clean runtime unless explicitly ported later; preserve via archive refs

### P1-DEP-001

- Category: dependencies
- Recommended action: approve dependency manifest/environment reconciliation for reachable third-party deps
- Path/dependency count: 8
- Scope sample: gurobipy, h5py, networkx, psutil, pyinstrument, pytest, tdigest, tqdm
- Risk: manifest/env mismatch
- Evidence: `cleanup/dependency-ledger.csv, cleanup/reports/dependency-audit.csv, cleanup/reports/entrypoint-smoke.md`
- If approved: Phase 2 may align `requirement.txt` or document that `conda activate gurobi` is the supported environment

### P1-DEP-002

- Category: dependencies
- Recommended action: review listed-but-not-statically-imported dependencies before removing
- Path/dependency count: 3
- Scope sample: bokeh, plotly, pyyaml
- Risk: unknown
- Evidence: `cleanup/dependency-ledger.csv`
- If approved: keep as REVIEW until runtime/plot tests prove stale

### P1-NODE-001

- Category: dependencies
- Recommended action: approve removal/ignore plan for empty Node manifests and local node_modules if no Node tooling is needed
- Path/dependency count: 3
- Scope sample: package.json, package-lock.json, node_modules/
- Risk: low
- Evidence: `cleanup/dependency-ledger.csv, cleanup/gitignore-cache-ledger.csv`
- If approved: Phase 2 can remove tracked empty manifests from clean main and add node_modules/ ignore

### P1-CACHE-001

- Category: cache-generated
- Recommended action: approve git rm --cached plus ignore for root generated artifacts
- Path/dependency count: 4
- Scope sample: output.pdf, output.txt, stats_collector.json, error_log.txt
- Risk: low
- Evidence: `cleanup/gitignore-cache-ledger.csv`
- If approved: Phase 2 untracks artifacts and updates ignore rules

### P1-CACHE-002

- Category: cache-generated
- Recommended action: approve archive-only/untrack plan for debug_case1 generated artifacts
- Path/dependency count: 3
- Scope sample: debug_case1/case1/case1_motiv1_style.pdf, debug_case1/case1/case1_satisfy_projection.pdf, debug_case1/case1/case1_summary.json
- Risk: medium
- Evidence: `cleanup/gitignore-cache-ledger.csv`
- If approved: Phase 2 removes/untracks debug_case1 from clean main

### P1-REMOVE-B7-DOCS

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk docs/agent/chat/history documents
- Path/dependency count: 27
- Scope sample: .claude/agents/conda-debug-runner.md, .claude/agents/doc-writer.md, .trae/documents/plan_20260212_195149.md, GUROBI_ENV_TESTING_SUMMARY.md, TEST_SCRIPT_UPDATE_SUMMARY.md, TODO.md, claude_talk/ala_coding.md, claude_talk/auto-repack.md, claude_talk/baseline_allocator.md, claude_talk/compared_allocator.md, doc/analytical_report/MOTIV_EXP_SUMMARY.md, doc/analytical_report/PARAMETER_MAPPING_SUMMARY.md, doc/dev/change_log_2023.md, doc/dev/change_log_2024.md, doc/dev/change_log_2026.md, doc/dev/fig_plotting_context.md, doc/dev/plans/2026-03-20-repack-fix-core.md, doc/dev/repack_debug_instrumentation.md, doc/spec/algorithm/paper/cursor_algo_paper.md, doc/temp/data_structure.md, ... (+7)
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B8-EXAMPLE

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk example bm1-bm3 files
- Path/dependency count: 3
- Scope sample: example/bm1.py, example/bm2.py, example/bm3.py
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B14-RUN

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk legacy run/*.sh wrappers
- Path/dependency count: 6
- Scope sample: run/1/all.sh, run/1/scan_core.sh, run/2/all2.sh, run/aba_scalability_scan copy.sh, run/abla_chain_sharing.sh, run/temp.sh
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B19-TESTS

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk stale tests outside current entrypoint contract
- Path/dependency count: 6
- Scope sample: scripts/test_alloc_lat.py, test_closure_fix.py, test_duplicate.py, test_event_update.py, test_mapping.py, test_updated_stats.py
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B3-ANALYZE

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk unreachable analysis helper
- Path/dependency count: 1
- Scope sample: analyze/xlsl_e2e_lat_abla.py
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B13-ROOT

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk root one-off debug helper
- Path/dependency count: 1
- Scope sample: debug_sink_constraint.py
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B11-OPTIMIZER

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk optimizer test helper
- Path/dependency count: 1
- Scope sample: optimizer/ops_test.py
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REMOVE-B15-SCHED

- Category: remove-from-clean
- Recommended action: approve archive-only exclusion for low-risk unused solver fitting helper
- Path/dependency count: 1
- Scope sample: sched/packing_solver/fit.py
- Risk: low
- Evidence: `cleanup/remove-from-clean-candidates.csv`
- If approved: Phase 2 removes these from clean branch only; recovery remains via archive ref

### P1-REVIEW-DOCS-001

- Category: review
- Recommended action: decide doc policy: keep authoritative doc/spec + doc/guide, archive dev logs and obsolete assistant notes
- Path/dependency count: 56
- Scope sample: .claude/skills/spec-writer/SKILL.md, claude_talk/auto-abla.md, claude_talk/explore.md, doc/ABLA_EXP_FIX_PLAN.md, doc/analytical_report/motiv_exp_implementation_check.md, doc/analytical_report/parameter_flow_exec_t_comp_ratio.md, doc/analytical_report/parameter_mapping_correction.md, doc/analytical_report/resource_constraint_dump_coupling.md, doc/assumptions.md, doc/dev/ablation_dev.md, doc/dev/change_log_2025.md, doc/dev/claude_revise.md, ... (+44)
- Risk: medium
- Evidence: `cleanup/review-candidates.csv`
- If approved: Phase 2 splits docs into KEEP vs archive-only batches

### P1-REVIEW-RUNTIME-001

- Category: review
- Recommended action: keep high-risk runtime-adjacent files under REVIEW until experiment-level or targeted tests pass
- Path/dependency count: 28
- Scope sample: analyze/UE_extract.py, analyze/__init__.py, analyze/analyze_ctx_switch.py, analyze/analyze_timing.py, analyze/analyze_tp.py, analyze/pattern.py, analyze/stat_num_exec.py, analyze/xlsl_e2e_latency.py, analyze/xlsl_max_tp.py, analyze/xlsl_min_core.py, analyze/xlsl_safe_scalable.py, model/__init__.py, model/streaming_processing/__init__.py, optimizer/Figure_1.png, ... (+14)
- Risk: high
- Evidence: `cleanup/review-candidates.csv, cleanup/reports/dynamic-risk-signals.csv`
- If approved: no deletion in Phase 2 without targeted tests or owner approval

### P1-SYMBOL-001

- Category: symbols
- Recommended action: keep all function/class candidates record-only; do not delete symbols in Phase 2 unless a later file batch and tests approve it
- Path/dependency count: 1179
- Scope sample: symbol records across Python files
- Risk: high
- Evidence: `cleanup/symbol-candidates.csv`
- If approved: symbol-candidates.csv remains advisory only

## 2. SAFE SUMMARY

- Archive refs exist before any cleanup planning.
- Original `test_pipeline` worktree had untracked files; they were not stashed, cleaned, or copied into the audit.
- `cleanup/*`, `cleanup/reports/*`, and `cleanup/requests/*` are audit outputs only.
- `cleanup/symbol-candidates.csv` is advisory only; no function/class/variable deletion is approved by Phase 1.

## 3. NO NEED TO REVIEW

- Raw import edges: `cleanup/reports/python-import-edges.csv`
- Raw dynamic risk hits: `cleanup/reports/dynamic-risk-signals.csv`
- Raw text-reference scan: `cleanup/reports/text-reference-scan.csv`
- Full file inventory/classification: `cleanup/reachable-files.csv`
- Ownership seed: `cleanup/ownership.csv` and `cleanup/ownership.md`
- Machine decision map: `cleanup/requests/decision-map.csv`
