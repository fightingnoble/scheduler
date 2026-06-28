"""global_sched.py — re-export shell (B6-SPLIT-001, 2026-06-29).

Original file split by purpose into:
  - global_sched_alloc.py  (Step 2 预分配: coleasing_alloc_cluster + 6 helpers)
  - global_sched_repack.py (Step 3 repack: push_task_into_bins_new + push_step_new)
This shell re-exports them so `from sched.global_sched import X` still works (variant B, zero external churn).
test_mem_planner kept here (test fn, only dead refs).
Recovery: git checkout archive/test_pipeline-20260612 -- sched/global_sched.py
"""
from sched.global_sched_alloc import (
    coleasing_alloc_cluster,
    coleasing_alloc_1bin,
    gurobi_split_solver,
    update_bp_result2_schedtab,
    rename_bins_and_relable_assignments,
    build_greedy_obj,
    build_search_obj,
)
from sched.global_sched_repack import (
    push_task_into_bins_new,
    push_step_new,
)

__all__ = [
    "coleasing_alloc_cluster", "coleasing_alloc_1bin", "gurobi_split_solver",
    "update_bp_result2_schedtab", "rename_bins_and_relable_assignments",
    "build_greedy_obj", "build_search_obj",
    "push_task_into_bins_new", "push_step_new",
]
