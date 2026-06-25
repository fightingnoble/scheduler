# B2-RUNTIME-TEST-INVENTORY

Date: 2026-06-16

Scope: inventory only. No cleanup executed. No source code changed.

## Inputs read

- `doc/spec/readme.md`
- `doc/spec/key_COT.md`
- `doc/spec/e2e_sched_sim_flow.md`
- `doc/spec/test_plan.md`
- `doc/spec/sim/approach_sim_spec.md`
- `doc/spec/algorithm/guided_hybrid_allocation_algorithm.md`
- `doc/spec/algorithm/chain_slack_assignment_algorithm.md`
- `doc/spec/algorithm/binpack_solver_spec.md`
- `doc/spec/stat/runtime_overhead_spec.md`
- `doc/guide/*.md`
- `CLAUDE.md`
- Phase 1 cleanup ledgers and import-edge reports

## Runtime classification

Current active runtime path:

```text
main_approach.py
  -> approach_setup.py
  -> sim_main.py::perform_bin_packing()
  -> approach_initiator.py
  -> approach_sim.py::run_simulation()
  -> approach_collector.py
```

`sim_main.py` remains active as the configuration and bin-packing backend. `approach_sim.py` is the current event-driven runtime simulator.

## Legacy runtime note

`sched/scheduler_agent.py` and `sched/monitor_agent.py` are documented as deprecated, but they are still imported by active configuration/bin-packing code paths. They are not deletion candidates in B2.

## Test inventory notes

- Keep protected tests.
- `scripts/test_alloc_lat.py` is blocked for execution because it hardcodes `/home/zhangchg/git_repo/scheduler` into `sys.path`.
- `test_event_update.py` and `test_mapping.py` need unresolved import triage for `approach_plot`.

## Verification

Environment:

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && ...'
```

PASS:

- WSL startup probe.
- `main_approach.py --help`.
- `python -m scripts.motiv_exp_runner --help`.
- `python -m scripts.abla_exp_runner --help`.
- Imports:
  - `main_approach`
  - `approach_setup`
  - `approach_sim`
  - `approach_def`
  - `approach_sched`
  - `approach_collector`
  - `sim_main`
  - `sched.global_sched`
  - `sched.slack_estim`
  - `sched.packing_solver.chain_slack_assign`
  - `mapper.mem_planner`

## Outputs

- `REVIEW_PACKET_BATCH_B2-RUNTIME-TEST-INVENTORY.md`
- `cleanup/reports/b2-runtime-test-inventory.csv`
- this report

## Next recommended batch

`B2-RUNTIME-SYMBOL-SPLIT-PREFLIGHT`: enumerate live/dead/ambiguous symbols in `sched/scheduler_agent.py`, `sched/monitor_agent.py`, and `allocator_agent.py`. Use a small runtime smoke matrix before any actual symbol deletion.
