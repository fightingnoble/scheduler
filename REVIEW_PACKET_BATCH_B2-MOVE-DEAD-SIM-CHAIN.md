# REVIEW_PACKET_BATCH_B2-MOVE-DEAD-SIM-CHAIN

Batch: `B2-MOVE-DEAD-SIM-CHAIN` (mode 1, conservative; **move**, not delete)
Strategy: `move-reference` — move dead code to `unused/` / `*_unused.py`; whole files to `unused/` / `scripts/old/`.

Per the corrected "delete vs rewrite" policy: this batch only **moves** code; it does not rewrite logic or interfaces. Class-internal dead methods of `Scheduler` are **out of scope** per user instruction ("先不管"). `sched_fn` / `state_trans` / `sched_utils` dead standalone functions are **deferred to a later batch** (they are tangled via `import *`).

## Background — the dead SCC (closed-loop evidence)

The old simulation loop forms a strongly-connected dead cluster, reachable only from `sim_main.py::main()` (the superseded entry, replaced by `main_approach.py`):

```
sim_main::main() [dead]  +  sim_main::others() [dead, 0 callers]
  └→ allocator_agent.{glb_sched, cyclic_sched, sched_step, period_boader_display, AllocatorInt} [dead]
      └→ Scheduler stepping methods [dead, but deferred — class-internal]
      └→ sched_fn / state_trans / sched_utils standalone fns [deferred]
```

User priors (2026-06-16):
- Q1: `sim_main::main()` + `allocator_agent.{glb_sched,cyclic_sched,sched_step,period_boader_display}` + `AllocatorInt` simulation chain is no longer used. **CONFIRMED DEAD.**
- `others()` = infrequently-used cases split out from main; move together with main. **CONFIRMED DEAD.**
- `allocator_agent.py`: move the **whole file** (not just functions). **CONFIRMED.**

## 1. MUST REVIEW

### Decision B2-MOVE-001 — move `allocator_agent.py` (whole file)

```text
Path: allocator_agent.py
Proposed action: move whole file to unused/allocator_agent.py
Reason: only importer is sim_main.py:17 (dead main). All 5 top-level symbols
        (AllocatorInt, sched_step, cyclic_sched, glb_sched, period_boader_display)
        are used only by the dead sim chain. Zero active-path dependency.
Risk: LOW. Verified: only sim_main.py imports it (line 17); AllocatorInt/sched_step/
      period_boader_display have 0 external users; cyclic_sched/glb_sched only in sim_main::main.
Evidence:
  - importer scan: grep "from allocator_agent|import allocator_agent" → only sim_main.py:17
  - symbol scan: 5 top-level symbols → all only referenced by dead main
  - import smoke baseline: scheduler_agent/monitor_agent/allocator_agent/approach_sim/approach_setup all OK in gurobi
Recovery: git checkout archive/test_pipeline-20260612 -- allocator_agent.py  (or move back)
Companion import cleanup: delete sim_main.py:17 (from allocator_agent import glb_sched, cyclic_sched)
Recommended decision: approve
```

### Decision B2-MOVE-002 — move `sim_main.py::others()` + `::main()` to `sim_main_unused.py`

```text
Path: sim_main.py
Proposed action: move two functions to a new file sim_main_unused.py (same dir, untracked-friendly name)
  - others():    L606–L691
  - main():      L726–L870
  KEEP in place (interleaved active functions):
  - preprocess_args (L692–L725): ACTIVE — called by approach_setup.py:106
Reason: others() has 0 callers; main() only called by scripts/repack_sweep.py (also dead, B2-MOVE-003).
        Both are the superseded entry/orchestration, replaced by main_approach.py.
Risk: LOW–MEDIUM. Risk is the interleaving: others(606) < preprocess_args(692, KEEP) < main(726).
      Must cut two non-contiguous ranges, NOT a single span. Surviving code must be byte-identical.
Evidence:
  - others() callers: 0 (grep "others(" excluding def/old/ref)
  - main() callers: only scripts/repack_sweep.py:32
  - preprocess_args callers: approach_setup.py:106 (active) + sim_main.py:728 (inside dead main)
Recovery: git checkout archive/test_pipeline-20260612 -- sim_main.py
Companion import cleanup: review sim_main.py top imports (L14/16/17/18) — keep any still used by
      active functions (create_common_scheduler_elements uses Scheduler L14, Monitor L16); delete
      only L17 (allocator_agent) and L18 (discrete_event_sim) if confirmed unused by surviving code.
Recommended decision: approve
```

### Decision B2-MOVE-003 — move `scripts/repack_sweep.py` to `scripts/old/`

```text
Path: scripts/repack_sweep.py
Proposed action: move whole file to scripts/old/repack_sweep.py
Reason: one-off sweep script whose only purpose is `from sim_main import main as sim_main` (line 32);
        the main it drives is dead. Not referenced by any .sh/.py or CLAUDE.md experiment command.
Risk: LOW. No references found outside itself.
Evidence:
  - grep "repack_sweep" across *.py *.sh → only the file itself
  - not in CLAUDE.md experiment commands
Recovery: git checkout archive/test_pipeline-20260612 -- scripts/repack_sweep.py
Recommended decision: approve
```

## 2. SAFE SUMMARY

- Whole-file moves: 2 (`allocator_agent.py` → `unused/`, `scripts/repack_sweep.py` → `scripts/old/`)
- Function-segment moves: 2 functions → new `sim_main_unused.py` (`others`, `main`)
- Companion import cleanups: `sim_main.py:17` (and conditionally `:18`); verify no breakage
- Out of scope (this batch): `Scheduler` class-internal dead methods (user: 先不管); `sched_fn`/`state_trans`/`sched_utils` standalone dead fns (deferred — `import *` tangle)
- Smoke test result: NOT YET RUN — to be run after execution as the regression gate (the 6-line import probe + 3× `--help`)
- Batch diff summary: none yet (execution pending approval)

## 3. NO NEED TO REVIEW

Evidence available:
- `cleanup/reports/b2-runtime-test-inventory.csv`
- `cleanup/reports/batch-B2-RUNTIME-TEST-INVENTORY.md`
- AST call-graph analysis (this session): `FILE_ADJUSTMENT_RECORD.md` 2026-06-16 B2 entries
- `cleanup/move-ledger.csv` (rows B2-MOVE-001/002/003 added with this packet)

## Response format

```text
approve all
approve B2-MOVE-001 B2-MOVE-002 B2-MOVE-003
reject B2-MOVE-002
pause batch
```

## Regression gate (post-execution, mandatory)

After moving, in the `gurobi` env, from the audit worktree:

1. Import probe — all must be OK:
   `sched.global_sched, sched.scheduler_agent, sched.monitor_agent, approach_sim, approach_setup` (note: allocator_agent no longer importable from repo root — that is expected; do NOT probe it)
2. `main_approach.py --help` PASS
3. `scripts/motiv_exp_runner.py --help` PASS
4. `scripts/abla_exp_runner.py --help` PASS

If any fails → restore from `archive/test_pipeline-20260612` and record in `FILE_ADJUSTMENT_RECORD.md`.
