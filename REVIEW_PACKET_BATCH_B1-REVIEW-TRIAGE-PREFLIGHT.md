# Batch preflight packet: B1-REVIEW-TRIAGE

Status: preflight complete, waiting for user decision before any execution.

This batch is a larger review triage. It does not delete files, move files, edit source code, or change dependencies.

## Scope

Included decisions:

- `P1-DEP-002`
- `P1-REVIEW-DOCS-001`
- `P1-REVIEW-RUNTIME-001`
- `P1-REUSE-001`
- `P1-SYMBOL-001`

Important interpretation:

- `P1-REUSE-001` is already active as a preserve/reuse constraint. It is not permission to delete reuse candidates.
- `P1-SYMBOL-001` is already active as a record-only constraint. It is not permission to delete functions, classes, or variables.

## Current review inventory

From `cleanup/review-candidates.csv`:

```text
REVIEW candidates: 112
docs: 55
python: 23
other: 14
script: 8
shared-config: 6
config: 3
test: 3
```

Largest batches still under review:

```text
B7-DOCS: 56
B13-ROOT: 11
B3-ANALYZE: 11
B14-RUN: 10
B0-SHARED: 6
```

## P1-DEP-002

Dependencies under keep-review:

```text
bokeh
plotly
pyyaml
```

Current evidence:

- Current `gurobi` Conda environment cannot import `bokeh`, `plotly`, or `yaml`.
- `pyyaml` has no tracked code reference outside `requirement.txt`.
- `bokeh` references are comments only.
- `plotly` has commented imports and an active branch in `sched/bin_list_utils.py` for `tool == "plotly"`, but the needed imports are commented out.

Preflight decision:

- Do not remove these dependencies in B1.
- Keep them under `P1-DEP-002` review until the plotting/YAML policy is explicitly decided.

Possible later batch:

- `B2-OPTIONAL-DEPS-DECISION`: decide whether to remove `bokeh`, `plotly`, and `pyyaml` from `requirement.txt`, or install/support them as optional plotting/config dependencies.

## P1-REVIEW-DOCS-001

User-protected documentation areas:

```text
.claude/
claude_talk/
doc/spec/
doc/guide/
doc/dev/
```

Preflight decision:

- Treat these as KEEP, not cleanup candidates.
- Do not bulk-delete docs.
- Docs outside those protected areas remain review, especially `doc/analytical_report/`, `doc/verification/`, `doc/imgs/`, `doc/setting.md`, `doc/sim_flow.md`, `doc/sparse_list.md`, and `path_migration_guide.md`.

Possible later batch:

- `B2-DOC-INVENTORY`: update only documentation ledger/status classifications. No deletion unless the user explicitly approves exact paths.

## P1-REVIEW-RUNTIME-001

Runtime-adjacent Python review files checked by AST parse:

```text
23 checked
23 AST_OK
0 AST_FAIL
```

These files still require targeted tests before any cleanup:

```text
analyze/*.py
appoach_plot6.py
approach_util33.py
run/cfg_parser.py
run/max_core_num_scan.py
simple_test_collector.py
task/throughput_cnt.py
package __init__.py files
```

Preflight decision:

- Do not delete or move runtime-adjacent review files in B1.
- Package `__init__.py` files should be treated as KEEP unless package import tests prove otherwise.
- Files with shell/subprocess/reflection signals need targeted tests before any action.

Possible later batch:

- `B2-RUNTIME-TEST-INVENTORY`: run import-only and targeted checks for runtime-adjacent review files, then propose exact KEEP / REVIEW / move-reference decisions.

## P1-REUSE-001

Reuse candidates remain preserved:

```text
model/noc.py
optimizer/delta_ver.py
optimizer/dist_custom.py
optimizer/ops.py
optimizer/scheduler_base.py
sched/packing_solver/gurobi_semi2Dclst_mapping.py
sched/packing_solver/gurobi_semi2Dclst_mapping2.py
scripts/e2e_exp_runner.py
scripts/e2e_hyperparam.py
scripts/repack_sweep.py
tile_2d_mesh_mapper_fixed.py
unused_fun.py
```

Preflight decision:

- Do not delete reuse candidates.
- If cleanup is needed later, use a separate reuse/porting decision. Default is preserve.

## P1-SYMBOL-001

Symbol candidate inventory:

```text
1179 records
record-only
no symbol deletion authority
```

Preflight decision:

- Do not delete functions, classes, or variables based on `symbol-candidates.csv`.
- Use the symbol list only as a risk index for later file-level work.

## Result

B1 produced no executable cleanup by itself.

The safe next step is to choose one follow-up batch:

1. `B2-RUNTIME-TEST-INVENTORY`: test runtime-adjacent review files.
2. `B2-DOC-INVENTORY`: reclassify docs into KEEP / REVIEW, without deleting.
3. `B2-OPTIONAL-DEPS-DECISION`: decide `bokeh`, `plotly`, and `pyyaml`.

No old `P1-REMOVE-*` or `P1-CACHE-*` action becomes approved because of this preflight.

