# Batch preflight packet: B0-SHARED-DEP

Status: preflight complete, waiting for user approval before any edit.

This packet covers only `P1-DEP-001`.

## Scope

Proposed batch:

- Batch: `B0-SHARED-DEP`
- Decision ID: `P1-DEP-001`
- Target file: `requirement.txt`
- Action type: dependency manifest reconciliation

This preflight did not edit `requirement.txt`, did not install packages, and did not touch Python source.

## Current manifest

Current `requirement.txt` contains:

```text
scipy
numpy
pandas
matplotlib
pyyaml
plotly
bokeh
```

## Findings

`P1-DEP-001` approved reconciliation for these reachable or test/runtime dependencies:

```text
gurobipy
h5py
networkx
psutil
pyinstrument
pytest
tdigest
tqdm
```

In the documented `gurobi` Conda environment:

- Import OK: `gurobipy`, `h5py`, `networkx`, `psutil`, `pyinstrument`, `pytest`, `tdigest`.
- Import FAIL: `tqdm`.

The `tqdm` miss is not only a stale manifest issue:

- `mapper/mem_planner.py` imports `tqdm`.
- `mapper/mem_planner.py` is classified as `KEEP`.
- `sched/global_sched.py` has a function-local import from `mapper.mem_planner`.

The current `--help` smoke checks still pass because they do not execute that function-local path.

## Keep-review dependencies

These dependencies are already listed in `requirement.txt`, but remain under `P1-DEP-002` keep-review:

```text
pyyaml
plotly
bokeh
```

Do not remove them in this batch.

## Recommended edit if approved

Append the approved `P1-DEP-001` dependencies to `requirement.txt` without removing the existing keep-review entries:

```text
gurobipy
h5py
networkx
psutil
pyinstrument
pytest
tdigest
tqdm
```

This is a repo manifest update only. It does not mutate the existing Conda environment.

## Validation plan if approved

After editing `requirement.txt`:

1. Re-run the three documented import/help probes in the `gurobi` Conda environment:
   - `scripts/motiv_exp_runner.py --help`
   - `scripts/abla_exp_runner.py --help`
   - `main_approach.py --help`
2. Record that `mapper.mem_planner` still requires the environment to install `tqdm`.
3. Do not run package installation unless the user explicitly approves environment mutation.

## Not in scope

- No deletion from old `P1-REMOVE-*` batches.
- No removal of `pyyaml`, `plotly`, or `bokeh`.
- No package install into the `gurobi` Conda environment.
- No source-code change.
- No `.gitignore` or cache cleanup.
- No operation on `/home/zhangchg/git_repo/scheduler`.

## Approval request

Approve `B0-SHARED-DEP` to update `requirement.txt` as above, or keep this as preflight only.

