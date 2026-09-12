# Batch review packet: B0-SHARED-DEP

Status: executed and accepted by user on 2026-06-13.

This batch only executed `P1-DEP-001`: reconcile the Python dependency manifest with reachable dependencies found during Phase 1.

## What changed

Updated `requirement.txt` by appending these approved dependencies:

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

The existing keep-review dependencies stayed in place:

```text
pyyaml
plotly
bokeh
```

## What did not change

- No package was installed into the `gurobi` Conda environment.
- No Python source file was changed.
- No dependency was removed.
- No `.gitignore` or cache cleanup was performed.
- No old `P1-REMOVE-*` action was executed.
- No operation was done in `/home/zhangchg/git_repo/scheduler`.

## Environment note

This batch did not install dependencies. At batch execution time, the repo manifest listed `tqdm`, but the active `gurobi` Conda environment did not have it installed, so this targeted import failed:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "import mapper.mem_planner"
```

The user later installed `tqdm` into the environment. After that environment update, the same targeted import passes.

## Validation

Smoke checks in the documented `gurobi` Conda environment:

- `scripts/motiv_exp_runner.py --help`: PASS
- `scripts/abla_exp_runner.py --help`: PASS
- `main_approach.py --help`: PASS

Targeted dependency check:

- `import mapper.mem_planner`: PASS after user installed `tqdm`.

Earlier batch-time error before the environment update:

```text
ModuleNotFoundError: No module named 'tqdm'
```

Follow-up verification after user installed `tqdm`:

```text
TQDM_IMPORT_PASS 4.68.2
MEM_PLANNER_IMPORT_PASS
```

## Recovery

To restore the original manifest from the archive ref:

```bash
git checkout archive/test_pipeline-20260612 -- requirement.txt
```
