# Smoke result: B0-SHARED-DEP

Batch: `B0-SHARED-DEP`

Decision: `P1-DEP-001`

The batch updated only `requirement.txt`. It did not install dependencies or mutate the `gurobi` Conda environment.

## Commands

```bash
wsl -d Ubuntu-20.04 --cd /home/zhangchg/git_repo/scheduler-audit-20260612 -- zsh -ic 'conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python scripts/motiv_exp_runner.py --help >/dev/null && PYTHONDONTWRITEBYTECODE=1 python scripts/abla_exp_runner.py --help >/dev/null && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help >/dev/null && echo B0_DEP_SMOKE_PASS'
```

```bash
wsl -d Ubuntu-20.04 --cd /home/zhangchg/git_repo/scheduler-audit-20260612 -- zsh -ic 'conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python -c "import mapper.mem_planner"'
```

## Result

- `scripts/motiv_exp_runner.py --help`: PASS
- `scripts/abla_exp_runner.py --help`: PASS
- `main_approach.py --help`: PASS

Output:

```text
B0_DEP_SMOKE_PASS
```

Targeted dependency check:

```text
ModuleNotFoundError: No module named 'tqdm'
```

This targeted failure was expected at batch execution time because the batch updated the manifest only. It did not install `tqdm` into the current `gurobi` Conda environment.

## Environment follow-up

The user later installed `tqdm` into the `gurobi` environment. Follow-up verification:

```bash
wsl -d Ubuntu-20.04 --cd /home/zhangchg/git_repo/scheduler-audit-20260612 -- zsh -ic 'conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python - <<\"PY\"
import importlib.metadata as md
import tqdm
import mapper.mem_planner
print(\"TQDM_IMPORT_PASS\", md.version(\"tqdm\"))
print(\"MEM_PLANNER_IMPORT_PASS\")
PY'
```

Output:

```text
TQDM_IMPORT_PASS 4.68.2
MEM_PLANNER_IMPORT_PASS
```
