# Local test plan

Phase 1 did not run experiments or install dependencies.

## Phase 2 minimum validation after dependency decision
1. Activate the intended WSL/Conda environment, required by `CLAUDE.md` as `conda activate gurobi`.
2. Re-run import-only probes:
   - `PYTHONDONTWRITEBYTECODE=1 python3 scripts/motiv_exp_runner.py --help`
   - `PYTHONDONTWRITEBYTECODE=1 python3 scripts/abla_exp_runner.py --help`
   - `PYTHONDONTWRITEBYTECODE=1 python3 main_approach.py --help`
3. Run dry/no-small experiment checks only after the import probes pass and data paths are confirmed.
4. Run targeted tests for any approved removal batch; do not use symbol-candidates as deletion authority without tests.

## Current status

- Import-only `--help` probes pass in the `gurobi` Conda environment.
- System `python3` is not a valid test environment for this repo.
- `cleanup/dependency-ledger.csv` is a manifest/environment reconciliation list, not proof that the Conda test environment is missing those packages.
- `task/task_agent.py` has unresolved reachable import `from task_cfg import task_attr_dict`; see `cleanup/reports/unresolved-imports.csv`.
