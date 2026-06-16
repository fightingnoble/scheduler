# Runtime contract

## Declared entrypoints
- `scripts/motiv_exp_runner.py`
- `scripts/abla_exp_runner.py`
- `main_approach.py`

## Required runtime inputs promoted to KEEP
- `cfgs/Bp_guided.json`: default `--bin_pack_cfg`
- `cfgs/Bp_scratch.json`: documented alternate bin-packing mode
- `cfgs/var_sim_cfg.json`: default `--var_sim_cfg`, loaded unconditionally by `utils.input_parser()`
- `profiling/profiling_light.csv`: current default profile for experiment runners
- `profiling/profiling.csv`: default profile for lower-level task/slack helpers

## Smoke result

The repository must be tested inside the `gurobi` Conda environment, as stated in `CLAUDE.md`.

Using system `python3` produced missing-module errors. That was an environment mistake, not a valid repo failure.

Using the required environment, all import-only `--help` probes pass:

- `scripts/motiv_exp_runner.py --help`
- `scripts/abla_exp_runner.py --help`
- `main_approach.py --help`

Command shape:

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help'
```

`cleanup/dependency-ledger.csv` now records manifest/environment reconciliation items, not confirmed runtime blockers.
