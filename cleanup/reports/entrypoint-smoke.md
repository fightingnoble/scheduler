# Entrypoint smoke results

No experiment execution was attempted. Only `--help` import probes were run.

The first probe used system `python3` and failed because it did not follow `CLAUDE.md`.
The repository requires:

```bash
conda activate gurobi
```

After rerunning through `zsh -ic` with the `gurobi` Conda environment, all three probes passed:

- `scripts/motiv_exp_runner.py --help`: PASS
- `scripts/abla_exp_runner.py --help`: PASS
- `main_approach.py --help`: PASS

Working command shape:

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python scripts/motiv_exp_runner.py --help'
```

Evidence files:

- `cleanup/reports/help-motiv-conda.txt`
- `cleanup/reports/help-abla-conda.txt`
- `cleanup/reports/help-main-approach-conda.txt`

The old `cleanup/reports/help-*.txt` files are base-Python failures and should not be read as repo runtime failures.
