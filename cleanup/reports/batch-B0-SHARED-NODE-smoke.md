# Smoke result: B0-SHARED-NODE

Batch: `B0-SHARED-NODE`

Decision: `P1-NODE-001`

The batch removed only tracked empty Node manifests:

- `package.json`
- `package-lock.json`

Smoke policy:

- Use the documented `gurobi` Conda environment for repository checks.
- Use import/help probes only; do not run experiments.

Commands:

```bash
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python scripts/motiv_exp_runner.py --help'
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python scripts/abla_exp_runner.py --help'
wsl -d Ubuntu-20.04 -- zsh -ic 'cd /home/zhangchg/git_repo/scheduler-audit-20260612 && conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help'
```

Result:

- PASS.
- `scripts/motiv_exp_runner.py --help`: PASS.
- `scripts/abla_exp_runner.py --help`: PASS.
- `main_approach.py --help`: PASS.

Final confirmation command:

```bash
wsl -d Ubuntu-20.04 --cd /home/zhangchg/git_repo/scheduler-audit-20260612 -- zsh -ic 'conda activate gurobi && PYTHONDONTWRITEBYTECODE=1 python scripts/motiv_exp_runner.py --help >/dev/null && PYTHONDONTWRITEBYTECODE=1 python scripts/abla_exp_runner.py --help >/dev/null && PYTHONDONTWRITEBYTECODE=1 python main_approach.py --help >/dev/null && echo B0_SMOKE_PASS'
```

Output:

```text
B0_SMOKE_PASS
```
