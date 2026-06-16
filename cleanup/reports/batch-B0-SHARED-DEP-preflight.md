# Preflight report: B0-SHARED-DEP

Date: 2026-06-13

Decision ID: `P1-DEP-001`

Mode: preflight only. No manifest edit and no package installation were performed.

## Read-only checks

Current `requirement.txt`:

```text
scipy
numpy
pandas
matplotlib
pyyaml
plotly
bokeh
```

`gurobi` Conda environment import check:

```text
OK,gurobipy,gurobipy,11.0.3
OK,h5py,h5py,3.14.0
OK,networkx,networkx,3.2.1
OK,psutil,psutil,7.1.0
OK,pyinstrument,pyinstrument,5.0.3
OK,pytest,pytest,8.4.2
OK,tdigest,tdigest,0.5.2.2
FAIL,tqdm,tqdm,ModuleNotFoundError:No module named 'tqdm'
FAIL,bokeh,bokeh,ModuleNotFoundError:No module named 'bokeh'
FAIL,plotly,plotly,ModuleNotFoundError:No module named 'plotly'
FAIL,pyyaml,yaml,ModuleNotFoundError:No module named 'yaml'
OK,numpy,numpy,2.0.2
OK,pandas,pandas,2.3.1
OK,matplotlib,matplotlib,3.9.4
OK,scipy,scipy,1.13.1
```

Targeted import check:

```text
PYTHONDONTWRITEBYTECODE=1 python -c "import mapper.mem_planner"
```

Result:

```text
ModuleNotFoundError: No module named 'tqdm'
```

## Preflight conclusion

Recommended next action is a repo-only manifest update:

- Keep current `requirement.txt` entries.
- Add `P1-DEP-001` dependencies:
  - `gurobipy`
  - `h5py`
  - `networkx`
  - `psutil`
  - `pyinstrument`
  - `pytest`
  - `tdigest`
  - `tqdm`

Do not install packages into the Conda environment without explicit approval.

