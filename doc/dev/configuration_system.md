# 配置系统详解

> 本文档说明参数从命令行到仿真函数的传递机制。

---

## 1. 配置流总览

```
JSON 文件 (cfgs/*.json)
    │
    ▼
BinPackConfig 类 (sched/binpack_config.py)
    │
    ▼
args.binpack_cfg (运行时)
    │
    ▼
binpack_cfg dict (函数参数)
    │
    ▼
调度算法 (sched/global_sched.py)
```

---

## 2. BinPackConfig 类

**位置**: `sched/binpack_config.py:7-146`

继承自 `dict`，同时支持字典访问和属性访问。

### 2.1 核心参数

| 参数 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `algorithm` | str | "coalescing" | 算法选择 |
| `sort` | str | "EAT" | Bin 排序策略 |
| `mode` | str | "non-block" | 插入模式 |
| `bin_sel_mod` | str | "search" | Bin 选择方式 |
| `reservation_policy` | str | "manual" | 预留策略 |
| `affinity_en` | bool | True | 启用亲和性 |
| `preempt_en` | bool | True | 允许抢占 |
| `quantile` | float | 0.99 | **运行时注入** |
| `var_dist_map` | Dict | {} | **运行时注入** |
| `mapping` | Dict | {} | **运行时注入** |
| `exec_t_comp_ratioB` | float | -1 | **运行时注入** |

### 2.2 参数来源

| 来源 | 示例参数 | 注入位置 |
|------|----------|----------|
| JSON 文件 | `algorithm`, `affinity_en` | `utils.input_parser()` |
| 运行时 | `quantile`, `var_dist_map` | `sim_main.prepare_binpack_cfg()` |
| 命令行 | `--bin_pack_cfg`, `--num_bins` | `argparse` |

---

## 3. 参数注入点

### 3.1 JSON 加载

**位置**: `utils.py:182-196`

```python
binpack_cfg_dict = json.load(open(cfg_path))
binpack_cfg_dict.update(args.bin_pack_para)
args.binpack_cfg = BinPackConfig(binpack_cfg_dict)

# 后续注入 exec_t_comp_ratioB
args.binpack_cfg.update({"exec_t_comp_ratioB": args.exec_t_comp_ratioB})
```

### 3.2 运行时注入

**位置**: `sim_main.py:423-429`

```python
def prepare_binpack_cfg(cfg, quantile, p_list, graph):
    new_cfg_dict = dict(cfg)
    new_cfg_dict['var_dist_map'] = {
        p.task.name: graph.nodes[p.task.name]['var_dist']
        for p in p_list
    }
    new_cfg_dict['quantile'] = quantile
    return BinPackConfig(new_cfg_dict)
```

### 3.3 全局参数注入

**位置**: `sched/global_sched.py` (coleasing_alloc_cluster 入口)

```python
# 注入 total_cores 到 binpack_cfg
binpack_cfg["total_cores"] = total_cores
```

---

## 4. 已知问题

### 4.1 僵尸参数

**位置**: `sched/global_sched.py:push_step_new()`

```python
def push_step_new(..., percentile: float, ...):
    # percentile 参数在函数签名中，但未在函数体中使用
    # 实际 quantile 通过 binpack_cfg['quantile'] 传递
```

### 4.2 双重传递

部分参数同时通过两种方式传递：
1. 直接函数参数
2. `binpack_cfg` 字典

可能导致不一致。

### 4.3 更新位置耦合

`exec_t_comp_ratioB` 在 `build_path_old()` 中更新，与路径构建耦合。

---

## 5. 配置文件

### 5.1 Bp_guided.json

```json
{
    "algorithm": "guided",
    "sort": "barycenter",
    "mode": "block",
    "bin_sel_mod": "search",
    "reservation_policy": "static_1_bin",
    "affinity_en": true,
    "affinity_level": 2,
    "preempt_en": false,
    "partial_alloc_en": false,
    "core_size": "induced"
}
```

### 5.2 Bp_scratch.json

```json
{
    "algorithm": "scratch",
    "sort": "barycenter",
    "mode": "block",
    "bin_sel_mod": "search",
    "reservation_policy": "manual",
    "affinity_en": true,
    "preempt_en": true,
    "core_size": "specified"
}
```

---

## 6. 相关文档

- [binpack_config_design.md](../spec/cfg/binpack_config_design.md) - 设计规范
- [parameter_flow_exec_t_comp_ratio.md](../parameter_flow_exec_t_comp_ratio.md) - 参数流向分析
