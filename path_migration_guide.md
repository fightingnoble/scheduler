# PathContext 迁移指南

## 概述

本指南说明如何从依赖 `utils.py` 中的 `get_cfg_n` 和 `args_postprocess` 函数迁移到统一的 `PathContext` 系统。

## 主要改进

### 1. 降低模块耦合度
- **旧方式**: 需要调用 `utils.py` 中的函数来生成路径
- **新方式**: `PathContext` 在 `__post_init__` 中自动生成静态路径

### 2. 扫描变量对齐
- **旧方式**: 扫描变量分散在不同地方
- **新方式**: `PathContext` 包含所有扫描相关变量

### 3. 动态调整方式
- **旧方式**: 路径生成和参数调整混合在一起
- **新方式**: 直接修改 `PathContext` 字段，若影响路径（cfg_n）则调用 `refresh_config()`

## 迁移步骤

### 步骤1: 替换路径生成逻辑

**旧代码:**
```python
from utils import get_cfg_n, args_postprocess

# 需要调用 utils.py 中的函数
cfg_para_dict, para_scan_group1, cfg_n = get_cfg_n(args)
cfg_para_dict, para_scan_group1, para_scan_group2, path_para_dict, bin_path_format, trace_path_para, plot_path_para, csv_xlxs_root = args_postprocess(args)
```

**新代码:**
```python
from paths import PathContext

# 直接创建 PathContext，自动生成所有路径
ctx = PathContext(
    root_dir=args.root_dir,
    case=args.case,
    exec_t_comp_ratioA=args.exec_t_comp_ratioA,
    exec_t_comp_ratioB=args.exec_t_comp_ratioB,
    jitter_t_comp_ratio=args.jitter_t_comp_ratio,
    wsc_slack_ratio=args.wsc_slack_ratio,
    aux_scale_factor=args.aux_scale_factor,
    e2e_latency=args.e2e_latency,
    num_cores=args.num_cores,
    seed=args.seed,
    jitter=args.jitter_sim_en
)

path_ctx = ctx
```

### 步骤2: 使用便利函数

**扫描类型特定的创建:**
```python
from paths import create_tp_scan_context, create_bin_scan_context, create_e2e_scan_context

# 线程池扫描
tp_ctx = create_tp_scan_context("tp_scan", num_cores=9, force_num_cores=True)

# Bin数量扫描  
bin_ctx = create_bin_scan_context("bin_scan", num_bins=24)

# 端到端扫描
e2e_ctx = create_e2e_scan_context("e2e_scan", aux_scale_factor=5, e2e_latency=0.09)
```

### 步骤3: 动态调整参数

**旧代码:**
```python
# 需要手动修改参数和重新生成路径
args.num_cores = new_num_cores
args.force_suffix = "force_" if args.force_num_cores else ""
# 重新调用 args_postprocess
```

**新代码:**
```python
# 直接修改 PathContext 字段
path_ctx.num_cores = new_num_cores
path_ctx.force_suffix = "force_"
```

### 步骤4: 获取路径

**旧代码:**
```python
# 从 args_postprocess 返回的字典中获取路径
bin_path = bin_path_format.format(num_cores=args.num_cores)
trace_path = trace_fn_w_seed_fmt.format(**trace_path_para)
plot_path = plt_fn_w_seed_fmt.format(**plot_path_para)
```

**新代码:**
```python
# 直接从 PathContext 获取路径
bin_path = path_ctx.get_bin_list_path()
trace_path = path_ctx.get_trace_path()
plot_path = path_ctx.get_plot_path("large")
```

## 扫描脚本迁移示例

### aba_scalability_scan.sh 对应的 Python 代码

**线程池扫描:**
```python
# 对应 scan_type="tp"
for num_cores in range(tp_start, tp_end + 1, tp_step):
    ctx = create_tp_scan_context(
        root_dir=root_dir,
        num_cores=num_cores,
        force_num_cores=True,
        exec_t_comp_ratioA=exec_t_comp_ratioA,
        exec_t_comp_ratioB=exec_t_comp_ratioB,
        jitter_t_comp_ratio=jitter_comp_cfg,
        wsc_slack_ratio=wsc_slack_ratio
    )
    # 直接使用 PathContext 生成各种路径
```

**Bin数量扫描:**
```python
# 对应 scan_type="bin"
for num_bins in list_n_bin_cfg:
    ctx = create_bin_scan_context(
        root_dir=root_dir,
        num_bins=num_bins,
        exec_t_comp_ratioA=exec_t_comp_ratioA,
        exec_t_comp_ratioB=exec_t_comp_ratioB
    )
    # 直接使用 PathContext 生成各种路径
```

**补偿比率B扫描:**
```python
# 对应 scan_type="ratioB"
for ratioB in ratioB_incr_list:
    new_ratioB = exec_t_comp_ratioB + ratioB
    ctx = create_compensation_scan_context(
        root_dir=root_dir,
        exec_t_comp_ratioA=exec_t_comp_ratioA,
        exec_t_comp_ratioB=new_ratioB
    )
    # 直接使用 PathContext，必要时拼接后缀
    ctx.file_suffix += f"_ov_{ratioB:.2f}_repack"
```

### abla_scalablility.sh 对应的 Python 代码

```python
# 对应延迟和辅助因子扫描
for lat in [0.1, 0.09, 0.08]:
    for n_aux in range(aux_start, aux_end + 1, aux_step):
        ctx = create_e2e_scan_context(
            root_dir=root_dir,
            aux_scale_factor=n_aux,
            e2e_latency=lat
        )
        # 直接使用 PathContext 生成路径
```

## 优势总结

1. **减少依赖**: 不再需要导入 `utils.py` 中的函数
2. **类型安全**: 使用 dataclass 提供更好的类型提示
3. **直接赋值**: 使用 PathContext 字段直接调整参数
4. **自动路径生成**: PathContext 自动生成所有静态路径
5. **扫描变量对齐**: 所有扫描相关变量都在 PathContext 中
6. **易于测试**: 可以轻松创建测试用的 PathContext 实例

## 注意事项

1. 所有路径在 `__post_init__` 中自动生成，无需手动调用
2. 修改会影响 cfg_n 的字段后需调用 `refresh_config()`（如 e2e/aux/补偿比率/late 模式等）
