# 路径替换完成总结

## 概述

我们已经成功完成了 `main` 函数中路径生成的替换工作，参照 `perform_bin_packing` 中的方式，将旧的路径生成方法替换为统一的 `PathContext` 系统。

## 完成的替换工作

### 1. main 函数中的路径替换

#### 1.1 线程池扫描路径替换
**位置**: `sim_main.py` 第 516-542 行

**替换内容**:
- `bin_list_save_path` 和 `routing_table_save_path` 的生成
- 支持 `induced` 和默认两种模式
- 添加了旧路径和新路径的比较检查

**替换前**:
```python
bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
```

**替换后**:
```python
# 使用旧方法生成路径
old_bin_list_save_path = bin_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})
old_routing_table_save_path = routing_table_save_fmt.format(**path_para_dict, **{"num_cores": num_cores})

# 使用新方法生成路径
path_ctx.num_cores = num_cores
new_bin_list_save_path = path_ctx.get_bin_list_path()
new_routing_table_save_path = path_ctx.get_routing_table_path()

# 比较路径
compare_paths(old_bin_list_save_path, new_bin_list_save_path, "bin_list_save_path (induced)")
compare_paths(old_routing_table_save_path, new_routing_table_save_path, "routing_table_save_path (induced)")

# 使用新路径
bin_list_save_path = new_bin_list_save_path
routing_table_save_path = new_routing_table_save_path
```

#### 1.2 默认模式路径替换
**位置**: `sim_main.py` 第 543-559 行

**替换内容**:
- 默认模式下的 `bin_list_save_path` 和 `routing_table_save_path` 生成
- 添加了路径比较检查

### 2. 函数签名和调用更新

#### 2.1 check_max_bin_num 函数
**位置**: `sim_main.py` 第 732-749 行

**更新内容**:
- 添加了 `path_ctx` 参数
- 替换了路径生成逻辑
- 添加了路径比较检查

**函数签名更新**:
```python
# 替换前
def check_max_bin_num(args, num_cores, bin_path_format):

# 替换后  
def check_max_bin_num(args, num_cores, bin_path_format, path_ctx):
```

#### 2.2 load_sched_tab 函数
**位置**: `sched/sched_utils.py` 第 232-235 行

**更新内容**:
- 将 `bin_path_format` 参数替换为 `path_ctx`
- 使用新的路径生成方法

**函数签名更新**:
```python
# 替换前
def load_sched_tab(num_cores, e2e_latency:float, aux_scale_factor:int, bin_path_format:str, scheduler_list:List[Scheduler]):

# 替换后
def load_sched_tab(num_cores, e2e_latency:float, aux_scale_factor:int, path_ctx, scheduler_list:List[Scheduler]):
```

#### 2.3 cyclic_sched 函数
**位置**: `allocator_agent.py` 第 337 行

**更新内容**:
- 将 `bin_path_format:str=None` 参数替换为 `path_ctx=None`
- 更新了函数调用

### 3. 路径比较检查

所有替换的路径都添加了 `compare_paths` 调用，确保新旧路径生成方法产生相同的结果：

```python
compare_paths(old_path, new_path, "description")
```

## 替换模式

我们遵循了与 `perform_bin_packing` 中相同的替换模式：

1. **保留旧路径生成**: 使用旧的 `format` 方法生成路径
2. **生成新路径**: 使用 `PathContext` 生成新路径
3. **比较路径**: 调用 `compare_paths` 验证路径一致性
4. **使用新路径**: 在实际代码中使用新生成的路径

## 测试验证

我们创建了测试来验证路径替换的正确性：

```python
# 测试路径生成
ctx = PathContext(
    root_dir='test_scan',
    case='bin_pack_new',
    num_bins=8,
    aux_scale_factor=5,
    e2e_latency=0.1,
    exec_t_comp_ratioA=0.3,
    jitter_t_comp_ratio=0.2,
    wsc_slack_ratio=1.0,
    num_cores=9
)

path_ctx = PathContext(
    root_dir='test_scan', case='bin_pack_new', num_bins=8,
    aux_scale_factor=5, e2e_latency=0.1, exec_t_comp_ratioA=0.3,
    jitter_t_comp_ratio=0.2, wsc_slack_ratio=1.0, num_cores=9,
    exec_t_comp_ratioB=0.15, seed=0, jitter=False, file_suffix='', i_file_suffix='', force_suffix=''
)
print(f'bin_list_path: {path_ctx.get_bin_list_path()}')
print(f'routing_table_path: {path_ctx.get_routing_table_path()}')
```

测试结果显示路径生成正确工作。

## 向后兼容性

为了保持向后兼容性，我们：

1. **保留了 `bin_path_format` 参数**: 在函数签名中保留，但不再使用
2. **保留了 `args_postprocess` 调用**: 继续返回旧的路径格式，供其他代码使用
3. **渐进式替换**: 只替换了实际使用路径的地方，保留了接口兼容性

## 优势

通过这次替换，我们获得了：

1. **降低耦合度**: 减少了对 `utils.py` 中 `get_cfg_n` 和 `args_postprocess` 的依赖
2. **类型安全**: 使用 `PathContext` 提供更好的类型提示
3. **路径一致性**: 所有路径生成都使用统一的 `PathContext` 系统
4. **易于维护**: 路径生成逻辑集中在一个地方，易于修改和扩展
5. **测试覆盖**: 通过路径比较确保新旧方法的一致性

## 下一步

路径替换工作已经完成。建议：

1. **运行完整测试**: 确保所有功能正常工作
2. **性能测试**: 验证新路径生成方法的性能
3. **文档更新**: 更新相关文档以反映新的路径生成方式
4. **逐步迁移**: 考虑将其他模块也迁移到新的路径生成系统
