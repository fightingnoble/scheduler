# 参数映射修正说明

## 问题分析

在motiv_exp_runner.py中，我们使用了几个参数名，但这些参数在原有的`utils.py`的`input_parser()`中并不存在。经过分析，需要将参数映射到现有的参数名。

## 参数映射关系

### ✅ 已修正的映射

| Runner参数 | 原有参数 | 说明 |
|-----------|---------|------|
| `num_tiles` | `num_cores` | 硬件tile数 → 核心数 |
| `num_chains` | `aux_scale_factor` | 任务链数 → 辅助缩放因子 |
| `output_dir` | `root_dir` | 输出目录 → 根目录 |
| `realloc_overhead_enabled` | `barrier_dis` | 切换开销开关 → 屏障禁用 |
| `num_hp` | `n_p` | 超周期数 → 周期数 |

### ✅ 新增参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `load_factor` | float | 1.0 | 负载倍数（用于Case 2） |

## 具体修改

### 1. utils.py 新增参数

```python
# 在input_parser()函数中添加：
parser.add_argument("--load_factor", default=1.0, type=float, 
                   help="load factor multiplier for motivation experiments")
```

### 2. main_approach.py 添加切换开销控制

```python
# 导入函数
from approach_def import set_verbose_output, set_realloc_disabled

# 在main函数中添加
set_verbose_output(args.verbose)

# 设置切换开销控制（用于Case 3对照实验）
realloc_disabled = getattr(args, 'barrier_dis', False)
set_realloc_disabled(realloc_disabled)
```

### 3. motiv_exp_runner.py 参数映射修正

#### Case 1 参数映射

```python
run_args = {
    'test_case': 'cyclic',  # 纯静态调度
    'exec_t_comp_ratioA': ratio,
    'exec_t_comp_ratioB': None,  # 纯静态不需要B
    'num_bins': -1,  # 每个任务一个bin
    'n_p': self.args.num_hp,  # ✅ 使用n_p而不是num_hp
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / f'ratio_{ratio:.2f}'),  # ✅ 使用root_dir
}
```

#### Case 2 参数映射

```python
run_args = {
    'test_case': 'dynamic',  # 纯动态调度（glb）
    'num_cores': tiles,  # ✅ num_tiles → num_cores
    'aux_scale_factor': chains,  # ✅ num_chains → aux_scale_factor
    'load_factor': load_factor,  # ✅ 新增load_factor参数
    'n_p': self.args.num_hp,  # ✅ 使用n_p而不是num_hp
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / f'tiles_{tiles}_chains_{chains}_load_{load_factor:.1f}'),  # ✅ 使用root_dir
}
```

#### Case 3 参数映射

```python
# 基线组（禁用切换开销）
run_args = {
    'test_case': 'dynamic',
    'n_p': self.num_periods,  # ✅ 使用n_p而不是num_hp
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / 'baseline'),  # ✅ 使用root_dir
    'barrier_dis': True,  # ✅ 禁用切换开销（基线组）
}

# 实验组（启用切换开销）
run_args = {
    'test_case': 'dynamic',
    'n_p': self.num_periods,  # ✅ 使用n_p而不是num_hp
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / 'experiment'),  # ✅ 使用root_dir
    'barrier_dis': False,  # ✅ 启用切换开销（实验组）
}
```

### 4. load_collector 函数修正

```python
def load_collector(output_dir: str) -> Any:
    """
    从main_approach.py的输出中加载StatisticsCollector
    """
    # 尝试从log目录中的collector文件加载
    log_dir = Path(output_dir) / 'log'
    if log_dir.exists():
        collector_files = list(log_dir.glob('collector_*.pkl'))
        if collector_files:
            # 使用最新的collector文件
            latest_file = max(collector_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'rb') as f:
                return pickle.load(f)
    
    # 尝试从根目录直接查找
    root_dir = Path(output_dir)
    collector_files = list(root_dir.glob('collector_*.pkl'))
    if collector_files:
        latest_file = max(collector_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Warning: Could not load collector from {output_dir}")
    return None
```

## 参数含义说明

### 原有参数含义

| 参数 | 含义 | 用途 |
|------|------|------|
| `num_cores` | 核心数量 | 硬件规模配置 |
| `aux_scale_factor` | 辅助缩放因子 | 影响任务链数量 |
| `root_dir` | 根目录 | 输出路径配置 |
| `barrier_dis` | 禁用屏障 | 控制切换行为 |
| `n_p` | 周期数 | 仿真长度控制 |
| `load_factor` | 负载倍数 | 负载压力调节 |

### 实验参数对应关系

#### Case 1: 纯静态调度
- **扫描参数**: `exec_t_comp_ratioA` (预留分位数)
- **固定参数**: `test_case='cyclic'`, `num_bins=-1`
- **控制参数**: `n_p` (仿真长度)

#### Case 2: 纯动态调度
- **扫描参数**: `num_cores` (硬件规模), `aux_scale_factor` (任务复杂度), `load_factor` (负载压力)
- **固定参数**: `test_case='dynamic'`
- **控制参数**: `n_p` (仿真长度)

#### Case 3: 切换不确定性
- **对照参数**: `barrier_dis` (True=基线组, False=实验组)
- **固定参数**: `test_case='dynamic'`
- **控制参数**: `n_p` (仿真长度), `case3_mode` (数据收集模式)

## 验证清单

### ✅ 已完成的修改

1. **utils.py**: 添加`load_factor`参数
2. **main_approach.py**: 添加`set_realloc_disabled`调用
3. **motiv_exp_runner.py**: 修正所有参数映射
4. **load_collector**: 修正路径查找逻辑

### 🔍 需要验证的功能

1. **参数传递**: 确认所有参数能正确传递到`setup_benchmark()`
2. **路径生成**: 确认`root_dir`能正确生成输出路径
3. **切换控制**: 确认`barrier_dis`能正确控制切换开销
4. **collector保存**: 确认collector能正确保存到指定路径
5. **collector加载**: 确认collector能正确从指定路径加载

### 🧪 测试建议

```bash
# 1. 测试参数解析
python main_approach.py --help | grep -E "(num_cores|aux_scale_factor|load_factor|barrier_dis|n_p)"

# 2. 测试Case 1
python scripts/motiv_exp_runner.py --case 1 --case1_ratios 0.8 --num_hp 10 --dry_run

# 3. 测试Case 2
python scripts/motiv_exp_runner.py --case 2 --case2_tiles 300 --case2_chains 1 --case2_loads 1.0 --num_hp 10 --dry_run

# 4. 测试Case 3
python scripts/motiv_exp_runner.py --case 3 --case3_baseline --case3_num_periods 100 --dry_run
```

## 注意事项

1. **路径结构**: `root_dir`会影响整个输出路径结构，需要确保与`path_ctx`兼容
2. **参数冲突**: 确保新参数不会与现有参数产生冲突
3. **默认值**: 所有新参数都有合理的默认值
4. **向后兼容**: 修改不影响现有功能

## 总结

通过这次参数映射修正，我们：

1. ✅ **复用现有参数**: 最大化利用现有参数解析器
2. ✅ **最小化修改**: 只添加了必要的`load_factor`参数
3. ✅ **保持兼容**: 不影响现有代码功能
4. ✅ **清晰映射**: 建立了明确的参数对应关系

现在motiv_exp_runner.py应该能够与现有的参数系统完全兼容。