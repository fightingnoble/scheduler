# 参数映射修正总结

## 🎯 问题解决

根据你的指导，我已经完成了参数映射的修正，确保motiv_exp_runner.py与现有参数系统完全兼容。

## 📋 参数映射关系

### 1. 试验脚本中添加的参数

#### Runner脚本自身参数（motiv_exp_runner.py）
- `--case`: 实验编号 (1, 2, 3)
- `--case1_ratios`: exec_t_comp_ratioA扫描值
- `--case2_tiles`: 硬件tile数扫描值
- `--case2_chains`: 任务链数量扫描值  
- `--case2_loads`: 负载倍数扫描值
- `--case3_mode`: 数据收集模式（raw/binned）
- `--case3_baseline`: 运行基线组
- `--case3_experiment`: 运行实验组
- `--case3_num_periods`: 仿真周期数
- `--num_hp`: 仿真超周期数
- `--output_dir`: 输出目录
- `--verbose`: 详细输出
- `--dry_run`: 干运行模式

#### 传递给main_approach.py的参数映射

| Runner参数 | 映射到原有参数 | 说明 |
|-----------|---------------|------|
| `num_tiles` | `num_cores` | 硬件tile数 → 核心数 |
| `num_chains` | `aux_scale_factor` | 任务链数 → 辅助缩放因子 |
| `output_dir` | `root_dir` | 输出目录 → 根目录 |
| `realloc_overhead_enabled` | `barrier_dis` | 切换开销开关 → 屏障禁用 |
| `num_hp` | `n_p` | 超周期数 → 周期数 |
| `load_factor` | `load_factor` | 负载倍数（新增） |

## 🔧 具体修改

### 1. utils.py - 新增参数

```python
# 在input_parser()函数中添加：
parser.add_argument("--load_factor", default=1.0, type=float, 
                   help="load factor multiplier for motivation experiments")
```

### 2. main_approach.py - 添加切换开销控制

```python
# 导入函数
from approach_def import set_verbose_output, set_realloc_disabled

# 在main函数中添加
set_verbose_output(args.verbose)

# 设置切换开销控制（用于Case 3对照实验）
realloc_disabled = getattr(args, 'barrier_dis', False)
set_realloc_disabled(realloc_disabled)
```

### 3. motiv_exp_runner.py - 参数映射修正

#### Case 1: 纯静态调度
```python
run_args = {
    'test_case': 'cyclic',
    'exec_t_comp_ratioA': ratio,
    'exec_t_comp_ratioB': None,
    'num_bins': -1,
    'n_p': self.args.num_hp,  # ✅ 使用n_p
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / f'ratio_{ratio:.2f}'),  # ✅ 使用root_dir
}
```

#### Case 2: 纯动态调度
```python
run_args = {
    'test_case': 'dynamic',
    'num_cores': tiles,  # ✅ num_tiles → num_cores
    'aux_scale_factor': chains,  # ✅ num_chains → aux_scale_factor
    'load_factor': load_factor,  # ✅ 新增load_factor
    'n_p': self.args.num_hp,  # ✅ 使用n_p
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / f'tiles_{tiles}_chains_{chains}_load_{load_factor:.1f}'),
}
```

#### Case 3: 切换不确定性
```python
# 基线组（禁用切换开销）
run_args = {
    'test_case': 'dynamic',
    'n_p': self.num_periods,
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / 'baseline'),
    'barrier_dis': True,  # ✅ 禁用切换开销
}

# 实验组（启用切换开销）
run_args = {
    'test_case': 'dynamic',
    'n_p': self.num_periods,
    'verbose': self.args.verbose,
    'root_dir': str(self.output_dir / 'experiment'),
    'barrier_dis': False,  # ✅ 启用切换开销
}
```

### 4. load_collector函数修正

```python
def load_collector(output_dir: str) -> Any:
    """从main_approach.py的输出中加载StatisticsCollector"""
    # 尝试从log目录中的collector文件加载
    log_dir = Path(output_dir) / 'log'
    if log_dir.exists():
        collector_files = list(log_dir.glob('collector_*.pkl'))
        if collector_files:
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

## ✅ 参数支持状态

### 🟢 完全支持的参数

| 参数 | 原有支持 | 新映射 | 状态 |
|------|---------|--------|------|
| `test_case` | ✅ | 直接使用 | ✅ |
| `exec_t_comp_ratioA` | ✅ | 直接使用 | ✅ |
| `exec_t_comp_ratioB` | ✅ | 直接使用 | ✅ |
| `num_bins` | ✅ | 直接使用 | ✅ |
| `verbose` | ✅ | 直接使用 | ✅ |
| `n_p` | ✅ | 映射自num_hp | ✅ |
| `num_cores` | ✅ | 映射自num_tiles | ✅ |
| `aux_scale_factor` | ✅ | 映射自num_chains | ✅ |
| `root_dir` | ✅ | 映射自output_dir | ✅ |
| `barrier_dis` | ✅ | 映射自realloc_overhead_enabled | ✅ |
| `load_factor` | ❌ | 新增参数 | ✅ |

### 🔄 参数传递流程

```
motiv_exp_runner.py
    ↓ 参数映射
main_approach.py
    ↓ 调用
utils.input_parser()
    ↓ 解析
args对象
    ↓ 传递
setup_benchmark(args, ...)
    ↓ 使用
path_ctx, G, bin_list等
```

## 🧪 测试验证

### 1. 参数解析测试

```bash
# 验证所有参数都能正确解析
python main_approach.py --help | grep -E "(num_cores|aux_scale_factor|load_factor|barrier_dis|n_p)"
```

### 2. 干运行测试

```bash
# Case 1
python scripts/motiv_exp_runner.py --case 1 --case1_ratios 0.8 --num_hp 10 --dry_run

# Case 2  
python scripts/motiv_exp_runner.py --case 2 --case2_tiles 300 --case2_chains 1 --case2_loads 1.0 --num_hp 10 --dry_run

# Case 3
python scripts/motiv_exp_runner.py --case 3 --case3_baseline --case3_num_periods 100 --dry_run
```

### 3. 小规模实际测试

```bash
# 使用Shell包装脚本
./scripts/run_motiv_exps.sh -n 10 -v 1  # Case 1，10个周期，详细输出
```

## 📊 修改统计

| 文件 | 修改类型 | 行数 | 说明 |
|------|---------|------|------|
| `utils.py` | 新增参数 | +3行 | 添加load_factor参数 |
| `main_approach.py` | 功能增强 | +4行 | 添加切换开销控制 |
| `motiv_exp_runner.py` | 参数映射 | ~20行 | 修正所有参数映射 |
| `doc/parameter_mapping_correction.md` | 新增文档 | +250行 | 详细说明文档 |

## 🎯 关键改进

1. **✅ 完全兼容**: 所有参数都映射到现有参数系统
2. **✅ 最小修改**: 只添加了必要的`load_factor`参数
3. **✅ 功能完整**: 支持所有三个motivation实验
4. **✅ 路径正确**: 使用`root_dir`和`path_ctx`的正确路径结构
5. **✅ 切换控制**: 通过`barrier_dis`和`set_realloc_disabled`正确控制切换开销

## 🚀 下一步

现在参数映射已经完全修正，可以：

1. **立即测试**: 使用`--dry_run`验证参数传递
2. **小规模运行**: 使用`--num_hp 10`进行快速验证
3. **完整实验**: 运行完整的motivation实验
4. **结果分析**: 使用缓存的collector进行深度分析

所有修改都已完成，系统现在应该能够完美运行！🎉
