Case 1的柱状图：添加对数轴、三个柱子（idle/miss/realloc）和附轴显示miss_count。

**双轴设计**:
- **主轴（左Y轴）**: 对数轴显示三个ratio的百分比
- **附轴（右Y轴）**: 线性轴显示miss_count数量

**三个柱子**:
- 🟠 **橙色柱**: Idle Ratio（闲置算力占比）
- 🟢 **绿色柱**: Miss Ratio（miss负载占比）  
- 🔴 **红色柱**: Realloc Ratio（切换开销占比）

**附轴线图**:
- 🔵 **蓝色线**: Miss Count（miss任务数量）

### 📊 视觉改进

1. **对数轴优势**:
   - 能同时显示大值（如0.1）和小值（如0.001）
   - 更好地展示ratio的变化趋势
   - 添加了1%和10%参考线

2. **数值标注**:
   - 柱子顶部：科学计数法（<0.01）或小数（≥0.01）
   - 线上标注：miss_count的精确数值

3. **颜色编码**:
   - 主轴：黑色标签和网格
   - 附轴：蓝色标签
   - 图例：合并显示所有元素

### 🔧 技术实现

```python
# 主轴：对数轴柱状图
ax1.set_yscale('log')
bars1 = ax1.bar(x_pos - width, idle_ratios, ...)    # 橙色
bars2 = ax1.bar(x_pos, miss_ratios, ...)           # 绿色  
bars3 = ax1.bar(x_pos + width, realloc_ratios, ...) # 红色

# 附轴：线性轴线图
ax2 = ax1.twinx()
line = ax2.plot(x_pos, miss_counts, 'o-', ...)     # 蓝色线
```

### 📈 期望效果

对于纯静态调度（Case 1）：
- **Idle柱**: 随预留分位数增加而升高（p50→p99）
- **Miss柱**: 随预留分位数增加而降低
- **Realloc柱**: 应接近0（纯静态无切换开销）
- **Miss Count线**: 随预留分位数增加而下降

这种设计能清晰展示静态方法的"利用率-可靠性权衡"关系！🎯