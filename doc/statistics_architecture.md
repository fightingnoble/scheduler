# 统计功能架构说明

## 架构概述

统计信息收集功能采用工厂模式设计，统计收集器在工厂函数中自动创建并分配给各个处理器，无需层层传递参数。

## 架构图

```
用户代码
    ↓
instantiate_processors()
    ↓
acc_p_factory() → 创建 StatisticsCollector
    ↓                    ↓
创建 Acc_p 实例 ←── 分配统计收集器
    ↓
创建 Sen_p 实例 ←── 分配统计收集器
    ↓
返回 (processors, event_t, stats_collector)
    ↓
run_simulation() → 使用统计收集器输出摘要
```

## 核心组件

### 1. StatisticsCollector 类
- **位置**: `approach_def.py`
- **功能**: 收集和管理所有统计信息
- **生命周期**: 在工厂函数中创建，在整个仿真过程中使用

### 2. acc_p_factory 函数
- **位置**: `approach_def.py`
- **功能**: 
  - 自动创建 `StatisticsCollector` 实例
  - 为每个加速器处理器分配统计收集器
  - 自动初始化分区统计信息
- **返回值**: `(acc_p_list, stats_collector)`

### 3. instantiate_processors 函数
- **位置**: `approach_initiator.py`
- **功能**:
  - 调用 `acc_p_factory` 获取加速器处理器和统计收集器
  - 为传感器处理器分配统计收集器
  - 返回所有处理器、事件队列和统计收集器
- **返回值**: `(processors, event_t, stats_collector)`

## 数据流

### 1. 统计收集器创建流程
```
1. 用户调用 instantiate_processors()
2. instantiate_processors() 调用 acc_p_factory()
3. acc_p_factory() 创建 StatisticsCollector 实例
4. 为每个分区初始化统计信息
5. 将统计收集器分配给所有处理器
6. 返回统计收集器给调用者
```

### 2. 统计信息收集流程
```
1. 处理器执行任务时自动记录统计信息
2. 统计收集器实时更新各种指标
3. 仿真结束后调用 print_summary() 输出统计摘要
```

## 使用方式

### 1. 基本使用（推荐）
```python
# 无需手动创建统计收集器
processors, event_t, stats_collector = instantiate_processors(
    G, partition_cfg, event_t, policy="pglb"
)

# 直接使用返回的统计收集器
run_simulation(processors, event_t, G, stats_collector=stats_collector)
```

### 2. 高级使用
```python
# 如果需要自定义统计收集器
custom_stats = StatisticsCollector()
# 可以替换默认的统计收集器
```

## 优势

### 1. 简化使用
- **无需手动创建**: 统计收集器自动创建
- **无需层层传递**: 通过工厂函数自动分配
- **统一管理**: 所有处理器共享同一个统计收集器

### 2. 自动化管理
- **自动初始化**: 分区统计信息自动初始化
- **自动分配**: 统计收集器自动分配给所有处理器
- **自动清理**: 仿真结束后自动输出统计摘要

### 3. 扩展性好
- **易于添加新指标**: 在 `StatisticsCollector` 中添加新方法
- **易于集成新处理器**: 新处理器类型自动获得统计支持
- **易于修改输出格式**: 统一在 `print_summary()` 中修改

## 注意事项

### 1. 统计收集器生命周期
- 统计收集器在工厂函数中创建
- 在整个仿真过程中持续存在
- 仿真结束后通过 `print_summary()` 输出结果

### 2. 性能考虑
- 统计收集会增加少量计算开销
- 长时间仿真时注意内存使用
- 建议在需要分析时启用

### 3. 线程安全
- 当前实现不是线程安全的
- 多线程环境下需要加锁保护
- 单线程仿真环境无需额外处理

## 扩展指南

### 1. 添加新的统计指标
```python
class StatisticsCollector:
    def __init__(self):
        # 添加新的统计字段
        self.new_metric = 0.0
    
    def record_new_metric(self, value):
        # 添加新的记录方法
        self.new_metric += value
    
    def get_summary(self):
        # 在摘要中添加新指标
        summary['new_metric'] = self.new_metric
        return summary
```

### 2. 添加新的处理器类型
```python
class NewProcessor(BaseProcessor):
    def __init__(self, id, cap, base_pwr, G, mapped_node, stats_collector=None):
        super().__init__(id, cap, base_pwr, G, mapped_node, stats_collector)
        # 新处理器的特定逻辑
```

### 3. 自定义输出格式
```python
def print_summary(self):
    # 自定义输出格式
    print("自定义统计摘要")
    # ... 自定义输出逻辑
```

## 总结

新的架构设计大大简化了统计功能的使用，用户无需关心统计收集器的创建和管理细节，所有功能都通过工厂函数自动完成。这种设计既保持了功能的完整性，又提高了代码的可维护性和易用性。 