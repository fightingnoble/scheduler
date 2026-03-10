# T-Digest 流式统计系统规范

**一句话核心**：T-Digest 是流式分位数摘要算法，解决长尾分布的统计问题——用固定内存 O(K) 实现无限数据流的精确分位数估计。

---

## 1. 设计思想

### 1.1 三个关键问题

| 问题               | 传统方案          | T-Digest 方案                |
| ------------------ | ----------------- | ---------------------------- |
| **长尾分布** | 固定分箱难以处理  | 自适应质心，尾部保持高分辨率 |
| **内存消耗** | 存储原始数据 O(N) | 流式处理 O(K)，与数据量无关  |
| **实时性**   | 批量计算          | 均摊 O(log K) 添加，实时更新 |

### 1.2 核心原理

T-Digest 通过维护一组自适应质心 (centroids) 概括数据分布：

- **密集区域**（分布中心）：质心数量少，覆盖范围大
- **稀疏区域**（分布尾部）：质心数量多，覆盖范围小

这种特性使得高分位数（p99, p999）估计特别准确，非常适合调度器中的长尾延迟分析。

### 1.3 精度控制

| 参数      | 含义     | 典型值 | 影响                   |
| --------- | -------- | ------ | ---------------------- |
| `delta` | 精度参数 | 0.01   | 越小精度越高，质心越多 |
| `K`     | 质心容量 | 25     | 每个质心的最大容量     |

---

## 2. 系统架构
    TDigestStreamingHistogram (ref_tdigest.py)
        │
        ├── 提供统一接口
        │   ├── add(value)         # 流式添加   
        │   ├── percentile(p)      # 分位数查询 
        │   └── get_summary()      # 完整摘要   
        ├── 支持序列化
        │   ├── to_dict()          # 导出状态
        │   └── from_dict()        # 恢复状态
        └── 封装 T-Digest 算法
            └── tdigest.TDigest (第三方库) 核心算法：自适应质心维护、压缩、合并  

**使用模式**：`add` 累积数据 → `get_summary` 获取分布

---

## 3. 核心接口速查

### 3.1 方法列表

| 方法                              | 复杂度                    | 用途                         |
| --------------------------------- | ------------------------- | ---------------------------- |
| `add(value)`                    | O(log K)                  | 流式添加数据点               |
| `percentile(p)`                 | O(log K)                  | 获取分位数值 (p ∈ [0, 100]) |
| `cdf(value)`                    | O(log K)                  | 获取累积分布函数值           |
| `get_mean()`                    | O(K)                      | 获取分布均值                 |
| `get_summary(num_bins, p_list)` | O(num_bins + len(p_list)) | 完整统计摘要                 |
| `to_dict()` / `from_dict()`   | O(K)                      | 序列化/反序列化              |
| `__add__(other)`                | O(K1 + K2)                | 合并两个分布                 |

### 3.2 参数选择建议

| delta | 精度 | 质心数   | 适用场景           |
| ----- | ---- | -------- | ------------------ |
| 0.01  | 高   | ~100-200 | 生产环境，精确分析 |
| 0.1   | 中   | ~50-100  | 实时监控           |
| 0.5   | 低   | ~20-50   | 快速近似           |

---

## 4. 使用示例

### 4.1 基本使用

```python
from utils import TDigestStreamingHistogram

# 创建直方图
hist = TDigestStreamingHistogram(delta=0.01, K=25)

# 流式添加数据
for value in data_stream:
    hist.add(value)

# 查询分位数
p50 = hist.percentile(50)
p99 = hist.percentile(99)

# 获取完整摘要
summary = hist.get_summary(num_bins=20, p_list=[0.5, 0.9, 0.99, 0.999])
```

### 4.2 分布合并

```python
# 从不同数据源创建两个直方图
hist1 = TDigestStreamingHistogram()
hist2 = TDigestStreamingHistogram()

for v in data_source_1:
    hist1.add(v)
for v in data_source_2:
    hist2.add(v)

# 合并分布
combined = hist1 + hist2
```

---

## 5. 实现细节

### 5.1 类定义

**文件位置**：`ref_tdigest.py`

```python
from tdigest import TDigest

class TDigestStreamingHistogram:
    """
    基于 T-Digest 的流式直方图实现。

    支持流式添加、分位数查询、直方图生成、序列化、分布合并。
    """

    def __init__(self, delta: float = 0.01, K: int = 25):
        self.tdigest = TDigest(delta=delta, K=K)
        self.total_processed_count = 0
```

### 5.2 核心方法实现

**添加数据点**：

```python
def add(self, value: float) -> None:
    """添加单个数据点。时间复杂度：O(log K) 均摊。"""
    self.tdigest.add(value)
    self.total_processed_count += 1
```

**完整摘要**：

```python
def get_summary(self, num_bins: int = 20,
                p_list: List[float] = [0.5, 0.9, 0.99, 0.999]) -> Dict:
    """获取完整统计摘要。"""
    histogram = self.get_histogram_data(num_bins)
    percentiles = {f'p{int(p*100)}': self.percentile(p*100) for p in p_list}

    return {
        'histogram': histogram,
        'percentiles': percentiles,
        'total_processed_count': self.total_processed_count
    }
```

**序列化支持**：

```python
def to_dict(self) -> Dict:
    """导出为可序列化的字典。"""
    return {
        'centroids': [(c.mean, c.count) for c in self.tdigest.centroids()],
        'delta': self.tdigest.delta,
        'K': self.tdigest.K,
        'total_processed_count': self.total_processed_count
    }

@classmethod
def from_dict(cls, data: Dict) -> 'TDigestStreamingHistogram':
    """从字典恢复实例。"""
    hist = cls(delta=data['delta'], K=data['K'])
    hist.total_processed_count = data['total_processed_count']
    for mean, count in data['centroids']:
        hist.tdigest.add(mean, count)
    return hist
```

**分布合并**：

```python
def __add__(self, other: 'TDigestStreamingHistogram') -> 'TDigestStreamingHistogram':
    """合并两个分布。时间复杂度：O(K1 + K2)。"""
    result = TDigestStreamingHistogram(
        delta=min(self.tdigest.delta, other.tdigest.delta),
        K=max(self.tdigest.K, other.tdigest.K)
    )
    result.tdigest = self.tdigest + other.tdigest
    result.total_processed_count = (
        self.total_processed_count + other.total_processed_count
    )
    return result
```

### 5.3 性能特性

| 操作             | 复杂度     | 说明                 |
| ---------------- | ---------- | -------------------- |
| `add()`        | O(log K)   | 均摊，偶尔 O(K) 压缩 |
| `percentile()` | O(log K)   | 在质心上二分查找     |
| `merge()`      | O(K1 + K2) | 线性于质心总数       |

**空间复杂度**：O(K)，典型内存占用约 1KB（默认参数）。

**精度特性**：

| 指标         | 精度           | 说明               |
| ------------ | -------------- | ------------------ |
| 中位数 (p50) | ~0.1% 相对误差 | 密集区域，非常准确 |
| 尾部 (p99)   | ~1% 相对误差   | 稀疏区域保持精度   |
| 极端 (p999)  | ~2% 相对误差   | 优于均匀分箱       |

---

## 6. 注意事项

1. **样本量要求**：推荐样本数 > 1000，小样本估计可能不准确
2. **线程安全**：当前实现非线程安全，多线程环境需外部锁
3. **预热期**：建议收集足够数据后再依赖估计值

---

## 7. 文件位置

| 组件                          | 文件         | 说明            |
| ----------------------------- | ------------ | --------------- |
| `TDigestStreamingHistogram` | `ref_tdigest.py` | 主实现          |
| 第三方依赖                    | `tdigest`  | T-Digest 算法库 |

---

## 8. 参考文献

- 原始论文：Dunning & Ertl, "Computing Extremely Accurate Quantiles Using t-Digests"
- Python 实现：`tdigest` 包
