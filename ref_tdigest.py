from __future__ import annotations
import typing 
if typing.TYPE_CHECKING:
    from typing import Dict, List
import time
from tdigest import TDigest
import numpy as np # Import numpy for various distribution data generation

class TDigestStreamingHistogram:
    """
    基于 T-Digest 的流式直方图实现。
    T-Digest 是一种优秀的分位数摘要算法，特别适用于长尾分布，
    可以在有限内存下提供精确的分位数估计，并从中构建直方图。

    它通过维护一组自适应的质心来概括数据分布，
    在数据稀疏的区域（如长尾）保持更高的分辨率。

    时间复杂度：
    - `add` (添加数据点): 平摊 O(log K')，其中 K' 是 T-Digest 中质心的数量，通常远小于数据点总数。
                        这个操作非常高效。
    - `get_histogram_data` (获取直方图): O(num_bins)，因为它是基于已有的分位数数据进行计算。

    空间复杂度：
    - O(K')，其中 K' 是质心的最大数量，通常由 `delta` 和 `K` 参数间接决定，是固定且可控的。
    Streaming Histogram implementation based on T-Digest.
    T-Digest is an excellent quantile summary algorithm, especially suitable for long-tailed distributions,
    providing accurate quantile estimates with limited memory, from which histograms can be constructed.

    It maintains a set of adaptive centroids to summarize the data distribution,
    keeping higher resolution in sparse data areas (e.g., long tails).

    Time Complexity:
    - `add` (adding data point): Amortized O(log K'), where K' is the number of centroids in T-Digest,
                                 usually much smaller than the total number of data points. This operation is highly efficient.
    - `get_histogram_data` (getting histogram): O(num_bins), as it's computed based on existing quantile data.
    - `get_summary` (getting summary): O(num_bins + len(p_list)).

    Space Complexity:
    - O(K'), where K' is the maximum number of centroids, indirectly determined by `delta` and `K` parameters,
              fixed and controllable.
    """

    def __init__(self, delta: float = 0.01, K: int = 25):
        """
        初始化 TDigestStreamingHistogram。
        :param delta: T-Digest 的精度参数，控制相对误差。
                      较小的值（例如 0.005）会提高精度但增加内存和质心数量。
                      通常在 0.01 到 0.1 之间。
        :param K: T-Digest 的内部合并参数，表示每个质心的最大容量。
                  这个值通常不需要手动调整，默认值 25 即可。
        Initializes TDigestStreamingHistogram.
        :param delta: T-Digest precision parameter, controlling relative error.
                      Smaller values (e.g., 0.005) increase precision but also memory and centroid count.
                      Typically between 0.01 and 0.1.
        :param K: T-Digest internal merging parameter, representing the maximum capacity of each centroid.
                  This value usually doesn't need manual adjustment; a default of 25 is often sufficient.
        """
        self.tdigest = TDigest(delta=delta, K=K)
        self.total_processed_count = 0 # Track total number of original data points processed

    def __add__(self, other: TDigestStreamingHistogram) -> TDigestStreamingHistogram:
        """
        Merges two TDigestStreamingHistogram instances.
        Creates a new TDigestStreamingHistogram containing the merged data.
        """
        if not isinstance(other, TDigestStreamingHistogram):
            return NotImplemented # Or raise an error
        
        new_tdigest_hist = TDigestStreamingHistogram(self.tdigest.delta, self.tdigest.K)
        # The tdigest library implements __add__ for TDigest objects to merge them
        new_tdigest_hist.tdigest = self.tdigest + other.tdigest
        new_tdigest_hist.total_processed_count = self.total_processed_count + other.total_processed_count
        return new_tdigest_hist
    
    def __radd__(self, other):
        """
        Handles addition when TDigestStreamingHistogram is on the right side of the + operator.
        This is important for `reduce` with an initial value (like 0) or when summing with non-TDigest objects.
        """
        if other == 0: # Handle sum() or reduce() starting with 0
            return self
        return self.__add__(other)

    def add(self, value: float):
        """
        Adds a new data point to the T-Digest.
        This operation is the core of T-Digest and is highly efficient.
        """
        self.tdigest.update(value)
        self.total_processed_count += 1
    
    def cdf(self, value: float) -> float:
        """
        Returns the estimated cumulative distribution function (CDF) for a given value.
        """
        return self.tdigest.cdf(value)
    
    def percentile(self, value: float) -> float:
        """
        Returns the estimated percentile for a given value.
        """
        return self.tdigest.percentile(value)
    
    def get_mean(self) -> float:
        """
        Returns the estimated mean of the distribution.
        Returns 0.0 for empty distributions (TDigest.trimmed_mean returns 0 when empty).
        """
        return self.tdigest.trimmed_mean(0, 100)

    def get_histogram_data(self, num_bins: int = 10) -> List[tuple[float, float, float]]:
        """
        Generates an approximate histogram from the T-Digest.
        :param num_bins: Number of bins for the histogram.
        :return: A list where each element is a tuple (bin_start, bin_end, approximate_count).
                 approximate_count is the approximate total weight (total data points) within the bin.
        """

        if len(self.tdigest) == 0:
            return []

        min_val = self.tdigest.C.min_item()[1].mean
        max_val = self.tdigest.C.max_item()[1].mean

        if min_val == max_val:
            return [(min_val, max_val, self.total_processed_count)]

        bin_width = (max_val - min_val) / num_bins
        histogram_bins = []

        for i in range(num_bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            # 确保最后一个桶的结束值包含最大值
            if i == num_bins - 1:
                bin_end = max_val + 1e-9 # 加上一个很小的 epsilon

            # 计算落在当前桶内的近似计数
            # T-Digest 提供了 `cdf` 方法，可以用来计算分位数。
            # cdf(x) 返回值 x 对应的累积百分比
            # 因此，桶内的近似计数 = (cdf(bin_end) - cdf(bin_start)) * total_count
            
            # 注意：T-Digest 的 cdf/percentile 是基于其内部质心的平滑估计。
            # 直接使用 cdf 差异来计算桶计数，通常比简单地遍历质心并检查其均值是否在桶内更准确。

            cdf_end = self.tdigest.cdf(bin_end)
            cdf_start = self.tdigest.cdf(bin_start)
            
            approx_count = (cdf_end - cdf_start) * self.total_processed_count

            histogram_bins.append((bin_start, bin_end, approx_count))

        return histogram_bins
    
    def get_summary(self, num_bins: int = 20, p_list: List[float] = None, nbit=1) -> Dict:
        """
        Gets a statistical summary, including histogram, quantiles and total count.
        :param num_bins: Number of bins for the histogram.
        :param p_list: List of percentiles (e.g., [0.5, 0.9, 0.99]) to calculate.
        :return: A dictionary containing 'histogram', 'percentiles', and 'total_processed_count'.
        """
        # Handle empty distribution
        if self.total_processed_count == 0:
            percentiles = {f'p{round(p*100, nbit)}': float('nan') for p in p_list}
            summary = {
                'histogram': [],
                'percentiles': percentiles,
                'total_processed_count': 0
            }
            return summary
        
        # calculate the percentiles
        percentiles = {f'p{round(p*100, nbit)}': self.percentile(round(p*100, nbit)) for p in p_list}
        summary = {
            'histogram': self.get_histogram_data(num_bins),
            'percentiles': percentiles,
            'total_processed_count': self.total_processed_count
        }
        return summary

    def to_dict(self) -> Dict:
        """
        Serializes the TDigestStreamingHistogram object into a dictionary.
        This includes the underlying TDigest's state and the total processed count.
        """
        return {
            'delta': self.tdigest.delta,
            'K': self.tdigest.K,
            'tdigest_state': self.tdigest.to_dict(), # TDigest library has a to_dict() method
            'total_processed_count': self.total_processed_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> TDigestStreamingHistogram:
        """
        Deserializes a dictionary into a TDigestStreamingHistogram object.
        """
        td_hist = cls(delta=data['delta'], K=data['K'])
        td_hist.tdigest.update_from_dict(data['tdigest_state']) # TDigest library has update_from_dict()
        td_hist.total_processed_count = data['total_processed_count']
        return td_hist

# 示例用法
if __name__ == '__main__':
    # 辅助函数：运行测试并打印结果
    def run_test(distribution_name, data_generator, n_samples=50000, delta=0.01, K=25):
        print(f"\n--- 测试 {distribution_name} 分布下的 TDigestStreamingHistogram ---")
        td_hist = TDigestStreamingHistogram(delta=delta, K=K)
        
        # 生成完整数据集作为基准
        full_data = data_generator(n_samples)
        
        start_time = time.time()
        for i, x in enumerate(full_data):
            td_hist.add(x)
            # if (i + 1) % 5000 == 0:
            #     print(f"  处理了 {i+1} 个数据点")
        elapsed_time = time.time() - start_time
        print(f"  添加 {n_samples} 个数据点耗时: {elapsed_time:.4f} 秒")
        print(f"  T-Digest 中当前质心数量: {len(td_hist.tdigest)}")

        # 计算真实分位数作为基准
        true_p90 = np.percentile(full_data, 90)
        true_p99 = np.percentile(full_data, 99)
        true_p99_9 = np.percentile(full_data, 99.9)

        # 获取 T-Digest 估计的分位数
        est_p90 = td_hist.tdigest.percentile(90)
        est_p99 = td_hist.tdigest.percentile(99)
        est_p99_9 = td_hist.tdigest.percentile(99.9)

        print(f"\n  分位数准确度 ({distribution_name}):")
        print(f"  90% 分位数: 真实={true_p90:.4f}, 估计={est_p90:.4f}, 误差={(abs(true_p90 - est_p90)/true_p90 if true_p90 != 0 else abs(true_p90 - est_p90)):.4%}")
        print(f"  99% 分位数: 真实={true_p99:.4f}, 估计={est_p99:.4f}, 误差={(abs(true_p99 - est_p99)/true_p99 if true_p99 != 0 else abs(true_p99 - est_p99)):.4%}")
        print(f"  99.9% 分位数: 真实={true_p99_9:.4f}, 估计={est_p99_9:.4f}, 误差={(abs(true_p99_9 - est_p99_9)/true_p99_9 if true_p99_9 != 0 else abs(true_p99_9 - est_p99_9)):.4%}")

        # 打印部分直方图数据
        hist_data = td_hist.get_histogram_data(num_bins=20)
        print(f"\n  直方图桶数据 (前3个桶):")
        for i, (bin_start, bin_end, count) in enumerate(hist_data[:3]):
            print(f"    桶 {i+1}: [{bin_start:.4f}, {bin_end:.4f}): {count:.2f} (近似计数)")
        print(f"  ...")
        print(f"  直方图桶数据 (后3个桶):")
        for i, (bin_start, bin_end, count) in enumerate(hist_data[-3:]):
            print(f"    桶 {len(hist_data)-3+i+1}: [{bin_start:.4f}, {bin_end:.4f}): {count:.2f} (近似计数)")

    # 1. 指数分布 (Exponential Distribution)
    # scale 参数是均值（lambda的倒数），具有长尾特性
    run_test(
        "指数分布 (Exponential)",
        lambda n: np.random.exponential(scale=2.0, size=n)
    )

    # 2. 对数正态分布 (Log-Normal Distribution)
    # mean 和 sigma 是对数转换后正态分布的均值和标准差，本身具有长尾特性
    run_test(
        "对数正态分布 (Log-Normal)",
        lambda n: np.random.lognormal(mean=0.0, sigma=1.0, size=n)
    )

    # 3. Gumbel 分布 (Extreme Value Type I)
    # loc 是位置参数，scale 是尺度参数。Gumbel 分布也常用于建模极端值，具有长尾
    # 假设用户指的是 Gumbel 分布，而不是 Gumbel-Softmax（通常用于离散采样）
    run_test(
        "Gumbel 分布",
        lambda n: np.random.gumbel(loc=0.0, scale=1.0, size=n)
    )

