## 问题描述：

当前代码针对实时端到端应用进行资源分配。端到端有一个总体时间约束。考虑到任务执行延迟存在抖动，分配资源（core数量），需要为端到端任务链上的任务分配时间余量。
DistributeSlack 和GurobiDistributeSlack 的作用在于，将端到端延迟余量分配给每个任务。
任务链定义为从src到sink的连续节点序列。其中，src节点和中间的节点具有变化的延迟。但是两者的延迟建模是不同的。

src 执行sensor数据采集，为串行CPU任务，我们直接将其延迟建模为具有10ms 延迟的偏移量的正态分布

但是，当前调度参数生成的过程中对负载variation的建模不同于仿真模型(approach_eq)。

1. 之前的预留方案中，通过jitter_t_comp_ratio，和exec_t_comp_ratioA 规定时间余量
2. 这些ratio 通过手动指定，而不考虑分布，现在预留的执行时间需要通过分布的分位数决:
例如，sensor 执行时间分布是均值为0的正态分布，之前手动指定补偿比例是exec_t_comp_ratioA，那么预留的延迟等于其3sigma也就是（1/src_attr[src])，乘上jitter_t_comp_ratio（右侧的某个位置）。
而，对于执行时间抖动和负载抖动则是用 预留延迟 = 典型负载加速后的延迟/(1-  exec_t_comp_ratioA) * load_com_ratio

这样手动输入存在两个问题：1.会导致，各个任务的补偿比例有差异，没有统一标准 2. 没有考虑分布，导致预留的时间不准确

现在已经将分散在各个地方的延迟模型集中在approach_EQ中，
SenVarDist 是src 节点的延迟分布，而AccVarDist则对应op类型节点的延迟（由两部分组成lat_i = load_i/processor_power + exec_q_i）。
接下来请将DistributeSlack 和GurobiDistributeSlack 下的时间约束的定义进行替换：
之前：
（GurobiDistributeSlack）每条链上的任务，在给定算力情况下的延迟，经过cal_lat变换后的总和，不超过延迟约束。
（DistributeSlack）或者每条链上的任务，在给定算力情况下的延迟，不超过总延迟约束经过slack_comp变换后的值。
现在：

已知各种variation的分布，我们获取分位数下的延迟，不超过端到端延迟。
先给出修改的规划。
