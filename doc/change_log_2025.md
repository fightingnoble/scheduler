
## 20250703
完善physical 和 logical graph的构建流程：保证后续能够根据概率图获取调度，验证所需的所有信息

New features:
 使用简单的调度器，验证调度器是否能够正确调度

## 20250726
    统一acc_p和sen_p的接口，方便后续的扩展
    
## 20250806 - 0820
1. 扩展多个超周期下的case的生成：
    需要循环多个超周期
2. 每次生成负载的时候，通过seed生成可控的，exp_comp_t
    每个超周期下，任务的offset需要加上当前超周期的offset，进而影响到ert和ddl，以及静态调度表。
    参考scheduler_base 中的参数设置方法，
    为Mygraph 添加成员函数，输入为指示当前第几个超周期的hp_idx和一个种子seed，
    将节点复制一份，节点名称后面添加下划线+hp_idx，
    同时将这些信息添加到，n_pred_map, ..., ert_map 这些状态缓存中，
    除了sink_node之外的src和op予随机的执行时间。
3. python数值计算精度和仿真需要的时间精度不同，不希望存在这些游离态的时间计算。
    现在所有产生新的时间度量、时间比较、参考时间推进仿真、参考时间决策调度的行为，都包含在approach_Eq.py或者经过elim_nume_error处理。

特别处理：
    1. 在生成静态表的时候
    2. 为了减少运行时内存的负担，选择动态扩展任务图、调度事件、任务映射、静态表等。
        tsx表示第x个超周期开始时刻，那这个时候要加载下一个周期的负载和配置：
        a）复制图，b) 设置新节点的mapping，c) 添加调度算法所需要的外部调度事件。
        ts0需要加载 0,1 两个超周期的配置，ts1需要加载超周期2的配置，以此类推……
        当curr_hp< num_hp的时候，复制a-c，否则只复制c。
    3. 当前调度方式，仍然不能解决在截止期前被打断的问题。

## 20250820 - 0830
3. 扩展统计信息的收集：
    1. 端到端、单任务的timeout次数
    2. 端到端、单任务的延迟
    3. 每个分区，realloc次数，累计realloc经历时间
    4. 所有分区，reallc时间*分区cap*base_pwr的和

4. 添加find_legal函数，用于在分配资源的时候，根据任务的资源约束，找到最优的资源分配方案。相对于之前一大堆奇奇怪怪的函数，这个函数利用了区间操作来计算，并且有详细的注释，方便后续的扩展和维护。

## 20250830 
1. 添加verbose参数，用于控制输出详细信息。
2. 添加output_path参数，用于控制输出路径。
3. Fix bug: 复制节点的时候，复制属性字典，保证logic graph 不变。
4. 统计信息输出，添加total_processed_count，用于检查任务完成情况。

## 20250903
计算延迟break down
延迟和负载之间的相关性（按照周期统计）

当前的collector 能够记录调度开销和延迟，现在我需要添加下面几种统计：
端到端延迟的break down，包括：等待时间，调度开销，计算时间 （其中等待时间不用显示统计，用端到端延迟-累计调度开销-计算延迟
按周期统计，每个周期内：
（总负载，在周期内完成的任务的端到端最差延迟）
（闲置算力，miss任务剩余负载）

Fix: var_en参数 is added
Add: 更新了可执行任务过滤逻辑
    filter timeout task if drop
        if op_miss_en, only the sink will be dropped for timeout;
        otherwise, all the tasks will be dropped for timeout.
    filter ert < curr_t task if reserve
    filter R task

## 20250910
- 统一强制核数逻辑（force_num_cores）：抽取 `apply_forced_num_cores`，适配单/多分区
- 抽取绘图封装：`render_bin_pack_plots` 与 `render_runtime_full_plot`，减少重复
- CSV 初始化统一：新增 `ensure_csv`，替换分支内重复 DataFrame 初始化
- 修复：`bin_split` 分支调用 `get_core_num_from_trace_name` 时缺少 `path_ctx`
- 统一 runtime induced 环境准备：新增 `prepare_induced_env_if_needed`，收敛核数解析/路径生成/加载 bin_list，替换两阶段分支重复逻辑

## 20250911
- 新增 API 函数：`build_paths_and_ctx`、`build_workload_and_criticality`、`build_scheduler_elements`、`build_simulation_env`，支持 `approach_setup` 复用
- 重构 `main` 函数：使用新 API 函数替换原有代码段，保持功能不变
