
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

## 20250916
- 修复：`bin_split` 分时复用分区（num_bin = -1）case 下，选择将图完全展开导致的资源估计过多的问题：
    将优化总资源设置为最少化分区数量的次要目标。
- 添加：更新var_factor的含义为：执行时间变异因子，用于表示执行时间的变化范围。

## 20250917
- 新增：按"已知分布+分位数"推导补偿比例，保持下游接口不变
  - 参数：`--reserve_quantile`、`--reserve_quantile_cfg`、`--exec_dist_para`、`--load_dist`
  - 规则：
    - jitter（0均值正态）`ratio = z(q)/3`，绝对抖动时间为 `(1/freq)*ratio`
    - exec（指数 X~Exp(scale)）设 `F=1+X`，`exec_ratio = 1 - 1/F(q)` 并裁剪到 [0,0.99]
    - load（离散）按已知概率构造，默认离散CDF分位；可切换 TDigest 连续化
  - 位置：`model/performance.py`（分布工具/类），`approach_Eq.py`（离散负载封装），`utils.py`（参数与映射接线）

## 20250921
- 重构：统一延迟分布的序列化与反序列化机制
  - **问题**：`sim_main.py` 和 `main_approach.py` 程序解耦，通过 JSON 图传递数据，但函数对象（`rng_fn_list`）无法序列化
  - **解决方案**：将分布对象变为可序列化，传递"配方"而非函数本身
  - **核心改动**：
    - 为所有 `Variation` 子类添加 `to_dict()` 序列化方法
    - 新增全局工厂函数 `dist_from_dict()` 用于反序列化
    - 修改 `init_var_dist`：创建分布对象后序列化存入节点 `dist_info` 属性
    - 重构 `MyGraph._rebuild_distributions`：从节点属性重建完整分布对象
    - 删除旧的 `get_var_t_fn` 函数，统一使用 `init_var_dist`
  - **新增 `AccVarDist` 类**：组合负载变化（`LoadVarDist`）和执行时间变化（`ExecVarDist`），实现加速任务的联合延迟分布
  - **文件涉及**：`approach_Eq.py`、`approach_def.py`、`task/task_cfg.py`
  - **数据流**：`gen_workloads` → JSON序列化 → `MyGraph` 反序列化重建 → 仿真使用

- add input size: op_io_time=task_attr["Data Size(GB)"]*1e9/BW_DRAM

我统一了approach_eq 里面的accVarDist，SenVarDist，使其var_fn 都需要接受processing_power。
现在的延迟模型为：
延迟 = T_compute + T_memory
T_compute = load / processing power
因此，duplicate_for_hyperperiod 的时候，不能仅仅生成 exp_comp_t，还需要生成exp_io_t。
后续计算延迟和估计资源的时候也需要，综合考虑exp_comp_t和exp_io_t。
请帮我找到所有需要修改的地方，并给出修改建议。

## 20250922
- 统一：IO 延迟字段为 `exp_io_t`
  - 构图阶段在 `task/task_cfg.py` 为 op 节点写入 `exp_io_t`
  - 新路径不再使用 `seq_io_time`（旧路径保留兼容）
- 分布：序列化/反序列化与复建
  - 在 `init_var_dist` 将分布对象序列化至节点 `dist_info`，并附回 `var_dist`
  - `MyGraph._rebuild_distributions` 支持从 `dist_info` 重建并回写 `var_dist` 到节点，同时缓存到 `var_dist_map`
- 超周期复制：同时采样计算与访存
  - `duplicate_for_hyperperiod` 在 `var_en` 下：
    - 对 `AccVarDist` 分别采样 `load_dist` 与 `exec_dist`，写入 `exp_comp_t`/`exp_io_t`
    - 对单一分布（如 `SenVarDist`/`LoadVarDist`）仅更新 `exp_comp_t`，`exp_io_t=0`
- 延迟/资源估计纳入访存
  - `sim_comp_time(task_load, allocated_resources, base_power, exp_io_t=0.0)` 加入访存固定时间
  - `estimate_resource_requirement(task_load, slack_time, base_power, exp_io_t=0.0)` 先扣除访存再估算
  - `approach_sched.alloc_fn_pglb` 与 `Acc_p.predict_next` 传入节点 `exp_io_t`
- 对齐：`init_var_dist` 与节点参数
  - op 节点使用 `exp_io_t` 推导 `ExecVarDist`（移位指数，loc 可非零）
  - 负载分布当前基于 `flops` 推导（兼容旧定义）

## 20250923
- 重构：链式松弛时间分配算法统一化
  - **问题**：`GurobiDistributeSlack` 和 `DistributeSlack` 函数存在大量重复代码，且接口不统一
  - **解决方案**：
    1. 合并两个函数为统一的 `DistributeSlack` 函数，通过 `algorithm` 参数选择求解策略
    2. 统一接口：将 `GurobiRscSlackEstim` 的 `constr_core` 参数从字典格式改为 `TaskConstraints` 对象列表
    3. 提取独立的解决方案验证函数 `check_solution_validity`，供两个求解器共同使用
  - **核心改动**：
    - 创建统一的 `DistributeSlack` 函数，支持 "gurobi" 和 "avg" 两种算法
    - 更新 `GurobiRscSlackEstim` 类以使用 `TaskConstraints` 对象
    - 实现独立的 `check_solution_validity` 验证函数
    - 保留 `GurobiDistributeSlack` 作为向后兼容的包装器
  - **文件涉及**：`sched/slack_estim.py`、`sched/packing_solver/chain_slack_assign.py`
  - **优势**：代码复用性提高，接口统一，维护性增强

- 文档：链式松弛时间分配算法技术文档
  - 创建 `doc/chain_slack_assignment_algorithm.md` 技术文档
  - 详细描述算法理论基础、数学模型、实现细节
  - 包含精确求解器（MIQP）和启发式求解器的完整分析
  - 提供算法复杂度分析和性能比较

- 修正：TaskConstraints 约束处理逻辑
  - **问题**：所有节点类型都被分配了 TaskConstraints，但只有非 AccVarDist 类型的节点才需要复杂的核心约束
  - **解决方案**：
    1. 在 `DistributeSlack` 函数中根据节点类型（`var_dist`）区分处理约束创建
    2. AccVarDist 类型节点（通常是传感器任务）使用默认约束：核心数固定为1
    3. 其他类型节点从任务配置中获取完整的约束信息
  - **核心改动**：
    - 修改约束创建逻辑：`isinstance(var_dist, AccVarDist)` 判断节点类型
    - 为 AccVarDist 节点设置默认约束：`parallel_mode="none", core_min=1, core_max=1`
    - 添加 `AccVarDist` 导入到 `sched/slack_estim.py`
  - **文件涉及**：`sched/slack_estim.py`
  - **优势**：约束处理更加精确，避免为不需要复杂约束的节点分配不必要的约束

## 20250924
- 改进：路径生成时自动创建目录
  - **问题**：路径生成方法只返回路径字符串，不检查目录是否存在，可能导致文件操作失败
  - **解决方案**：
    1. 新增 `_ensure_dir_exists` 辅助方法，检查并创建目录
    2. 更新所有路径生成方法，在返回路径前确保目录存在
  - **核心改动**：
    - 为 `PathContext` 类添加 `_ensure_dir_exists` 私有方法
    - 更新以下方法：`get_bin_list_path`、`get_routing_table_path`、`get_trace_path`、`get_plot_path`、`get_csv_path`、`get_log_path`、`get_stat_log_path`、`get_cfg_path`
    - 优化 `refresh_config` 方法，使用 `exist_ok=True` 避免重复创建
  - **文件涉及**：`paths.py`
  - **优势**：避免因目录不存在导致的文件操作错误，提高系统稳定性

- 重构：统一延迟补偿参数对路径生成的影响
  - **问题**：路径生成中使用了多个延迟补偿参数（`jitter_t_comp_ratio`、`exec_t_comp_ratioA`、`wsc_slack_ratio`），导致路径格式复杂且不一致
  - **解决方案**：
    1. 统一路径格式：只使用 `exec_t_comp_ratioA` 和 `lateness_mode` 生成路径
    2. 清理不再使用的参数和代码
    3. 保持 `exec_t_comp_ratioB` 用于 `repack` 模式的文件后缀
  - **核心改动**：
    - 更新 `global_var.py` 中的 `cfg_root_fmt`：格式简化为 `..._rda-{exec_t_comp_ratioA:.2%}(T)_{lateness_mode:s}`
    - 修改 `paths.py` 中的 `PathContext.refresh_config()`：只使用 `exec_t_comp_ratioA` 和 `lateness_mode`
    - 更新 `utils.py` 中的 `get_cfg_n()` 函数：与新的路径格式保持一致
    - 从 `global_sched.py` 中移除未使用的 `wsc_slack_ratio` 参数
    - 从 `sim_main.py` 和 `allocator_agent.py` 中移除对 `wsc_slack_ratio` 的函数调用
    - 清理 `PathContext` 类中不再需要的 `jitter_t_comp_ratio` 参数
    - 移除 `utils.py` 中未使用的 `var_estimation` 相关代码
  - **文件涉及**：`global_var.py`、`paths.py`、`utils.py`、`sched/global_sched.py`、`sim_main.py`、`allocator_agent.py`
  - **优势**：路径管理更加简洁统一，移除了冗余参数，提高了代码的可维护性