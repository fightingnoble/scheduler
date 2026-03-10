# 仿真流程总览

本文档描述实时任务调度器仿真系统的端到端流程、关键模块职责和参数流动路径。

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              仿真系统架构                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐                                                      │
│   │ main_approach.py │ ◄─── 主入口 (策略驱动仿真流程)                         │
│   └────────┬─────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌──────────────────┐                                                      │
│   │ approach_setup.py│ ◄─── 基准设置 (bin_list 生成)                         │
│   └────────┬─────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌──────────────────┐     ┌───────────────────┐                           │
│   │   sim_main.py    │◄───►│ sched/global_sched│ ◄─── 装箱算法              │
│   └────────┬─────────┘     └───────────────────┘                           │
│            │                                                                │
│            ▼                                                                │
│   ┌──────────────────┐                                                      │
│   │  approach_sim.py │ ◄─── 事件驱动仿真运行时                               │
│   └──────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 端到端流程图

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                              main_approach.py: main()                               │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  1. 参数解析                                                                       │
│     args = input_parser()                                                          │
│     ├── 设置时间单位: set_time_unit(1e-6, False)                                   │
│     └── 初始化 verbose, realloc_disabled                                          │
│                                                                                    │
│  2. 基准设置 ─────────────────────────────────────────────────────────────────►   │
│     setup_benchmark(args, time_norm_factor)                                        │
│     │                                                                              │
│     │  返回: G, pid2name, bin_list, policy, path_ctx, T_hp                        │
│     │                                                                              │
│     ▼                                                                              │
│  3. 创建分区配置                                                                   │
│     get_partition_info(bin_list, G, pid2name)                                      │
│     │                                                                              │
│     │  返回: num_partitions, partition_size, flops_per_core,                      │
│     │        partition_task_map, TSMap_list                                        │
│     │                                                                              │
│     ▼                                                                              │
│  4. 构建 PartitionConfig                                                           │
│     PartitionConfig(num_partitions, cap_list, ...)                                 │
│                                                                                    │
│  5. 初始化仿真事件                                                                 │
│     initialize_events(policy, G, TSMap_list)                                       │
│     │                                                                              │
│     │  根据策略类型初始化事件集:                                                   │
│     │  ├── cyc: table 事件 (静态调度表)                                           │
│     │  ├── reserv: external 事件 (ert触发)                                        │
│     │  └── pglb/glb: 仅 sensor 触发事件                                           │
│     │                                                                              │
│     ▼                                                                              │
│  6. 实例化处理器                                                                   │
│     instantiate_processors(partition_cfg, event_t, policy)                         │
│     │                                                                              │
│     │  返回: processors, event_t_rt, stats_collector                              │
│     │                                                                              │
│     ▼                                                                              │
│  7. 运行仿真                                                                       │
│     run_simulation(processors, event_t, G, num_hp, T_hp)                           │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 3. setup_benchmark 详细流程

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                     approach_setup.py: setup_benchmark()                           │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 0: 预处理参数                                                           │ │
│  │ preprocess_args(args)                                                        │ │
│  │ ├── 强制后缀处理: force_suffix                                                │ │
│  │ ├── enforce_wc 设置                                                          │ │
│  │ └── test_case 特殊检查                                                       │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 1: 构建路径和上下文                                                     │ │
│  │ build_paths_and_ctx(args)                                                    │ │
│  │ │                                                                            │ │
│  │ │  返回: path_params, path_ctx                                               │ │
│  │ │  ├── path_ctx: PathContext 实例 (统一路径管理)                             │ │
│  │ │  └── path_params: 路径参数元组                                             │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ 判断是否需要 repack                                                          │ │
│  │ need_repack = (exec_t_comp_ratioB != -1 and exec_t_comp_ratioA > ratioB)    │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│              ┌───────────────────────┴───────────────────────┐                    │
│              ▼                                               ▼                    │
│  ┌────────────────────────┐                   ┌────────────────────────┐          │
│  │ Phase 1: Split (non-repack)│               │ Phase 2: Repack (可选) │          │
│  │ args.quantile = ratioA     │               │ args.quantile = ratioB │          │
│  │                            │               │                        │          │
│  │ run_benchmark_setup_pipeline│─────────────►│ run_benchmark_setup_   │          │
│  │   (..., need_repack=False) │               │ pipeline(..., True)    │          │
│  └────────────────────────┘                   └────────────────────────┘          │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step Final: 加载逻辑图                                                       │ │
│  │ instantiate_mygraph_from_json(path_ctx.graph_fn, time_norm_factor)          │ │
│  │                                                                              │ │
│  │ 返回: G, pid2name, bin_list, policy, path_ctx, hyper_p                       │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 4. run_benchmark_setup_pipeline 详细步骤

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│             approach_setup.py: run_benchmark_setup_pipeline()                      │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  输入: args, path_ctx, path_params, need_repack, hyper_p, bin_list, num_cores     │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 2: 生成 workload 并设置 criticality                                     │ │
│  │ build_workload_and_criticality(args)                                         │ │
│  │ │                                                                            │ │
│  │ │  调用: gen_workloads(args) → sim_main.py → task/task_cfg.py               │ │
│  │ │                                                                            │ │
│  │ │  返回: hyper_p, glb_n_task_dict, physical_graph_nx, glb_p_list            │ │
│  │ │  ├── hyper_p: 超周期 (e2e_latency)                                         │ │
│  │ │  ├── glb_p_list: ProcessInt 列表 (进程实例)                                │ │
│  │ │  └── physical_graph_nx: NetworkX DAG 图                                   │ │
│  │ │                                                                            │ │
│  │ │  导出: export_json_graph_utils(physical_graph_nx, path_ctx.graph_fn)      │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 3: 创建调度器元素                                                       │ │
│  │ create_scheduler_elements_with_config(args, path_params, ...)               │ │
│  │ │                                                                            │ │
│  │ │  调用: create_common_scheduler_elements() → sim_main.py                    │ │
│  │ │                                                                            │ │
│  │ │  返回: task_spec, rsc_list, msg_dispatcher, a_data_pipe, w_data_pipe,     │ │
│  │ │        scheduler_list, monitor_list, trace_path, sim_step                  │ │
│  │ │                                                                            │ │
│  │ │  关键组件:                                                                 │ │
│  │ │  ├── rsc_list: Resource_model_int 列表 (每个 bin 一个)                     │ │
│  │ │  ├── scheduler_list: Scheduler 列表                                        │ │
│  │ │  └── monitor_list: Monitor 列表 (运行时记录)                               │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 4: 构建仿真环境                                                         │ │
│  │ build_simulation_env(args, workload, sim_step)                              │ │
│  │ │                                                                            │ │
│  │ │  返回: num_periods, warmup, quantumSize, event_range, event_iter_dict     │ │
│  │ │                                                                            │ │
│  │ │  事件生成: TaskInt.get_event_generator(glb_p_list, ...)                   │ │
│  │ │  └── 生成到达时间、截止时间、负载的迭代器                                   │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 5: 执行装箱算法 (核心)                                                  │ │
│  │ perform_bin_packing(args, glb_p_list, num_cores, ...)                       │ │
│  │ │                                                                            │ │
│  │ │  算法分支:                                                                 │ │
│  │ │  ├── algorithm="scratch": push_task_into_bins_new()                       │ │
│  │ │  │   └── 从零开始，完全动态装箱                                            │ │
│  │ │  │                                                                        │ │
│  │ │  └── algorithm="guided": coleasing_alloc_cluster()                        │ │
│  │ │      ├── need_repack=False: Phase 1 (空间划分)                            │ │
│  │ │      │   └── 使用 exec_t_comp_ratioA (保守分位数，如 0.99)                │ │
│  │ │      │                                                                    │ │
│  │ │      └── need_repack=True: Phase 2 (时间窗口重分配)                       │ │
│  │ │          └── push_task_into_bins_new() + 预定义映射                       │ │
│  │ │          └── 使用 exec_t_comp_ratioB (激进分位数，如 0.80)                │ │
│  │ │                                                                            │ │
│  │ │  返回: bin_list, max_core_num, glb_p_list, hyper_p                        │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 6: 应用资源约束 (仅 non-repack)                                         │ │
│  │ if not need_repack:                                                          │ │
│  │     num_cores = apply_forced_num_cores(bin_list, max_core_num, args.num_cores)│
│  │                                                                              │ │
│  │ 关键约束: repack 不改变资源数量，只重新分配时间窗口                           │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│                                      ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ Step 7-10: 输出和持久化                                                      │ │
│  │ ├── 生成 dump 路径: generate_bin_paths()                                     │ │
│  │ ├── 打印布局: Bin_list_print()                                               │ │
│  │ ├── 绘制图表: render_bin_pack_plots() (可选)                                 │ │
│  │ └── 保存 bin_list: dump_and_check()                                          │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                    │
│  输出: hyper_p, bin_list, num_cores                                               │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 5. 事件驱动仿真流程 (approach_sim.py)

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                      approach_sim.py: run_simulation()                             │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│  输入: processors, event_t, G, num_hp, T_hp, verbose, var_en                      │
│                                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────────────┐ │
│  │ 主循环: while (G.nodes() or curr_hp < num_hp) or time_eq(curr_t, 0)         │ │
│  └──────────────────────────────────────────────────────────────────────────────┘ │
│                                      │                                             │
│              ┌───────────────────────┴───────────────────────┐                    │
│              ▼                                               ▼                    │
│  ┌────────────────────────┐                   ┌────────────────────────┐          │
│  │ 超周期边界处理         │                   │ 事件处理循环           │          │
│  │ if curr_t >= boundary  │                   │                        │          │
│  │ │                     │                   │ 1. update_run()        │          │
│  │ ├── record_miss()     │                   │    更新运行队列        │          │
│  │ ├── forward_hyperperiod│                   │    计算剩余工作量      │          │
│  │ ├── curr_hp++         │                   │    检查完成事件        │          │
│  │ ├── duplicate_for_    │                   │                        │          │
│  │ │   hyperperiod()     │                   │ 2. update_ready()      │          │
│  │ └── add_events_for_   │                   │    处理事件            │          │
│  │     hyperperiod()     │                   │    更新就绪队列        │          │
│  └────────────────────────┘                   │                        │          │
│                                               │ 3. sched()             │          │
│                                               │    调度决策            │          │
│                                               │    │                   │          │
│                                               │    ├── Sen_p: FCFS    │          │
│                                               │    └── Acc_p: 优先级  │          │
│                                               │                        │          │
│                                               │ 4. 计算下一个事件时间  │          │
│                                               │    next_t = min(       │          │
│                                               │      curr_t + duration,│          │
│                                               │      next_timer_event, │          │
│                                               │      sim_hp * T_hp     │          │
│                                               │    )                   │          │
│                                               └────────────────────────┘          │
│                                                                                    │
│  事件类型:                                                                         │
│  ├── "finish": 任务完成                                                           │
│  ├── "external": 外部触发 (sensor, ert)                                           │
│  └── "table": 静态调度表切换 (cyc 策略)                                           │
│                                                                                    │
│  处理器类型:                                                                       │
│  ├── Sen_p: 传感器处理器                                                          │
│  │   └── 调度策略: FCFS (First-Come-First-Served)                                │
│  └── Acc_p: 加速器处理器                                                          │
│      └── 调度策略: 优先级调度 (参数化优先级定义)                                   │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 6. 关键函数调用链

### 6.1 主入口调用链

```
main_approach.py:main()
    │
    ├── utils.py:input_parser()                    # 解析命令行参数
    │
    ├── approach_Eq.py:set_time_unit()             # 设置时间单位
    │
    ├── approach_setup.py:setup_benchmark()        # 基准设置 (见 6.2)
    │
    ├── approach_initiator.py:get_partition_info() # 从 bin_list 提取分区信息
    │
    ├── approach_sched.py:PartitionConfig()        # 创建分区配置
    │
    ├── approach_initiator.py:initialize_events()  # 初始化事件集
    │
    ├── approach_initiator.py:instantiate_processors() # 实例化处理器
    │   │
    │   ├── Sen_p()                                # 传感器处理器
    │   └── acc_p_factory()                        # 加速器处理器工厂
    │
    └── approach_sim.py:run_simulation()           # 运行仿真
```

### 6.2 setup_benchmark 调用链

```
approach_setup.py:setup_benchmark()
    │
    ├── sim_main.py:preprocess_args()              # 预处理参数
    │
    ├── sim_main.py:build_paths_and_ctx()          # 构建路径
    │   │
    │   └── utils.py:build_path_old()              # 旧路径生成方法
    │
    └── run_benchmark_setup_pipeline() [x1 or x2]  # 可能执行两次 (split + repack)
        │
        ├── sim_main.py:build_workload_and_criticality()
        │   │
        │   └── task/task_cfg.py:gen_workloads()   # 生成工作负载
        │
        ├── task/task_cfg.py:export_json_graph_utils() # 导出图
        │
        ├── sim_main.py:create_scheduler_elements_with_config()
        │   │
        │   └── sim_main.py:create_common_scheduler_elements()
        │       │
        │       ├── sched/scheduler_agent.py:Scheduler()
        │       └── sched/monitor_agent.py:Monitor()
        │
        ├── sim_main.py:build_simulation_env()
        │   │
        │   └── task/task_agent.py:TaskInt.get_event_generator()
        │
        ├── sim_main.py:perform_bin_packing()      # 核心装箱算法
        │   │
        │   ├── [algorithm="guided"]
        │   │   └── sched/global_sched.py:coleasing_alloc_cluster() # Phase 1
        │   │
        │   └── [algorithm="guided" + need_repack]
        │       └── sched/global_sched.py:push_task_into_bins_new()  # Phase 2
        │
        ├── sim_main.py:apply_forced_num_cores()   # 应用资源约束
        │
        ├── sim_main.py:generate_bin_paths()       # 生成路径
        │
        ├── sched/bin_list_utils.py:Bin_list_print() # 打印
        │
        └── utils.py:dump_and_check()              # 保存
```

## 7. 参数流动路径

### 7.1 核心参数流向

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              参数流动路径                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  命令行参数 (input_parser)                                                       │
│  ├── exec_t_comp_ratioA (0.99) ──────────────────────────────────────────────┐ │
│  ├── exec_t_comp_ratioB (0.80) ─────────────────────────────────────────────┐│ │
│  ├── num_bins (-1, 1, >=2) ────────────────────────────────────────────────┐││ │
│  ├── num_cores (None or int) ─────────────────────────────────────────────┐│││ │
│  ├── e2e_latency (0.1s) ─────────────────────────────────────────────────┐││││ │
│  ├── policy (cyc, glb, pglb, reserv) ───────────────────────────────────┐│││││ │
│  └── binpack_cfg (JSON) ───────────────────────────────────────────────┐││││││ │
│                                                                        │││││││ │
│  gen_workloads()                                                       │││││││ │
│  │   └──► hyper_p (= e2e_latency)                                      │││││││ │
│  │   └──► glb_p_list (ProcessInt 列表)                                 │││││││ │
│  │   └──► physical_graph_nx (DAG 图)                                   │││││││ │
│  │                                                                      │││││││ │
│  coleasing_alloc_cluster() [Phase 1]                                    │││││││ │
│  │   └──► bin_list (任务到 bin 的映射)                                  │││││││ │
│  │   └──► max_core_num (估算核心数)                                     │││││││ │
│  │   └──► 使用 exec_t_comp_ratioA ◄─────────────────────────────────────┘││││││ │
│  │                                                                        ││││││ │
│  push_task_into_bins_new() [Phase 2, 可选]                                ││││││ │
│  │   └──► 更新 bin_list (时间窗口分配)                                    ││││││ │
│  │   └──► 使用 exec_t_comp_ratioB ◄───────────────────────────────────────┘│││││ │
│  │                                                                          │││││ │
│  apply_forced_num_cores() [仅 non-repack]                                   │││││ │
│  │   └──► num_cores (最终核心数) ◄──────────────────────────────────────────┘││││ │
│  │                                                                            ││││ │
│  get_partition_info()                                                         ││││ │
│  │   └──► partition_task_map ◄────────────────────────────────────────────────┘│││ │
│  │   └──► partition_size ◄──────────────────────────────────────────────────────┘││ │
│  │   └──► TSMap_list (静态调度表) ◄───────────────────────────────────────────────┘│ │
│  │                                                                              ││ │
│  PartitionConfig()                                                              ││ │
│  │   └──► 包含所有分区信息 ◄──────────────────────────────────────────────────────┘│ │
│  │                                                                                ││
│  instantiate_processors()                                                         ││
│  │   └──► processors [Sen_p, Acc_p, ...] ◄────────────────────────────────────────┘│
│  │   └──► stats_collector                                                          ││
│  │                                                                                  ││
│  run_simulation()                                                                   ││
│  │   └──► 事件驱动执行                                                              ││
│  │   └──► 统计收集 ◄────────────────────────────────────────────────────────────────┘│
│  │                                                                                    │
│  输出: stats_collector.export_summary()                                               │
│                                                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 关键参数说明

| 参数 | 来源 | 作用范围 | 影响的函数 |
|------|------|----------|------------|
| `exec_t_comp_ratioA` | CLI | Phase 1 (Split) | `coleasing_alloc_cluster()` - 资源估算 |
| `exec_t_comp_ratioB` | CLI | Phase 2 (Repack) | `push_task_into_bins_new()` - 时间窗口 |
| `num_bins` | CLI | Split | `coleasing_alloc_cluster()` - 分区数量 |
| `num_cores` | CLI | 约束 | `apply_forced_num_cores()` - 资源约束 |
| `e2e_latency` | CLI | 全局 | `gen_workloads()` → `hyper_p` |
| `policy` | CLI | 运行时 | `acc_p_factory()`, `initialize_events()` |
| `lateness_mode` | CLI | 运行时 | 任务 criticality 设置 |

## 8. 各阶段职责说明

### 8.1 预处理阶段 (preprocess_args)

**职责**: 参数验证和初始化

- 设置 `force_suffix` (强制核心数标识)
- 配置 `enforce_wc` (worst-case 执行时间强制)
- 检查 test_case 兼容性
- 处理 `core_size` 配置 (induced vs specified)

### 8.2 工作负载生成阶段 (build_workload_and_criticality)

**职责**: 生成任务图和进程实例

- 从 profiling 数据生成 DAG 任务图
- 创建 ProcessInt 实例 (包含执行时间分布)
- 设置任务 criticality (hard/soft)
- 计算 hyper_p (超周期)

### 8.3 调度器创建阶段 (create_scheduler_elements_with_config)

**职责**: 初始化调度和监控组件

- 创建 Resource_model_int (资源模型)
- 创建 Scheduler (调度器)
- 创建 Monitor (运行时监控)
- 创建 MsgDispatcher, DataPipe (通信管道)

### 8.4 仿真环境构建阶段 (build_simulation_env)

**职责**: 设置仿真参数

- 计算 num_periods, warmup
- 生成事件迭代器 (到达时间, 截止时间, 负载)
- 设置随机种子

### 8.5 装箱算法阶段 (perform_bin_packing)

**职责**: 核心调度决策

**算法分支**:

1. **scratch**: 从零开始完全动态装箱
2. **guided**: 两阶段引导式混合分配
   - Phase 1: `coleasing_alloc_cluster()` - 空间划分
   - Phase 2: `push_task_into_bins_new()` - 时间窗口

### 8.6 事件驱动仿真阶段 (run_simulation)

**职责**: 运行时调度执行

- 处理器更新循环: update_run → update_ready → sched
- 超周期边界处理: 复制图、更新映射、添加事件
- 统计收集: record_miss, forward_hyperperiod

## 9. 模块依赖关系

```
main_approach.py
    │
    ├── approach_setup.py ─────────────────────────────────────────────┐
    │   │                                                              │
    │   ├── sim_main.py ────────────────────────────────────────────┐  │
    │   │   │                                                        │  │
    │   │   ├── task/task_cfg.py (gen_workloads)                    │  │
    │   │   ├── sched/global_sched.py (coleasing_alloc_cluster)     │  │
    │   │   ├── sched/scheduler_agent.py (Scheduler)                │  │
    │   │   ├── sched/monitor_agent.py (Monitor)                    │  │
    │   │   ├── model/resource_agent.py (Resource_model_int)        │  │
    │   │   ├── model/message/ (MsgDispatcher, DataPipe)            │  │
    │   │   └── paths.py (PathContext)                              │  │
    │   │                                                            │  │
    │   └── approach_initiator.py ───────────────────────────────┐  │  │
    │                                                            │  │  │
    ├── approach_sched.py (PartitionConfig, acc_p_factory)      │  │  │
    │                                                            │  │  │
    ├── approach_sim.py (run_simulation)                        │  │  │
    │   │                                                        │  │  │
    │   ├── approach_def.py (Sen_p, Acc_p, MyGraph) ◄───────────┘  │  │
    │   └── approach_collector.py (StatisticsCollector)            │  │
    │                                                               │  │
    └── utils.py (input_parser, dump_and_check, load_pickle) ◄──────┘  │
                                                                      │
    └── global_var.py (全局常量) ◄─────────────────────────────────────┘
```

## 10. 关键数据结构

| 数据结构 | 定义位置 | 用途 |
|----------|----------|------|
| `ProcessInt` | task/task_agent.py | 进程实例，包含任务属性和执行状态 |
| `SchedulingTableInt` | sched/scheduling_table.py | 调度表，管理时间槽分配 |
| `BinPackConfig` | sched/binpack_config.py | 装箱算法配置封装 |
| `PartitionConfig` | approach_sched.py | 分区配置，包含映射和调度表 |
| `PathContext` | paths.py | 统一路径管理上下文 |
| `MyGraph` | approach_def.py | 逻辑图，管理任务 DAG |
| `StatisticsCollector` | approach_collector.py | 统计收集器 |

---

*文档生成时间: 2026-03-10*
